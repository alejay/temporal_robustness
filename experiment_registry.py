"""
Registry-driven dataset/task/model/baseline orchestration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from datasets.common import (
    PatientDataset,
    clone_records,
)
from datasets.physionet2012 import (
    load_physionet_2012_source,
)
from models.grud import GRUD, train_grud
from models.landmarking import LandmarkingModel, train_landmarking
from models.state_space import StateSpaceRiskModel, train_state_space
from models.transformer import TimeAwareTransformer, train_transformer
from models.latent_ode import LatentODE, train_latent_ode
from baselines.guideline import QSOFABaseline, SIRSBaseline, SOFABaseline


class DatasetAdapter:
    """Abstract dataset loader used by the registry-based workflow."""
    name = ""

    def load(self, cfg):
        raise NotImplementedError


class PhysioNet2012Adapter(DatasetAdapter):
    name = "physionet2012"

    def load(self, cfg):
        """Load raw PhysioNet-derived splits plus dataset metadata."""
        return load_physionet_2012_source(
            raw_dir=cfg.raw_dir,
            max_patients=cfg.max_patients,
            seed=cfg.seed,
        )


class TaskAdapter:
    """Abstract task wrapper that annotates generic trajectories for one task."""
    name = ""

    def apply(self, raw_splits, raw_regimes, meta, cfg):
        raise NotImplementedError


def _annotate_mortality_records(records, labels, meta):
    """Attach mortality labels and prediction metadata to raw patient records."""
    out = []
    for record in clone_records(records):
        times = record["times"]
        eligible_mask = record.get(
            "eligible_mask",
            record["mask"].sum(axis=1) > 0,
        ).astype(bool)
        record["label"] = int(labels.get(record["patient_id"], 0))
        record["label_time"] = float(times[-1]) if len(times) else 0.0
        record["prediction_times"] = times.copy()
        record["eligible_mask"] = eligible_mask
        record["dataset_name"] = meta["dataset_name"]
        record["task_name"] = "mortality"
        out.append(record)
    return out


class MortalityTaskAdapter(TaskAdapter):
    name = "mortality"

    def apply(self, raw_splits, raw_regimes, meta, cfg):
        """Wrap raw splits/regimes as mortality datasets with task metadata."""
        labels = meta["outcome_labels"]
        task_meta = deepcopy(meta)
        task_meta.update({
            "task_name": self.name,
            "task_mode": cfg.task_mode,
            "prediction_horizon_hours": cfg.prediction_horizon_hours,
            "label_window_hours": cfg.label_window_hours,
        })
        splits = {}
        regimes = {}

        for split_name, records in raw_splits.items():
            task_records = _annotate_mortality_records(records, labels, task_meta)
            splits[split_name] = PatientDataset(task_records, labels=labels, meta=task_meta)

        for regime_name, split_map in raw_regimes.items():
            regimes[regime_name] = {}
            for split_name, records in split_map.items():
                task_records = _annotate_mortality_records(records, labels, task_meta)
                regimes[regime_name][split_name] = PatientDataset(
                    task_records,
                    labels=labels,
                    meta=task_meta,
                )
        return splits, regimes, task_meta


@dataclass
class ModelSpec:
    """Registry entry describing how to build, train, and save one model family."""
    builder: Callable
    trainer: Callable
    checkpoint_suffix: str = ".pt"


def build_model_registry():
    """Assemble supported learned models with builder/trainer metadata."""
    return {
        "grud": ModelSpec(
            builder=lambda cfg, meta, device: GRUD(
                input_size=meta["n_features"],
                hidden_size=cfg.hidden_dim,
                dropout=0.3,
            ).to(device),
            trainer=lambda model, train_loader, val_loader, device, cfg, meta: train_grud(
                model,
                train_loader,
                val_loader,
                device,
                n_epochs=cfg.epochs,
                lr=cfg.lr,
                pos_weight=(1 - meta["label_prevalence"]) / max(meta["label_prevalence"], 1e-4),
                patience=cfg.patience,
            ),
        ),
        "latent_ode": ModelSpec(
            builder=lambda cfg, meta, device: _build_latent_ode(cfg, meta, device),
            trainer=lambda model, train_loader, val_loader, device, cfg, meta: _train_latent_ode(
                model,
                train_loader,
                val_loader,
                device,
                cfg,
            ),
        ),
        "transformer": ModelSpec(
            builder=lambda cfg, meta, device: TimeAwareTransformer(
                input_size=meta["n_features"],
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.transformer_heads,
                num_layers=cfg.transformer_layers,
                dropout=0.2,
            ).to(device),
            trainer=lambda model, train_loader, val_loader, device, cfg, meta: train_transformer(
                model,
                train_loader,
                val_loader,
                device,
                n_epochs=cfg.epochs,
                lr=cfg.lr,
                patience=cfg.patience,
            ),
        ),
        "state_space": ModelSpec(
            builder=lambda cfg, meta, device: StateSpaceRiskModel(
                input_size=meta["n_features"],
                state_dim=cfg.state_dim,
            ).to(device),
            trainer=lambda model, train_loader, val_loader, device, cfg, meta: train_state_space(
                model,
                train_loader,
                val_loader,
                device,
                n_epochs=cfg.epochs,
                lr=cfg.lr,
                patience=cfg.patience,
            ),
        ),
        "landmarking": ModelSpec(
            builder=lambda cfg, meta, device: LandmarkingModel(
                landmark_grid_hours=cfg.landmark_grid_hours,
                n_bootstrap_samples=cfg.n_bootstrap_samples,
            ),
            trainer=lambda model, train_loader, val_loader, device, cfg, meta: train_landmarking(
                model,
                train_loader.dataset if hasattr(train_loader, "dataset") else train_loader,
                val_loader.dataset if hasattr(val_loader, "dataset") else val_loader,
            ),
            checkpoint_suffix=".pkl",
        ),
    }


def build_baseline_registry():
    """Assemble supported clinical baseline-score implementations."""
    return {
        "qsofa": QSOFABaseline,
        "sirs": SIRSBaseline,
        "sofa": SOFABaseline,
    }


DATASET_REGISTRY: Dict[str, DatasetAdapter] = {
    "physionet2012": PhysioNet2012Adapter(),
}

TASK_REGISTRY: Dict[str, TaskAdapter] = {
    "mortality": MortalityTaskAdapter(),
}

MODEL_REGISTRY = build_model_registry()
BASELINE_REGISTRY = build_baseline_registry()


def checkpoint_path(method_name: str, method_type: str, out_dir: Path, suffix: str = None):
    """Return the canonical checkpoint path for a trained method."""
    if suffix is None:
        suffix = ".pkl" if method_type == "baseline" else ".pt"
    return out_dir / f"{method_type}_{method_name}{suffix}"


def build_baseline(name: str):
    """Instantiate one baseline class from the registry by name."""
    baseline_cls = BASELINE_REGISTRY[name]
    return baseline_cls()


def _build_latent_ode(cfg, meta, device):
    """Construct the latent ODE model."""
    return LatentODE(
        input_size=meta["n_features"],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        n_samples=cfg.n_ode_samples,
    ).to(device)


def _train_latent_ode(model, train_loader, val_loader, device, cfg):
    """Train the latent ODE model through its dedicated trainer."""
    return train_latent_ode(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=cfg.epochs,
        lr=cfg.lr,
        patience=cfg.patience,
    )
