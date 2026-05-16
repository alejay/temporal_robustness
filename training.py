"""
Training entry point for temporal perturbation experiments.

This public v1 runner prepares task-specific datasets, trains selected models
and baselines on the dense training split, and saves checkpoints for the
perturbation scripts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from experiment_registry import (
    BASELINE_REGISTRY,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    TASK_REGISTRY,
    build_baseline,
    checkpoint_path,
)
from models.landmarking import LandmarkingModel
from tracking import (
    build_data_provenance,
    collect_git_metadata,
    default_run_name,
    log_git_tags,
    log_params_from_namespace,
    maybe_log_artifacts,
    maybe_log_metrics,
    set_common_tags,
    start_run_if_enabled,
    write_provenance_files,
)


DEFAULTS = dict(
    raw_dir="data/raw",
    results_dir="results",
    run_subdir="",
    max_patients=None,
    batch_size=32,
    epochs=30,
    lr=1e-3,
    hidden_dim=64,
    latent_dim=32,
    n_ode_samples=20,
    seed=42,
    patience=5,
    models="grud,latent_ode,transformer,state_space,landmarking",
    baselines="qsofa,sirs,sofa",
    dataset="physionet2012",
    task="mortality",
    prediction_horizon_hours=6.0,
    label_window_hours=24.0,
    task_mode="patient_binary",
    transformer_heads=4,
    transformer_layers=2,
    state_dim=32,
    landmark_grid_hours=6.0,
    n_bootstrap_samples=10,
    fast=False,
    skip_existing=False,
    track=False,
    tracking_uri="file:./mlruns",
    experiment_name="temporal_robustness",
    run_name="",
)


def parse_args():
    """Build the training CLI with grouped, user-facing argument descriptions."""
    parser = argparse.ArgumentParser(
        description="Train learned models and clinical baselines for perturbation tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Paths")
    io_group.add_argument(
        "--raw_dir",
        type=str,
        default=DEFAULTS["raw_dir"],
        help="Directory containing raw dataset files or automatic downloads.",
    )
    io_group.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULTS["results_dir"],
        help="Directory where checkpoints and provenance files are written.",
    )
    io_group.add_argument(
        "--run_subdir",
        type=str,
        default=DEFAULTS["run_subdir"],
        help="Optional isolated run root under results_dir for checkpoints and provenance.",
    )

    data_group = parser.add_argument_group("Data And Task")
    data_group.add_argument(
        "--dataset",
        type=str,
        default=DEFAULTS["dataset"],
        help="Dataset adapter to load. Public v1 uses physionet2012.",
    )
    data_group.add_argument(
        "--task",
        type=str,
        default=DEFAULTS["task"],
        help="Task adapter to apply after dataset loading.",
    )
    data_group.add_argument(
        "--task_mode",
        type=str,
        default=DEFAULTS["task_mode"],
        help="Task labeling mode used by the task adapter.",
    )
    data_group.add_argument(
        "--prediction_horizon_hours",
        type=float,
        default=DEFAULTS["prediction_horizon_hours"],
        help="Prediction horizon metadata in hours for the selected task.",
    )
    data_group.add_argument(
        "--label_window_hours",
        type=float,
        default=DEFAULTS["label_window_hours"],
        help="Label window metadata in hours for the selected task.",
    )
    data_group.add_argument(
        "--max_patients",
        type=int,
        default=DEFAULTS["max_patients"],
        help="Optional cap on loaded patients, mainly for smoke or debug runs.",
    )
    data_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Global random seed for splitting and training setup.",
    )

    selection_group = parser.add_argument_group("Method Selection")
    selection_group.add_argument(
        "--models",
        type=str,
        default=DEFAULTS["models"],
        help="Comma-separated learned models to train.",
    )
    selection_group.add_argument(
        "--baselines",
        type=str,
        default=DEFAULTS["baselines"],
        help="Comma-separated clinical baseline scores to fit.",
    )
    selection_group.add_argument(
        "--skip_existing",
        action="store_true",
        default=DEFAULTS["skip_existing"],
        help="Skip methods whose checkpoint already exists in the active run root.",
    )

    optim_group = parser.add_argument_group("Training")
    optim_group.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULTS["batch_size"],
        help="Mini-batch size for learned models.",
    )
    optim_group.add_argument(
        "--epochs",
        type=int,
        default=DEFAULTS["epochs"],
        help="Maximum number of training epochs for learned models.",
    )
    optim_group.add_argument(
        "--lr",
        type=float,
        default=DEFAULTS["lr"],
        help="Learning rate for learned-model optimizers.",
    )
    optim_group.add_argument(
        "--patience",
        type=int,
        default=DEFAULTS["patience"],
        help="Early-stopping patience measured in validation epochs.",
    )
    optim_group.add_argument(
        "--fast",
        action="store_true",
        default=DEFAULTS["fast"],
        help="Run a reduced configuration for quick verification.",
    )

    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=DEFAULTS["hidden_dim"],
        help="Shared hidden dimension for GRU-D, transformer, and latent ODE components.",
    )
    model_group.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Latent state dimension for the latent ODE model.",
    )
    model_group.add_argument(
        "--n_ode_samples",
        type=int,
        default=DEFAULTS["n_ode_samples"],
        help="Number of posterior samples used by latent ODE prediction routines.",
    )
    model_group.add_argument(
        "--transformer_heads",
        type=int,
        default=DEFAULTS["transformer_heads"],
        help="Number of attention heads in the transformer encoder.",
    )
    model_group.add_argument(
        "--transformer_layers",
        type=int,
        default=DEFAULTS["transformer_layers"],
        help="Number of transformer encoder layers.",
    )
    model_group.add_argument(
        "--state_dim",
        type=int,
        default=DEFAULTS["state_dim"],
        help="Latent state dimension for the state-space model.",
    )
    model_group.add_argument(
        "--landmark_grid_hours",
        type=float,
        default=DEFAULTS["landmark_grid_hours"],
        help="Spacing between landmark times for the landmarking baseline.",
    )
    model_group.add_argument(
        "--n_bootstrap_samples",
        type=int,
        default=DEFAULTS["n_bootstrap_samples"],
        help="Number of bootstrap logistic models per landmark time.",
    )

    tracking_group = parser.add_argument_group("Tracking")
    tracking_group.add_argument(
        "--track",
        action="store_true",
        default=DEFAULTS["track"],
        help="Enable optional MLflow tracking and provenance logging.",
    )
    tracking_group.add_argument(
        "--tracking_uri",
        type=str,
        default=DEFAULTS["tracking_uri"],
        help="MLflow tracking URI, typically a local file store.",
    )
    tracking_group.add_argument(
        "--experiment_name",
        type=str,
        default=DEFAULTS["experiment_name"],
        help="MLflow experiment name used when --track is enabled.",
    )
    tracking_group.add_argument(
        "--run_name",
        type=str,
        default=DEFAULTS["run_name"],
        help="Optional MLflow run name. Defaults to an auto-generated name.",
    )

    return parser.parse_args()


def parse_name_list(value: str):
    """Parse a comma-separated CLI list into non-empty names."""
    return [item.strip() for item in value.split(",") if item.strip()]


def get_device():
    """Choose the preferred torch device available in the current environment."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size, shuffle, seed: int | None = None):
    """Construct the padded DataLoader shared by model trainers."""
    from datasets.common import pad_collate

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=pad_collate,
        num_workers=0,
    )


def save_checkpoint(method_name, method_type, model, out_dir):
    """Persist one trained model or baseline using the repo naming convention."""
    suffix = ".pkl" if (method_type == "baseline" or isinstance(model, LandmarkingModel)) else ".pt"
    path = checkpoint_path(method_name, method_type, out_dir, suffix=suffix)
    if suffix == ".pkl":
        if hasattr(model, "save"):
            model.save(path)
        else:
            with open(path, "wb") as handle:
                pickle.dump(model, handle)
    else:
        torch.save(model.state_dict(), path)
    return path


def checkpoint_exists(method_name, method_type, out_dir):
    """Check whether the expected checkpoint file already exists."""
    suffix = ".pkl" if method_type == "baseline" else MODEL_REGISTRY[method_name].checkpoint_suffix
    return checkpoint_path(method_name, method_type, out_dir, suffix=suffix).exists()


def train_methods(cfg, splits, meta, device, out_dir, selected_models, selected_baselines, provenance_dir, git_meta):
    """Train selected methods and optionally attach nested MLflow runs."""
    train_loader = make_loader(splits["train"], cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(splits["val"], cfg.batch_size, shuffle=False, seed=cfg.seed)

    summary = {"models": {}, "baselines": {}}

    for model_name in selected_models:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
        if cfg.skip_existing and checkpoint_exists(model_name, "model", out_dir):
            print(f"\n── Skipping {model_name} (checkpoint exists) ───────────────")
            continue

        print(f"\n── Training {model_name} ─────────────────────────────────────")
        spec = MODEL_REGISTRY[model_name]
        child_name = f"train-{model_name}"
        # Nested runs keep per-method checkpoints and metrics attributable even
        # when one command trains several methods in the same top-level run.
        with start_run_if_enabled(
            cfg.track,
            cfg.experiment_name,
            child_name,
            tracking_uri=cfg.tracking_uri,
            nested=True,
        ) as child_mlflow:
            if child_mlflow is not None:
                set_common_tags(child_mlflow, "training", cfg, {"component": model_name, "method_type": "model"})
                log_git_tags(child_mlflow, git_meta)
                log_params_from_namespace(child_mlflow, cfg, exclude={"run_name", "experiment_name", "tracking_uri", "track"})
            model = spec.builder(cfg, meta, device)
            metric = spec.trainer(model, train_loader, val_loader, device, cfg, meta)
            path = save_checkpoint(model_name, "model", model, out_dir)
            maybe_log_metrics(child_mlflow, {"val_metric": metric})
            if child_mlflow is not None:
                child_mlflow.log_param("checkpoint_path", str(path.resolve()))
                maybe_log_artifacts(child_mlflow, path.parent, artifact_path=f"checkpoints/{model_name}")
                maybe_log_artifacts(child_mlflow, provenance_dir, artifact_path="provenance")
            summary["models"][model_name] = {
                "checkpoint": str(path),
                "val_metric": None if metric is None else float(metric),
            }
            if metric is not None:
                print(f"  Validation metric: {metric:.4f}")
            print(f"  Saved checkpoint: {path}")

    for baseline_name in selected_baselines:
        if baseline_name not in BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline '{baseline_name}'. Available: {sorted(BASELINE_REGISTRY)}"
            )
        if cfg.skip_existing and checkpoint_exists(baseline_name, "baseline", out_dir):
            print(f"\n── Skipping baseline {baseline_name} (checkpoint exists) ───")
            continue

        print(f"\n── Fitting baseline {baseline_name} ───────────────────────────")
        child_name = f"train-{baseline_name}"
        with start_run_if_enabled(
            cfg.track,
            cfg.experiment_name,
            child_name,
            tracking_uri=cfg.tracking_uri,
            nested=True,
        ) as child_mlflow:
            if child_mlflow is not None:
                set_common_tags(child_mlflow, "training", cfg, {"component": baseline_name, "method_type": "baseline"})
                log_git_tags(child_mlflow, git_meta)
                log_params_from_namespace(child_mlflow, cfg, exclude={"run_name", "experiment_name", "tracking_uri", "track"})
            baseline = build_baseline(baseline_name)
            baseline.fit(splits["train"], meta)
            path = save_checkpoint(baseline_name, "baseline", baseline, out_dir)
            if child_mlflow is not None:
                child_mlflow.log_param("checkpoint_path", str(path.resolve()))
                maybe_log_artifacts(child_mlflow, path.parent, artifact_path=f"checkpoints/{baseline_name}")
                maybe_log_artifacts(child_mlflow, provenance_dir, artifact_path="provenance")
            summary["baselines"][baseline_name] = {
                "checkpoint": str(path),
                "val_metric": None,
            }
            print(f"  Saved checkpoint: {path}")

    return summary


def main():
    """Run the end-to-end training workflow and write provenance artifacts."""
    cfg = parse_args()
    if cfg.fast:
        cfg.max_patients = cfg.max_patients or 300
        cfg.epochs = min(cfg.epochs, 5)
        cfg.n_ode_samples = min(cfg.n_ode_samples, 5)
        cfg.n_bootstrap_samples = min(cfg.n_bootstrap_samples, 3)
        print("Fast mode: reduced patients, epochs, ODE samples, and bootstrap samples.")

    selected_models = parse_name_list(cfg.models)
    selected_baselines = parse_name_list(cfg.baselines)
    out_dir = Path(cfg.results_dir)
    if cfg.run_subdir:
        out_dir = out_dir / cfg.run_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    set_global_seed(cfg.seed)

    print("\n── Loading data ────────────────────────────────────────────────")
    raw_splits, raw_regimes, meta = DATASET_REGISTRY[cfg.dataset].load(cfg)
    splits, regimes, meta = TASK_REGISTRY[cfg.task].apply(raw_splits, raw_regimes, meta, cfg)
    print(f"  Dataset: {meta['dataset_name']} | Task: {meta['task_name']}")
    print(
        f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | "
        f"Test: {len(splits['test'])} | Features: {meta['n_features']}"
    )

    git_meta = collect_git_metadata(Path(__file__).resolve().parent)
    provenance_dir = out_dir / "provenance" / "training"
    write_provenance_files(
        provenance_dir,
        {
            "config.json": vars(cfg),
            "git.json": git_meta,
            "git_status.txt": git_meta.get("git_status_short") or "",
            "git_diff.patch": git_meta.get("git_diff") or "",
            "git_diff_cached.patch": git_meta.get("git_diff_cached") or "",
            "data.json": build_data_provenance(cfg, meta, splits=splits),
        },
    )

    run_name = cfg.run_name or default_run_name(
        "training",
        cfg,
        [*selected_models, *selected_baselines],
    )
    with start_run_if_enabled(
        cfg.track,
        cfg.experiment_name,
        run_name,
        tracking_uri=cfg.tracking_uri,
    ) as mlflow:
        if mlflow is not None:
            set_common_tags(mlflow, "training", cfg)
            log_git_tags(mlflow, git_meta)
            log_params_from_namespace(mlflow, cfg, exclude={"run_name", "experiment_name", "tracking_uri", "track"})
            mlflow.log_param("results_dir_resolved", str(out_dir.resolve()))
            mlflow.log_param("run_root_resolved", str(out_dir.resolve()))
            maybe_log_artifacts(mlflow, provenance_dir, artifact_path="provenance")

        summary = train_methods(
            cfg,
            splits,
            meta,
            device,
            out_dir,
            selected_models,
            selected_baselines,
            provenance_dir,
            git_meta,
        )
        summary_path = provenance_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        maybe_log_artifacts(mlflow, summary_path.parent, artifact_path="provenance")

    print("\nDone.")
    print(f"Checkpoints directory: {out_dir}")
    if not summary["models"] and not summary["baselines"]:
        print("No new checkpoints were created.")


if __name__ == "__main__":
    main()
