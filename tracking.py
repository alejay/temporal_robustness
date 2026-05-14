"""
Optional MLflow tracking helpers for local experiment provenance.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable


def ensure_mlflow():
    """Import MLflow lazily and raise a Conda-oriented setup error if missing."""
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow tracking was requested, but `mlflow` is not installed in the "
            "active Conda environment. Install dependencies from environment.yml "
            "or add mlflow to the environment first."
        ) from exc
    return mlflow


def configure_mlflow(tracking_uri: str | None = None):
    """Resolve and configure the active MLflow tracking URI."""
    mlflow = ensure_mlflow()
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    return mlflow


def sanitize_tag_value(value):
    """Convert tag values to MLflow-safe strings."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def log_params_from_namespace(mlflow, cfg, exclude: Iterable[str] = ()):
    """Log argparse namespace fields as MLflow params with JSON fallback."""
    excluded = set(exclude)
    params = {}
    for key, value in sorted(vars(cfg).items()):
        if key in excluded:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
        else:
            params[key] = json.dumps(value, sort_keys=True, default=str)
    if params:
        mlflow.log_params(params)


def _git_cmd(repo_dir: Path, args: list[str]) -> str | None:
    """Run one Git command and return stripped stdout when available."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def collect_git_metadata(repo_dir: Path) -> Dict[str, object]:
    """Collect commit, branch, dirty-state, and diff metadata for provenance."""
    metadata = {
        "git_repo": (repo_dir / ".git").exists(),
        "git_commit": _git_cmd(repo_dir, ["rev-parse", "HEAD"]),
        "git_commit_short": _git_cmd(repo_dir, ["rev-parse", "--short", "HEAD"]),
        "git_branch": _git_cmd(repo_dir, ["branch", "--show-current"]),
        "git_status_short": _git_cmd(repo_dir, ["status", "--short"]),
        "git_diff": _git_cmd(repo_dir, ["diff", "--no-ext-diff"]),
        "git_diff_cached": _git_cmd(repo_dir, ["diff", "--cached", "--no-ext-diff"]),
    }
    status = metadata.get("git_status_short") or ""
    metadata["git_dirty"] = bool(status.strip())
    metadata["git_has_commit"] = bool(metadata.get("git_commit"))
    return metadata


def write_provenance_files(out_dir: Path, payloads: Dict[str, object]) -> Dict[str, Path]:
    """Write structured provenance payloads alongside run outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in payloads.items():
        if payload is None:
            continue
        path = out_dir / name
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        paths[name] = path
    return paths


def build_data_provenance(cfg, meta: dict, splits: dict | None = None, split_name: str | None = None):
    """Assemble dataset/task/path metadata for tracked runs and saved artifacts."""
    payload = {
        "dataset": meta.get("dataset_name"),
        "task": meta.get("task_name"),
        "raw_dir": str(Path(cfg.raw_dir).resolve()),
        "results_dir": str(Path(cfg.results_dir).resolve()),
        "feature_count": meta.get("n_features"),
        "feature_names": meta.get("feature_names"),
        "normalizer_present": bool(meta.get("normalizer")),
        "label_prevalence": meta.get("label_prevalence"),
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "n_test": meta.get("n_test"),
        "max_patients": cfg.max_patients,
    }
    if splits is not None:
        payload["split_sizes"] = {name: len(ds) for name, ds in splits.items()}
    if split_name is not None:
        payload["active_split"] = split_name
    return payload


def default_run_name(stage: str, cfg, names: Iterable[str]) -> str:
    """Build the default run name from stage, dataset/task, methods, and seed."""
    selected = "-".join(names) if names else "none"
    return f"{stage}-{cfg.dataset}-{cfg.task}-{selected}-seed{cfg.seed}"


def set_common_tags(mlflow, stage: str, cfg, extra: Dict[str, object] | None = None):
    """Attach stable environment and task tags to the active MLflow run."""
    tags = {
        "stage": stage,
        "dataset": cfg.dataset,
        "task": cfg.task,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    if extra:
        tags.update(extra)
    clean_tags = {k: sanitize_tag_value(v) for k, v in tags.items() if sanitize_tag_value(v) is not None}
    if clean_tags:
        mlflow.set_tags(clean_tags)


def log_git_tags(mlflow, git_meta: Dict[str, object]):
    """Attach Git provenance tags to the active MLflow run."""
    tags = {
        "git.repo_present": git_meta.get("git_repo"),
        "git.has_commit": git_meta.get("git_has_commit"),
        "git.commit": git_meta.get("git_commit"),
        "git.commit_short": git_meta.get("git_commit_short"),
        "git.branch": git_meta.get("git_branch"),
        "git.dirty": git_meta.get("git_dirty"),
    }
    clean_tags = {k: sanitize_tag_value(v) for k, v in tags.items() if sanitize_tag_value(v) is not None}
    if clean_tags:
        mlflow.set_tags(clean_tags)


@contextmanager
def start_run_if_enabled(enabled: bool, experiment_name: str, run_name: str, tracking_uri: str | None = None, nested: bool = False):
    """Context manager that yields an MLflow client only when tracking is enabled."""
    if not enabled:
        yield None
        return
    mlflow = configure_mlflow(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested):
        yield mlflow


def maybe_log_artifacts(mlflow, path: Path, artifact_path: str | None = None):
    """Log one file or directory as MLflow artifacts when tracking is active."""
    if mlflow is None or not path.exists():
        return
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def maybe_log_metrics(mlflow, metrics: Dict[str, object]):
    """Log only numeric metrics, silently skipping missing or non-numeric values."""
    if mlflow is None:
        return
    clean = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            clean[key] = float(value)
        except (TypeError, ValueError):
            continue
    if clean:
        mlflow.log_metrics(clean)
