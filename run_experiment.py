"""Run training, perturbation tests, and stress tests as one experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULTS = dict(
    raw_dir="data/raw",
    results_dir="results",
    run_subdir="",
    max_patients=None,
    batch_size=32,
    epochs=30,
    lr=1e-3,
    patience=5,
    hidden_dim=64,
    latent_dim=32,
    n_ode_samples=20,
    transformer_heads=4,
    transformer_layers=2,
    state_dim=32,
    landmark_grid_hours=6.0,
    n_bootstrap_samples=10,
    dataset="physionet2012",
    task="mortality",
    task_mode="patient_binary",
    prediction_horizon_hours=6.0,
    label_window_hours=24.0,
    split="test",
    seed=42,
    models="grud,latent_ode,transformer,state_space,landmarking",
    baselines="qsofa,sirs,sofa",
    retention_levels="1.0,0.75,0.5,0.25,0.1",
    n_repeats=10,
    timestamp_factors="0.5,1.0,2.0,4.0",
    fast=False,
    skip_existing=False,
    skip_training=False,
    skip_perturbation=False,
    skip_stress=False,
    track=False,
    tracking_uri="file:./mlruns",
    experiment_name="temporal_robustness",
)


def parse_args():
    """Build the experiment-runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run training, perturbation tests, and stress tests sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Paths")
    io_group.add_argument("--raw_dir", type=str, default=DEFAULTS["raw_dir"],
                          help="Directory containing raw dataset files or automatic downloads.")
    io_group.add_argument("--results_dir", type=str, default=DEFAULTS["results_dir"],
                          help="Base results directory for checkpoints, outputs, and provenance.")
    io_group.add_argument("--run_subdir", type=str, default=DEFAULTS["run_subdir"],
                          help="Optional isolated run root under results_dir used by all three stages.")

    data_group = parser.add_argument_group("Data And Task")
    data_group.add_argument("--dataset", type=str, default=DEFAULTS["dataset"],
                            help="Dataset adapter to load.")
    data_group.add_argument("--task", type=str, default=DEFAULTS["task"],
                            help="Task adapter to apply after dataset loading.")
    data_group.add_argument("--task_mode", type=str, default=DEFAULTS["task_mode"],
                            help="Task labeling mode used by the task adapter.")
    data_group.add_argument("--prediction_horizon_hours", type=float,
                            default=DEFAULTS["prediction_horizon_hours"],
                            help="Prediction horizon metadata in hours for the selected task.")
    data_group.add_argument("--label_window_hours", type=float,
                            default=DEFAULTS["label_window_hours"],
                            help="Label window metadata in hours for the selected task.")
    data_group.add_argument("--max_patients", type=int, default=DEFAULTS["max_patients"],
                            help="Optional cap on loaded patients.")
    data_group.add_argument("--split", type=str, default=DEFAULTS["split"],
                            help="Dense split used by the stress-test stage.")
    data_group.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                            help="Random seed shared across all stages.")

    selection_group = parser.add_argument_group("Method Selection")
    selection_group.add_argument("--models", type=str, default=DEFAULTS["models"],
                                 help="Comma-separated learned models to train and evaluate.")
    selection_group.add_argument("--baselines", type=str, default=DEFAULTS["baselines"],
                                 help="Comma-separated baseline scores to train and evaluate.")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"],
                             help="Mini-batch size for learned-model training and inference loops.")
    train_group.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                             help="Maximum number of training epochs for learned models.")
    train_group.add_argument("--lr", type=float, default=DEFAULTS["lr"],
                             help="Learning rate for learned-model optimizers.")
    train_group.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                             help="Early-stopping patience measured in validation epochs.")
    train_group.add_argument("--skip_existing", action="store_true",
                             default=DEFAULTS["skip_existing"],
                             help="Skip methods whose checkpoints already exist in the active run root.")

    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"],
                             help="Shared hidden dimension for GRU-D, transformer, and latent ODE components.")
    model_group.add_argument("--latent_dim", type=int, default=DEFAULTS["latent_dim"],
                             help="Latent state dimension for latent ODE.")
    model_group.add_argument("--n_ode_samples", type=int, default=DEFAULTS["n_ode_samples"],
                             help="Number of posterior samples used by latent ODE prediction routines.")
    model_group.add_argument("--transformer_heads", type=int, default=DEFAULTS["transformer_heads"],
                             help="Number of attention heads in the transformer encoder.")
    model_group.add_argument("--transformer_layers", type=int, default=DEFAULTS["transformer_layers"],
                             help="Number of transformer encoder layers.")
    model_group.add_argument("--state_dim", type=int, default=DEFAULTS["state_dim"],
                             help="Latent state dimension for the state-space model.")
    model_group.add_argument("--landmark_grid_hours", type=float,
                             default=DEFAULTS["landmark_grid_hours"],
                             help="Spacing between landmark times for landmarking.")
    model_group.add_argument("--n_bootstrap_samples", type=int,
                             default=DEFAULTS["n_bootstrap_samples"],
                             help="Number of bootstrap logistic models per landmark time.")

    test_group = parser.add_argument_group("Test Configuration")
    test_group.add_argument("--retention_levels", type=str, default=DEFAULTS["retention_levels"],
                            help="Comma-separated retention fractions for thinning-based tests.")
    test_group.add_argument("--n_repeats", type=int, default=DEFAULTS["n_repeats"],
                            help="Number of repeated random thinnings per retention level.")
    test_group.add_argument("--timestamp_factors", type=str, default=DEFAULTS["timestamp_factors"],
                            help="Comma-separated multiplicative factors for timestamp stretching.")
    test_group.add_argument("--fast", action="store_true", default=DEFAULTS["fast"],
                            help="Run reduced configurations for quick verification.")

    stage_group = parser.add_argument_group("Stage Selection")
    stage_group.add_argument("--skip_training", action="store_true",
                             default=DEFAULTS["skip_training"],
                             help="Skip the training stage.")
    stage_group.add_argument("--skip_perturbation", action="store_true",
                             default=DEFAULTS["skip_perturbation"],
                             help="Skip the perturbation-test stage.")
    stage_group.add_argument("--skip_stress", action="store_true",
                             default=DEFAULTS["skip_stress"],
                             help="Skip the stress-test stage.")

    tracking_group = parser.add_argument_group("Tracking")
    tracking_group.add_argument("--track", action="store_true", default=DEFAULTS["track"],
                                help="Enable optional MLflow tracking and provenance logging.")
    tracking_group.add_argument("--tracking_uri", type=str, default=DEFAULTS["tracking_uri"],
                                help="MLflow tracking URI, typically a local file store.")
    tracking_group.add_argument("--experiment_name", type=str, default=DEFAULTS["experiment_name"],
                                help="MLflow experiment name used when --track is enabled.")

    return parser.parse_args()


def _add_common_args(cmd: list[str], cfg, include_baselines: bool = True) -> list[str]:
    """Append arguments shared by all three stage scripts."""
    cmd.extend([
        "--raw_dir", cfg.raw_dir,
        "--results_dir", cfg.results_dir,
        "--dataset", cfg.dataset,
        "--task", cfg.task,
        "--task_mode", cfg.task_mode,
        "--prediction_horizon_hours", str(cfg.prediction_horizon_hours),
        "--label_window_hours", str(cfg.label_window_hours),
        "--seed", str(cfg.seed),
        "--models", cfg.models,
    ])
    if include_baselines:
        cmd.extend(["--baselines", cfg.baselines])
    if cfg.run_subdir:
        cmd.extend(["--run_subdir", cfg.run_subdir])
    if cfg.max_patients is not None:
        cmd.extend(["--max_patients", str(cfg.max_patients)])
    if cfg.fast:
        cmd.append("--fast")
    if cfg.track:
        cmd.extend([
            "--track",
            "--tracking_uri", cfg.tracking_uri,
            "--experiment_name", cfg.experiment_name,
        ])
    return cmd


def build_training_command(cfg, repo_root: Path) -> list[str]:
    """Build the training stage command."""
    cmd = [sys.executable, str(repo_root / "training.py")]
    _add_common_args(cmd, cfg, include_baselines=True)
    cmd.extend([
        "--batch_size", str(cfg.batch_size),
        "--epochs", str(cfg.epochs),
        "--lr", str(cfg.lr),
        "--patience", str(cfg.patience),
        "--hidden_dim", str(cfg.hidden_dim),
        "--latent_dim", str(cfg.latent_dim),
        "--n_ode_samples", str(cfg.n_ode_samples),
        "--transformer_heads", str(cfg.transformer_heads),
        "--transformer_layers", str(cfg.transformer_layers),
        "--state_dim", str(cfg.state_dim),
        "--landmark_grid_hours", str(cfg.landmark_grid_hours),
        "--n_bootstrap_samples", str(cfg.n_bootstrap_samples),
    ])
    if cfg.skip_existing:
        cmd.append("--skip_existing")
    return cmd


def build_perturbation_command(cfg, repo_root: Path) -> list[str]:
    """Build the perturbation-test stage command."""
    cmd = [sys.executable, str(repo_root / "scripts" / "run_perturbation_test.py")]
    _add_common_args(cmd, cfg, include_baselines=False)
    cmd.extend([
        "--batch_size", str(cfg.batch_size),
        "--hidden_dim", str(cfg.hidden_dim),
        "--latent_dim", str(cfg.latent_dim),
        "--n_ode_samples", str(cfg.n_ode_samples),
        "--transformer_heads", str(cfg.transformer_heads),
        "--transformer_layers", str(cfg.transformer_layers),
        "--state_dim", str(cfg.state_dim),
        "--landmark_grid_hours", str(cfg.landmark_grid_hours),
        "--n_bootstrap_samples", str(cfg.n_bootstrap_samples),
        "--retention_levels", cfg.retention_levels,
        "--n_repeats", str(cfg.n_repeats),
        "--timestamp_factors", cfg.timestamp_factors,
    ])
    return cmd


def build_stress_command(cfg, repo_root: Path) -> list[str]:
    """Build the stress-test stage command."""
    cmd = [sys.executable, str(repo_root / "scripts" / "run_stress_tests.py")]
    _add_common_args(cmd, cfg, include_baselines=True)
    cmd.extend([
        "--batch_size", str(cfg.batch_size),
        "--retention_levels", cfg.retention_levels,
        "--n_repeats", str(cfg.n_repeats),
        "--split", cfg.split,
    ])
    return cmd


def run_stage(name: str, cmd: list[str], repo_root: Path) -> None:
    """Run one stage command and stop immediately on failure."""
    print(f"\n== {name} ==")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def main():
    """Run the configured experiment stages sequentially."""
    cfg = parse_args()
    repo_root = Path(__file__).resolve().parent

    if not cfg.skip_training:
        run_stage("Training", build_training_command(cfg, repo_root), repo_root)
    if not cfg.skip_perturbation:
        run_stage("Perturbation Test", build_perturbation_command(cfg, repo_root), repo_root)
    if not cfg.skip_stress:
        run_stage("Stress Test", build_stress_command(cfg, repo_root), repo_root)

    print("\nExperiment run complete.")


if __name__ == "__main__":
    main()
