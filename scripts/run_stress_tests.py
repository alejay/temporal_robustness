"""
Run baseline availability and robustness stress tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnostics.stress_test import (
    run_all_stress_tests,
    save_stress_test_outputs,
)
from diagnostics.perturbation_test import parse_float_list
from experiment_registry import (
    BASELINE_REGISTRY,
    DATASET_REGISTRY,
    TASK_REGISTRY,
    checkpoint_path,
)
from tracking import (
    build_data_provenance,
    collect_git_metadata,
    default_run_name,
    log_git_tags,
    log_params_from_namespace,
    maybe_log_metrics,
    maybe_log_artifacts,
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
    seed=42,
    baselines="qsofa,sirs,sofa",
    dataset="physionet2012",
    task="mortality",
    prediction_horizon_hours=6.0,
    label_window_hours=24.0,
    task_mode="patient_binary",
    retention_levels="1.0,0.75,0.5,0.25,0.1",
    n_repeats=10,
    split="test",
    fast=False,
    track=False,
    tracking_uri="file:./mlruns",
    experiment_name="temporal_robustness",
    run_name="",
)


def parse_args():
    """Build the stress-test CLI with grouped, user-facing descriptions."""
    parser = argparse.ArgumentParser(
        description="Run baseline availability and robustness stress tests.",
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
        help="Base results directory containing baseline checkpoints and stress-test outputs.",
    )
    io_group.add_argument(
        "--run_subdir",
        type=str,
        default=DEFAULTS["run_subdir"],
        help="Optional isolated run root under results_dir for checkpoints, outputs, and provenance.",
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
        "--split",
        type=str,
        default=DEFAULTS["split"],
        help="Dense split to evaluate, usually test.",
    )
    data_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Random seed used for thinning and repeated availability tests.",
    )

    selection_group = parser.add_argument_group("Baseline Selection")
    selection_group.add_argument(
        "--baselines",
        type=str,
        default=DEFAULTS["baselines"],
        help="Comma-separated clinical baseline scores to evaluate.",
    )

    stress_group = parser.add_argument_group("Stress Tests")
    stress_group.add_argument(
        "--retention_levels",
        type=str,
        default=DEFAULTS["retention_levels"],
        help="Comma-separated retention fractions for observation thinning.",
    )
    stress_group.add_argument(
        "--n_repeats",
        type=int,
        default=DEFAULTS["n_repeats"],
        help="Number of repeated random thinnings per retention level.",
    )
    stress_group.add_argument(
        "--fast",
        action="store_true",
        default=DEFAULTS["fast"],
        help="Run a reduced configuration for quick verification.",
    )
    stress_group.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULTS["batch_size"],
        help="Retained for CLI consistency; baseline scoring is not minibatch-based.",
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

    args = parser.parse_args()
    args.retention_levels = parse_float_list(args.retention_levels)
    return args


def parse_name_list(value: str):
    """Parse a comma-separated CLI list into non-empty names."""
    return [item.strip() for item in value.split(",") if item.strip()]


def load_baseline_checkpoint(baseline_name: str, out_dir: Path):
    """Load one trained baseline checkpoint for stress-test evaluation."""
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {sorted(BASELINE_REGISTRY)}"
        )
    path = checkpoint_path(baseline_name, "baseline", out_dir, ".pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for baseline {baseline_name}: {path}. "
            "Create the checkpoint first, or use --baselines to select "
            "baselines with existing checkpoints."
        )
    return BASELINE_REGISTRY[baseline_name].load(path)


def main():
    """Run the baseline stress-test workflow."""
    cfg = parse_args()
    if cfg.fast:
        cfg.max_patients = cfg.max_patients or 300
        cfg.n_repeats = min(cfg.n_repeats, 3)
        print("Fast mode: reduced patients and repeats.")

    selected_baselines = parse_name_list(cfg.baselines)
    out_dir = Path(cfg.results_dir)
    if cfg.run_subdir:
        out_dir = out_dir / cfg.run_subdir

    print("\n── Loading data ────────────────────────────────────────────────")
    raw_splits, raw_regimes, meta = DATASET_REGISTRY[cfg.dataset].load(cfg)
    splits, regimes, meta = TASK_REGISTRY[cfg.task].apply(
        raw_splits, raw_regimes, meta, cfg
    )
    if cfg.split not in regimes["dense"]:
        raise ValueError(
            f"Unknown split '{cfg.split}'. Available: {sorted(regimes['dense'])}"
        )
    test_ds = regimes["dense"][cfg.split]
    print(f"  Dataset: {meta['dataset_name']} | Task: {meta['task_name']}")
    print(f"  Split: {cfg.split} | Patients: {len(test_ds)} | Features: {meta['n_features']}")

    print("\n── Loading baseline checkpoints ───────────────────────────────")
    baselines = {}
    for baseline_name in selected_baselines:
        baselines[baseline_name] = load_baseline_checkpoint(baseline_name, out_dir)
        print(f"  Loaded {baseline_name}")

    print("\n── Running stress tests ───────────────────────────────────────")
    perturb_dir = out_dir / "stress_test"
    git_meta = collect_git_metadata(Path(__file__).resolve().parent.parent)
    provenance_dir = out_dir / "provenance" / "stress_test"
    write_provenance_files(
        provenance_dir,
        {
            "config.json": vars(cfg),
            "git.json": git_meta,
            "git_status.txt": git_meta.get("git_status_short") or "",
            "git_diff.patch": git_meta.get("git_diff") or "",
            "git_diff_cached.patch": git_meta.get("git_diff_cached") or "",
            "data.json": build_data_provenance(cfg, meta, split_name=cfg.split),
            "loaded_checkpoints.json": {
                baseline_name: str(checkpoint_path(baseline_name, "baseline", out_dir, ".pkl").resolve())
                for baseline_name in selected_baselines
            },
        },
    )

    run_name = cfg.run_name or default_run_name("stress_test", cfg, selected_baselines)
    with start_run_if_enabled(
        cfg.track,
        cfg.experiment_name,
        run_name,
        tracking_uri=cfg.tracking_uri,
    ) as mlflow:
        if mlflow is not None:
            set_common_tags(mlflow, "stress_test", cfg, {"component_count": len(selected_baselines)})
            log_git_tags(mlflow, git_meta)
            log_params_from_namespace(mlflow, cfg, exclude={"run_name", "experiment_name", "tracking_uri", "track"})
            mlflow.log_param("results_dir_resolved", str(out_dir.resolve()))
            mlflow.log_param("run_root_resolved", str(out_dir.resolve()))
            mlflow.log_param("output_dir_resolved", str(perturb_dir.resolve()))
            maybe_log_artifacts(mlflow, provenance_dir, artifact_path="provenance")

        results = run_all_stress_tests(
            baselines, test_ds, meta, cfg
        )
        save_stress_test_outputs(results, perturb_dir)
        metrics = {}
        for baseline_name, baseline_result in results.items():
            coverage_decay = baseline_result.get("coverage_decay", {}).get("by_retention", {})
            full_retention = coverage_decay.get("1.0", {})
            metrics[f"{baseline_name}_coverage_retention_1_0"] = full_retention.get("coverage")
            sparsity = baseline_result.get("sparsity_performance_covered", {}).get("by_retention", {})
            full_perf = sparsity.get("1.0", {})
            metrics[f"{baseline_name}_auc_retention_1_0"] = full_perf.get("auc")
        maybe_log_metrics(mlflow, metrics)
        maybe_log_artifacts(mlflow, perturb_dir, artifact_path="outputs/stress_test")

    print(f"\nStress test outputs saved to {perturb_dir}")
    print(f"  {perturb_dir / 'summary.md'}")
    print(f"  {perturb_dir / 'stress_test_results.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
