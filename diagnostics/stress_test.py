"""
Baseline-focused perturbation tests for clinical guideline scores.

These tests emphasize score availability, component coverage, and covered-subset
prediction stability under sparse observations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.common import PatientDataset
from diagnostics.perturbation_test import (
    _dataset,
    _trajectory_stats,
    paired_drift,
    predictive_metrics,
    prefix_records,
    thin_records,
)


BASELINE_LABELS = {
    "qsofa": "qSOFA",
    "sirs": "SIRS",
    "sofa": "SOFA",
}


BASELINE_COLORS = {
    "qsofa": "#4C78A8",
    "sirs": "#F58518",
    "sofa": "#54A24B",
}


BASELINE_COMPONENTS = {
    "qsofa": {
        "RespRate": ("single", ["RespRate"]),
        "SysABP": ("single", ["SysABP"]),
        "GCS": ("single", ["GCS"]),
    },
    "sirs": {
        "Temp": ("single", ["Temp"]),
        "HR": ("single", ["HR"]),
        "RespRate": ("single", ["RespRate"]),
        "WBC": ("single", ["WBC"]),
    },
    "sofa": {
        "Platelets": ("single", ["Platelets"]),
        "Bilirubin": ("single", ["Bilirubin"]),
        "Creatinine": ("single", ["Creatinine"]),
        "GCS": ("single", ["GCS"]),
        "MAP": ("single", ["MAP"]),
        "PaO2/FiO2": ("same_step_all", ["PaO2", "FiO2"]),
    },
}


def predict_baseline_patient_rows(baseline, dataset: PatientDataset,
                                  meta: dict) -> Dict[str, dict]:
    """Return one patient-level row per record, preserving uncovered patients."""
    rows = {}
    for output in baseline.predict_dataset(dataset, meta):
        pid = str(output["patient_id"])
        points = output.get("points", [])
        row = {
            "patient_id": pid,
            "label": float(output.get("label", dataset.labels.get(pid, 0))),
            "covered": bool(points),
            "n_score_points": int(len(points)),
            "final_time": None,
            "final_score": None,
            "final_probability": None,
            "prob": None,
            "score": None,
            "available_components": 0,
            "total_components": int(getattr(baseline, "total_components", 0)),
            "complete": False,
            "component_flags": {},
        }
        if points:
            point = points[-1]
            row.update({
                "final_time": float(point.time),
                "final_score": float(point.score),
                "final_probability": float(point.calibrated_prob),
                "prob": float(point.calibrated_prob),
                "score": float(point.score),
                "available_components": int(point.available_components),
                "total_components": int(point.total_components),
                "complete": bool(point.complete),
                "component_flags": dict(point.component_flags),
            })
        rows[pid] = row
    return rows


def _feature_idx(meta: dict, name: str):
    """Look up one feature index from dataset metadata."""
    return meta.get("feature_map", {}).get(name)


def _component_mask(record: dict, meta: dict, mode: str,
                    feature_names: List[str]) -> np.ndarray:
    """Return the time-step mask for one score component definition."""
    masks = []
    for name in feature_names:
        idx = _feature_idx(meta, name)
        if idx is None:
            masks.append(np.zeros(len(record["times"]), dtype=bool))
        else:
            masks.append(np.asarray(record["mask"][:, idx] > 0.5, dtype=bool))
    if not masks:
        return np.zeros(len(record["times"]), dtype=bool)
    if mode == "same_step_all":
        out = masks[0].copy()
        for mask in masks[1:]:
            out &= mask
        return out
    return masks[0]


def component_availability_profile(baseline_name: str, records: Iterable[dict],
                                   meta: dict) -> dict:
    """Summarize patient- and step-level availability for each score component."""
    components = BASELINE_COMPONENTS.get(baseline_name, {})
    by_component = {}
    records = list(records)
    n_records = len(records)
    for component, (mode, names) in components.items():
        patient_any = []
        step_fracs = []
        for record in records:
            mask = _component_mask(record, meta, mode, names)
            patient_any.append(bool(mask.any()))
            step_fracs.append(float(mask.mean()) if len(mask) else 0.0)
        by_component[component] = {
            "patient_fraction_any": float(np.mean(patient_any)) if patient_any else None,
            "mean_step_fraction": float(np.mean(step_fracs)) if step_fracs else None,
            "n_patients": int(n_records),
        }
    return by_component


def _coverage_summary(rows: Dict[str, dict], total_patients: int) -> dict:
    """Aggregate score coverage statistics over one set of patient-level rows."""
    values = list(rows.values())
    covered = [r for r in values if r["covered"]]
    complete = [r for r in covered if r["complete"]]
    n_points = np.array([r["n_score_points"] for r in values], dtype=float)
    available = np.array([r["available_components"] for r in covered], dtype=float)
    return {
        "n_total": int(total_patients),
        "n_covered": int(len(covered)),
        "coverage": float(len(covered) / total_patients) if total_patients else None,
        "mean_score_points": float(n_points.mean()) if len(n_points) else None,
        "median_score_points": float(np.median(n_points)) if len(n_points) else None,
        "complete_rate_among_covered": (
            float(len(complete) / len(covered)) if covered else None
        ),
        "mean_available_components_final": (
            float(available.mean()) if len(available) else None
        ),
        "median_available_components_final": (
            float(np.median(available)) if len(available) else None
        ),
    }


def _covered_pred_map(rows: Dict[str, dict]) -> Dict[str, dict]:
    """Keep only covered patient probabilities for drift/performance comparisons."""
    return {
        pid: {"prob": row["prob"], "label": row["label"]}
        for pid, row in rows.items()
        if row["covered"] and row["prob"] is not None
    }


def _covered_score_map(rows: Dict[str, dict]) -> Dict[str, dict]:
    """Keep only covered raw scores for paired drift comparisons."""
    return {
        pid: {"prob": row["score"], "label": row["label"]}
        for pid, row in rows.items()
        if row["covered"] and row["score"] is not None
    }


def run_baseline_coverage_decay(baseline_name, baseline, dataset, meta, cfg) -> dict:
    """Track how often a baseline remains scorable as retention decreases."""
    by_retention = {}
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        rows = predict_baseline_patient_rows(baseline, _dataset(recs, dataset), meta)
        item = _coverage_summary(rows, len(dataset.records))
        item["rows"] = list(rows.values())
        by_retention[str(retention)] = item
    return {"by_retention": by_retention}


def run_baseline_component_availability(baseline_name, dataset, meta, cfg) -> dict:
    """Track which score components remain observed under thinning."""
    by_retention = {}
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        by_retention[str(retention)] = component_availability_profile(
            baseline_name, recs, meta
        )
    return {"by_retention": by_retention}


def run_baseline_thinning_sensitivity(baseline_name, baseline, dataset,
                                      meta, cfg) -> dict:
    """Measure score/probability drift among patients still covered after thinning."""
    base_rows = predict_baseline_patient_rows(baseline, dataset, meta)
    base_probs = _covered_pred_map(base_rows)
    base_scores = _covered_score_map(base_rows)
    by_retention = {}
    total = len(dataset.records)
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        rows = predict_baseline_patient_rows(baseline, _dataset(recs, dataset), meta)
        pred_drift = paired_drift(base_probs, _covered_pred_map(rows))
        score_drift = paired_drift(base_scores, _covered_score_map(rows))
        full_covered = {pid for pid, row in base_rows.items() if row["covered"]}
        thin_covered = {pid for pid, row in rows.items() if row["covered"]}
        item = {
            "n_total": int(total),
            "n_reference_covered": int(len(full_covered)),
            "n_comparison_covered": int(len(thin_covered)),
            "n_pairs": int(pred_drift["n_pairs"]),
            "paired_coverage": float(pred_drift["n_pairs"] / total) if total else None,
            "lost_coverage": int(len(full_covered - thin_covered)),
            "gained_coverage": int(len(thin_covered - full_covered)),
            "probability_drift": pred_drift,
            "score_drift": score_drift,
        }
        by_retention[str(retention)] = item
    return {"reference_n": len(base_probs), "by_retention": by_retention}


def run_baseline_repeated_thinning_instability(baseline_name, baseline, dataset,
                                               meta, cfg) -> dict:
    """Measure instability of score availability across repeated thinnings."""
    by_retention = {}
    pids = [str(r["patient_id"]) for r in dataset.records]
    for retention in cfg.retention_levels:
        if retention >= 1.0:
            continue
        per_patient = {
            pid: {"covered": [], "probs": [], "scores": []}
            for pid in pids
        }
        for repeat in range(cfg.n_repeats):
            recs = thin_records(dataset.records, retention, cfg.seed + repeat)
            rows = predict_baseline_patient_rows(
                baseline, _dataset(recs, dataset), meta
            )
            for pid in pids:
                row = rows.get(pid)
                covered = bool(row and row["covered"])
                per_patient[pid]["covered"].append(covered)
                if covered:
                    per_patient[pid]["probs"].append(row["prob"])
                    per_patient[pid]["scores"].append(row["score"])

        rows = []
        for pid, vals in per_patient.items():
            coverage_frequency = float(np.mean(vals["covered"]))
            probs = np.array(vals["probs"], dtype=float)
            scores = np.array(vals["scores"], dtype=float)
            rows.append({
                "patient_id": pid,
                "coverage_frequency": coverage_frequency,
                "availability_instability": bool(0.0 < coverage_frequency < 1.0),
                "n_covered_repeats": int(len(probs)),
                "prob_sd": float(probs.std(ddof=1)) if len(probs) >= 2 else None,
                "prob_range": float(probs.max() - probs.min()) if len(probs) else None,
                "score_sd": float(scores.std(ddof=1)) if len(scores) >= 2 else None,
                "score_range": float(scores.max() - scores.min()) if len(scores) else None,
            })
        prob_sds = np.array([r["prob_sd"] for r in rows if r["prob_sd"] is not None])
        by_retention[str(retention)] = {
            "n_patients": int(len(rows)),
            "n_repeats": int(cfg.n_repeats),
            "mean_coverage_frequency": float(np.mean([
                r["coverage_frequency"] for r in rows
            ])) if rows else None,
            "availability_instability_rate": float(np.mean([
                r["availability_instability"] for r in rows
            ])) if rows else None,
            "mean_prob_sd_among_repeat_covered": (
                float(prob_sds.mean()) if len(prob_sds) else None
            ),
            "median_prob_sd_among_repeat_covered": (
                float(np.median(prob_sds)) if len(prob_sds) else None
            ),
            "rows": rows,
        }
    return {"n_repeats": int(cfg.n_repeats), "by_retention": by_retention}


def run_baseline_prefix_availability_volatility(baseline_name, baseline, dataset,
                                                meta, cfg) -> dict:
    """Summarize when baseline scores appear and how much they move over prefixes."""
    pref_records, index = prefix_records(dataset.records)
    if not pref_records:
        return {"n_patients": 0, "summary": {}, "rows": []}
    rows_by_prefix = predict_baseline_patient_rows(
        baseline, _dataset(pref_records, dataset), meta
    )
    rows = []
    for pid, pref_ids in index.items():
        covered_probs = []
        coverage_flags = []
        first_prefix_length = None
        first_score_time = None
        for pref_id in pref_ids:
            row = rows_by_prefix.get(pref_id)
            covered = bool(row and row["covered"])
            coverage_flags.append(covered)
            if covered:
                covered_probs.append(row["prob"])
                if first_prefix_length is None:
                    first_prefix_length = int(pref_id.split("::prefix::")[-1])
                    first_score_time = row["final_time"]
        stats = _trajectory_stats(covered_probs)
        unavailable_after_available = 0
        seen_available = False
        for flag in coverage_flags:
            if flag:
                seen_available = True
            elif seen_available:
                unavailable_after_available += 1
        rows.append({
            "patient_id": pid,
            "n_prefixes": int(len(pref_ids)),
            "n_covered_prefixes": int(sum(coverage_flags)),
            "coverage_fraction": (
                float(np.mean(coverage_flags)) if coverage_flags else None
            ),
            "first_covered_prefix_length": first_prefix_length,
            "first_score_time": first_score_time,
            "unavailable_after_available": int(unavailable_after_available),
            **stats,
        })

    def _summ(key):
        vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        return {
            "mean": float(vals.mean()) if len(vals) else None,
            "median": float(np.median(vals)) if len(vals) else None,
        }

    return {
        "n_patients": int(len(rows)),
        "summary": {
            "coverage_fraction": _summ("coverage_fraction"),
            "first_covered_prefix_length": _summ("first_covered_prefix_length"),
            "total_variation": _summ("total_variation"),
            "max_jump": _summ("max_jump"),
            "oscillation_count": _summ("oscillation_count"),
            "unavailable_after_available": _summ("unavailable_after_available"),
        },
        "rows": rows,
    }


def run_baseline_sparsity_performance(baseline_name, baseline, dataset,
                                      meta, cfg) -> dict:
    """Recompute covered-subset predictive metrics at each retention level."""
    by_retention = {}
    total = len(dataset.records)
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        rows = predict_baseline_patient_rows(baseline, _dataset(recs, dataset), meta)
        metrics = predictive_metrics(_covered_pred_map(rows))
        metrics["n_total"] = int(total)
        metrics["coverage"] = float(metrics["n"] / total) if total else None
        by_retention[str(retention)] = metrics
    return {"by_retention": by_retention}


def run_all_stress_tests(baselines: Dict[str, object],
                         dataset: PatientDataset,
                         meta: dict,
                         cfg) -> dict:
    """Run the full stress-test suite for every supplied clinical baseline."""
    results = {}
    for baseline_name, baseline in baselines.items():
        print(f"\n── Stress tests: {baseline_name} ─────────────────────────────")
        results[baseline_name] = {
            "coverage_decay": run_baseline_coverage_decay(
                baseline_name, baseline, dataset, meta, cfg),
            "component_availability": run_baseline_component_availability(
                baseline_name, dataset, meta, cfg),
            "thinning_sensitivity_covered": run_baseline_thinning_sensitivity(
                baseline_name, baseline, dataset, meta, cfg),
            "repeated_thinning_availability": run_baseline_repeated_thinning_instability(
                baseline_name, baseline, dataset, meta, cfg),
            "prefix_availability_volatility": (
                run_baseline_prefix_availability_volatility(
                    baseline_name, baseline, dataset, meta, cfg)
            ),
            "sparsity_performance_covered": run_baseline_sparsity_performance(
                baseline_name, baseline, dataset, meta, cfg),
            "timestamp_stretching": {
                "not_applicable": True,
                "reason": (
                    "Guideline scores do not directly model elapsed-time dynamics."
                ),
            },
        }
    return results


def _fmt(value, digits=4):
    """Format scalar summary values for Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    return f"{value:.{digits}f}"


def format_stress_test_summary(results: dict) -> str:
    """Render a compact Markdown summary of baseline stress-test outputs."""
    lines = ["# Stress Test Results", ""]
    lines += [
        "These are availability and robustness stress tests for guideline "
        "scores under sparse observations. Missing coverage is a result, not "
        "a low-risk prediction.",
        "",
    ]

    lines += ["## Coverage Decay", ""]
    lines.append(
        "| Baseline | Retention | N | Covered | Coverage | Mean score points | "
        "Complete rate | Mean final components |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for baseline, res in results.items():
        for retention, item in res["coverage_decay"]["by_retention"].items():
            lines.append(
                f"| {baseline} | {retention} | {item['n_total']} | "
                f"{item['n_covered']} | {_fmt(item['coverage'])} | "
                f"{_fmt(item['mean_score_points'])} | "
                f"{_fmt(item['complete_rate_among_covered'])} | "
                f"{_fmt(item['mean_available_components_final'])} |"
            )

    lines += ["", "## Component Availability", ""]
    lines.append(
        "| Baseline | Retention | Component | Patient fraction any | "
        "Mean step fraction |"
    )
    lines.append("|---|---:|---|---:|---:|")
    for baseline, res in results.items():
        for retention, comps in res["component_availability"]["by_retention"].items():
            for component, item in comps.items():
                lines.append(
                    f"| {baseline} | {retention} | {component} | "
                    f"{_fmt(item['patient_fraction_any'])} | "
                    f"{_fmt(item['mean_step_fraction'])} |"
                )

    lines += ["", "## Covered-Patient Thinning Sensitivity", ""]
    lines.append(
        "| Baseline | Retention | Ref covered | Thin covered | Pairs | "
        "Paired coverage | Lost coverage | Mean prob drift | Mean score drift |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for baseline, res in results.items():
        for retention, item in res["thinning_sensitivity_covered"]["by_retention"].items():
            lines.append(
                f"| {baseline} | {retention} | {item['n_reference_covered']} | "
                f"{item['n_comparison_covered']} | {item['n_pairs']} | "
                f"{_fmt(item['paired_coverage'])} | {item['lost_coverage']} | "
                f"{_fmt(item['probability_drift']['mean_abs_drift'])} | "
                f"{_fmt(item['score_drift']['mean_abs_drift'])} |"
            )

    lines += ["", "## Repeated-Thinning Availability", ""]
    lines.append(
        "| Baseline | Retention | N patients | Scorable thinned versions | "
        "Patients scorable in some repeats but not others | "
        "Mean SD of predicted risk across scorable repeats |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for baseline, res in results.items():
        for retention, item in res["repeated_thinning_availability"]["by_retention"].items():
            lines.append(
                f"| {baseline} | {retention} | {item['n_patients']} | "
                f"{_fmt(100.0 * item['mean_coverage_frequency'])}% | "
                f"{int(round(item['availability_instability_rate'] * item['n_patients']))} / "
                f"{item['n_patients']} "
                f"({_fmt(100.0 * item['availability_instability_rate'])}%) | "
                f"{_fmt(item['mean_prob_sd_among_repeat_covered'])} |"
            )

    lines += ["", "## Prefix Availability and Volatility", ""]
    lines.append(
        "| Baseline | N patients | Mean prefix coverage | Mean first covered prefix | "
        "Mean TV | Mean max jump | Mean unavailable-after-available |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for baseline, res in results.items():
        item = res["prefix_availability_volatility"]
        summ = item.get("summary", {})
        lines.append(
            f"| {baseline} | {item.get('n_patients', 0)} | "
            f"{_fmt(summ.get('coverage_fraction', {}).get('mean'))} | "
            f"{_fmt(summ.get('first_covered_prefix_length', {}).get('mean'))} | "
            f"{_fmt(summ.get('total_variation', {}).get('mean'))} | "
            f"{_fmt(summ.get('max_jump', {}).get('mean'))} | "
            f"{_fmt(summ.get('unavailable_after_available', {}).get('mean'))} |"
        )

    lines += ["", "## Covered-Subset Performance Under Sparsity", ""]
    lines.append("| Baseline | Retention | Covered N | Coverage | AUC | Brier | Log loss | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for baseline, res in results.items():
        for retention, item in res["sparsity_performance_covered"]["by_retention"].items():
            lines.append(
                f"| {baseline} | {retention} | {item['n']} | "
                f"{_fmt(item['coverage'])} | {_fmt(item['auc'])} | "
                f"{_fmt(item['brier'])} | {_fmt(item['log_loss'])} | "
                f"{_fmt(item['ece'])} |"
            )

    lines += [
        "",
        "## Timestamp Stretching",
        "",
        "Not applicable in v1: guideline scores do not directly model elapsed-time dynamics.",
        "",
    ]
    return "\n".join(lines)


def _plot_by_retention(results, section, y_getter, ylabel, title, out_path):
    """Plot one retention-indexed scalar summary across baselines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for baseline, res in results.items():
        xs, ys = [], []
        for retention, item in res[section]["by_retention"].items():
            y = y_getter(item)
            if y is None:
                continue
            xs.append(float(retention))
            ys.append(float(y))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o",
                    label=BASELINE_LABELS.get(baseline, baseline))
    ax.set_xlabel("Retention")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_stress_test_figures(results: dict, out_dir: Path) -> None:
    """Generate the standard figure set for baseline stress-test outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_by_retention(
        results,
        "coverage_decay",
        lambda item: item.get("coverage"),
        "Patient coverage",
        "Coverage Decay Under Thinning",
        out_dir / "coverage_decay.png",
    )
    _plot_by_retention(
        results,
        "thinning_sensitivity_covered",
        lambda item: item.get("probability_drift", {}).get("mean_abs_drift"),
        "Mean probability drift among covered pairs",
        "Covered-Patient Thinning Sensitivity",
        out_dir / "thinning_sensitivity_covered.png",
    )
    _plot_by_retention(
        results,
        "repeated_thinning_availability",
        lambda item: item.get("availability_instability_rate"),
        "Availability instability rate",
        "Repeated-Thinning Availability Instability",
        out_dir / "repeated_thinning_availability.png",
    )
    _plot_by_retention(
        results,
        "sparsity_performance_covered",
        lambda item: item.get("brier"),
        "Brier score among covered patients",
        "Covered-Subset Performance Under Sparsity",
        out_dir / "sparsity_performance_covered.png",
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, vals, colors = [], [], []
    for baseline, res in results.items():
        retention_items = res["component_availability"]["by_retention"]
        if not retention_items:
            continue
        lowest = min(retention_items, key=lambda x: float(x))
        for component, item in retention_items[lowest].items():
            labels.append(f"{BASELINE_LABELS.get(baseline, baseline)}\n{component}")
            vals.append(item.get("patient_fraction_any") or 0.0)
            colors.append(BASELINE_COLORS.get(baseline, "#9A9A9A"))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Patient fraction with any component observation")
    ax.set_title("Component Availability at Lowest Retention")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out_dir / "component_availability.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels, vals = [], []
    for baseline, res in results.items():
        summ = res["prefix_availability_volatility"].get("summary", {})
        labels.append(BASELINE_LABELS.get(baseline, baseline))
        vals.append(summ.get("coverage_fraction", {}).get("mean") or 0.0)
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean prefix coverage fraction")
    ax.set_title("Prefix Availability")
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_availability_volatility.png", bbox_inches="tight")
    plt.close(fig)


def save_stress_test_outputs(results: dict, out_dir: Path) -> None:
    """Write JSON, Markdown, and figure artifacts for stress tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stress_test_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (out_dir / "summary.md").write_text(
        format_stress_test_summary(results)
    )
    generate_stress_test_figures(results, out_dir / "figures")
