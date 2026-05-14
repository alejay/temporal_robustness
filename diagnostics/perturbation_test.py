"""
Controlled perturbation tests for irregular-time model behavior.

These tests compare predictions for counterfactual versions of the same
patient histories. They are behavioral stress tests, not formal coherence
diagnostics.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from datasets.common import PatientDataset, pad_collate


MODEL_LABELS = {
    "grud": "GRU-D",
    "latent_ode": "Latent ODE",
    "transformer": "Transformer",
    "state_space": "State-Space",
    "landmarking": "Landmarking",
}


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated list of numeric CLI values."""
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def clone_record(record: dict) -> dict:
    """Deep-copy one patient record before applying perturbations."""
    return copy.deepcopy(record)


def thin_record_fraction(record: dict, keep_fraction: float,
                         rng: np.random.Generator) -> dict:
    """Thin one record while preserving endpoints and record metadata."""
    out = clone_record(record)
    times = np.asarray(out["times"])
    T = len(times)
    if keep_fraction >= 1.0 or T <= 2:
        return out

    interior = np.arange(1, T - 1)
    keep_mask = rng.random(len(interior)) < keep_fraction
    keep_idx = np.unique(np.concatenate([[0], interior[keep_mask], [T - 1]]))

    out["times"] = out["times"][keep_idx]
    out["values"] = out["values"][keep_idx]
    out["mask"] = out["mask"][keep_idx]
    if "eligible_mask" in out and len(out["eligible_mask"]) == T:
        out["eligible_mask"] = out["eligible_mask"][keep_idx]
    if "prediction_times" in out and len(out["prediction_times"]) == T:
        out["prediction_times"] = out["prediction_times"][keep_idx]
    return out


def thin_records(records: Iterable[dict], keep_fraction: float,
                 seed: int) -> List[dict]:
    """Apply independent retrospective thinning to every record in a collection."""
    rng = np.random.default_rng(seed)
    return [thin_record_fraction(r, keep_fraction, rng) for r in records]


def stretch_record_times(record: dict, factor: float) -> dict:
    """Keep values/masks fixed and stretch timestamps relative to first time."""
    out = clone_record(record)
    times = np.asarray(out["times"], dtype=np.float32)
    if len(times) == 0:
        return out
    origin = float(times[0])
    out["times"] = (origin + factor * (times - origin)).astype(np.float32)
    if "prediction_times" in out and len(out["prediction_times"]) == len(times):
        out["prediction_times"] = out["times"].copy()
    return out


def prefix_records(records: Iterable[dict]) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Create all length >= 2 prefixes and map source patient ids to prefix ids."""
    out = []
    index = {}
    for record in records:
        pid = str(record["patient_id"])
        index[pid] = []
        T = len(record["times"])
        for L in range(2, T + 1):
            pref = clone_record(record)
            pref_id = f"{pid}::prefix::{L}"
            pref["patient_id"] = pref_id
            pref["source_patient_id"] = pid
            pref["prefix_length"] = int(L)
            pref["times"] = pref["times"][:L]
            pref["values"] = pref["values"][:L]
            pref["mask"] = pref["mask"][:L]
            if "eligible_mask" in pref and len(pref["eligible_mask"]) >= L:
                pref["eligible_mask"] = pref["eligible_mask"][:L]
            if "prediction_times" in pref and len(pref["prediction_times"]) >= L:
                pref["prediction_times"] = pref["prediction_times"][:L]
            out.append(pref)
            index[pid].append(pref_id)
    return out, index


def _chunked(items: List[dict], batch_size: int):
    """Yield fixed-size record batches for lightweight prediction loops."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _with_labels(records: List[dict], dataset: PatientDataset) -> List[dict]:
    """Ensure each record carries a label before it is collated for inference."""
    out = []
    for record in records:
        r = record if "label" in record else clone_record(record)
        if "label" not in r:
            pid = str(r.get("source_patient_id", r["patient_id"]))
            r["label"] = float(dataset.labels.get(pid, 0))
        out.append(r)
    return out


def predict_patient_probs(method_name: str, method, dataset: PatientDataset,
                          device, batch_size: int = 32,
                          n_ode_samples: int = 20,
                          torch_seed: int = 0) -> Dict[str, dict]:
    """
    Return {patient_id: {prob, label}} for covered records.

    This uses lightweight final prediction paths and avoids full diagnostic
    trajectory collection.
    """
    if method_name == "landmarking":
        out = {}
        for record in dataset.records:
            pred = method.predict_at(record, float(record["times"][-1]))
            if pred is None:
                continue
            pid = str(record["patient_id"])
            out[pid] = {
                "prob": float(pred["mean"]),
                "label": float(record.get("label", dataset.labels.get(pid, 0))),
            }
        return out

    method.to(device)
    method.eval()
    preds = {}
    for records_b in _chunked(dataset.records, batch_size):
        records_b = _with_labels(records_b, dataset)
        batch = pad_collate(records_b)
        times = batch["times"].to(device)
        values = batch["values"].to(device)
        mask = batch["mask"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        lengths = batch["lengths"].to(device)

        with torch.no_grad():
            if method_name == "latent_ode":
                torch.manual_seed(int(torch_seed))
                probs_t, _ = method.predict(
                    times, values, mask, obs_mask, lengths,
                    n_samples=n_ode_samples,
                )
                probs = probs_t.cpu().numpy()
            else:
                logits = method(times, values, mask, obs_mask, lengths)
                probs = torch.sigmoid(logits).cpu().numpy()

        labels = batch["labels"].numpy()
        for record, prob, label in zip(records_b, probs, labels):
            preds[str(record["patient_id"])] = {
                "prob": float(prob),
                "label": float(label),
            }
    return preds


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute a simple equal-width expected calibration error estimate."""
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (probs >= lo) & (probs < hi)
        if idx.sum() == 0:
            continue
        total += (idx.sum() / len(probs)) * abs(labels[idx].mean() - probs[idx].mean())
    return float(total)


def predictive_metrics(pred_map: Dict[str, dict]) -> dict:
    """Summarize predictive performance for one patient-level prediction map."""
    probs = np.array([v["prob"] for v in pred_map.values()], dtype=float)
    labels = np.array([v["label"] for v in pred_map.values()], dtype=float)
    if len(probs) == 0:
        return {"n": 0, "auc": None, "brier": None, "log_loss": None, "ece": None}
    try:
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None
    except Exception:
        auc = None
    try:
        ll = float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        ll = None
    return {
        "n": int(len(probs)),
        "auc": auc,
        "brier": float(np.mean((probs - labels) ** 2)),
        "log_loss": ll,
        "ece": _ece(probs, labels),
    }


def paired_drift(reference: Dict[str, dict], comparison: Dict[str, dict]) -> dict:
    """Summarize prediction drift over patients present in both prediction maps."""
    rows = []
    for pid, ref in reference.items():
        comp = comparison.get(pid)
        if comp is None:
            continue
        rows.append({
            "patient_id": pid,
            "p_reference": ref["prob"],
            "p_comparison": comp["prob"],
            "abs_drift": abs(ref["prob"] - comp["prob"]),
            "signed_drift": comp["prob"] - ref["prob"],
            "label": ref["label"],
        })
    drift = np.array([r["abs_drift"] for r in rows], dtype=float)
    return {
        "n_pairs": int(len(rows)),
        "mean_abs_drift": float(drift.mean()) if len(drift) else None,
        "median_abs_drift": float(np.median(drift)) if len(drift) else None,
        "p90_abs_drift": float(np.quantile(drift, 0.9)) if len(drift) else None,
        "rows": rows,
    }


def _dataset(records: List[dict], base_dataset: PatientDataset) -> PatientDataset:
    """Rewrap perturbed records with labels/metadata from the source dataset."""
    return PatientDataset(records, labels=base_dataset.labels, meta=base_dataset.meta)


def run_thinning_sensitivity(method_name, method, dataset, device, cfg) -> dict:
    """Compare original predictions with predictions on thinned trajectories."""
    base_preds = predict_patient_probs(
        method_name, method, dataset, device, cfg.batch_size, cfg.n_ode_samples,
        cfg.seed,
    )
    by_retention = {}
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        preds = predict_patient_probs(
            method_name, method, _dataset(recs, dataset), device,
            cfg.batch_size, cfg.n_ode_samples, cfg.seed,
        )
        by_retention[str(retention)] = paired_drift(base_preds, preds)
    return {"reference_n": len(base_preds), "by_retention": by_retention}


def run_repeated_thinning_instability(method_name, method, dataset, device, cfg) -> dict:
    """Measure variability across repeated random thinnings of the same records."""
    by_retention = {}
    for retention in cfg.retention_levels:
        if retention >= 1.0:
            continue
        per_patient = {}
        for repeat in range(cfg.n_repeats):
            recs = thin_records(dataset.records, retention, cfg.seed + repeat)
            preds = predict_patient_probs(
                method_name, method, _dataset(recs, dataset), device,
                cfg.batch_size, cfg.n_ode_samples, cfg.seed,
            )
            for pid, item in preds.items():
                per_patient.setdefault(pid, []).append(item["prob"])

        rows = []
        for pid, vals in per_patient.items():
            if len(vals) < 2:
                continue
            arr = np.array(vals, dtype=float)
            rows.append({
                "patient_id": pid,
                "n_repeats": int(len(arr)),
                "sd": float(arr.std(ddof=1)),
                "range": float(arr.max() - arr.min()),
                "mean": float(arr.mean()),
            })
        sd = np.array([r["sd"] for r in rows], dtype=float)
        ranges = np.array([r["range"] for r in rows], dtype=float)
        by_retention[str(retention)] = {
            "n_patients": int(len(rows)),
            "mean_sd": float(sd.mean()) if len(sd) else None,
            "median_sd": float(np.median(sd)) if len(sd) else None,
            "mean_range": float(ranges.mean()) if len(ranges) else None,
            "median_range": float(np.median(ranges)) if len(ranges) else None,
            "rows": rows,
        }
    return {"n_repeats": int(cfg.n_repeats), "by_retention": by_retention}


def run_sparsity_response(method_name, method, dataset, device, cfg) -> dict:
    """Recompute predictive metrics as observation retention is reduced."""
    by_retention = {}
    for retention in cfg.retention_levels:
        recs = thin_records(dataset.records, retention, cfg.seed)
        preds = predict_patient_probs(
            method_name, method, _dataset(recs, dataset), device,
            cfg.batch_size, cfg.n_ode_samples, cfg.seed,
        )
        by_retention[str(retention)] = predictive_metrics(preds)
    return {"by_retention": by_retention}


def _trajectory_stats(probs: List[float]) -> dict:
    """Summarize movement statistics for one sequence of prefix predictions."""
    arr = np.array(probs, dtype=float)
    if len(arr) < 2:
        return {
            "n_points": int(len(arr)),
            "total_variation": None,
            "max_jump": None,
            "oscillation_count": None,
        }
    diffs = np.diff(arr)
    signs = np.sign(diffs[np.abs(diffs) > 1e-8])
    oscillations = int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0
    return {
        "n_points": int(len(arr)),
        "total_variation": float(np.abs(diffs).sum()),
        "max_jump": float(np.abs(diffs).max()),
        "oscillation_count": oscillations,
    }


def run_prefix_volatility(method_name, method, dataset, device, cfg) -> dict:
    """Measure how predictions evolve over progressively longer prefixes."""
    pref_records, index = prefix_records(dataset.records)
    if not pref_records:
        return {"n_patients": 0, "rows": [], "summary": {}}
    preds = predict_patient_probs(
        method_name, method, _dataset(pref_records, dataset), device,
        cfg.batch_size, cfg.n_ode_samples, cfg.seed,
    )
    rows = []
    for pid, pref_ids in index.items():
        probs = []
        for pref_id in pref_ids:
            if pref_id in preds:
                probs.append(preds[pref_id]["prob"])
        stats = _trajectory_stats(probs)
        if stats["total_variation"] is None:
            continue
        stats["patient_id"] = pid
        rows.append(stats)

    def _summ(key):
        vals = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        return {
            "mean": float(vals.mean()) if len(vals) else None,
            "median": float(np.median(vals)) if len(vals) else None,
        }

    return {
        "n_patients": int(len(rows)),
        "summary": {
            "total_variation": _summ("total_variation"),
            "max_jump": _summ("max_jump"),
            "oscillation_count": _summ("oscillation_count"),
        },
        "rows": rows,
    }


def run_timestamp_stretching(method_name, method, dataset, device, cfg) -> dict:
    """Compare predictions after rescaling elapsed times but not observations."""
    if method_name == "landmarking":
        return {"not_applicable": True, "reason": "landmarking uses fixed landmark times"}
    base_preds = predict_patient_probs(
        method_name, method, dataset, device, cfg.batch_size, cfg.n_ode_samples,
        cfg.seed,
    )
    by_factor = {}
    for factor in cfg.timestamp_factors:
        recs = [stretch_record_times(r, factor) for r in dataset.records]
        preds = predict_patient_probs(
            method_name, method, _dataset(recs, dataset), device,
            cfg.batch_size, cfg.n_ode_samples, cfg.seed,
        )
        by_factor[str(factor)] = paired_drift(base_preds, preds)
    return {"reference_n": len(base_preds), "by_factor": by_factor}


def run_all_perturbation_tests(methods: Dict[str, object], dataset: PatientDataset,
                               device, cfg) -> dict:
    """Run the full perturbation test suite for every supplied learned method."""
    results = {}
    for method_name, method in methods.items():
        print(f"\n── Perturbation tests: {method_name} ─────────────────────────")
        results[method_name] = {
            "thinning_sensitivity": run_thinning_sensitivity(
                method_name, method, dataset, device, cfg),
            "repeated_thinning_instability": run_repeated_thinning_instability(
                method_name, method, dataset, device, cfg),
            "sparsity_response": run_sparsity_response(
                method_name, method, dataset, device, cfg),
            "prefix_volatility": run_prefix_volatility(
                method_name, method, dataset, device, cfg),
            "timestamp_stretching": run_timestamp_stretching(
                method_name, method, dataset, device, cfg),
        }
    return results


def _fmt(value, digits=4):
    """Format scalar summary values for Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    return f"{value:.{digits}f}"


def format_perturbation_summary(results: dict) -> str:
    """Render a compact Markdown summary of perturbation test outputs."""
    lines = ["# Controlled Perturbation Test Results", ""]

    lines += ["## Thinning Sensitivity", ""]
    lines.append("| Model | Retention | N pairs | Mean drift | Median drift | P90 drift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, res in results.items():
        for retention, item in res["thinning_sensitivity"]["by_retention"].items():
            lines.append(
                f"| {model} | {retention} | {item['n_pairs']} | "
                f"{_fmt(item['mean_abs_drift'])} | {_fmt(item['median_abs_drift'])} | "
                f"{_fmt(item['p90_abs_drift'])} |"
            )

    lines += ["", "## Repeated-Thinning Instability", ""]
    lines.append("| Model | Retention | N patients | Mean SD | Median SD | Mean range |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, res in results.items():
        for retention, item in res["repeated_thinning_instability"]["by_retention"].items():
            lines.append(
                f"| {model} | {retention} | {item['n_patients']} | "
                f"{_fmt(item['mean_sd'])} | {_fmt(item['median_sd'])} | "
                f"{_fmt(item['mean_range'])} |"
            )

    lines += ["", "## Sparsity Response", ""]
    lines.append("| Model | Retention | N | AUC | Brier | Log loss | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, res in results.items():
        for retention, item in res["sparsity_response"]["by_retention"].items():
            lines.append(
                f"| {model} | {retention} | {item['n']} | {_fmt(item['auc'])} | "
                f"{_fmt(item['brier'])} | {_fmt(item['log_loss'])} | {_fmt(item['ece'])} |"
            )

    lines += ["", "## Prefix Volatility", ""]
    lines.append("| Model | N patients | Mean TV | Median TV | Mean max jump | Mean oscillations |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, res in results.items():
        item = res["prefix_volatility"]
        summ = item.get("summary", {})
        lines.append(
            f"| {model} | {item.get('n_patients', 0)} | "
            f"{_fmt(summ.get('total_variation', {}).get('mean'))} | "
            f"{_fmt(summ.get('total_variation', {}).get('median'))} | "
            f"{_fmt(summ.get('max_jump', {}).get('mean'))} | "
            f"{_fmt(summ.get('oscillation_count', {}).get('mean'))} |"
        )

    lines += ["", "## Timestamp Stretching", ""]
    lines.append("| Model | Factor | N pairs | Mean drift | Median drift | P90 drift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, res in results.items():
        item = res["timestamp_stretching"]
        if item.get("not_applicable"):
            lines.append(f"| {model} | n/a | 0 | n/a | n/a | n/a |")
            continue
        for factor, drift in item["by_factor"].items():
            lines.append(
                f"| {model} | {factor} | {drift['n_pairs']} | "
                f"{_fmt(drift['mean_abs_drift'])} | {_fmt(drift['median_abs_drift'])} | "
                f"{_fmt(drift['p90_abs_drift'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _plot_metric_by_x(results, section, x_key, y_getter, ylabel, title, out_path):
    """Plot one scalar metric against a shared x-axis across methods."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, res in results.items():
        data = res.get(section, {})
        if x_key not in data:
            continue
        xs, ys = [], []
        for key, item in data[x_key].items():
            y = y_getter(item)
            if y is None:
                continue
            xs.append(float(key))
            ys.append(float(y))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o",
                    label=MODEL_LABELS.get(model, model))
    ax.set_xlabel("Retention" if "retention" in x_key else "Timestamp factor")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_sparsity_auc_ece(results, out_path):
    """Plot AUC and ECE together for the sparsity-response analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, metric, ylabel, title in [
        (axes[0], "auc", "AUC", "Discrimination"),
        (axes[1], "ece", "ECE", "Calibration error"),
    ]:
        for model, res in results.items():
            data = res.get("sparsity_response", {}).get("by_retention", {})
            xs, ys = [], []
            for key, item in data.items():
                y = item.get(metric)
                if y is None:
                    continue
                xs.append(float(key))
                ys.append(float(y))
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o",
                        label=MODEL_LABELS.get(model, model))
        ax.set_xlabel("Retention")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Performance Under Sparsity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_perturbation_figures(results: dict, out_dir: Path) -> None:
    """Generate the standard figure set for learned-model perturbation outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_by_x(
        results, "thinning_sensitivity", "by_retention",
        lambda item: item.get("mean_abs_drift"),
        "Mean |p(full) - p(thinned)|",
        "Thinning Sensitivity",
        out_dir / "thinning_sensitivity.png",
    )
    _plot_metric_by_x(
        results, "repeated_thinning_instability", "by_retention",
        lambda item: item.get("mean_sd"),
        "Mean within-patient SD",
        "Repeated-Thinning Instability",
        out_dir / "repeated_thinning_instability.png",
    )
    _plot_sparsity_auc_ece(results, out_dir / "sparsity_response_auc_ece.png")
    # Prefix volatility: bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models, vals = [], []
    for model, res in results.items():
        tv = res["prefix_volatility"].get("summary", {}).get("total_variation", {}).get("mean")
        if tv is None:
            continue
        models.append(MODEL_LABELS.get(model, model))
        vals.append(tv)
    ax.bar(models, vals)
    ax.set_ylabel("Mean total variation")
    ax.set_title("Prefix Volatility")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_volatility.png", bbox_inches="tight")
    plt.close(fig)
    _plot_metric_by_x(
        results, "timestamp_stretching", "by_factor",
        lambda item: item.get("mean_abs_drift"),
        "Mean |p(original) - p(stretched)|",
        "Timestamp Sensitivity",
        out_dir / "timestamp_sensitivity.png",
    )


def save_perturbation_outputs(results: dict, out_dir: Path) -> None:
    """Write JSON, Markdown, and figure artifacts for perturbation tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "perturbation_test_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "summary.md").write_text(format_perturbation_summary(results))
    generate_perturbation_figures(results, out_dir / "figures")
