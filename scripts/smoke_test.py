"""
End-to-end smoke test using purely synthetic data.
Runs in ~60 seconds on CPU.  No data download required.

Tests:
  1. Synthetic patient generation
  2. Observation thinning
  3. GRU-D forward pass and MC Dropout
  4. Latent ODE forward pass and posterior sampling
  5. Controlled perturbation tests
  6. Baseline stress tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.guideline import QSOFABaseline, SIRSBaseline, SOFABaseline
from datasets.common import (
    PatientDataset,
    build_sparsity_regimes,
    pad_collate,
)
from datasets.physionet2012 import (
    PHYSIONET_VARS,
)
from models.grud import GRUD, _forward_train
from models.latent_ode import LatentODE
from models.transformer import TimeAwareTransformer
from models.state_space import StateSpaceRiskModel
from models.landmarking import LandmarkingModel
from diagnostics.perturbation_test import (
    predict_patient_probs,
    run_all_perturbation_tests,
    save_perturbation_outputs,
    stretch_record_times,
    thin_record_fraction,
    thin_records,
)
from diagnostics.stress_test import (
    predict_baseline_patient_rows,
    run_all_stress_tests,
    save_stress_test_outputs,
)


# ── Synthetic data factory ────────────────────────────────────────────────────

def make_synthetic_patients(n: int = 80, D: int = 10, T_max: int = 20,
                             seed: int = 0):
    """Generate random patient records and binary labels."""
    rng = np.random.default_rng(seed)
    records = []
    labels  = {}
    for i in range(n):
        T = rng.integers(5, T_max + 1)
        times  = np.sort(rng.uniform(0, 48, T)).astype(np.float32)
        values = rng.standard_normal((T, D)).astype(np.float32)
        mask   = (rng.random((T, D)) > 0.4).astype(np.float32)
        pid    = str(i)
        records.append({"patient_id": pid, "times": times,
                        "values": values * mask, "mask": mask})
        labels[pid] = int(rng.random() < 0.25)
    return records, labels


def test_thinning(records, labels):
    """Validate deterministic retrospective thinning and sparsity regimes."""
    print("  Testing observation thinning...")
    regimes = build_sparsity_regimes(records, seed=42)
    for name, recs in regimes.items():
        mean_len = np.mean([len(r["times"]) for r in recs])
        print(f"    {name}: mean T = {mean_len:.1f}")
    assert regimes["sparse"][0]["times"].shape[0] <= regimes["dense"][0]["times"].shape[0]
    print("  OK")
    return regimes


def test_grud(records, labels, regimes, D):
    """Check GRU-D training-time, MC-dropout, and trajectory inference paths."""
    print("  Testing GRU-D...")
    device = torch.device("cpu")
    model  = GRUD(input_size=D, hidden_size=16, dropout=0.3).to(device)

    ds  = PatientDataset(records, labels)
    dl  = DataLoader(ds, batch_size=8, collate_fn=pad_collate, shuffle=False)

    batch = next(iter(dl))
    times    = batch["times"].to(device)
    values   = batch["values"].to(device)
    mask     = batch["mask"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    lengths  = batch["lengths"].to(device)

    # Training forward
    model.train()
    logits = _forward_train(model, times, values, mask, obs_mask, lengths)
    assert logits.shape == (len(batch["labels"]),), f"bad shape: {logits.shape}"

    # MC Dropout
    mc = model(times, values, mask, obs_mask, lengths, n_mc_samples=5)
    assert mc.shape == (len(batch["labels"]), 5)

    # Trajectory
    out = model(times, values, mask, obs_mask, lengths,
                n_mc_samples=1, return_trajectory=True)
    logit_f, h_traj, risk_traj, delta_h = out
    B = len(batch["labels"])
    T = times.shape[1]
    assert h_traj.shape[0]    == B
    assert risk_traj.shape[1] == T
    print(f"    logits: {logits.shape}, mc: {mc.shape}, traj: {risk_traj.shape}")
    print("  OK")
    return model


def test_latent_ode(records, labels, D):
    """Check latent ODE ELBO and prediction paths."""
    print("  Testing Latent ODE...")
    device = torch.device("cpu")
    model  = LatentODE(input_size=D, hidden_dim=16, latent_dim=8,
                       n_samples=3).to(device)

    ds     = PatientDataset(records, labels)
    dl     = DataLoader(ds, batch_size=4, collate_fn=pad_collate, shuffle=False)
    batch  = next(iter(dl))

    times    = batch["times"].to(device)
    values   = batch["values"].to(device)
    mask_b   = batch["mask"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    labels_t = batch["labels"].to(device)
    lengths  = batch["lengths"].to(device)

    # ELBO
    model.train()
    loss = model.elbo(times, values, mask_b, obs_mask, labels_t, lengths)
    assert loss.ndim == 0 and torch.isfinite(loss), f"bad loss: {loss}"

    # Predict
    mean_risk, epi_u = model.predict(times, values, mask_b, obs_mask, lengths)
    assert mean_risk.shape == (len(batch["labels"]),)
    print(f"    loss: {loss.item():.4f}, risk: {mean_risk[:3].tolist()}")
    print("  OK")
    return model


def test_transformer(records, labels, D):
    """Check transformer logits, MC-dropout sampling, and trajectory outputs."""
    print("  Testing Transformer...")
    device = torch.device("cpu")
    model = TimeAwareTransformer(input_size=D, hidden_dim=16,
                                 num_heads=4, num_layers=2, dropout=0.2).to(device)
    ds = PatientDataset(records, labels)
    dl = DataLoader(ds, batch_size=4, collate_fn=pad_collate, shuffle=False)
    batch = next(iter(dl))

    times = batch["times"].to(device)
    values = batch["values"].to(device)
    mask = batch["mask"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    lengths = batch["lengths"].to(device)

    logits = model(times, values, mask, obs_mask, lengths)
    mc = model(times, values, mask, obs_mask, lengths, n_mc_samples=4)
    _, _, risk_traj, delta_h = model(
        times, values, mask, obs_mask, lengths,
        n_mc_samples=1, return_trajectory=True
    )
    assert logits.shape == (len(batch["labels"]),)
    assert mc.shape == (len(batch["labels"]), 4)
    assert risk_traj.shape == delta_h.shape
    print(f"    logits: {logits.shape}, mc: {mc.shape}, traj: {risk_traj.shape}")
    print("  OK")
    return model


def test_state_space(records, labels, D):
    """Check state-space logits and filtered trajectory diagnostics."""
    print("  Testing State-Space...")
    device = torch.device("cpu")
    model = StateSpaceRiskModel(input_size=D, state_dim=8).to(device)
    ds = PatientDataset(records, labels)
    dl = DataLoader(ds, batch_size=4, collate_fn=pad_collate, shuffle=False)
    batch = next(iter(dl))

    times = batch["times"].to(device)
    values = batch["values"].to(device)
    mask = batch["mask"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    lengths = batch["lengths"].to(device)
    logits = model(times, values, mask, obs_mask, lengths)
    result = model(times, values, mask, obs_mask, lengths, return_trajectory=True)
    assert logits.shape == (len(batch["labels"]),)
    assert result["risk_after"].shape[0] == len(batch["labels"])
    print(f"    logits: {logits.shape}, traj: {result['risk_after'].shape}")
    print("  OK")
    return model


def test_landmarking(records, labels):
    """Fit the landmarking baseline and verify basic prediction outputs."""
    print("  Testing Landmarking...")
    ds = PatientDataset(records, labels)
    model = LandmarkingModel(landmark_grid_hours=6.0, n_bootstrap_samples=3)
    model.fit(ds)
    probs, y, gaps = model.predict_dataset(ds)
    assert probs.ndim == 1
    print(f"    patients: {len(probs)}, landmarks: {len(model.bundles)}")
    print("  OK")
    return model


def test_guideline_baselines():
    """Fit guideline baselines on synthetic rows and run the stress-test suite."""
    print("  Testing guideline baselines...")
    D = len(PHYSIONET_VARS)
    fmap = {name: idx for idx, name in enumerate(PHYSIONET_VARS)}

    def make_record(pid, measurements, label):
        times = np.array([0.0, 6.0], dtype=np.float32)
        values = np.zeros((2, D), dtype=np.float32)
        mask = np.zeros((2, D), dtype=np.float32)
        for t, row in enumerate(measurements):
            for name, value in row.items():
                idx = fmap[name]
                values[t, idx] = float(value)
                mask[t, idx] = 1.0
        return {
            "patient_id": pid,
            "times": times,
            "values": values,
            "mask": mask,
            "label": float(label),
            "prediction_times": times.copy(),
            "eligible_mask": np.array([True, True]),
        }

    records = [
        make_record("0", [
            {"RespRate": 18, "SysABP": 120, "GCS": 15, "Temp": 37.0, "HR": 82, "WBC": 7.0, "Platelets": 250, "Bilirubin": 0.7, "Creatinine": 0.8, "MAP": 85, "PaO2": 120, "FiO2": 0.3},
            {"RespRate": 26, "SysABP": 92, "GCS": 13, "Temp": 39.0, "HR": 112, "WBC": 15.0, "Platelets": 90, "Bilirubin": 3.5, "Creatinine": 2.4, "MAP": 65, "PaO2": 80, "FiO2": 0.5},
        ], 1),
        make_record("1", [
            {"RespRate": 16, "SysABP": 130, "GCS": 15, "Temp": 36.7, "HR": 78, "WBC": 8.0, "Platelets": 210, "Bilirubin": 0.8, "Creatinine": 0.9, "MAP": 82, "PaO2": 115, "FiO2": 0.4},
            {"RespRate": 19, "SysABP": 118, "GCS": 15, "Temp": 37.0, "HR": 84, "WBC": 8.5, "Platelets": 180, "Bilirubin": 0.9, "Creatinine": 1.0, "MAP": 78, "PaO2": 110, "FiO2": 0.4},
        ], 0),
        make_record("2", [
            {"RespRate": 20, "SysABP": 122, "GCS": 15, "Temp": 37.1, "HR": 88, "WBC": 6.5, "Platelets": 170, "Bilirubin": 1.0, "Creatinine": 1.1, "MAP": 76, "PaO2": 105, "FiO2": 0.4},
            {"RespRate": 28, "SysABP": 88, "GCS": 12, "Temp": 39.3, "HR": 120, "WBC": 16.0, "Platelets": 110, "Bilirubin": 2.8, "Creatinine": 3.2, "MAP": 62, "PaO2": 70, "FiO2": 0.6},
        ], 1),
        make_record("3", [
            {"RespRate": 17, "SysABP": 124, "GCS": 15, "Temp": 36.6, "HR": 74, "WBC": 7.2, "Platelets": 230, "Bilirubin": 0.7, "Creatinine": 0.8, "MAP": 88, "PaO2": 125, "FiO2": 0.3},
            {"RespRate": 18, "SysABP": 120, "GCS": 15, "Temp": 36.8, "HR": 80, "WBC": 7.5, "Platelets": 210, "Bilirubin": 0.8, "Creatinine": 0.9, "MAP": 83, "PaO2": 118, "FiO2": 0.3},
        ], 0),
    ]
    labels = {r["patient_id"]: int(r["label"]) for r in records}
    meta = {"feature_map": fmap}
    ds = PatientDataset(records, labels=labels, meta=meta)

    fitted = {}
    for baseline_cls in [QSOFABaseline, SIRSBaseline, SOFABaseline]:
        baseline = baseline_cls().fit(ds, meta)
        fitted[baseline.name] = baseline

    rows = predict_baseline_patient_rows(fitted["qsofa"], ds, meta)
    assert len(rows) == len(ds)
    assert all("covered" in row for row in rows.values())

    class Cfg:
        retention_levels = [1.0, 0.5]
        n_repeats = 2
        seed = 13

    baseline_results = run_all_stress_tests(fitted, ds, meta, Cfg)
    expected = {
        "coverage_decay",
        "component_availability",
        "thinning_sensitivity_covered",
        "repeated_thinning_availability",
        "prefix_availability_volatility",
        "sparsity_performance_covered",
    }
    assert expected.issubset(baseline_results["qsofa"].keys())
    out_dir = Path("/tmp/temporal_coherence_stress_test_smoke")
    save_stress_test_outputs(baseline_results, out_dir)
    assert (out_dir / "stress_test_results.json").exists()
    assert (out_dir / "summary.md").exists()
    print("  OK")


def test_perturbation_suite(records, labels, model):
    """Run the learned-model perturbation suite on a small synthetic subset."""
    print("  Testing controlled perturbation suite...")
    device = torch.device("cpu")
    ds = PatientDataset(records[:12], labels)
    rng = np.random.default_rng(123)
    original = records[0]

    thinned_a = thin_record_fraction(original, 0.5, rng)
    assert thinned_a["times"][0] == original["times"][0]
    assert thinned_a["times"][-1] == original["times"][-1]

    same_1 = thin_records(records[:4], 0.5, seed=99)
    same_2 = thin_records(records[:4], 0.5, seed=99)
    diff_1 = thin_records(records[:4], 0.5, seed=100)
    assert all(np.array_equal(a["times"], b["times"]) for a, b in zip(same_1, same_2))
    assert any(not np.array_equal(a["times"], b["times"]) for a, b in zip(same_1, diff_1))

    stretched = stretch_record_times(original, 2.0)
    assert np.array_equal(stretched["values"], original["values"])
    assert np.array_equal(stretched["mask"], original["mask"])
    assert not np.array_equal(stretched["times"], original["times"])

    preds = predict_patient_probs("grud", model, ds, device, batch_size=4)
    assert len(preds) == len(ds)
    assert all(0.0 <= item["prob"] <= 1.0 for item in preds.values())

    class Cfg:
        batch_size = 4
        n_ode_samples = 2
        retention_levels = [1.0, 0.5]
        n_repeats = 2
        timestamp_factors = [1.0, 2.0]
        seed = 17

    results = run_all_perturbation_tests({"grud": model}, ds, device, Cfg)
    expected = {
        "thinning_sensitivity",
        "repeated_thinning_instability",
        "sparsity_response",
        "prefix_volatility",
        "timestamp_stretching",
    }
    assert expected.issubset(results["grud"].keys())
    out_dir = Path("/tmp/temporal_coherence_perturbation_smoke")
    save_perturbation_outputs(results, out_dir)
    assert (out_dir / "perturbation_test_results.json").exists()
    assert (out_dir / "summary.md").exists()
    print("  OK")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Perturbation Suite — Smoke Test")
    print("=" * 60)

    D        = 10
    records, labels = make_synthetic_patients(n=80, D=D, seed=0)

    regimes = test_thinning(records, labels)
    grud    = test_grud(records, labels, regimes, D)
    lode    = test_latent_ode(records, labels, D)
    transformer = test_transformer(records, labels, D)
    state_space = test_state_space(records, labels, D)
    landmarking = test_landmarking(records, labels)
    test_guideline_baselines()
    test_perturbation_suite(records, labels, grud)

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
    print("\nTo run the perturbation scripts:")
    print("  python scripts/run_perturbation_test.py")
    print("  python scripts/run_stress_tests.py")
