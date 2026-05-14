"""PhysioNet 2012 dataset download, parsing, and split construction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from datasets.common import (
    PatientDataset,
    apply_normalizer,
    build_sparsity_regimes,
    clone_records,
    fit_normalizer,
    stratified_split_records,
)


PHYSIONET_URL = "https://physionet.org/files/challenge-2012/1.0.0/set-a.zip"
OUTCOMES_URL = "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"

PHYSIONET_VARS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP",
    "NIMAP", "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
]


def download_physionet_2012(raw_dir: Path) -> None:
    """Download and unpack PhysioNet 2012 set-a if it is not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "set-a.zip"
    outcomes_path = raw_dir / "Outcomes-a.txt"

    if not zip_path.exists():
        print("Downloading PhysioNet 2012 set-a (~11 MB)...")
        r = requests.get(PHYSIONET_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if not (raw_dir / "set-a").exists():
        print("Unpacking set-a.zip...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    if not outcomes_path.exists():
        print("Downloading outcomes...")
        r = requests.get(OUTCOMES_URL, timeout=60)
        r.raise_for_status()
        outcomes_path.write_bytes(r.content)

    print("PhysioNet 2012 data ready.")


def parse_physionet_patient(txt_path: Path) -> pd.DataFrame:
    """Parse one PhysioNet patient file into a long-format observation table."""
    df = pd.read_csv(txt_path)
    df = df[~df["Parameter"].isin(["RecordID", "Age", "Gender", "Height",
                                    "ICUType", "Weight"])]
    df = df.dropna(subset=["Value"])
    df = df.rename(columns={"Time": "time_str", "Parameter": "variable",
                             "Value": "value"})

    def parse_time(s):
        """Convert PhysioNet HH:MM timestamps into elapsed hours."""
        h, m = s.split(":")
        return int(h) + int(m) / 60.0

    df["time"] = df["time_str"].apply(parse_time)
    return df[["time", "variable", "value"]].reset_index(drop=True)


def build_physionet_tensors(raw_dir: Path, max_patients: int = None):
    """Build patient trajectory tensors and mortality labels from PhysioNet files."""
    set_a_dir = raw_dir / "set-a"
    outcomes_path = raw_dir / "Outcomes-a.txt"

    outcomes_df = pd.read_csv(outcomes_path)
    labels = dict(zip(outcomes_df["RecordID"].astype(str),
                      outcomes_df["In-hospital_death"].astype(int)))

    var_index = {v: i for i, v in enumerate(PHYSIONET_VARS)}
    D = len(PHYSIONET_VARS)

    patient_files = sorted(set_a_dir.glob("*.txt"))
    if max_patients:
        patient_files = patient_files[:max_patients]

    records = []
    for path in tqdm(patient_files, desc="Parsing patients"):
        pid = path.stem
        if pid not in labels:
            continue
        try:
            df = parse_physionet_patient(path)
        except Exception:
            continue

        df = df[df["variable"].isin(var_index)]
        if df.empty:
            continue

        df["time_bin"] = (df["time"] / 0.5).round() * 0.5
        time_points = sorted(df["time_bin"].unique())

        values = np.zeros((len(time_points), D), dtype=np.float32)
        mask = np.zeros((len(time_points), D), dtype=np.float32)

        tp_index = {t: i for i, t in enumerate(time_points)}
        for _, row in df.iterrows():
            ti = tp_index[row["time_bin"]]
            vi = var_index[row["variable"]]
            values[ti, vi] = row["value"]
            mask[ti, vi] = 1.0

        records.append({
            "patient_id": pid,
            "times": np.array(time_points, dtype=np.float32),
            "values": values,
            "mask": mask,
        })

    return records, labels


def load_physionet_2012_source(raw_dir: str = "data/raw",
                               max_patients: int = None,
                               val_frac: float = 0.15,
                               test_frac: float = 0.15,
                               seed: int = 42):
    """Load PhysioNet 2012 into generic trajectory splits plus outcome metadata."""
    raw_dir = Path(raw_dir)
    download_physionet_2012(raw_dir)
    records, labels = build_physionet_tensors(raw_dir, max_patients)

    split_records = stratified_split_records(
        records,
        labels,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    mean, std = fit_normalizer(split_records["train"])

    splits = {}
    for split_name, split_recs in split_records.items():
        splits[split_name] = apply_normalizer(clone_records(split_recs), mean, std)

    regimes = {}
    val_regimes = build_sparsity_regimes(splits["val"], seed=seed)
    test_regimes = build_sparsity_regimes(splits["test"], seed=seed)
    for regime_name in ["dense", "moderate", "sparse"]:
        regimes[regime_name] = {
            "train": clone_records(splits["train"]),
            "val": clone_records(val_regimes[regime_name]),
            "test": clone_records(test_regimes[regime_name]),
        }

    n_pos = sum(labels.get(r["patient_id"], 0) for r in split_records["train"])
    meta = {
        "dataset_name": "physionet2012",
        "n_features": len(PHYSIONET_VARS),
        "n_train": len(split_records["train"]),
        "n_val": len(split_records["val"]),
        "n_test": len(split_records["test"]),
        "label_prevalence": n_pos / max(len(split_records["train"]), 1),
        "feature_names": PHYSIONET_VARS,
        "feature_map": {name: idx for idx, name in enumerate(PHYSIONET_VARS)},
        "normalizer": (mean, std),
        "outcome_labels": labels,
        "raw_dir": str(raw_dir),
    }
    return splits, regimes, meta
