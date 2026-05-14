"""Shared dataset wrappers and generic sequence preprocessing utilities."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import Dataset


def clone_records(records):
    """Deep-copy record lists so task adapters can annotate safely."""
    return [copy.deepcopy(r) for r in records]


def fit_normalizer(records):
    """Compute per-feature mean and std from observed values."""
    all_vals = []
    all_masks = []
    for r in records:
        obs = r["values"] * r["mask"]
        all_vals.append(obs.reshape(-1, obs.shape[-1]))
        all_masks.append(r["mask"].reshape(-1, r["mask"].shape[-1]))
    all_vals = np.concatenate(all_vals, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    counts = all_masks.sum(axis=0).astype(np.float32)
    mean = all_vals.sum(axis=0) / np.maximum(counts, 1)
    sq = (all_vals ** 2).sum(axis=0) / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(sq - mean ** 2, 1e-8))
    return mean, std


def apply_normalizer(records, mean, std):
    """Normalise observed values in-place; unobserved remain zero."""
    for r in records:
        norm = (r["values"] - mean) / std
        r["values"] = norm * r["mask"]
    return records


def thin_record(record: dict, keep_fraction: float, rng: np.random.Generator) -> dict:
    """Retrospectively thin one record while preserving the first and last step."""
    T = len(record["times"])
    if T <= 2:
        return record.copy()

    interior = np.arange(1, T - 1)
    keep_mask = rng.random(len(interior)) < keep_fraction
    keep_idx = np.concatenate([[0], interior[keep_mask], [T - 1]])
    keep_idx = np.unique(keep_idx)

    return {
        "patient_id": record["patient_id"],
        "times": record["times"][keep_idx],
        "values": record["values"][keep_idx],
        "mask": record["mask"][keep_idx],
    }


def build_sparsity_regimes(records, seed: int = 42):
    """Build dense, moderate, and sparse views of the same record collection."""
    regimes = {}
    for name, frac in [("dense", 1.0), ("moderate", 0.5), ("sparse", 0.2)]:
        rng_local = np.random.default_rng(seed)
        regimes[name] = [thin_record(r, frac, rng_local) for r in records]
    return regimes


def stratified_split_records(records, labels, val_frac: float = 0.15,
                             test_frac: float = 0.15, seed: int = 42):
    """Create reproducible stratified train/val/test splits."""
    rng = np.random.default_rng(seed)
    pos = [r for r in records if labels.get(r["patient_id"], 0) == 1]
    neg = [r for r in records if labels.get(r["patient_id"], 0) == 0]

    def split_list(lst):
        """Split one class-specific record list into train/val/test partitions."""
        n = len(lst)
        idx = rng.permutation(n)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        return (
            [lst[i] for i in idx[n_val + n_test:]],
            [lst[i] for i in idx[:n_val]],
            [lst[i] for i in idx[n_val:n_val + n_test]],
        )

    pos_tr, pos_va, pos_te = split_list(pos)
    neg_tr, neg_va, neg_te = split_list(neg)
    return {
        "train": pos_tr + neg_tr,
        "val": pos_va + neg_va,
        "test": pos_te + neg_te,
    }


def pad_collate(batch):
    """Collate variable-length sequences into padded tensors."""
    times, values, masks, labels, lengths = zip(*[
        (b["times"], b["values"], b["mask"], b["label"], len(b["times"]))
        for b in batch
    ])

    B = len(times)
    T = max(lengths)
    D = values[0].shape[1]

    t_pad = torch.zeros(B, T)
    v_pad = torch.zeros(B, T, D)
    m_pad = torch.zeros(B, T, D)
    o_pad = torch.zeros(B, T)

    for i, (t, v, m, l) in enumerate(zip(times, values, masks, lengths)):
        t_pad[i, :l] = torch.tensor(t)
        v_pad[i, :l] = torch.tensor(v)
        m_pad[i, :l] = torch.tensor(m)
        o_pad[i, :l] = 1.0

    return {
        "times": t_pad,
        "values": v_pad,
        "mask": m_pad,
        "obs_mask": o_pad,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


class PatientDataset(Dataset):
    """Wrap patient records and labels in a PyTorch-compatible dataset."""

    def __init__(self, records, labels=None, meta=None):
        self.records = records
        self.labels = labels or {}
        self.meta = meta or {}

    def __len__(self):
        """Return the number of patient trajectories in the dataset."""
        return len(self.records)

    def __getitem__(self, idx):
        """Return one patient trajectory with its scalar outcome label."""
        r = self.records[idx]
        label = r.get("label", self.labels.get(r["patient_id"], 0))
        return {
            "times": r["times"],
            "values": r["values"],
            "mask": r["mask"],
            "label": float(label),
        }

