"""
Landmarking baseline using per-landmark logistic regression models.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _patient_last_values(record, landmark_time):
    """Build landmark features from history available up to one landmark time."""
    values = record["values"]
    mask = record["mask"]
    times = record["times"]
    D = values.shape[1]

    latest = np.zeros(D, dtype=np.float32)
    since = np.full(D, landmark_time, dtype=np.float32)
    counts = np.zeros(D, dtype=np.float32)

    valid_idx = np.where(times <= landmark_time)[0]
    if len(valid_idx) == 0:
        return np.concatenate([latest, since, counts], axis=0)

    for idx in valid_idx:
        t = times[idx]
        obs = mask[idx] > 0.5
        latest[obs] = values[idx, obs]
        since[obs] = landmark_time - t
        counts += mask[idx]

    return np.concatenate([latest, since, counts], axis=0)


@dataclass
class LandmarkModelBundle:
    """One fitted landmark ensemble plus its feature standardization state."""
    landmark_time: float
    models: list
    mean: np.ndarray
    std: np.ndarray


def _fit_standardizer(X: np.ndarray):
    """Compute per-feature mean/std with a floor for near-constant features."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Apply the stored landmark feature standardization."""
    return ((X - mean) / std).astype(np.float32)


class LandmarkingModel:
    """Fit one bootstrap logistic-regression ensemble per landmark time."""
    def __init__(self, landmark_grid_hours: float = 6.0,
                 n_bootstrap_samples: int = 10,
                 min_samples: int = 12):
        self.landmark_grid_hours = landmark_grid_hours
        self.n_bootstrap_samples = n_bootstrap_samples
        self.min_samples = min_samples
        self.bundles = []

    def fit(self, dataset):
        """Fit one bootstrap logistic-regression ensemble per landmark time."""
        records = dataset.records
        labels_map = dataset.labels
        max_t = max(float(r["times"][-1]) for r in records)
        landmarks = np.arange(self.landmark_grid_hours, max_t + 1e-6,
                              self.landmark_grid_hours)

        rng = np.random.default_rng(42)
        bundles = []
        for landmark in landmarks:
            X, y = [], []
            for record in records:
                if record["times"][0] > landmark:
                    continue
                features = _patient_last_values(record, landmark)
                label = labels_map.get(record["patient_id"], 0)
                X.append(features)
                y.append(int(label))

            if len(X) < self.min_samples:
                continue
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            if len(np.unique(y)) < 2:
                continue
            mean, std = _fit_standardizer(X)
            X_scaled = _apply_standardizer(X, mean, std)

            models = []
            for _ in range(self.n_bootstrap_samples):
                sample_idx = rng.integers(0, len(X_scaled), size=len(X_scaled))
                model = LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                )
                model.fit(X_scaled[sample_idx], y[sample_idx])
                models.append(model)
            bundles.append(LandmarkModelBundle(float(landmark), models, mean, std))

        self.bundles = bundles
        return self

    def predict_at(self, record, landmark_time):
        """Predict risk at the latest fitted landmark not after `landmark_time`."""
        if not self.bundles:
            raise RuntimeError("landmarking model is not fitted")
        eligible = [b for b in self.bundles if b.landmark_time <= landmark_time + 1e-6]
        if not eligible:
            return None
        bundle = eligible[-1]
        feat = _patient_last_values(record, bundle.landmark_time).reshape(1, -1)
        feat = _apply_standardizer(feat, bundle.mean, bundle.std)
        probs = np.array([m.predict_proba(feat)[0, 1] for m in bundle.models], dtype=float)
        return {
            "landmark_time": bundle.landmark_time,
            "mean": float(probs.mean()),
            "var": float(probs.var()),
        }

    def predict_dataset(self, dataset):
        """Predict one final risk per record and return auxiliary gap summaries."""
        probs, labels, gaps = [], [], []
        for record in dataset.records:
            pred = self.predict_at(record, float(record["times"][-1]))
            if pred is None:
                continue
            probs.append(pred["mean"])
            labels.append(float(dataset.labels.get(record["patient_id"], 0)))
            if len(record["times"]) >= 2:
                gaps.append(float(np.diff(record["times"]).mean()))
            else:
                gaps.append(0.0)
        return np.asarray(probs), np.asarray(labels), np.asarray(gaps)

    def score_auc(self, dataset):
        """Compute validation AUC for the current fitted landmark ensembles."""
        probs, labels, _ = self.predict_dataset(dataset)
        if len(probs) == 0 or len(np.unique(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, probs))
        except Exception:
            return 0.5

    def save(self, path: Path):
        """Serialize the fitted landmarking model with pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        """Load a serialized landmarking model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def train_landmarking(model, train_dataset, val_dataset,
                      verbose: bool = True):
    """Fit the landmarking model and report validation AUC."""
    model.fit(train_dataset)
    auc = model.score_auc(val_dataset)
    if verbose:
        print(f"  Landmarking val AUC {auc:.4f}")
    return auc
