"""
Guideline-style clinical score baselines with calibrated probabilities.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def _sigmoid(x):
    """Numerically stable sigmoid used by the fallback score-to-risk mapping."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _obs_step(mask_row):
    """Return whether a time step has any observed variables at all."""
    return bool(np.any(mask_row > 0.5))


@dataclass
class ScorePoint:
    """One time-stamped clinical score evaluation with calibration metadata."""
    time: float
    score: float
    available_components: int
    total_components: int
    component_flags: dict
    complete: bool
    proxy_uncertainty: float | None = None
    calibrated_prob: float | None = None


class GuidelineBaseline:
    """Base class for score-based clinical baselines with probability calibration."""
    name = "baseline"
    total_components = 0
    supports_partial_score = False

    def __init__(self):
        self.calibrator = None
        self.score_stats = {}
        self.global_variance = 0.25
        self.global_prevalence = 0.5

    def save(self, path: Path):
        """Serialize the fitted baseline and its calibrator."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        """Load a serialized baseline from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _feature_index(self, meta, name):
        """Resolve one feature name to its dataset column index."""
        return meta["feature_map"].get(name)

    def _value_at(self, record, t, meta, name):
        """Read one observed feature value at a given step, or return None."""
        idx = self._feature_index(meta, name)
        if idx is None:
            return None
        if record["mask"][t, idx] <= 0.5:
            return None
        return float(record["values"][t, idx])

    def _score_point(self, record, t, meta):
        raise NotImplementedError

    def score_trajectory(self, record, meta):
        """Compute all eligible score evaluations for one patient trajectory."""
        points = []
        eligible = record.get("eligible_mask")
        if eligible is None:
            eligible = np.ones(len(record["times"]), dtype=bool)
        for t in range(len(record["times"])):
            if not eligible[t] or not _obs_step(record["mask"][t]):
                continue
            point = self._score_point(record, t, meta)
            if point is not None:
                points.append(point)
        return points

    def _train_rows(self, dataset, meta):
        """Build calibrator feature rows from all score points and final scores."""
        rows = []
        labels = []
        final_rows = []
        final_labels = []
        for record in dataset.records:
            label = float(record.get("label", dataset.labels.get(record["patient_id"], 0)))
            points = self.score_trajectory(record, meta)
            for point in points:
                rows.append(self._row_features(point))
                labels.append(label)
            if points:
                final_rows.append(self._row_features(points[-1]))
                final_labels.append(label)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float), np.asarray(final_rows, dtype=float), np.asarray(final_labels, dtype=float)

    def _row_features(self, point: ScorePoint):
        """Map one score point to the calibrator feature vector."""
        return np.array([
            float(point.score),
            float(point.available_components),
            float(point.total_components),
            float(int(point.complete)),
        ], dtype=float)

    def fit(self, train_dataset, meta):
        """Fit score statistics and an optional logistic calibrator."""
        rows, labels, final_rows, final_labels = self._train_rows(train_dataset, meta)
        prevalence = float(np.mean(labels)) if len(labels) else 0.5
        self.global_prevalence = prevalence
        self.global_variance = max(prevalence * (1.0 - prevalence), 1e-4)

        stats = {}
        if len(rows):
            for row, label in zip(rows, labels):
                key = (int(round(row[0])), int(row[1]), int(row[3]))
                stats.setdefault(key, []).append(float(label))
        self.score_stats = {
            key: {
                "count": len(vals),
                "prevalence": float(np.mean(vals)),
                "variance": float(max(np.mean(vals) * (1.0 - np.mean(vals)), 1e-4)),
            }
            for key, vals in stats.items()
        }

        if len(final_rows) >= 8 and len(np.unique(final_labels)) > 1:
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(final_rows, final_labels.astype(int))
            self.calibrator = model
        else:
            self.calibrator = None
        return self

    def _fallback_variance(self, point: ScorePoint):
        """Pool nearby empirical score bins when exact variance stats are sparse."""
        if not self.score_stats:
            return self.global_variance
        target_score = float(point.score)
        target_avail = int(point.available_components)
        target_complete = int(point.complete)
        candidates = []
        for key, stats in self.score_stats.items():
            score_key, avail_key, complete_key = key
            penalty = abs(score_key - target_score)
            if avail_key != target_avail:
                penalty += 0.5
            if complete_key != target_complete:
                penalty += 0.25
            candidates.append((penalty, stats["variance"]))
        if not candidates:
            return self.global_variance
        candidates.sort(key=lambda x: x[0])
        pooled = [var for _, var in candidates[:3]]
        return float(np.mean(pooled))

    def proxy_uncertainty(self, point: ScorePoint):
        """Estimate score uncertainty from empirical label variance by score state."""
        key = (int(round(point.score)), int(point.available_components), int(point.complete))
        stats = self.score_stats.get(key)
        if stats and stats["count"] >= 4:
            return float(stats["variance"])
        return float(self._fallback_variance(point))

    def calibrated_probability(self, point: ScorePoint):
        """Map one raw score point to a mortality probability."""
        row = self._row_features(point).reshape(1, -1)
        if self.calibrator is not None:
            return float(self.calibrator.predict_proba(row)[0, 1])
        logit = np.log(self.global_prevalence / max(1.0 - self.global_prevalence, 1e-6))
        logit += 0.35 * float(point.score)
        return float(np.clip(_sigmoid(logit), 1e-4, 1 - 1e-4))

    def predict_dataset(self, dataset, meta):
        """Return score trajectories and calibrated outputs for each patient."""
        outputs = []
        for record in dataset.records:
            points = self.score_trajectory(record, meta)
            for point in points:
                point.proxy_uncertainty = self.proxy_uncertainty(point)
                point.calibrated_prob = self.calibrated_probability(point)
            outputs.append({
                "patient_id": record["patient_id"],
                "label": float(record.get("label", dataset.labels.get(record["patient_id"], 0))),
                "points": points,
            })
        return outputs


class QSOFABaseline(GuidelineBaseline):
    name = "qsofa"
    total_components = 3
    supports_partial_score = False

    def _score_point(self, record, t, meta):
        rr = self._value_at(record, t, meta, "RespRate")
        sbp = self._value_at(record, t, meta, "SysABP")
        gcs = self._value_at(record, t, meta, "GCS")
        flags = {
            "resp_rate_ge_22": rr is not None,
            "sbp_le_100": sbp is not None,
            "gcs_lt_15": gcs is not None,
        }
        if not all(flags.values()):
            return None
        score = float((rr >= 22.0) + (sbp <= 100.0) + (gcs < 15.0))
        return ScorePoint(
            time=float(record["times"][t]),
            score=score,
            available_components=3,
            total_components=3,
            component_flags=flags,
            complete=True,
        )


class SIRSBaseline(GuidelineBaseline):
    name = "sirs"
    total_components = 4
    supports_partial_score = False

    def _score_point(self, record, t, meta):
        temp = self._value_at(record, t, meta, "Temp")
        hr = self._value_at(record, t, meta, "HR")
        rr = self._value_at(record, t, meta, "RespRate")
        wbc = self._value_at(record, t, meta, "WBC")
        flags = {
            "temp_available": temp is not None,
            "hr_available": hr is not None,
            "rr_available": rr is not None,
            "wbc_available": wbc is not None,
        }
        if not all(flags.values()):
            return None
        score = float((temp > 38.0 or temp < 36.0) + (hr > 90.0) + (rr > 20.0) + (wbc > 12.0 or wbc < 4.0))
        return ScorePoint(
            time=float(record["times"][t]),
            score=score,
            available_components=4,
            total_components=4,
            component_flags=flags,
            complete=True,
        )


class SOFABaseline(GuidelineBaseline):
    name = "sofa"
    total_components = 6
    supports_partial_score = True

    def _score_point(self, record, t, meta):
        platelets = self._value_at(record, t, meta, "Platelets")
        bilirubin = self._value_at(record, t, meta, "Bilirubin")
        creatinine = self._value_at(record, t, meta, "Creatinine")
        gcs = self._value_at(record, t, meta, "GCS")
        map_v = self._value_at(record, t, meta, "MAP")
        pao2 = self._value_at(record, t, meta, "PaO2")
        fio2 = self._value_at(record, t, meta, "FiO2")

        components = {}
        if platelets is not None:
            if platelets < 20:
                components["coagulation"] = 4
            elif platelets < 50:
                components["coagulation"] = 3
            elif platelets < 100:
                components["coagulation"] = 2
            elif platelets < 150:
                components["coagulation"] = 1
            else:
                components["coagulation"] = 0

        if bilirubin is not None:
            if bilirubin >= 12.0:
                components["liver"] = 4
            elif bilirubin >= 6.0:
                components["liver"] = 3
            elif bilirubin >= 2.0:
                components["liver"] = 2
            elif bilirubin >= 1.2:
                components["liver"] = 1
            else:
                components["liver"] = 0

        if creatinine is not None:
            if creatinine >= 5.0:
                components["renal"] = 4
            elif creatinine >= 3.5:
                components["renal"] = 3
            elif creatinine >= 2.0:
                components["renal"] = 2
            elif creatinine >= 1.2:
                components["renal"] = 1
            else:
                components["renal"] = 0

        if gcs is not None:
            if gcs < 6:
                components["cns"] = 4
            elif gcs < 10:
                components["cns"] = 3
            elif gcs < 13:
                components["cns"] = 2
            elif gcs < 15:
                components["cns"] = 1
            else:
                components["cns"] = 0

        if map_v is not None:
            components["cardiovascular"] = 1 if map_v < 70.0 else 0

        if pao2 is not None and fio2 is not None and fio2 > 0:
            ratio = pao2 / max(fio2, 1e-6)
            if ratio < 100:
                components["respiratory"] = 4
            elif ratio < 200:
                components["respiratory"] = 3
            elif ratio < 300:
                components["respiratory"] = 2
            elif ratio < 400:
                components["respiratory"] = 1
            else:
                components["respiratory"] = 0

        if not components:
            return None

        score = float(sum(components.values()))
        available = len(components)
        return ScorePoint(
            time=float(record["times"][t]),
            score=score,
            available_components=available,
            total_components=self.total_components,
            component_flags={k: True for k in components},
            complete=(available == self.total_components),
        )
