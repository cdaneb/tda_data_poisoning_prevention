"""Clean-only novelty detectors and calibration for Phase Q Frame B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM


class KNNDistance:
    def __init__(self, n_neighbors: int = 10, n_jobs: int = 1):
        self.n_neighbors, self.n_jobs = n_neighbors, n_jobs
    def fit(self, X):
        self.model_ = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs).fit(X)
        return self
    def score_samples(self, X):
        return self.model_.kneighbors(X, return_distance=True)[0][:, -1]


class HigherIsAnomaly:
    def __init__(self, estimator): self.estimator = estimator
    def fit(self, X): self.estimator.fit(X); return self
    def score_samples(self, X): return -self.estimator.score_samples(X)


def detector_factories(seed: int, n_jobs: int = 1) -> dict[str, Callable[[], object]]:
    return {
        "knn_distance": lambda: KNNDistance(10, n_jobs),
        "empirical_knn": lambda: KNNDistance(10, n_jobs),
        "lof_novelty": lambda: HigherIsAnomaly(LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=n_jobs)),
        "isolation_forest": lambda: HigherIsAnomaly(IsolationForest(n_estimators=200, random_state=seed, n_jobs=n_jobs)),
        "one_class_svm": lambda: HigherIsAnomaly(OneClassSVM(kernel="rbf", gamma="scale", nu=.01)),
    }


def clean_threshold(scores: np.ndarray, removal_budget: float) -> float:
    """Conservative higher-tail threshold fitted solely from calibration clean."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("calibration scores must be a finite nonempty vector")
    if not 0 < removal_budget < 1: raise ValueError("budget must lie in (0, 1)")
    return float(np.quantile(scores, 1-removal_budget, method="higher"))


def calibrated_pvalues(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Finite-sample upper-tail conformal p-values (larger score is stranger)."""
    cal = np.sort(np.asarray(calibration_scores, dtype=float))
    scores = np.asarray(scores, dtype=float)
    return (len(cal) - np.searchsorted(cal, scores, side="left") + 1) / (len(cal)+1)


def evaluate_threshold(clean_scores, poison_scores, threshold):
    cr = np.asarray(clean_scores) >= threshold; pr = np.asarray(poison_scores) >= threshold
    removed = int(cr.sum()+pr.sum())
    return {"clean_removed": int(cr.sum()), "poison_removed": int(pr.sum()),
            "clean_removal_rate": float(cr.mean()), "poison_capture": float(pr.mean()),
            "precision": float(pr.sum()/removed) if removed else 0.0}


def fit_calibrate_evaluate(factory, X_train, X_cal, X_clean_eval, X_poison, budgets):
    """API deliberately accepts no poison labels during fitting or calibration."""
    model = factory().fit(X_train)
    cal = model.score_samples(X_cal)
    clean = model.score_samples(X_clean_eval)
    poison = model.score_samples(X_poison)
    return model, {str(b): evaluate_threshold(clean, poison, clean_threshold(cal, b)) |
                   {"threshold": clean_threshold(cal, b)} for b in budgets}, (cal, clean, poison)
