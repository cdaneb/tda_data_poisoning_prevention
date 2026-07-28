"""Frozen Phase Q evaluation metrics for cluster-based poison removal."""
from __future__ import annotations

import numpy as np


PURITY_THRESHOLDS = (1.0, 0.95, 0.90, 0.80, 0.50)


def removal_metrics(cluster_labels, is_poisoned, purity_threshold=1.0):
    """Score removal of clusters meeting an oracle poison-purity threshold.

    Cluster label -1 is retained as unclustered, matching the project's capture
    convention.  Ground-truth poison membership is used only for retrospective
    evaluation; this is not a deployable cluster-labeling rule.
    """
    labels = np.asarray(cluster_labels)
    poisoned = np.asarray(is_poisoned, dtype=bool)
    if labels.shape != poisoned.shape:
        raise ValueError("cluster_labels and is_poisoned must have the same shape")
    if not 0.0 <= purity_threshold <= 1.0:
        raise ValueError("purity_threshold must lie in [0, 1]")

    removed = np.zeros(len(labels), dtype=bool)
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        purity = float(poisoned[mask].mean())
        qualifies = purity == 1.0 if purity_threshold == 1.0 else purity > purity_threshold
        if qualifies:
            removed |= mask

    tp = int(np.count_nonzero(removed & poisoned))
    fp = int(np.count_nonzero(removed & ~poisoned))
    n_poison = int(poisoned.sum())
    n_clean = int((~poisoned).sum())
    n_removed = int(removed.sum())
    poison_noise = int(np.count_nonzero((labels == -1) & poisoned))
    clean_noise = int(np.count_nonzero((labels == -1) & ~poisoned))
    return {
        "purity_threshold": float(purity_threshold),
        "n_removed": n_removed,
        "true_poison_removed": tp,
        "clean_removed": fp,
        "poison_removal_rate": tp / n_poison if n_poison else 0.0,
        "clean_false_removal_rate": fp / n_clean if n_clean else 0.0,
        "removal_precision": tp / n_removed if n_removed else None,
        "poison_unclustered_fraction": poison_noise / n_poison if n_poison else 0.0,
        "clean_unclustered_fraction": clean_noise / n_clean if n_clean else 0.0,
    }


def removal_curve(cluster_labels, is_poisoned, thresholds=PURITY_THRESHOLDS):
    return [removal_metrics(cluster_labels, is_poisoned, t) for t in thresholds]


def matched_clean_cost(control_curve, repair_curve, atol=1e-12):
    """Compare repair recall to its best point under each control clean cost."""
    matches = []
    for control in control_curve:
        budget = control["clean_false_removal_rate"]
        feasible = [
            point for point in repair_curve
            if point["clean_false_removal_rate"] <= budget + atol
        ]
        best = max(feasible, key=lambda p: p["poison_removal_rate"]) if feasible else None
        matches.append({
            "clean_cost_budget": budget,
            "control_purity_threshold": control["purity_threshold"],
            "control_poison_removal_rate": control["poison_removal_rate"],
            "repair_best": best,
            "poison_removal_rate_delta": (
                best["poison_removal_rate"] - control["poison_removal_rate"]
                if best is not None else None
            ),
        })
    return matches
