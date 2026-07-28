"""Controlled TDA representations and diagram access for Phase Q.

The repair changes one representation variable: one Binarizer cutoff becomes a
fixed bank of nine cutoffs.  Every threshold block uses the existing Monkam
reconstruction unchanged and blocks are concatenated in declared order.
"""
from __future__ import annotations

import numpy as np
from gtda.homology import CubicalPersistence
from gtda.images import Binarizer, HeightFiltration, RadialFiltration
from sklearn.pipeline import make_pipeline

from tda_pipeline import build_tda_pipeline, reshape_for_tda


THRESHOLD_STACK = tuple(round(i / 10, 1) for i in range(1, 10))
CONTROL_THRESHOLD = 0.4


def validate_thresholds(thresholds):
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds:
        raise ValueError("threshold stack cannot be empty")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold stack contains duplicates")
    if any(not 0 < t < 1 for t in thresholds):
        raise ValueError("all thresholds must lie strictly between 0 and 1")
    if CONTROL_THRESHOLD not in thresholds:
        raise ValueError("threshold stack must contain the 0.4 control")
    return thresholds


def extract_multithreshold_features(X, thresholds=THRESHOLD_STACK, return_blocks=False):
    """Fit the legacy 60-feature pipeline once per threshold and concatenate.

    All threshold blocks see the identical input batch.  No additional feature
    standardization or weighting is introduced, because either would be a
    second method change.
    """
    thresholds = validate_thresholds(thresholds)
    images = reshape_for_tda(X)
    blocks = {}
    pipelines = {}
    for threshold in thresholds:
        print(f"  Threshold {threshold:.1f}: extracting legacy TDA block...")
        pipeline = build_tda_pipeline(threshold=threshold)
        block = pipeline.fit_transform(images)
        if block.shape[1] != 60:
            raise AssertionError(
                f"threshold {threshold} produced {block.shape[1]} features; expected 60"
            )
        if not np.isfinite(block).all():
            raise ValueError(f"threshold {threshold} produced non-finite features")
        blocks[threshold] = block
        pipelines[threshold] = pipeline
    # Dimension compensation is part of the fixed representation definition.
    # If all m blocks happened to be identical, raw concatenation would inflate
    # every Euclidean distance by sqrt(m), silently making OPTICS max_eps=2 a
    # different decision rule.  Dividing by sqrt(m) preserves that reference
    # scale while retaining equal, non-learned weight for every threshold.
    stacked = np.concatenate([blocks[t] for t in thresholds], axis=1) / np.sqrt(len(thresholds))
    if return_blocks:
        return stacked, blocks, pipelines
    return stacked, pipelines


def filtration_specs():
    """Names and constructors matching the five legacy filtrations exactly."""
    return (
        ("height_0_1", lambda: HeightFiltration(direction=np.array([0, 1]), n_jobs=-1)),
        ("height_1_0", lambda: HeightFiltration(direction=np.array([1, 0]), n_jobs=-1)),
        ("radial_0_50", lambda: RadialFiltration(center=np.array([0, 50]), n_jobs=-1)),
        ("radial_0_25", lambda: RadialFiltration(center=np.array([0, 25]), n_jobs=-1)),
        ("radial_30_0", lambda: RadialFiltration(center=np.array([30, 0]), n_jobs=-1)),
    )


def extract_unscaled_diagrams(X, threshold=CONTROL_THRESHOLD):
    """Return pre-Scaler cubical diagrams for each legacy filtration.

    The caller supplies one combined comparison batch, so every Binarizer is
    fitted once across clean and perturbed observations.
    """
    images = reshape_for_tda(X)
    diagrams = {}
    pipelines = {}
    for name, make_filtration in filtration_specs():
        pipeline = make_pipeline(
            Binarizer(threshold=threshold, n_jobs=-1),
            make_filtration(),
            CubicalPersistence(n_jobs=-1),
        )
        diagrams[name] = pipeline.fit_transform(images)
        pipelines[name] = pipeline
    return diagrams, pipelines
