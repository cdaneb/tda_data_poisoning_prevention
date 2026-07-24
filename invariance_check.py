"""
Count-invariance instrumentation (Phase P, Step 0 / Test B).

Backs the project's primary deductive claim: binarization is a fixed
pointwise threshold, hence equivariant under permutation of byte positions;
summation is symmetric; therefore the foreground-pixel count is invariant
under every permutation. This module measures that claim empirically rather
than assuming it.

Reuses gtda.images.Binarizer directly — the exact class tda_pipeline.py
builds into the real pipeline — rather than reimplementing its threshold
rule (`X / max_value_ > threshold`, confirmed from gtda's own source:
max_value_ = np.max(X) over the whole fitted collection). What this module
measures is therefore provably what the pipeline actually does, not an
approximation of it.
"""
import numpy as np
from gtda.images import Binarizer

from tda_pipeline import reshape_for_tda


def _as_images(X):
    """Accepts (N, 1500) raw bytes or (N, 30, 50) already-reshaped images."""
    return X if X.ndim == 3 else reshape_for_tda(X)


def binarize(X, threshold=0.4):
    """
    Binarized images plus the fitted max_value_, using the pipeline's own
    Binarizer, fit across the whole collection X (matching how
    build_tda_pipeline() fits it — one max_value_ per batch, not per sample).

    Callers that need the images themselves (e.g. rendering the binarized
    triptych) use this so the threshold rule lives in exactly one place;
    foreground_count() is a thin count on top of it.

    Args:
        X: (N, 1500) raw payload bytes, or (N, 30, 50) reshaped images.
        threshold: Binarizer threshold (fraction of fitted max_value_).

    Returns:
        binarized: (N, 30, 50) bool array.
        max_value_: float — the fitted max_value_.
    """
    images = _as_images(X)
    binarizer = Binarizer(threshold=threshold, n_jobs=-1)
    binarized = binarizer.fit_transform(images)
    return binarized, binarizer.max_value_


def foreground_count(X, threshold=0.4):
    """
    Per-sample count of activated (foreground) pixels after binarization.

    Args:
        X: (N, 1500) raw payload bytes, or (N, 30, 50) reshaped images.
        threshold: Binarizer threshold (fraction of fitted max_value_).

    Returns:
        counts: (N,) int array — foreground pixel count per sample.
        max_value_: float — the fitted max_value_, for max_value_check /
            crossed_threshold.
    """
    binarized, max_value_ = binarize(X, threshold=threshold)
    counts = binarized.reshape(len(binarized), -1).sum(axis=1)
    return counts, max_value_


def positions_changed(x_clean, x_perm):
    """
    Number of byte indices where value differs, on raw byte vectors.

    Args:
        x_clean, x_perm: (1500,) or (N, 1500) raw byte vectors (same shape).

    Returns:
        int, or (N,) int array for batched input.
    """
    diff = (x_clean != x_perm)
    return diff.sum(axis=-1)


def crossed_threshold(x_clean, x_perm, threshold=0.4, max_value=255.0):
    """
    Of the positions where value differs, how many actually changed which
    side of the binarization cutoff they fall on (flipped the binarized
    bit) — using the identical rule gtda.images.Binarizer applies
    (`value / max_value > threshold`).

    Args:
        x_clean, x_perm: (1500,) or (N, 1500) raw byte vectors.
        threshold: Binarizer threshold fraction.
        max_value: the max_value_ to binarize against. Defaults to 255.0
            (byte data's theoretical max); pass the value foreground_count()
            actually returned for an exact match to a specific fitted batch.

    Returns:
        int, or (N,) int array for batched input.
    """
    b_clean = (x_clean / max_value) > threshold
    b_perm = (x_perm / max_value) > threshold
    return (b_clean != b_perm).sum(axis=-1)


def max_value_check(images_clean, images_perm):
    """
    Assert the fitted Binarizer.max_value_ (np.max over the collection) is
    identical for clean and permuted data — the "a maximum is itself
    permutation-invariant" step of the proof, verified empirically rather
    than assumed.

    Args:
        images_clean, images_perm: (N, 1500) or (N, 30, 50).

    Returns:
        (max_clean: float, max_perm: float, equal: bool)
    """
    ic = _as_images(images_clean)
    ip = _as_images(images_perm)
    max_clean = float(np.max(ic))
    max_perm = float(np.max(ip))
    return max_clean, max_perm, max_clean == max_perm
