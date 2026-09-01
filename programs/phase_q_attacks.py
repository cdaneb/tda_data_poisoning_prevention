"""Controlled permutation attacks for the multithreshold repair study.

The legacy Test B generators sample offsets across all 1500 stored positions,
including synthetic zero padding.  This module leaves those generators intact
and defines a new experimental substrate whose permutations are restricted to
a conservative, observable payload support.

Payload-Byte converts a variable-length payload to 1500 bytes with
``ndarray.resize``.  Expansion is zero-filled, but zero is also a legitimate
payload byte.  The last nonzero byte is therefore a conservative lower bound
on payload length: it may omit legitimate trailing zero bytes, but it cannot
include appended padding.  We call this the conservative support throughout.
"""
from __future__ import annotations

import numpy as np

from programs.adversarial_attack import label_to_binary


MIN_COMMON_SUPPORT = 120
MAX_PARAMETER_DRAWS = 256


def conservative_support_lengths(X):
    """Return one-past-last-nonzero indices for a batch of padded payloads."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    positions = np.arange(1, X.shape[1] + 1, dtype=np.int32)
    return np.where(X != 0, positions, 0).max(axis=1)


def common_attackable_mask(X, lengths, k=60):
    """Sufficient mask for all four fixed-strength attacks to act nontrivially.

    The first 2*k bytes must contain at least two values, must not be a
    palindrome (block reversal witness at offset zero), and its first and
    second k-byte blocks must differ (block-swap witness at offsets 0 and k).
    This is deliberately sufficient rather than maximal so eligibility is
    deterministic, shared across families, and cheap to audit.
    """
    X = np.asarray(X)
    if X.shape[1] < 2 * k:
        return np.zeros(len(X), dtype=bool)
    prefix = X[:, :2 * k]
    nonconstant = np.any(prefix != prefix[:, :1], axis=1)
    reversal_witness = np.any(prefix != prefix[:, ::-1], axis=1)
    swap_witness = np.any(prefix[:, :k] != prefix[:, k:2 * k], axis=1)
    return (lengths >= 2 * k) & nonconstant & reversal_witness & swap_witness


def _select_common_targets(X, y, poison_rate, random_state, min_support):
    rng = np.random.default_rng(random_state)
    lengths = conservative_support_lengths(X)
    malicious = label_to_binary(y) == 1
    attackable = common_attackable_mask(X, lengths, k=MIN_COMMON_SUPPORT // 2)
    eligible = np.flatnonzero(malicious & (lengths >= min_support) & attackable)
    n_poison = int(len(X) * poison_rate)
    if n_poison > len(eligible):
        raise ValueError(
            f"poison_rate requires {n_poison} targets, but only {len(eligible)} "
            f"malicious samples satisfy the shared support/attackability criteria"
        )
    targets = rng.choice(eligible, size=n_poison, replace=False)
    return rng, targets, lengths


def _finish(X, y, targets, X_poison, attack_log):
    n = len(X)
    return (
        np.vstack([X, X_poison]),
        np.concatenate([y, y[targets].copy()]),
        np.concatenate([np.zeros(n, dtype=bool), np.ones(len(targets), dtype=bool)]),
        attack_log,
    )


def _log(target, support, changed, **parameters):
    return {
        "target_index": int(target),
        "support_length": int(support),
        "positions_changed": int(changed),
        "raw_noop": bool(changed == 0),
        "valid": True,
        **parameters,
    }


def _require_changed(sample, draw, max_parameter_draws, fallback):
    """Draw attack parameters until the permutation acts nontrivially."""
    for attempt in range(1, max_parameter_draws + 1):
        perturbed, parameters = draw()
        changed = int(np.count_nonzero(sample != perturbed))
        if changed:
            return perturbed, changed, attempt, parameters
    perturbed, parameters = fallback()
    changed = int(np.count_nonzero(sample != perturbed))
    if not changed:
        raise AssertionError("Shared attackability criterion supplied a trivial fallback")
    return perturbed, changed, max_parameter_draws + 1, parameters


def supported_transposition_attack(
    X, y, poison_rate=0.10, random_state=42, n_swaps=60,
    min_support=MIN_COMMON_SUPPORT, max_parameter_draws=MAX_PARAMETER_DRAWS,
):
    """Apply 60 disjoint transpositions inside conservative support."""
    if 2 * n_swaps > min_support:
        raise ValueError("min_support must be at least 2*n_swaps")
    rng, targets, lengths = _select_common_targets(
        X, y, poison_rate, random_state, min_support
    )
    poison = np.empty((len(targets), X.shape[1]), dtype=np.uint8)
    log = []
    for out_i, target in enumerate(targets):
        sample = X[target]
        support = int(lengths[target])
        def draw():
            positions = rng.choice(support, size=2 * n_swaps, replace=False)
            candidate = sample.copy()
            for a, b in positions.reshape(-1, 2):
                candidate[a], candidate[b] = candidate[b], candidate[a]
            return candidate, {"n_swaps": int(n_swaps)}
        def fallback():
            different = int(np.flatnonzero(sample[:support] != sample[0])[0])
            remaining = np.setdiff1d(np.arange(support), [0, different], assume_unique=True)
            extra = rng.choice(remaining, size=2 * (n_swaps - 1), replace=False)
            pairs = np.concatenate(([0, different], extra)).reshape(-1, 2)
            candidate = sample.copy()
            for a, b in pairs:
                candidate[a], candidate[b] = candidate[b], candidate[a]
            return candidate, {"n_swaps": int(n_swaps), "fallback": True}
        perturbed, changed, attempts, parameters = _require_changed(
            sample, draw, max_parameter_draws, fallback
        )
        poison[out_i] = perturbed
        log.append(_log(target, support, changed, draw_attempts=attempts, **parameters))
    return _finish(X, y, targets, poison, log)


def supported_block_reversal_attack(
    X, y, poison_rate=0.10, random_state=42, k=120,
    min_support=MIN_COMMON_SUPPORT, max_parameter_draws=MAX_PARAMETER_DRAWS,
):
    """Reverse one fixed-width block wholly inside conservative support."""
    if k > min_support:
        raise ValueError("min_support must be at least k")
    rng, targets, lengths = _select_common_targets(
        X, y, poison_rate, random_state, min_support
    )
    poison = np.empty((len(targets), X.shape[1]), dtype=np.uint8)
    log = []
    for out_i, target in enumerate(targets):
        sample = X[target]
        support = int(lengths[target])
        def draw():
            offset = int(rng.integers(0, support - k + 1))
            candidate = sample.copy()
            candidate[offset:offset + k] = sample[offset:offset + k][::-1]
            return candidate, {"offset": offset, "k": int(k)}
        def fallback():
            candidate = sample.copy()
            candidate[:k] = sample[:k][::-1]
            return candidate, {"offset": 0, "k": int(k), "fallback": True}
        perturbed, changed, attempts, parameters = _require_changed(
            sample, draw, max_parameter_draws, fallback
        )
        poison[out_i] = perturbed
        log.append(_log(target, support, changed, draw_attempts=attempts, **parameters))
    return _finish(X, y, targets, poison, log)


def supported_block_swap_attack(
    X, y, poison_rate=0.10, random_state=42, k=60,
    min_support=MIN_COMMON_SUPPORT, max_parameter_draws=MAX_PARAMETER_DRAWS,
):
    """Swap two disjoint fixed-width blocks inside conservative support."""
    if 2 * k > min_support:
        raise ValueError("min_support must be at least 2*k")
    rng, targets, lengths = _select_common_targets(
        X, y, poison_rate, random_state, min_support
    )
    poison = np.empty((len(targets), X.shape[1]), dtype=np.uint8)
    log = []
    for out_i, target in enumerate(targets):
        sample = X[target]
        support = int(lengths[target])
        def draw():
            while True:
                a = int(rng.integers(0, support - k + 1))
                b = int(rng.integers(0, support - k + 1))
                if abs(a - b) >= k:
                    break
            candidate = sample.copy()
            candidate[a:a + k] = sample[b:b + k]
            candidate[b:b + k] = sample[a:a + k]
            return candidate, {"offset_a": a, "offset_b": b, "k": int(k)}
        def fallback():
            candidate = sample.copy()
            candidate[:k] = sample[k:2 * k]
            candidate[k:2 * k] = sample[:k]
            return candidate, {"offset_a": 0, "offset_b": int(k), "k": int(k),
                               "fallback": True}
        perturbed, changed, attempts, parameters = _require_changed(
            sample, draw, max_parameter_draws, fallback
        )
        poison[out_i] = perturbed
        log.append(_log(target, support, changed, draw_attempts=attempts, **parameters))
    return _finish(X, y, targets, poison, log)


def supported_cyclic_shift_attack(
    X, y, poison_rate=0.10, random_state=42,
    min_support=MIN_COMMON_SUPPORT, max_parameter_draws=MAX_PARAMETER_DRAWS,
):
    """Rotate only the conservative-support prefix, leaving padding fixed."""
    rng, targets, lengths = _select_common_targets(
        X, y, poison_rate, random_state, min_support
    )
    poison = np.empty((len(targets), X.shape[1]), dtype=np.uint8)
    log = []
    for out_i, target in enumerate(targets):
        sample = X[target]
        support = int(lengths[target])
        def draw():
            shift = int(rng.integers(1, support))
            candidate = sample.copy()
            candidate[:support] = np.roll(sample[:support], shift)
            return candidate, {"shift": shift}
        def fallback():
            candidate = sample.copy()
            candidate[:support] = np.roll(sample[:support], 1)
            return candidate, {"shift": 1, "fallback": True}
        perturbed, changed, attempts, parameters = _require_changed(
            sample, draw, max_parameter_draws, fallback
        )
        poison[out_i] = perturbed
        log.append(_log(target, support, changed, draw_attempts=attempts, **parameters))
    return _finish(X, y, targets, poison, log)


SUPPORTED_FAMILIES = {
    "transpositions": (supported_transposition_attack, {"n_swaps": 60}),
    "block_reversal": (supported_block_reversal_attack, {"k": 120}),
    "block_swap": (supported_block_swap_attack, {"k": 60}),
    "cyclic_shift": (supported_cyclic_shift_attack, {}),
}
