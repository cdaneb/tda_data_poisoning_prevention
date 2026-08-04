"""Frame construction for the preregistered Phase Q4 mechanism test.

Q4 changes one thing relative to the standing Q3/R60 frame: after the seeded
5,000-row draw, exact duplicate 1,500-byte payloads are removed before poison
is generated.  The first sampled occurrence is retained, order is preserved,
and the frame is not backfilled.  Attack code and parameters are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from adversarial_attack import malicious_random_attack
from tools.phase_q2_common import (
    MAX_SAMPLES,
    N_SWAPS,
    POISON_RATE,
    array_hash,
    build_realization,
    load_unsw_once,
    realization_provenance,
)


def stable_exact_payload_deduplicate(X, y):
    """Deduplicate exact payload rows, retaining the first sampled occurrence.

    Deduplication keys use all 1,500 payload-byte coordinates and do not use
    labels.  Label-conflict diagnostics are reported retrospectively but never
    influence which row is retained.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 1500:
        raise ValueError(f"X must have shape (n, 1500), got {X.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    _, first, inverse, counts = np.unique(
        X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    keep = np.sort(first)

    repeated_classes = np.flatnonzero(counts >= 2)
    conflict_classes = []
    conflict_rows = 0
    for cls in repeated_classes:
        members = np.flatnonzero(inverse == cls)
        if len(np.unique(y[members])) > 1:
            conflict_classes.append(int(cls))
            conflict_rows += int(len(members))

    metadata = {
        "rule": (
            "exact equality over all 1500 payload bytes; retain the first row "
            "in seeded sample order; labels are not used; no backfill"
        ),
        "n_before": int(len(X)),
        "n_after": int(len(keep)),
        "n_removed": int(len(X) - len(keep)),
        "removed_fraction": float((len(X) - len(keep)) / len(X)) if len(X) else 0.0,
        "n_repeated_payload_classes_before": int(len(repeated_classes)),
        "n_repeated_member_rows_before": int(counts[counts >= 2].sum()),
        "n_label_conflict_payload_classes_before": int(len(conflict_classes)),
        "n_rows_in_label_conflict_classes_before": int(conflict_rows),
        "kept_sample_position_hash": array_hash(keep.astype(np.int64)),
        "clean_payload_hash_before": array_hash(X),
        "clean_payload_hash_after": array_hash(X[keep]),
    }
    return X[keep].copy(), y[keep].copy(), keep, metadata


def attack_change_diagnostics(X_clean, X_combined, attack_log):
    """Measure raw byte changes without redefining the legacy validity flag."""
    n_clean = len(X_clean)
    X_poison = np.asarray(X_combined)[n_clean:]
    targets = np.asarray([entry["target_index"] for entry in attack_log], dtype=int)
    if len(targets) != len(X_poison):
        raise AssertionError("attack log and appended poison length disagree")
    changed = np.count_nonzero(X_poison != np.asarray(X_clean)[targets], axis=1)
    return {
        "positions_changed": {
            "per_poison": [int(v) for v in changed],
            "mean": float(changed.mean()) if len(changed) else 0.0,
            "median": float(np.median(changed)) if len(changed) else 0.0,
            "min": int(changed.min()) if len(changed) else 0,
            "max": int(changed.max()) if len(changed) else 0,
        },
        "raw_noop_count": int(np.count_nonzero(changed == 0)),
        "raw_noop_fraction": float(np.mean(changed == 0)) if len(changed) else 0.0,
        "legacy_validity_count": int(sum(bool(e["valid"]) for e in attack_log)),
        "note": (
            "legacy validity checks byte range and multiset preservation; it does "
            "not imply that the raw byte vector changed"
        ),
    }


def _poison_frame(X_clean, y_clean, seed):
    Xc, yc, poisoned, log = malicious_random_attack(
        X_clean, y_clean, poison_rate=POISON_RATE,
        random_state=seed, n_swaps=N_SWAPS)
    return {
        "seed": int(seed),
        "dataset": "UNSW-NB15 Payload-Byte",
        "attack": "malicious_random_attack",
        "n_swaps": int(N_SWAPS),
        "poison_rate": float(POISON_RATE),
        "n_clean": int(len(X_clean)),
        "n_poison": int(poisoned.sum()),
        "n_total": int(len(poisoned)),
        "X_combined": Xc,
        "y_combined": yc,
        "is_poisoned": poisoned,
        "attack_target_index": [int(e["target_index"]) for e in log],
        "input_hash": array_hash(Xc),
        "poison_mask_hash": array_hash(poisoned),
        "clean_frame_hash": array_hash(X_clean),
        "attack_diagnostics": attack_change_diagnostics(X_clean, Xc, log),
    }


def build_q4_frames(seed):
    """Build the unchanged standing probe and the one-variable Q4 arm."""
    X_all, y_all = load_unsw_once()
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(X_all), size=MAX_SAMPLES, replace=False)
    X_sample = X_all[sampled_indices]
    y_sample = y_all[sampled_indices]

    # Replay the control attack only to prove that Q4 started from the exact Q3
    # realization.  Its expensive TDA/OPTICS results are reused from Q3.
    standing_replay = _poison_frame(X_sample, y_sample, seed)
    standing_reference = build_realization(seed)
    if standing_replay["input_hash"] != standing_reference["input_hash"]:
        raise AssertionError("standing replay does not reproduce Q3 input")
    if standing_replay["poison_mask_hash"] != standing_reference["poison_mask_hash"]:
        raise AssertionError("standing replay does not reproduce Q3 poison mask")

    X_dedup, y_dedup, keep, dedup = stable_exact_payload_deduplicate(X_sample, y_sample)
    deduplicated = _poison_frame(X_dedup, y_dedup, seed)
    deduplicated["subsample_index_hash"] = array_hash(sampled_indices)
    deduplicated["dedup_kept_sample_positions"] = keep
    deduplicated["deduplication"] = dedup

    standing_probe = {
        **realization_provenance(standing_reference),
        "clean_frame_hash": array_hash(X_sample),
        "attack_diagnostics": standing_replay["attack_diagnostics"],
        "replay_matches_q3_input": True,
        "replay_matches_q3_poison_mask": True,
    }
    return standing_probe, deduplicated


def q4_realization_provenance(real):
    """Return the JSON-safe portion of a Q4 realization."""
    excluded = {
        "X_combined", "y_combined", "is_poisoned",
        "dedup_kept_sample_positions", "attack_target_index",
    }
    return {k: v for k, v in real.items() if k not in excluded}
