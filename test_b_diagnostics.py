"""
Phase P — Step 0 (count-invariance, blocking gate) and effective-swap-
fraction / bit-identity diagnostics for Test B's four permutation families,
plus a Gaussian-noise positive control.

Cheap by design: everything here operates at the Binarizer layer (via
invariance_check.py) or the raw byte level. The one exception is the P5
bit-identity feature-vector check, which needs one small (200-sample) TDA
extraction pass per family — still far cheaper than a capture run.

All four attack functions (malicious_random_attack, block_reversal_attack,
block_swap_attack, cyclic_shift_attack) start with the identical sequence
`rng = np.random.default_rng(random_state); ...; rng.choice(malicious_idx,
size=n_poison, replace=False)`. Calling each with the same random_state and
a poison_rate tuned to yield exactly n_poison=200 therefore draws the exact
same 200 target indices for every family — verified below, not assumed —
making the four families' diagnostics genuinely apples-to-apples.

Usage:
    python test_b_diagnostics.py            # run P4 + P5, report, exit
                                              # nonzero if the P4 gate fails
"""
import sys
import json
import numpy as np

from data_loader import load_unsw
from adversarial_attack import (
    malicious_random_attack, block_reversal_attack, block_swap_attack,
    cyclic_shift_attack, label_to_binary,
)
from invariance_check import foreground_count, max_value_check, positions_changed, crossed_threshold
from tda_pipeline import extract_tda_features
from paths import RESULTS_DIR

N_DIAG_SAMPLES = 200
RANDOM_STATE = 42
THRESHOLDS = [0.4, 0.3]

# Pre-committed reference values (see PHASE_P instructions / docs/PROJECT_HANDOFF_1.md
# §4, docs/ABC_PHASE_REPORT.md). Written before this script runs anything, per the
# phase's pre-commitment rule. Population (ddof=0) statistics throughout, matching
# the project's established convention.
REFERENCE = {
    "transpositions": {"count_invariant": "0/200", "positions_changed_mean": 21.96,
                        "positions_changed_std": 30.4, "capture_pct_mean": 1.80, "capture_pct_std": 0.51},
    "block_reversal": {"count_invariant": "0/200", "positions_changed_mean": 16.42,
                        "positions_changed_std": 39.5, "capture_pct_mean": 0.00, "capture_pct_std": 0.00},
    "block_swap": {"count_invariant": "0/200", "positions_changed_mean": 14.70,
                   "positions_changed_std": 37.2, "capture_pct_mean": 0.00, "capture_pct_std": 0.00},
    "cyclic_shift": {"count_invariant": "0/200", "positions_changed_mean": 266.6,
                      "positions_changed_std": 375.6, "capture_pct_mean": 6.28, "capture_pct_std": 1.31},
    "noise_control": {"threshold_0.4": {"mean_delta": 10.6, "mean_abs_delta": 15.8},
                       "threshold_0.3": {"mean_abs_delta": 36.1}},
    "clean_mean_foreground_count": 190.6,
}

FAMILIES = {
    "transpositions": (malicious_random_attack, {"n_swaps": 60}),
    "block_reversal": (block_reversal_attack, {"k": 120}),
    "block_swap": (block_swap_attack, {"k": 60}),
    "cyclic_shift": (cyclic_shift_attack, {}),
}


def get_clean_and_perturbed(fn, kwargs, X, y, n_samples=N_DIAG_SAMPLES, random_state=RANDOM_STATE):
    """Runs an attack function with poison_rate tuned to yield exactly
    n_samples poisoned targets. Returns (X_clean_subset, X_perturbed,
    target_indices), paired via the attack's own target_index log."""
    poison_rate = n_samples / len(X)
    Xc, yc, ip, log = fn(X, y, poison_rate=poison_rate, random_state=random_state, **kwargs)
    n_poison = int(len(X) * poison_rate)
    assert n_poison == n_samples, f"poison_rate arithmetic gave {n_poison}, expected {n_samples}"
    target_indices = np.array([l["target_index"] for l in log])
    X_clean_subset = X[target_indices]
    X_perturbed = Xc[len(X):]  # appended tail = the n_poison perturbed rows, same order as log
    return X_clean_subset, X_perturbed, target_indices


def noise_only(X, y, n_samples=N_DIAG_SAMPLES, random_state=RANDOM_STATE, noise_scale=30):
    """Gaussian-noise-only positive control (no swap component — isolates
    noise from poison.py's coupled noise+swap), using the identical
    target-selection convention as the attack functions so it draws the
    SAME 200 samples they do."""
    rng = np.random.default_rng(random_state)
    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]
    target_indices = rng.choice(malicious_idx, size=n_samples, replace=False)
    X_clean_subset = X[target_indices]
    noise = rng.normal(0, noise_scale, size=X_clean_subset.shape)
    X_noisy = np.clip(X_clean_subset.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return X_clean_subset, X_noisy, target_indices


def verify_same_targets(X, y):
    """Confirms all four families draw the same 200 target indices, as the
    module docstring claims — checked, not assumed."""
    all_targets = {}
    for family, (fn, kwargs) in FAMILIES.items():
        _, _, targets = get_clean_and_perturbed(fn, kwargs, X, y)
        all_targets[family] = targets
    ref = all_targets["transpositions"]
    all_match = all(np.array_equal(ref, t) for t in all_targets.values())
    return all_match, all_targets


def run_p4(X, y):
    """Step 0: count-invariance gate, both thresholds, all 4 families + noise control."""
    results = {}
    gate_failures = []

    for threshold in THRESHOLDS:
        key = f"threshold_{threshold}"
        results[key] = {}
        for family, (fn, kwargs) in FAMILIES.items():
            X_clean, X_perm, _ = get_clean_and_perturbed(fn, kwargs, X, y)
            counts_clean, max_clean = foreground_count(X_clean, threshold=threshold)
            counts_perm, max_perm = foreground_count(X_perm, threshold=threshold)
            n_changed = int((counts_clean != counts_perm).sum())
            mvc = max_value_check(X_clean, X_perm)
            results[key][family] = {
                "n_changed": n_changed, "n_total": int(len(counts_clean)),
                "max_value_clean": mvc[0], "max_value_perm": mvc[1], "max_value_equal": bool(mvc[2]),
                "clean_mean_foreground_count": float(counts_clean.mean()),
            }
            if n_changed != 0:
                gate_failures.append((threshold, family, n_changed))

        X_clean, X_noisy, _ = noise_only(X, y)
        counts_clean, _ = foreground_count(X_clean, threshold=threshold)
        counts_noisy, _ = foreground_count(X_noisy, threshold=threshold)
        delta = counts_noisy.astype(int) - counts_clean.astype(int)
        results[key]["noise_control"] = {
            "mean_delta": float(delta.mean()), "mean_abs_delta": float(np.abs(delta).mean()),
            "clean_mean_foreground_count": float(counts_clean.mean()),
        }

    return results, gate_failures


def run_p5(X, y):
    """Effective swap fraction (positions_changed / crossed_threshold stats)
    and bit-identity (block_reversal / block_swap only), threshold=0.4."""
    threshold = 0.4
    results = {}

    for family, (fn, kwargs) in FAMILIES.items():
        X_clean, X_perm, _ = get_clean_and_perturbed(fn, kwargs, X, y)
        _, max_value = foreground_count(X_clean, threshold=threshold)

        pc = positions_changed(X_clean, X_perm)
        ct = crossed_threshold(X_clean, X_perm, threshold=threshold, max_value=max_value)
        crossed_frac = np.divide(ct, pc, out=np.zeros_like(ct, dtype=float), where=pc != 0)

        entry = {
            "positions_changed_mean": float(pc.mean()),
            "positions_changed_std": float(pc.std()),  # population (ddof=0)
            "positions_changed_median": float(np.median(pc)),
            "positions_changed_frac_zero": float((pc == 0).mean()),
            "crossed_threshold_mean": float(ct.mean()),
            "crossed_threshold_frac_of_changed_mean": float(crossed_frac.mean()),
        }

        if family in ("block_reversal", "block_swap"):
            # Bit-identity: exact equality of binarized images and of the
            # resulting 60-dim feature vectors. Both computed from ONE
            # combined clean+perturbed batch (fit once, then split) —
            # matching how the real pipeline is actually used everywhere
            # else in this repo (poison.py appends poisoned samples into
            # X_combined, which is fit as a single batch). Fitting
            # Binarizer/Scaler separately per batch was tried first and
            # produced a spurious "binarized-identical but feature-different"
            # gap (16/200 for block_reversal) traced entirely to Scaler's
            # per-batch normalization constant differing between the two
            # independently-fit batches — an artifact of the diagnostic, not
            # the pipeline. Verified: with a shared fit, the two identical-
            # fractions match exactly.
            from gtda.images import Binarizer
            from tda_pipeline import reshape_for_tda
            n = len(X_clean)
            X_both = np.vstack([X_clean, X_perm])
            images_both = reshape_for_tda(X_both)
            binarizer = Binarizer(threshold=threshold, n_jobs=-1)
            bin_both = binarizer.fit_transform(images_both)
            bin_clean, bin_perm = bin_both[:n], bin_both[n:]
            per_sample_bin_identical = np.all(
                bin_clean.reshape(n, -1) == bin_perm.reshape(n, -1), axis=1)

            X_tda_both, _ = extract_tda_features(X_both, threshold=threshold)
            X_tda_clean, X_tda_perm = X_tda_both[:n], X_tda_both[n:]
            per_sample_feat_identical = np.all(
                np.isclose(X_tda_clean, X_tda_perm, rtol=0, atol=0), axis=1)

            entry["bit_identity"] = {
                "binarized_images_identical_frac": float(per_sample_bin_identical.mean()),
                "feature_vectors_identical_frac": float(per_sample_feat_identical.mean()),
                "binarized_but_not_feature_identical_count":
                    int(np.sum(per_sample_bin_identical & ~per_sample_feat_identical)),
            }

        results[family] = entry

    return results


def main():
    print("Loading full UNSW-NB15 dataset...")
    X, y = load_unsw(max_samples=None)

    print("\n=== Verifying all 4 families draw the same 200 target samples ===")
    same_targets, _ = verify_same_targets(X, y)
    print(f"  Same targets across all families: {same_targets}")
    if not same_targets:
        print("  WARNING: families are NOT operating on identical sample sets — "
              "family-to-family comparisons below are not apples-to-apples.")

    print("\n=== P4: Step 0 count-invariance gate ===")
    p4_results, gate_failures = run_p4(X, y)
    print(json.dumps(p4_results, indent=2))

    print("\n=== P5: effective swap fraction / bit-identity ===")
    p5_results = run_p5(X, y)
    print(json.dumps(p5_results, indent=2))

    output = {
        "n_diag_samples": N_DIAG_SAMPLES,
        "random_state": RANDOM_STATE,
        "same_targets_across_families": bool(same_targets),
        "reference": REFERENCE,
        "p4_count_invariance": p4_results,
        "p5_swap_fraction_bit_identity": p5_results,
        "p4_gate_failures": [{"threshold": t, "family": f, "n_changed": n} for t, f, n in gate_failures],
    }
    out_path = RESULTS_DIR / "test_b_diagnostics.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWritten to {out_path}")

    if gate_failures:
        print("\n=== P4 GATE: FAIL ===")
        for threshold, family, n_changed in gate_failures:
            print(f"  threshold={threshold} family={family}: {n_changed}/{N_DIAG_SAMPLES} changed count "
                  f"(expected 0/{N_DIAG_SAMPLES})")
        print("This contradicts the primary claim. STOPPING per Phase P's gate rule — "
              "do not proceed to P6 without human judgment on this result.")
        sys.exit(1)
    else:
        print(f"\n=== P4 GATE: PASS — 0/{N_DIAG_SAMPLES} changed count, all families, both thresholds ===")


if __name__ == "__main__":
    main()
