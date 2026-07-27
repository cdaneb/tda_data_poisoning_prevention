"""
Phase P — Step 0 (count-invariance, blocking gate) and effective-swap-
fraction / bit-identity diagnostics for Test B's four permutation families,
plus a Gaussian-noise positive control.

Cheap by design: everything here operates at the Binarizer layer (via
invariance_check.py) or the raw byte level. The one exception is the P5
bit-identity feature-vector check, which needs one small (200-sample) TDA
extraction pass per family — still far cheaper than a capture run.

All five attack functions (malicious_random_attack, block_reversal_attack,
block_swap_attack, cyclic_shift_attack, gaussian_noise_attack) start with the
identical sequence `rng = np.random.default_rng(random_state); ...;
rng.choice(malicious_idx, size=n_poison, replace=False)`. Calling each with the
same random_state and a poison_rate tuned to yield exactly n_poison=200
therefore draws the exact same 200 target indices for every family — verified
below, not assumed — making the families' diagnostics genuinely
apples-to-apples.

Phase M / M6 adds Gaussian noise as a declared NON-invariant fifth family. It is
not a Test B family and never appears on figure V3; it is here as the count-
channel positive control, so its Step 0 gate is inverted rather than skipped.
Also added: the per-seed max_value_ gate (A14), the clip_frac/Delta-count
correlation (A12), and a cross-check of the two independent noise
implementations (A4). Output goes to a Phase M artifact, leaving Phase P's
committed results/test_b_diagnostics.json alone.

Usage:
    python test_b_diagnostics.py            # run A14 + P4 + M6 + P5, report,
                                              # exit nonzero if a gate fails
"""
import sys
import json
import numpy as np

from data_loader import load_unsw
from adversarial_attack import (
    malicious_random_attack, block_reversal_attack, block_swap_attack,
    cyclic_shift_attack, gaussian_noise_attack, label_to_binary,
)
from invariance_check import (
    binarize, foreground_count, max_value_check, positions_changed, crossed_threshold,
)
from tda_pipeline import extract_tda_features
from paths import RESULTS_DIR
from phase_m_env import env_block

N_DIAG_SAMPLES = 200
RANDOM_STATE = 42
THRESHOLDS = [0.4, 0.3]

# A14: the max_value_ gate is per-seed, not one draw. M5 measured seed 42 only;
# nothing guarantees the other four seeds' draws also contain a 255 byte.
SEEDS = [42, 123, 456, 789, 1024]

# Phase M writes to its own artifact. results/test_b_diagnostics.json is the
# committed Phase P record and its four-family content is unchanged by the noise
# addition below — overwriting it would edit a committed artifact for no gain.
# The name must match .gitignore's `!results/phase_m_*.json` negation, which
# catches direct children ending in .json only.
OUTPUT_NAME = "phase_m_m6_diagnostics.json"

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

# Phase M / M6 benchmarks: the values Phase P actually produced and CLAUDE.md §6
# now records, as distinct from REFERENCE above (Phase P's pre-commitment, which
# §9 items 3 and 5 supersede — 21.96/16.42/14.70/266.6 and the 190.6 clean count
# are the stale draft figures and must not be used as the comparison here).
# Provenance: all rebuilt and backed by committed code, except `noise_positive_
# control`, which is the recorded Step 0 positive control this run must match.
REFERENCE_M6 = {
    "positions_changed_mean": {"block_reversal": 14.08, "block_swap": 17.63,
                                "transpositions": 22.93, "cyclic_shift": 281.09},
    "positions_changed_median": {"block_reversal": 0.0, "block_swap": 0.0,
                                  "transpositions": 8.0, "cyclic_shift": 89.0},
    "frac_zero_footprint": {"block_reversal": 0.845, "block_swap": 0.785,
                             "transpositions": 0.125, "cyclic_shift": 0.0},
    "bit_identical_frac": {"block_reversal": 0.880, "block_swap": 0.830},
    "noise_positive_control": {"threshold_0.4": {"mean_delta": 11.5, "mean_abs_delta": 14.02},
                                "threshold_0.3": {"mean_delta": 19.945, "mean_abs_delta": 29.705}},
    "clean_mean_foreground_count": {"threshold_0.4": 62.305, "threshold_0.3": 114.305},
}

# (fn, kwargs, invariant_expected). `invariant_expected` is the *direction* of
# the Step 0 count gate, not a switch that turns it off (Phase M, M6): the four
# multiset-preserving permutation families must leave the foreground count
# untouched (Delta == 0 for all N), whereas Gaussian noise must move it
# (Delta != 0 for at least one sample). A family declared non-invariant is
# therefore a positive control whose gate is *inverted*; bypassing it would
# produce no evidence at all.
#
# A13: this dict and run_test_b_capture.py's FAMILIES must agree on key set and
# on (fn, kwargs) for the four permutation families — asserted at M7's parity
# check. make_figure_v3.py's list is deliberately NOT coupled to either: V3 is a
# four-family Test B figure and noise is not a Test B family.
FAMILIES = {
    "transpositions": (malicious_random_attack, {"n_swaps": 60}, True),
    "block_reversal": (block_reversal_attack, {"k": 120}, True),
    "block_swap": (block_swap_attack, {"k": 60}, True),
    "cyclic_shift": (cyclic_shift_attack, {}, True),
    "noise": (gaussian_noise_attack, {"sigma": 30}, False),
}


def get_clean_and_perturbed(fn, kwargs, X, y, n_samples=N_DIAG_SAMPLES, random_state=RANDOM_STATE):
    """Runs an attack function with poison_rate tuned to yield exactly
    n_samples poisoned targets. Returns (X_clean_subset, X_perturbed,
    target_indices, attack_log), paired via the attack's own target_index log.

    The log is returned (Phase M, A12) so per-sample fields the attack recorded
    — `clip_frac` for gaussian_noise_attack — can be correlated against the
    per-sample count change measured here, without re-running the attack."""
    poison_rate = n_samples / len(X)
    Xc, yc, ip, log = fn(X, y, poison_rate=poison_rate, random_state=random_state, **kwargs)
    n_poison = int(len(X) * poison_rate)
    assert n_poison == n_samples, f"poison_rate arithmetic gave {n_poison}, expected {n_samples}"
    target_indices = np.array([l["target_index"] for l in log])
    X_clean_subset = X[target_indices]
    X_perturbed = Xc[len(X):]  # appended tail = the n_poison perturbed rows, same order as log
    return X_clean_subset, X_perturbed, target_indices, log


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
    """Confirms all families draw the same 200 target indices, as the module
    docstring claims — checked, not assumed. Includes the noise family (A2):
    gaussian_noise_attack makes its target draw its FIRST rng use, exactly as
    the four permutation families do, so at a given seed it must select the
    same targets."""
    all_targets = {}
    for family, (fn, kwargs, _) in FAMILIES.items():
        _, _, targets, _ = get_clean_and_perturbed(fn, kwargs, X, y)
        all_targets[family] = targets
    ref = all_targets["transpositions"]
    all_match = all(np.array_equal(ref, t) for t in all_targets.values())
    return all_match, all_targets


def run_p4(X, y):
    """Step 0: count gate, both thresholds, all families + the noise_only control.

    The gate is directional, not unconditional (M6): families declared
    `invariant_expected=True` fail on any count change; families declared False
    fail on *no* count change, since for them a null result means the positive
    control produced no signal."""
    results = {}
    gate_failures = []

    for threshold in THRESHOLDS:
        key = f"threshold_{threshold}"
        results[key] = {}
        for family, (fn, kwargs, invariant_expected) in FAMILIES.items():
            X_clean, X_perm, _, _ = get_clean_and_perturbed(fn, kwargs, X, y)
            counts_clean, fitted_max_clean = foreground_count(X_clean, threshold=threshold)
            counts_perm, fitted_max_perm = foreground_count(X_perm, threshold=threshold)
            delta = counts_perm.astype(int) - counts_clean.astype(int)
            n_changed = int((delta != 0).sum())
            mvc = max_value_check(X_clean, X_perm)
            results[key][family] = {
                "invariant_expected": invariant_expected,
                "n_changed": n_changed, "n_total": int(len(counts_clean)),
                "mean_delta": float(delta.mean()),
                "mean_abs_delta": float(np.abs(delta).mean()),
                "max_value_clean": mvc[0], "max_value_perm": mvc[1], "max_value_equal": bool(mvc[2]),
                # A14: the *fitted* Binarizer.max_value_ from each of the two
                # fits, recorded rather than inferred from np.max.
                "fitted_max_value_clean": float(fitted_max_clean),
                "fitted_max_value_perm": float(fitted_max_perm),
                "clean_mean_foreground_count": float(counts_clean.mean()),
            }
            if invariant_expected and n_changed != 0:
                gate_failures.append((threshold, family, "expected_invariant", n_changed))
            elif not invariant_expected and n_changed == 0:
                gate_failures.append((threshold, family, "expected_non_invariant", n_changed))

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

    for family, (fn, kwargs, _) in FAMILIES.items():
        X_clean, X_perm, _, _ = get_clean_and_perturbed(fn, kwargs, X, y)
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

        # "noise" is included here (M6) because the bit-identical fraction is
        # the direct measurement of the interpretation the phase asks for: how
        # much of what noise does survives binarization at all. The two silent
        # permutation families sit at 88.0% / 83.0%; noise is expected far
        # lower, and the gap is the quantization argument made quantitative.
        if family in ("block_reversal", "block_swap", "noise"):
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


def run_m6_noise_cross_check(X, y, threshold=0.4):
    """
    Two independent paths to the noise count-change statistic, reported side by
    side (A4). `noise_only()` predates Phase M and is left untouched;
    `gaussian_noise_attack` is M4's family-shaped implementation. They should
    agree exactly, not merely closely: both consume the same Generator in the
    same order (target draw first, then normals), and numpy's Generator fills a
    (200, 1500) normal draw from the same sequential bit stream as 200
    consecutive (1500,) draws. Disagreement is therefore a finding, and the
    phase stops on it rather than picking a number.
    """
    fn, kwargs, _ = FAMILIES["noise"]
    X_clean_f, X_perm_f, targets_f, log_f = get_clean_and_perturbed(fn, kwargs, X, y)
    X_clean_n, X_perm_n, targets_n = noise_only(X, y)

    def delta_stats(X_clean, X_pert):
        c_clean, _ = foreground_count(X_clean, threshold=threshold)
        c_pert, _ = foreground_count(X_pert, threshold=threshold)
        d = c_pert.astype(int) - c_clean.astype(int)
        return d

    d_f = delta_stats(X_clean_f, X_perm_f)
    d_n = delta_stats(X_clean_n, X_perm_n)

    return {
        "threshold": threshold,
        "families_path": {"mean_delta": float(d_f.mean()),
                          "mean_abs_delta": float(np.abs(d_f).mean())},
        "noise_only_path": {"mean_delta": float(d_n.mean()),
                            "mean_abs_delta": float(np.abs(d_n).mean())},
        "same_targets": bool(np.array_equal(targets_f, targets_n)),
        # Stronger than agreement on the summary statistic: the two paths'
        # perturbed byte arrays themselves.
        "perturbed_arrays_identical": bool(np.array_equal(X_perm_f, X_perm_n)),
        "per_sample_delta_identical": bool(np.array_equal(d_f, d_n)),
        "mean_abs_delta_agree_exactly": bool(
            float(np.abs(d_f).mean()) == float(np.abs(d_n).mean())),
    }


def run_m6_clip_correlation(X, y, threshold=0.4):
    """
    A12: correlate per-sample `clip_frac` (logged by gaussian_noise_attack at
    M4) against per-sample Delta foreground count.

    Motivation and the expected answer, stated before the number is read: a
    padding byte at 0 under N(0, 30) clips at zero with probability 1/2, and the
    clipped variate has expectation sigma/sqrt(2*pi) ~= 12. The binarization
    cutoff is 102 and P(N(0, 30) > 102) ~= 3e-4. So 0 and 12 binarize to the
    same background pixel: the headline clipping figure is largely inert in the
    space the detector sees. Delta count should be driven by bytes already near
    the cutoff, which is a different population from the clipped ones. A strong
    correlation would instead mean the noise cell's across-seed variance has a
    known source, and that belongs in M10.
    """
    from scipy import stats

    fn, kwargs, _ = FAMILIES["noise"]
    X_clean, X_perm, _, log = get_clean_and_perturbed(fn, kwargs, X, y)
    clip_frac = np.array([l["clip_frac"] for l in log], dtype=float)
    n_clipped = np.array([l["n_clipped"] for l in log], dtype=float)

    counts_clean, _ = foreground_count(X_clean, threshold=threshold)
    counts_perm, _ = foreground_count(X_perm, threshold=threshold)
    delta = counts_perm.astype(int) - counts_clean.astype(int)

    pearson_r, pearson_p = stats.pearsonr(clip_frac, delta)
    spearman_r, spearman_p = stats.spearmanr(clip_frac, delta)
    # The clean foreground count is the plausible confounder: a packet with more
    # non-padding bytes has both fewer clippable zeros and more bytes near the
    # cutoff. Reported so the A12 correlation can be read against it.
    pearson_clean_r, _ = stats.pearsonr(counts_clean.astype(float), delta)
    pearson_clipfrac_vs_clean_r, _ = stats.pearsonr(clip_frac, counts_clean.astype(float))

    return {
        "threshold": threshold,
        "n_samples": int(len(delta)),
        "clip_frac_mean": float(clip_frac.mean()),
        "clip_frac_std": float(clip_frac.std()),
        "n_clipped_mean": float(n_clipped.mean()),
        "delta_mean": float(delta.mean()),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "delta_std": float(delta.std()),
        "pearson_clipfrac_vs_delta": {"r": float(pearson_r), "p": float(pearson_p)},
        "spearman_clipfrac_vs_delta": {"rho": float(spearman_r), "p": float(spearman_p)},
        "pearson_cleancount_vs_delta_r": float(pearson_clean_r),
        "pearson_clipfrac_vs_cleancount_r": float(pearson_clipfrac_vs_clean_r),
    }


def run_a14_max_value_gate(X, y, threshold=0.4, seeds=SEEDS):
    """
    A14: the max_value_ gate, per seed rather than on one draw.

    M5 verified max_value_ == 255.0 for clean-alone, noised-alone and the
    concatenation at seed 42, and passed because that particular 200-sample
    batch happens to contain a 255 byte. That is safe-by-coincidence, not
    safe-by-theorem (the permutation families are the latter — a maximum is
    permutation-invariant). Nothing carries the coincidence to the other four
    seeds, so every fit is recorded.

    A seed returning anything other than 255.0, or disagreeing between clean and
    perturbed, means that seed's count and capture numbers are confounded and
    the phase pauses. Such a seed is reported, never dropped.
    """
    records = {}
    violations = []
    for seed in seeds:
        records[str(seed)] = {}
        for family, (fn, kwargs, _) in FAMILIES.items():
            X_clean, X_perm, _, _ = get_clean_and_perturbed(
                fn, kwargs, X, y, random_state=seed)
            _, mv_clean = binarize(X_clean, threshold=threshold)
            _, mv_perm = binarize(X_perm, threshold=threshold)
            _, mv_both = binarize(np.vstack([X_clean, X_perm]), threshold=threshold)
            rec = {
                "max_value_clean": float(mv_clean),
                "max_value_perturbed": float(mv_perm),
                "max_value_combined": float(mv_both),
                "effective_cutoff_clean": float(mv_clean) * threshold,
                "effective_cutoff_perturbed": float(mv_perm) * threshold,
                "effective_cutoff_combined": float(mv_both) * threshold,
                "all_255": bool(mv_clean == 255.0 and mv_perm == 255.0 and mv_both == 255.0),
                "clean_equals_perturbed": bool(mv_clean == mv_perm),
            }
            records[str(seed)][family] = rec
            if not rec["all_255"] or not rec["clean_equals_perturbed"]:
                violations.append({"seed": seed, "family": family, **rec})
    return records, violations


def main():
    print("Loading full UNSW-NB15 dataset...")
    X, y = load_unsw(max_samples=None)

    print(f"\n=== Verifying all {len(FAMILIES)} families draw the same "
          f"{N_DIAG_SAMPLES} target samples ===")
    same_targets, _ = verify_same_targets(X, y)
    print(f"  Same targets across all families: {same_targets}")
    if not same_targets:
        print("  WARNING: families are NOT operating on identical sample sets — "
              "family-to-family comparisons below are not apples-to-apples.")

    print("\n=== A14: per-seed max_value_ gate ===")
    a14_records, a14_violations = run_a14_max_value_gate(X, y)
    print(f"{'seed':>6} {'family':<16} {'clean':>8} {'perturbed':>10} {'combined':>9} {'cutoff':>7}")
    for seed in SEEDS:
        for family in FAMILIES:
            r = a14_records[str(seed)][family]
            print(f"{seed:>6} {family:<16} {r['max_value_clean']:>8.1f} "
                  f"{r['max_value_perturbed']:>10.1f} {r['max_value_combined']:>9.1f} "
                  f"{r['effective_cutoff_combined']:>7.1f}")

    print("\n=== P4: Step 0 count gate (directional) ===")
    p4_results, gate_failures = run_p4(X, y)
    print(json.dumps(p4_results, indent=2))

    print("\n=== M6: noise cross-check — FAMILIES path vs noise_only() path ===")
    m6_cross = run_m6_noise_cross_check(X, y)
    print(json.dumps(m6_cross, indent=2))

    print("\n=== M6 / A12: clip_frac vs Delta count correlation ===")
    m6_clip = run_m6_clip_correlation(X, y)
    print(json.dumps(m6_clip, indent=2))

    print("\n=== P5: effective swap fraction / bit-identity ===")
    p5_results = run_p5(X, y)
    print(json.dumps(p5_results, indent=2))

    output = {
        "n_diag_samples": N_DIAG_SAMPLES,
        "random_state": RANDOM_STATE,
        "seeds": SEEDS,
        "same_targets_across_families": bool(same_targets),
        "reference": REFERENCE,
        "reference_m6": REFERENCE_M6,
        "p4_count_invariance": p4_results,
        "p5_swap_fraction_bit_identity": p5_results,
        "p4_gate_failures": [{"threshold": t, "family": f, "expected": e, "n_changed": n}
                             for t, f, e, n in gate_failures],
        "m6_noise_cross_check": m6_cross,
        "m6_clip_frac_correlation": m6_clip,
        "a14_max_value_per_seed": a14_records,
        "a14_violations": a14_violations,
        # A9/A10: provenance travels with the artifact, dirty flag included.
        "env": env_block(),
    }
    out_path = RESULTS_DIR / OUTPUT_NAME
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWritten to {out_path}")

    if a14_violations:
        print("\n=== A14 GATE: FAIL ===")
        for v in a14_violations:
            print(f"  seed={v['seed']} family={v['family']}: clean={v['max_value_clean']} "
                  f"perturbed={v['max_value_perturbed']} combined={v['max_value_combined']} "
                  f"(expected 255.0 throughout)")
        print("Count and capture numbers for the affected seed(s) are confounded. "
              "STOPPING per A14 — the seed is reported, not dropped.")
        sys.exit(1)
    else:
        print(f"\n=== A14 GATE: PASS — max_value_ == 255.0 on every fit, "
              f"{len(SEEDS)} seeds x {len(FAMILIES)} families x 3 fits ===")

    if gate_failures:
        print("\n=== P4 GATE: FAIL ===")
        for threshold, family, expected, n_changed in gate_failures:
            if expected == "expected_invariant":
                print(f"  threshold={threshold} family={family}: {n_changed}/{N_DIAG_SAMPLES} "
                      f"changed count (expected 0/{N_DIAG_SAMPLES})")
            else:
                print(f"  threshold={threshold} family={family}: 0/{N_DIAG_SAMPLES} changed count, "
                      f"but this family is declared non-invariant (expected > 0) — the positive "
                      f"control produced no signal")
        print("STOPPING per the phase's gate rule — do not proceed without human "
              "judgment on this result.")
        sys.exit(1)
    else:
        n_inv = sum(1 for _, _, inv in FAMILIES.values() if inv)
        print(f"\n=== P4 GATE: PASS — {n_inv} invariant-declared families at "
              f"0/{N_DIAG_SAMPLES}, {len(FAMILIES) - n_inv} non-invariant-declared "
              f"family(ies) nonzero, both thresholds ===")


if __name__ == "__main__":
    main()
