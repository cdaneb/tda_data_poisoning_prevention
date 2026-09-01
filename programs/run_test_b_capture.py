"""
Phase P, P6 — Test B capture runs.

All four multiset-preserving permutation families (transpositions, block
reversal, block swap, cyclic shift), 5 seeds, threshold 0.4, no noise, no
guidance, malicious targeting, all 4 clustering algorithms (OPTICS is the
project's primary; the others are reported for completeness, matching every
other driver in this repo).

Mirrors run_lens4_baseline.py's single-pass baseline pattern
(subsample_for_seed, one TDA extraction + all-algorithm clustering per
run), applied to the four family functions built in Phase P
(adversarial_attack.py) instead of the L/R60/G60-MLP/G60-RF ladder.
Transpositions here IS the existing R60/"S" path (malicious_random_attack,
n_swaps=60) — included as an internal control that must reproduce the
recorded per-seed values exactly, not just approximately.
"""
import sys
import time
import json
import numpy as np

from programs.data_loader import load_unsw
from programs.tda_pipeline import extract_tda_features
from programs.clustering import run_all_clustering, classify_clusters
from programs.results_io import convert_for_json
from programs.adversarial_attack import (
    malicious_random_attack, block_reversal_attack, block_swap_attack, cyclic_shift_attack,
    gaussian_noise_attack,
)
from programs.paths import RESULTS_DIR
from programs.phase_m_env import env_block

SEEDS = [42, 123, 456, 789, 1024]
MAX_SAMPLES = 5000
POISON_RATE = 0.10
THRESHOLD = 0.4

OUTPUT_NAME = "phase_m_m7_capture.json"

# A13: key set and (fn, kwargs) for the four permutation families must match
# test_b_diagnostics.py's FAMILIES — asserted by parity_check() before the run.
# The fifth entry, noise, is the count-channel positive control and is NOT a Test
# B family; make_figure_v3.py's list is deliberately decoupled from this dict so
# noise can never become a fifth bar on the poster's centerpiece figure.
FAMILIES = {
    "transpositions": (malicious_random_attack, {"n_swaps": 60}),
    "block_reversal": (block_reversal_attack, {"k": 120}),
    "block_swap": (block_swap_attack, {"k": 60}),
    "cyclic_shift": (cyclic_shift_attack, {}),
    "noise": (gaussian_noise_attack, {"sigma": 30}),
}

# Pre-committed reference values (recorded, OPTICS, threshold 0.4), written
# before this script runs anything.
#
# A6 — PROVENANCE TIER IS MARKED IN CODE, deliberately. The four permutation rows
# are backed by committed code (Phase P, rebuilt). The noise row is NOT a
# reproduction target and must never be scored against:
#   * cell N's 4.96 +/- 1.64 is orphaned — produced by a scratchpad script that was
#     never committed (CLAUDE.md §6 provenance status), and
#   * poison.py cannot have produced cell N at all: it always adds uncontrolled
#     byte swaps, whereas N is noise-only. M5's finding about poison.py's
#     whole-array sampling remains valid, but does not identify N's frame.
# M7 agrees with the orphaned N value to both reported digits (mean and population
# SD) on the committed malicious-only frame. This is numerical corroboration, not
# provenance: N remains unscored until its scratchpad source is recovered or a
# dedicated whole-array run is authorized.
REFERENCE_CAPTURE = {
    "transpositions": {"mean": 1.80, "std": 0.51, "is_target": True,
                        "provenance": "committed-code backed (Phase P, rebuilt)",
                        "per_seed": {"42": 2.2, "123": 2.2, "456": 2.2, "789": 1.0, "1024": 1.4}},
    "block_reversal": {"mean": 0.00, "std": 0.00, "is_target": True,
                        "provenance": "committed-code backed (Phase P, rebuilt)"},
    "block_swap": {"mean": 0.00, "std": 0.00, "is_target": True,
                    "provenance": "committed-code backed (Phase P, rebuilt)"},
    "cyclic_shift": {"mean": 6.28, "std": 1.31, "is_target": True,
                      "provenance": "committed-code backed (Phase P, rebuilt)"},
    "noise": {"mean": None, "std": None, "is_target": False,
               "provenance": "orphaned, numerically corroborated by M7 — NOT a target",
               "orphaned_cell_N": {"mean": 4.96, "std": 1.64},
               "why": "M7 matches the recorded mean and population SD on the malicious-only "
                      "frame; the original N script/frame remains unknown. See A19."},
}

# --- A17 pre-registration -------------------------------------------------------
# Recorded into the M7 artifact BEFORE the run begins and never edited afterwards.
#
# TWO ceiling mechanisms exist, they overlap, and their union is what binds:
#   1. unclustered ceiling — clusterer-dependent; MeanShift has no -1 label so it
#      is 100% there by construction.
#   2. duplicate ceiling  — algorithm-independent. If a poisoned row's 60-dim
#      feature vector exactly equals ANY clean row's, every deterministic
#      clusterer co-locates them, that cluster holds >=1 clean point, its poison
#      fraction is <100%, it is not Red, and the sample is not captured. This is a
#      property of the feature map, not of the clustering algorithm.
#   3. union ceiling      — 1 - (poison in either set / total poison).
PREREGISTRATION = {
    "recorded_before_run": True,
    "duplicate_ceiling_predictions": {
        "block_reversal": {"bit_identical_recorded": 0.880, "predicted_ceiling_at_most": 0.12},
        "block_swap": {"bit_identical_recorded": 0.830, "predicted_ceiling_at_most": 0.17},
        "transpositions": {"bit_identical_recorded": None,
                            "zero_footprint_recorded": 0.125,
                            "predicted_ceiling_at_most": 0.87,
                            "note": "never measured; anywhere in [12.5%, 87%]. A16 measures it "
                                    "for the first time and this is the family that matters — it "
                                    "carries the 2.1999999999999997 internal control."},
        "cyclic_shift": {"bit_identical_recorded": None, "zero_footprint_recorded": 0.0,
                          "predicted_ceiling_at_most": 1.00,
                          "note": "essentially unconstrained by duplicates"},
        "noise": {"bit_identical_recorded": 0.025, "feature_identical_recorded": 0.045,
                   "predicted_ceiling_at_most": 0.95},
    },
    # Stated approximate on purpose: the bit-identity figures come from the
    # 200-target diagnostic frame, M7 runs a 500-poison capture frame. The
    # prediction transfers approximately; a modest discrepancy is expected and is
    # not a failure.
    "transfer_caveat": "bit-identity measured on the 200-target diagnostic frame; "
                       "M7 is a 500-poison capture frame. Approximate transfer only.",
    "purity_curve_crossing_40pct": {
        "block_reversal": "NEVER — duplicate ceiling ~12% caps capture an order of "
                           "magnitude below 40%",
        "block_swap": "NEVER — duplicate ceiling ~17%",
        "transpositions": "NEVER — recorded 1.80% at 100% purity; relaxing purity to >50% is "
                           "not expected to yield the ~22x increase a crossing would require",
        "cyclic_shift": "NEVER — recorded 6.28%; and if it approaches 40% that is saturation "
                         "against the ~42% unclustered ceiling, not a crossing",
        "noise": "NEVER",
        "overall": "NO family is expected to cross 40% at any purity threshold down to >50%. "
                    "If this holds, the zero-tolerance-purity hypothesis for the 40-70% "
                    "reconstruction gap (CLAUDE.md §8 item 6) is RETIRED. That is the less "
                    "comfortable outcome and it is pre-committed here as the expectation.",
    },
    "interpretation_rule": "For any family whose capture curve comes within a few points of "
                            "its own UNION ceiling, the result is SATURATION, not confirmation. "
                            "'Crossed 40%' and 'saturated at 42%' are different findings and "
                            "must not be reported as the same one.",
    "hypothesis_noise_vs_families": {
        "hypothesis": "noise alone yields a HIGHER capture rate than the permutation families",
        "predictions": {"block_reversal": "noise above (0.00)", "block_swap": "noise above (0.00)",
                         "transpositions": "noise above (1.80)", "cyclic_shift": "noise below (6.28)",
                         "SG_surrogate_guided": "noise below (6.48; §6 main factorial, NOT a "
                                                 "Test B family — frame-compatible only)"},
        "pre_commitment": "Report whichever way it lands. If noise comes in above cyclic shift, "
                           "that contradicts the 'capture tracks spatial disruption' framing in "
                           "the poster Discussion — stop and report; do not rerun, do not rescope, "
                           "do not sweep sigma.",
    },
}


def subsample_for_seed(X_full, y_full, seed):
    """Matches run_lens4_baseline.py::subsample_for_seed exactly."""
    rng = np.random.RandomState(seed)
    if len(X_full) > MAX_SAMPLES:
        idx = rng.choice(len(X_full), size=MAX_SAMPLES, replace=False)
        return X_full[idx], y_full[idx]
    return X_full.copy(), y_full.copy()


PERMUTATION_FAMILIES = ("transpositions", "block_reversal", "block_swap", "cyclic_shift")


def parity_check():
    """
    A13: the two drivers must agree; the figure script must not be coupled to
    either. A one-off assertion here is sufficient for this phase — the durable
    version (a `families.py` owning the registry for the two drivers only) is
    proposed at M10 and built after MathFest.
    """
    from programs.test_b_diagnostics import FAMILIES as DIAG_FAMILIES
    from programs.make_figure_v3 import FAMILIES as V3_FAMILIES

    report = {}
    keys_here, keys_diag = set(FAMILIES), set(DIAG_FAMILIES)
    report["key_sets_identical"] = keys_here == keys_diag
    report["keys_capture"] = sorted(keys_here)
    report["keys_diagnostics"] = sorted(keys_diag)
    assert keys_here == keys_diag, f"FAMILIES key sets differ: {keys_here ^ keys_diag}"

    mismatches = []
    for fam in PERMUTATION_FAMILIES:
        fn_here, kw_here = FAMILIES[fam]
        fn_diag, kw_diag, _ = DIAG_FAMILIES[fam]
        if fn_here is not fn_diag or kw_here != kw_diag:
            mismatches.append({"family": fam,
                               "capture": (fn_here.__name__, kw_here),
                               "diagnostics": (fn_diag.__name__, kw_diag)})
    report["permutation_fn_kwargs_identical"] = not mismatches
    report["mismatches"] = mismatches
    assert not mismatches, f"fn/kwargs differ for {mismatches}"

    v3_names = [f[0] for f in V3_FAMILIES]
    report["v3_names"] = v3_names
    report["v3_is_exactly_the_four_permutations"] = v3_names == sorted(
        v3_names, key=lambda n: n) or set(v3_names) == set(PERMUTATION_FAMILIES)
    report["v3_excludes_noise"] = "noise" not in v3_names
    assert set(v3_names) == set(PERMUTATION_FAMILIES), \
        f"figure V3 list is not exactly the four permutation families: {v3_names}"
    assert "noise" not in v3_names, "figure V3 list contains 'noise'"
    report["v3_display_order"] = v3_names
    return report


def a18_superset_check():
    """
    A18: 'superseded' must be a verified claim, not an assertion. Every key in the
    committed Phase P artifact must exist in the M6 artifact with an equal value
    wherever both are populated. If the M6 artifact is not a strict superset it is
    not a successor and the amendment does not apply.
    """
    import json as _json
    p_path = RESULTS_DIR / "test_b_diagnostics.json"
    m_path = RESULTS_DIR / "phase_m_m6_diagnostics.json"
    if not (p_path.exists() and m_path.exists()):
        return {"ran": False, "reason": f"missing artifact: {p_path.exists()=} {m_path.exists()=}"}
    p, m = _json.load(open(p_path)), _json.load(open(m_path))
    missing, unequal, n_leaves = [], [], [0]

    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                unequal.append({"path": path, "kind": "type"}); return
            for k, v in a.items():
                if k not in b:
                    missing.append(f"{path}.{k}")
                else:
                    walk(v, b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                unequal.append({"path": path, "kind": "list"}); return
            for i, v in enumerate(a):
                walk(v, b[i], f"{path}[{i}]")
        else:
            n_leaves[0] += 1
            if a != b:
                unequal.append({"path": path, "phase_p": a, "m6": b})

    walk(p, m, "")
    strict = (not missing) and (not unequal)
    return {"ran": True, "strict_superset": strict, "keys_only_in_phase_p": missing,
            "unequal_shared_values": unequal, "scalar_leaves_compared": n_leaves[0],
            "phase_p_artifact_status": "SUPERSEDED" if strict else "NOT A SUCCESSOR"}


def pipeline_max_values(pipeline):
    """
    A14: the fitted Binarizer.max_value_ from each of the five filtration branches
    of the real feature pipeline — the fits that actually happen at M7, read off
    the fitted estimator rather than recomputed.
    """
    return {name: float(sub.steps[0][1].max_value_)
            for name, sub in pipeline.transformer_list}


def exact_duplicate_analysis(X_tda, is_poisoned, attack_log):
    """
    A16: exact feature-duplicate detection over the already-fitted X_tda. No new
    extraction, no second fit — one comparison pass over an array M7 already holds.

    Exact equality on the fitted array, no tolerance. Rows are compared by their
    raw float64 bytes; signed zero is normalised first so -0.0 and 0.0 (which are
    numerically equal but byte-different) cannot manufacture a false distinction.
    NaN count is reported because NaN would break value equality while leaving byte
    equality intact.
    """
    from collections import Counter

    n_total = len(X_tda)
    n_poison = int(is_poisoned.sum())
    n_clean = n_total - n_poison

    # The duplicate argument requires the clean original to still be present in the
    # clustered set. Verify the harness APPENDS rather than replaces.
    clean_block_is_first = not bool(is_poisoned[:n_clean].any())
    poison_block_is_tail = bool(is_poisoned[n_clean:].all())
    appended = clean_block_is_first and poison_block_is_tail and n_poison == len(attack_log)

    Xk = np.where(X_tda == 0.0, 0.0, np.ascontiguousarray(X_tda, dtype=np.float64))
    keys = [row.tobytes() for row in Xk]
    clean_keys = set(keys[:n_clean])
    poison_keys = keys[n_clean:]

    dup_any_clean = np.array([k in clean_keys for k in poison_keys], dtype=bool)
    if appended:
        own = np.array([keys[n_clean + i] == keys[attack_log[i]["target_index"]]
                        for i in range(n_poison)], dtype=bool)
    else:
        own = np.zeros(n_poison, dtype=bool)
    cnt = Counter(poison_keys)
    dup_within = np.array([cnt[k] >= 2 for k in poison_keys], dtype=bool)

    return {
        "harness_appends_not_replaces": appended,
        "clean_block_is_first": clean_block_is_first,
        "poison_block_is_tail": poison_block_is_tail,
        "x_tda_shape": list(X_tda.shape),
        "n_clean": n_clean, "n_poison": n_poison,
        "n_nan_in_x_tda": int(np.isnan(X_tda).sum()),
        "tolerance": "exact (byte equality on float64, signed zero normalised)",
        "n_poison_dup_any_clean": int(dup_any_clean.sum()),
        "frac_poison_dup_any_clean": float(dup_any_clean.mean()),
        "n_poison_dup_own_original": int(own.sum()),
        "frac_poison_dup_own_original": float(own.mean()),
        "n_poison_dup_within_poison_block": int(dup_within.sum()),
        "frac_poison_dup_within_poison_block": float(dup_within.mean()),
        "n_distinct_poison_feature_vectors": int(len(cnt)),
        "_dup_any_clean_mask": dup_any_clean,   # consumed by ceilings(), not serialised
    }


def ceilings(labels, is_poisoned, dup_mask):
    """
    A17: decompose the capture ceiling into its two mechanisms and their union.
    Reporting only the union would let an OPTICS result read as a clustering
    artifact when it is a feature-map property, or the reverse.
    """
    poison_labels = labels[is_poisoned]
    n_poison = len(poison_labels)
    in_noise = poison_labels == -1
    has_noise_label = bool((labels == -1).any())
    union = in_noise | dup_mask
    return {
        "algorithm_has_noise_label": has_noise_label,
        "n_poison_total": int(n_poison),
        "n_poison_unclustered": int(in_noise.sum()),
        "frac_poison_unclustered": float(in_noise.mean()),
        "unclustered_ceiling": float(1.0 - in_noise.mean()),
        "n_poison_duplicate": int(dup_mask.sum()),
        "frac_poison_duplicate": float(dup_mask.mean()),
        "duplicate_ceiling": float(1.0 - dup_mask.mean()),
        "n_poison_union": int(union.sum()),
        "frac_poison_union": float(union.mean()),
        "union_ceiling": float(1.0 - union.mean()),
        "n_poison_both_mechanisms": int((in_noise & dup_mask).sum()),
    }


def run_single_pass(X_combined, is_poisoned, attack_log, threshold=THRESHOLD):
    t0 = time.time()
    X_tda, pipeline = extract_tda_features(X_combined, threshold=threshold)
    tda_time = time.time() - t0

    # A14: max_value_ from the five Binarizer fits that actually happened here.
    max_values = pipeline_max_values(pipeline)

    # A16: one comparison pass over the fitted array. Algorithm-independent, so it
    # is computed once per pass and shared across all four clusterers below.
    dup = exact_duplicate_analysis(X_tda, is_poisoned, attack_log)
    dup_mask = dup.pop("_dup_any_clean_mask")

    t0 = time.time()
    results = run_all_clustering(X_tda)
    cluster_time = time.time() - t0

    per_algo = {}
    for algo_name, labels in results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned)
        # A11: the -1 record in raw counts, not derived fractions.
        minus_one = next((c for c in cluster_info if c["cluster_id"] == -1), None)
        per_algo[algo_name] = {
            "red_poison_capture_pct": summary["red_poison_capture_pct"],
            "colors": summary["colors"],
            "unclustered_record": {
                "present": minus_one is not None,
                "n_poisoned": int(minus_one["n_poisoned"]) if minus_one else 0,
                "n_clean": int(minus_one["n_clean"]) if minus_one else 0,
                "size": int(minus_one["size"]) if minus_one else 0,
            },
            "n_poison_total_this_pass": int(is_poisoned.sum()),
            "ceilings": ceilings(labels, is_poisoned, dup_mask),
            # M3 (Phase M): persist the full per-cluster purity records that
            # classify_clusters already computes and the driver previously
            # discarded (CLAUDE.md §8 item 5). Additive only — no existing key
            # changes. Each record carries cluster_id (-1 = OPTICS noise bucket,
            # already emitted as color "Noise"), size, n_poisoned, n_clean,
            # poison_fraction, color, size_pct, dpdc. Serialized via the file's
            # json.dump(default=convert_for_json), which handles the numpy leaves.
            "clusters": cluster_info,
        }
    return {"tda_time_s": tda_time, "cluster_time_s": cluster_time, "per_algo": per_algo,
            "max_value_per_filtration": max_values, "duplicate_analysis": dup}


def main():
    # --- A13 parity check, run BEFORE the capture run ---------------------------
    print("=== A13 parity check ===")
    parity = parity_check()
    print(f"  FAMILIES key sets identical (capture vs diagnostics): "
          f"{parity['key_sets_identical']}  {parity['keys_capture']}")
    print(f"  fn/kwargs identical for the four permutation families: "
          f"{parity['permutation_fn_kwargs_identical']}")
    print(f"  figure V3 list = {parity['v3_names']}")
    print(f"  figure V3 excludes 'noise': {parity['v3_excludes_noise']}")

    # --- A18 superset check ----------------------------------------------------
    print("\n=== A18 superset check (Phase P artifact vs M6 artifact) ===")
    superset = a18_superset_check()
    print(f"  strict superset: {superset.get('strict_superset')}  "
          f"leaves compared: {superset.get('scalar_leaves_compared')}  "
          f"keys only in Phase P: {superset.get('keys_only_in_phase_p')}")
    print(f"  Phase P artifact status: {superset.get('phase_p_artifact_status')}")

    print("\nLoading full UNSW-NB15 dataset (once, for all seeds)...")
    X_all, y_all = load_unsw(max_samples=None)

    # Phase M artifact. The Phase P capture artifact
    # (results/test_b_permutation_families.json) is committed and is left alone;
    # this run is a five-family superset of it, so it gets its own name.
    out_path = RESULTS_DIR / OUTPUT_NAME
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {family: {} for family in FAMILIES}
    all_results["_reference"] = REFERENCE_CAPTURE
    all_results["_meta"] = {"seeds": SEEDS, "max_samples": MAX_SAMPLES,
                             "poison_rate": POISON_RATE, "threshold": THRESHOLD}
    # A17: pre-registration is written to disk BEFORE the first pass runs, so it
    # cannot be edited after the numbers arrive.
    all_results["_preregistration"] = PREREGISTRATION
    all_results["_parity_check"] = parity
    all_results["_a18_superset_check"] = superset
    all_results["env"] = env_block()
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=convert_for_json)
    print(f"\nPre-registration written to {out_path} before the run began.")

    t_start = time.time()
    for family, (fn, kwargs) in FAMILIES.items():
        for seed in SEEDS:
            print(f"\n--- {family}, seed {seed} ---")
            X, y = subsample_for_seed(X_all, y_all, seed)
            t0 = time.time()
            Xc, yc, ip, log = fn(X, y, poison_rate=POISON_RATE, random_state=seed, **kwargs)
            gen_time = time.time() - t0
            validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
            baseline = run_single_pass(Xc, ip, log)
            print(f"  gen_time={gen_time:.1f}s validity={validity_pct:.1f}% "
                  f"tda={baseline['tda_time_s']:.1f}s cluster={baseline['cluster_time_s']:.1f}s")
            for algo, m in baseline["per_algo"].items():
                print(f"    {algo:<10} capture={m['red_poison_capture_pct']:.2f}%")

            all_results[family][str(seed)] = {
                "gen_time_s": gen_time, "validity_pct": validity_pct, "n_poison": int(ip.sum()),
                **baseline,
            }
            # Namespaced by (family, seed) key within one file, written after
            # every (family, seed) pair so a crash preserves partial progress
            # and the file is always internally consistent to inspect.
            with open(out_path, "w") as fh:
                json.dump(all_results, fh, indent=2, default=convert_for_json)

    total_time = time.time() - t_start
    print(f"\nTotal wall-clock: {total_time:.1f}s")

    def cap(family, seed, algo="OPTICS"):
        return all_results[family][str(seed)]["per_algo"][algo]["red_poison_capture_pct"]

    print("\n=== SUMMARY (OPTICS, population std) ===")
    print(f"{'Family':<15} {'Observed':>18}   {'Recorded':>18}   {'Per-seed'}")
    for family in FAMILIES:
        captures = [cap(family, s) for s in SEEDS]
        mean, std = float(np.mean(captures)), float(np.std(captures))  # population (ddof=0)
        ref = REFERENCE_CAPTURE[family]
        if ref.get("is_target"):
            ref_txt = f"{ref['mean']:>7.2f}% +/- {ref['std']:<6.2f}%"
        else:
            ref_txt = f"{'superseded':>18}"   # A6: never scored against
        print(f"{family:<15} {mean:>7.2f}% +/- {std:<6.2f}%   {ref_txt}   {captures}")

    print("\n=== ALL FOUR CLUSTERERS (mean +/- population sd) ===")
    algos = list(all_results[next(iter(FAMILIES))][str(SEEDS[0])]["per_algo"].keys())
    print(f"{'Family':<15}" + "".join(f"{a:>20}" for a in algos))
    for family in FAMILIES:
        cells = []
        for a in algos:
            v = [cap(family, s, a) for s in SEEDS]
            cells.append(f"{np.mean(v):>6.2f} +/- {np.std(v):<5.2f}")
        print(f"{family:<15}" + "".join(f"{c:>20}" for c in cells))

    print("\n=== A17 CEILINGS (OPTICS; duplicate ceiling is algorithm-independent) ===")
    print(f"{'Family':<15}{'dup ceil':>11}{'unclust ceil':>14}{'union ceil':>12}"
          f"{'capture':>10}{'headroom':>10}")
    for family in FAMILIES:
        d = [all_results[family][str(s)]["per_algo"]["OPTICS"]["ceilings"] for s in SEEDS]
        dc = float(np.mean([x["duplicate_ceiling"] for x in d]))
        uc = float(np.mean([x["unclustered_ceiling"] for x in d]))
        nc = float(np.mean([x["union_ceiling"] for x in d]))
        c = float(np.mean([cap(family, s) for s in SEEDS])) / 100.0
        print(f"{family:<15}{100*dc:>10.2f}%{100*uc:>13.2f}%{100*nc:>11.2f}%"
              f"{100*c:>9.2f}%{100*(nc-c):>9.2f}%")

    all_results["_summary"] = {
        family: {
            "mean": float(np.mean([cap(family, s) for s in SEEDS])),
            "std": float(np.std([cap(family, s) for s in SEEDS])),
            "per_seed": {str(s): cap(family, s) for s in SEEDS},
            "per_algo": {a: {"mean": float(np.mean([cap(family, s, a) for s in SEEDS])),
                             "std": float(np.std([cap(family, s, a) for s in SEEDS])),
                             "per_seed": {str(s): cap(family, s, a) for s in SEEDS}}
                         for a in algos},
        }
        for family in FAMILIES
    }
    all_results["_total_wall_clock_s"] = total_time

    # --- Blocking regression gate (§6 internal control) -------------------------
    # Transpositions must return 2.1999999999999997 at full floating-point
    # precision, not 2.20 and not 2.2000. A literal mismatch means the M3/M7
    # instrumentation altered behaviour somewhere repro_check.py does not look.
    EXPECT = 2.1999999999999997
    control = {str(s): cap("transpositions", s) for s in SEEDS}
    recorded = {"42": 2.1999999999999997, "123": 2.1999999999999997,
                "456": 2.1999999999999997, "789": 1.0, "1024": 1.4000000000000001}
    literal_match = {s: (control[s] == recorded[s]) for s in control}
    all_results["_control_gate"] = {
        "expect_seed42_literal": repr(EXPECT),
        "observed_seed42_literal": repr(control["42"]),
        "seed42_literal_match": control["42"] == EXPECT,
        "observed_all_seeds": {s: repr(v) for s, v in control.items()},
        "recorded_all_seeds": {s: repr(v) for s, v in recorded.items()},
        "all_seeds_literal_match": all(literal_match.values()),
        "per_seed_match": literal_match,
    }
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=convert_for_json)
    print(f"\nWritten to {out_path}")

    print("\n=== BLOCKING CONTROL GATE: transpositions, full float precision ===")
    for s in SEEDS:
        print(f"  seed {s:>4}: observed {control[str(s)]!r}  recorded {recorded[str(s)]!r}  "
              f"{'MATCH' if literal_match[str(s)] else '*** MISMATCH ***'}")
    if not all(literal_match.values()):
        print("\nControl drifted. The M3/M7 instrumentation altered behaviour somewhere "
              "repro_check.py did not catch. STOPPING — do not proceed to M8.")
        sys.exit(1)
    print("  Control reproduces at full floating-point precision on all five seeds.")


if __name__ == "__main__":
    main()
