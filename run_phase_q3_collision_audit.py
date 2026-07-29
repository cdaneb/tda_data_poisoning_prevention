"""Phase Q3 driver: collision-provenance audit of the legacy 60-feature pipeline.

Q3-A  reproduce and *define* the Q2 collision observation on the frozen frame
Q3-B  trace the largest final-vector class upstream to its earliest merger
Q3-C  population-wide earliest-merger attribution + per-filtration ablation
Q3-D  decompose strict-100%-purity capture failure for every poisoned row
Q3-E  five-seed confirmation

This is a diagnostic phase.  Nothing here filters rows, deduplicates, changes
binarization, changes a filtration, or tunes OPTICS.  Ground-truth labels are
used only retrospectively, for composition and attribution; every stage
signature is built label-free by ``phase_q3_stage_pipeline``.

Usage:
    python run_phase_q3_collision_audit.py                # full audit
    python run_phase_q3_collision_audit.py --seeds 42     # seed 42 only
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from clustering import classify_clusters
from phase_q3_collisions import (
    class_size_array,
    class_stats,
    earliest_merger_attribution,
    equivalence_classes,
    exact_hash,
    label_free_signature_violations,
    transition_report,
)
from phase_q3_stage_pipeline import CHAIN_STAGES, extract_all_stages
from tools.phase_q2_common import (
    CONFIRMATION_SEEDS,
    build_realization,
    environment_block,
    float_array_hash,
    realization_provenance,
    write_json,
)

OPTICS_PARAMS = {"min_samples": 5, "max_eps": 2.0}
OUT_PATH = Path(__file__).resolve().parent / "results" / "phase_q3_collision_audit.json"

# Q2 recorded these for the legacy 30x50 / threshold-0.4 arm at seed 42.
Q2_REFERENCE = {
    "source_artifact": "results/phase_q2_geometry.json  arms.legacy_30x50_t04",
    "input_hash": "fce036c2424196ef",
    "poison_mask_hash": "8f8ee5534151e4fe",
    "feature_hash_rounded9": "ffce38eb0cd462df",
    "n_exact_duplicate_rows": 2849,
    "max_duplicate_multiplicity": 1043,
}


def run_optics(X_tda):
    model = OPTICS(**OPTICS_PARAMS, n_jobs=-1)
    labels = model.fit_predict(X_tda)
    return labels, model


def quant(a, qs=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(a)
    return {str(q): float(np.percentile(a, q)) for q in qs} if a.size else {}


def summarize_ints(a):
    a = np.asarray(a)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else 0.0,
            "min": int(a.min()) if a.size else 0,
            "max": int(a.max()) if a.size else 0,
            "quantiles": quant(a)}


# ---------------------------------------------------------------------------
# Q3-A
# ---------------------------------------------------------------------------

def q3a(real, extracted, labels):
    poisoned = real["is_poisoned"]
    final = extracted["stages"]["final_60_vector"]
    stats = class_stats(final, poisoned, "final_60_vector")
    X_tda = extracted["X_tda"]

    n_rows = stats["n_rows"]
    redundancy = stats["redundancy_fraction"]
    repeated_member = stats["repeated_member_fraction"]

    # Which statistic was Q2's "51.8%"?
    q2_dup_rows = Q2_REFERENCE["n_exact_duplicate_rows"]
    q2_value = q2_dup_rows / n_rows
    matches_redundancy = abs(q2_value - redundancy) < 1e-12
    matches_repeated = abs(q2_value - repeated_member) < 1e-12

    classes = equivalence_classes(final)
    largest_sig = stats["largest_class_signature"]
    largest_idx = classes[largest_sig]

    # OPTICS label behaviour inside every exact coordinate class.
    labels = np.asarray(labels)
    n_labels_per_repeated_class = []
    split_classes = []
    n_split_across_two_real_clusters = 0
    n_split_only_by_noise = 0
    for sig, idx in classes.items():
        if len(idx) < 2:
            continue
        uniq = np.unique(labels[idx])
        n_labels_per_repeated_class.append(len(uniq))
        if len(uniq) > 1:
            real_cluster_ids = [int(v) for v in uniq if v != -1]
            if len(real_cluster_ids) > 1:
                n_split_across_two_real_clusters += 1
            else:
                n_split_only_by_noise += 1
            split_classes.append({
                "signature": sig, "size": int(len(idx)),
                "n_distinct_labels": int(len(uniq)),
                "labels": [int(v) for v in uniq[:10]],
                "split_kind": ("two_real_clusters" if len(real_cluster_ids) > 1
                               else "one_real_cluster_plus_noise"),
            })
    n_labels_per_repeated_class = np.array(n_labels_per_repeated_class)

    return {
        "definition_note": (
            "repeated_member_fraction counts rows in a class of size >= 2; "
            "redundancy_fraction is (n_rows - n_unique_classes) / n_rows. "
            "The phrase 'duplicate fraction' is not used."
        ),
        "final_vector_class_stats": stats,
        "feature_hash_exact": exact_hash(X_tda),
        "feature_hash_rounded9": float_array_hash(X_tda),
        "q2_reference": Q2_REFERENCE,
        "q2_statistic_resolution": {
            "q2_reported_value": float(q2_value),
            "q2_reported_n_rows": int(q2_dup_rows),
            "reproduced_redundancy_fraction": float(redundancy),
            "reproduced_n_redundant_rows": int(stats["n_redundant_rows"]),
            "reproduced_repeated_member_fraction": float(repeated_member),
            "reproduced_n_repeated_member_rows": int(stats["n_repeated_member_rows"]),
            "q2_51_8_pct_is_redundancy_fraction": bool(matches_redundancy),
            "q2_51_8_pct_is_repeated_member_fraction": bool(matches_repeated),
            "input_hash_matches_q2": real["input_hash"] == Q2_REFERENCE["input_hash"],
            "poison_mask_hash_matches_q2": (
                real["poison_mask_hash"] == Q2_REFERENCE["poison_mask_hash"]),
            "feature_hash_matches_q2": (
                float_array_hash(X_tda) == Q2_REFERENCE["feature_hash_rounded9"]),
        },
        "largest_final_class": {
            "size": int(len(largest_idx)),
            "share_of_all_rows": float(len(largest_idx) / n_rows),
            "signature": largest_sig,
            "reproduces_q2_1043": bool(len(largest_idx) == 1043),
            "n_clean": int((~poisoned[largest_idx]).sum()),
            "n_poison": int(poisoned[largest_idx].sum()),
        },
        "optics_label_behaviour_in_exact_classes": {
            "n_repeated_classes_examined": int(len(n_labels_per_repeated_class)),
            "max_distinct_labels_in_one_class": (
                int(n_labels_per_repeated_class.max())
                if n_labels_per_repeated_class.size else 0),
            "n_classes_split_across_labels": len(split_classes),
            "n_classes_split_across_two_real_clusters": n_split_across_two_real_clusters,
            "n_classes_split_only_by_noise_assignment": n_split_only_by_noise,
            "split_class_examples": split_classes[:8],
            "all_exact_classes_co_assigned": len(split_classes) == 0,
            "no_exact_class_spans_two_real_clusters": n_split_across_two_real_clusters == 0,
            "scope_note": (
                "Verified empirically for this fitted OPTICS "
                f"({OPTICS_PARAMS}) at this seed. Not proven for OPTICS in "
                "general and not claimed for any other algorithm. The "
                "distinction matters: a class split only because one twin was "
                "left at -1 still cannot produce two competing real clusters, "
                "but it does mean a 100%-poison cluster can exist while an "
                "identical clean row sits in noise."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Q3-B
# ---------------------------------------------------------------------------

def q3b(real, extracted, labels, attribution):
    poisoned = np.asarray(real["is_poisoned"])
    y = np.asarray(real["y_combined"])
    n_clean_rows = real["n_clean"]
    stages = extracted["stages"]
    desc = extracted["descriptive"]
    final = stages["final_60_vector"]
    classes = equivalence_classes(final)
    sig = max(classes, key=lambda h: len(classes[h]))
    idx = classes[sig]

    per_stage = []
    for name in CHAIN_STAGES:
        h = np.asarray(stages[name], dtype=object)[idx]
        per_stage.append({
            "stage": name,
            "n_distinct_upstream_signatures": int(len(set(h.tolist()))),
        })

    # Poison members: is each identical to its own clean source row at each stage?
    poison_members = idx[poisoned[idx]]
    src_of = {}
    for i in poison_members:
        src_of[int(i)] = int(real["attack_target_index"][int(i) - n_clean_rows])
    identical_to_source = {}
    for name in CHAIN_STAGES:
        h = np.asarray(stages[name], dtype=object)
        identical_to_source[name] = int(sum(
            1 for i in poison_members if h[i] == h[src_of[int(i)]]))

    src_in_class = int(sum(1 for i in poison_members if src_of[int(i)] in set(idx.tolist())))
    lbls, cnts = np.unique(y[idx], return_counts=True)

    earliest = attribution["_earliest_stage_index"]
    e_in_class = earliest[idx]
    stage_names = attribution["stage_order"]
    earliest_hist = {}
    for s, nm in enumerate(stage_names):
        c = int(np.count_nonzero(e_in_class == s))
        if c:
            earliest_hist[nm] = c
    first_stage = min((s for s in e_in_class if s >= 0), default=-1)

    optics_lbls, optics_cnts = np.unique(np.asarray(labels)[idx], return_counts=True)

    return {
        "signature": sig,
        "size": int(len(idx)),
        "share_of_all_rows": float(len(idx) / len(final)),
        "n_clean": int((~poisoned[idx]).sum()),
        "n_poison": int(poisoned[idx].sum()),
        "n_source_rows": int(np.count_nonzero(idx < n_clean_rows)),
        "n_appended_poison_rows": int(np.count_nonzero(idx >= n_clean_rows)),
        "n_poison_whose_clean_source_is_also_in_this_class": src_in_class,
        "unsw_label_composition": {str(k): int(v) for k, v in zip(lbls, cnts)},
        "per_stage_distinct_signatures": per_stage,
        "earliest_merger_stage_histogram": earliest_hist,
        "earliest_merger_stage_for_class": (
            stage_names[first_stage] if first_stage >= 0 else None),
        "poison_members_identical_to_own_clean_source": identical_to_source,
        "n_poison_members": int(len(poison_members)),
        "padding_profile": {
            "support_end": summarize_ints(desc["support_end"][idx]),
            "nonzero_count": summarize_ints(desc["nonzero_count"][idx]),
            "n_all_zero_payload": int(desc["all_zero_payload"][idx].sum()),
            "all_zero_share_of_class": float(desc["all_zero_payload"][idx].mean()),
            "foreground_count_at_threshold_0_4": summarize_ints(
                desc["foreground_count"][idx]),
        },
        "optics_labels_in_class": {
            "labels": [int(v) for v in optics_lbls],
            "counts": [int(v) for v in optics_cnts],
        },
        "example_row_indices": [int(v) for v in idx[:5]],
        "example_note": (
            "Ordinary 0-based row numbers into the 5500-row combined matrix "
            "(0..4999 = subsampled clean rows, 5000..5499 = appended poison). "
            "No payload bytes are recorded anywhere in this artifact."
        ),
    }


# ---------------------------------------------------------------------------
# Q3-C
# ---------------------------------------------------------------------------

def q3c(real, extracted, attribution):
    poisoned = np.asarray(real["is_poisoned"])
    stages = extracted["stages"]
    desc = extracted["descriptive"]

    stage_stats = [class_stats(stages[n], poisoned, n) for n in CHAIN_STAGES]

    transitions = []
    for a, b in zip(CHAIN_STAGES[:-1], CHAIN_STAGES[1:]):
        rep = transition_report(stages[a], stages[b], poisoned)
        rep["from"], rep["to"] = a, b
        transitions.append(rep)

    # Padding characterisation of the raw-collision mass.  The support record is
    # a bijection with the raw row, so it can never merge anything on its own;
    # "padding artifact" is a *description* of the raw-collision class, not a
    # separate merger stage, and is reported that way.
    raw_sizes = class_size_array(stages["raw_payload"])
    raw_repeated = raw_sizes >= 2
    padding = {
        "n_raw_repeated_member_rows": int(raw_repeated.sum()),
        "n_all_zero_payload_rows": int(desc["all_zero_payload"].sum()),
        "all_zero_share_of_raw_repeated_mass": (
            float(desc["all_zero_payload"][raw_repeated].mean())
            if raw_repeated.any() else 0.0),
        "support_end_of_raw_repeated_rows": summarize_ints(
            desc["support_end"][raw_repeated]),
        "support_end_of_raw_singleton_rows": summarize_ints(
            desc["support_end"][~raw_repeated]),
        "foreground_count_of_raw_repeated_rows": summarize_ints(
            desc["foreground_count"][raw_repeated]),
        "note": (
            "(support_end, row[:support_end]) determines the whole zero-padded "
            "row, so the supported-payload record is a bijection with the raw "
            "row and merges nothing. Its attribution row is expected to be 0 "
            "and serves as a self-check."
        ),
    }

    # Per-filtration ablation (diagnostic only; no feature selection follows).
    final_repeated = class_size_array(stages["final_60_vector"]) >= 2
    per_filt = []
    for fname in extracted["filtration_names"]:
        bh = extracted["per_filtration"][fname]["feature_block"]
        st = class_stats(bh, poisoned, f"feature_block::{fname}")
        merged_from_binary = transition_report(stages["binary_mask"], bh, poisoned)
        merged_from_raw = transition_report(stages["raw_payload"], bh, poisoned)
        block_repeated = class_size_array(bh) >= 2
        per_filt.append({
            "filtration": fname,
            "n_unique_classes": st["n_unique_classes"],
            "repeated_member_fraction": st["repeated_member_fraction"],
            "redundancy_fraction": st["redundancy_fraction"],
            "largest_class_size": st["largest_class_size"],
            "n_block_classes_merging_multiple_binary_classes":
                merged_from_binary["n_downstream_classes_merging_multiple_upstream_classes"],
            "n_rows_merged_relative_to_binary":
                merged_from_binary["n_rows_in_newly_merged_classes"],
            "n_block_classes_merging_multiple_raw_classes":
                merged_from_raw["n_downstream_classes_merging_multiple_upstream_classes"],
            "n_rows_repeated_in_this_block_but_singleton_at_final":
                int(np.count_nonzero(block_repeated & ~final_repeated)),
        })

    final_stats = next(s for s in stage_stats if s["stage"] == "final_60_vector")
    dominant = max(per_filt, key=lambda p: p["repeated_member_fraction"])
    return {
        "stage_class_stats": stage_stats,
        "earliest_merger_attribution": {
            k: v for k, v in attribution.items() if not k.startswith("_")},
        "stage_transitions": transitions,
        "padding_characterisation_of_raw_collisions": padding,
        "per_filtration_ablation": {
            "blocks": per_filt,
            "final_repeated_member_fraction": final_stats["repeated_member_fraction"],
            "most_collision_prone_block": dominant["filtration"],
            "note": (
                "Diagnostic only. Q3 does not select, reweight, or drop any "
                "filtration on the basis of this table."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Q3-D
# ---------------------------------------------------------------------------

def q3d(real, extracted, labels):
    poisoned = np.asarray(real["is_poisoned"])
    labels = np.asarray(labels)
    final = extracted["stages"]["final_60_vector"]
    classes = equivalence_classes(final)

    exact_mixed = np.zeros(len(poisoned), dtype=bool)
    for idx in classes.values():
        n_p = int(poisoned[idx].sum())
        if 0 < n_p < len(idx):
            exact_mixed[idx] = True

    cluster_info, summary = classify_clusters(labels, poisoned)
    red_ids = {c["cluster_id"] for c in cluster_info if c["color"] == "Red"}
    mixed_ids = {c["cluster_id"] for c in cluster_info
                 if c["color"] in ("Yellow", "Pink")}

    p_idx = np.flatnonzero(poisoned)
    cat = np.zeros(len(p_idx), dtype=int)
    for j, i in enumerate(p_idx):
        lab = labels[i]
        if lab != -1 and lab in red_ids:
            cat[j] = 1
        elif lab == -1:
            cat[j] = 2
        elif exact_mixed[i]:
            cat[j] = 3
        elif lab in mixed_ids:
            cat[j] = 4
        else:
            cat[j] = 5

    n_p = len(p_idx)
    names = {
        1: "captured_in_100pct_poison_non_noise_cluster",
        2: "label_minus_1_unclustered",
        3: "non_noise_and_shares_exact_final_vector_with_a_clean_row",
        4: "non_noise_distinct_vector_but_assigned_to_a_mixed_cluster",
        5: "residual_unexplained",
    }
    decomposition = {names[k]: {"n": int((cat == k).sum()),
                                "pct_of_poison": float(100.0 * (cat == k).sum() / n_p)}
                     for k in sorted(names)}

    obstruction = float(exact_mixed[poisoned].mean())
    captured = np.zeros(len(poisoned), dtype=bool)
    captured[p_idx[cat == 1]] = True
    # Does the obstruction actually bind?  A poisoned row that is BOTH captured
    # and in a mixed exact-vector class would be a real counterexample to
    # treating 1 - obstruction as a ceiling for this fit.
    n_captured_in_mixed_class = int(np.count_nonzero(captured & exact_mixed))
    return {
        "n_poison": int(n_p),
        "categories_priority_order": [names[k] for k in sorted(names)],
        "decomposition": decomposition,
        "categories_sum_to_all_poison": int(sum(
            d["n"] for d in decomposition.values())) == n_p,
        "residual_is_empty": int(decomposition[names[5]]["n"]) == 0,
        "residual_note": (
            "A non-noise poisoned row is in a cluster that is either 100% "
            "poison (category 1) or mixed (categories 3/4), so category 5 is "
            "expected to be empty; it is reported rather than assumed."
        ),
        "coordinate_class_obstruction": {
            "numerator_poison_sharing_exact_vector_with_clean": int(exact_mixed[poisoned].sum()),
            "denominator_all_poison": int(n_p),
            "obstruction_fraction": obstruction,
            "one_minus_obstruction": 1.0 - obstruction,
            "n_captured_rows_also_in_a_mixed_exact_vector_class": n_captured_in_mixed_class,
            "obstruction_binds_under_this_fit": n_captured_in_mixed_class == 0,
            "scope_note": (
                "1 - obstruction is NOT claimed as an algorithm-independent "
                "capture ceiling. It is a ceiling exactly to the extent that "
                "the clustering rule cannot split identical coordinates; that "
                "co-assignment is verified empirically for the standing OPTICS "
                "fit in Q3-A and claimed only at that scope."
            ),
        },
        "comparisons": {
            "observed_exact_purity_capture_pct": float(summary["red_poison_capture_pct"]),
            "poison_unclustered_fraction": float((labels == -1)[poisoned].mean()),
            "clean_unclustered_fraction": float((labels == -1)[~poisoned].mean()),
            "n_clusters": int(summary["n_clusters"]),
            "colors": summary["colors"],
        },
    }


# ---------------------------------------------------------------------------
# per-seed and driver
# ---------------------------------------------------------------------------

def audit_seed(seed, full=True):
    t0 = time.time()
    print(f"\n{'=' * 70}\n=== Phase Q3 audit, seed {seed} ===\n{'=' * 70}")
    real = build_realization(seed)
    # Map each appended poison row to the clean row it was derived from.
    real["attack_target_index"] = None  # filled below from a fresh attack log

    from adversarial_attack import malicious_random_attack
    from tools.phase_q2_common import MAX_SAMPLES, N_SWAPS, POISON_RATE, load_unsw_once
    X_all, y_all = load_unsw_once()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_all), size=MAX_SAMPLES, replace=False)
    _, _, _, log = malicious_random_attack(
        X_all[idx], y_all[idx], poison_rate=POISON_RATE,
        random_state=seed, n_swaps=N_SWAPS)
    real["attack_target_index"] = [e["target_index"] for e in log]

    extracted = extract_all_stages(real["X_combined"], threshold=0.4)
    print("  [q3] instrumented 60-vector equals production:",
          extracted["equality_check"]["instrumented_equals_production_bitwise"])

    print("  [q3] fitting OPTICS...")
    labels, model = run_optics(extracted["X_tda"])

    attribution = earliest_merger_attribution(
        list(CHAIN_STAGES),
        [extracted["stages"][n] for n in CHAIN_STAGES],
        real["is_poisoned"])

    block = {
        "seed": int(seed),
        "realization": realization_provenance(real | {"attack_target_index": None}),
        "equality_check": extracted["equality_check"],
        "fitted_state": extracted["fitted_state"],
        "effective_byte_cut": extracted["effective_byte_cut"],
        "optics_params": OPTICS_PARAMS,
        "labels_hash": exact_hash(np.asarray(labels)),
        "Q3_A_definition_and_reproduction": q3a(real, extracted, labels),
        "Q3_D_strict_purity_failure_decomposition": q3d(real, extracted, labels),
    }
    if full:
        block["Q3_B_largest_class_trace"] = q3b(real, extracted, labels, attribution)
        block["Q3_C_population_attribution"] = q3c(real, extracted, attribution)
    block["elapsed_seconds"] = round(time.time() - t0, 1)
    print(f"  [q3] seed {seed} done in {block['elapsed_seconds']}s")
    return block


def five_seed_summary(blocks):
    def collect(fn):
        vals = [fn(b) for b in blocks]
        a = np.array(vals, dtype=float)
        return {"per_seed": [float(v) for v in vals],
                "mean": float(a.mean()), "sd_pop": float(a.std(ddof=0))}

    def stage(b, name, key):
        return next(s[key] for s in b["Q3_C_population_attribution"]["stage_class_stats"]
                    if s["stage"] == name)

    dec = lambda b, k: b["Q3_D_strict_purity_failure_decomposition"]["decomposition"][k]["pct_of_poison"]
    return {
        "seeds": [b["seed"] for b in blocks],
        "raw_repeated_member_fraction": collect(lambda b: stage(b, "raw_payload", "repeated_member_fraction")),
        "raw_redundancy_fraction": collect(lambda b: stage(b, "raw_payload", "redundancy_fraction")),
        "binary_repeated_member_fraction": collect(lambda b: stage(b, "binary_mask", "repeated_member_fraction")),
        "binary_redundancy_fraction": collect(lambda b: stage(b, "binary_mask", "redundancy_fraction")),
        "final_repeated_member_fraction": collect(
            lambda b: b["Q3_A_definition_and_reproduction"]["final_vector_class_stats"]["repeated_member_fraction"]),
        "final_redundancy_fraction": collect(
            lambda b: b["Q3_A_definition_and_reproduction"]["final_vector_class_stats"]["redundancy_fraction"]),
        "largest_final_class_size": collect(
            lambda b: b["Q3_A_definition_and_reproduction"]["largest_final_class"]["size"]),
        "largest_final_class_share": collect(
            lambda b: b["Q3_A_definition_and_reproduction"]["largest_final_class"]["share_of_all_rows"]),
        "mixed_final_class_poison_obstruction": collect(
            lambda b: b["Q3_D_strict_purity_failure_decomposition"]["coordinate_class_obstruction"]["obstruction_fraction"]),
        "poison_unclustered_fraction": collect(
            lambda b: b["Q3_D_strict_purity_failure_decomposition"]["comparisons"]["poison_unclustered_fraction"]),
        "clean_unclustered_fraction": collect(
            lambda b: b["Q3_D_strict_purity_failure_decomposition"]["comparisons"]["clean_unclustered_fraction"]),
        "exact_purity_capture_pct": collect(
            lambda b: b["Q3_D_strict_purity_failure_decomposition"]["comparisons"]["observed_exact_purity_capture_pct"]),
        "failure_decomposition_pct": {
            k: collect(lambda b, k=k: dec(b, k))
            for k in blocks[0]["Q3_D_strict_purity_failure_decomposition"]["decomposition"]
        },
        "raw_first_merger_share": collect(
            lambda b: next(r["share_of_final_repeated_member_mass"]
                           for r in b["Q3_C_population_attribution"]
                           ["earliest_merger_attribution"]["rows"]
                           if r["stage"] == "raw_payload")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(CONFIRMATION_SEEDS))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    violations = label_free_signature_violations()
    if violations:
        raise AssertionError(f"stage-signature functions accept labels: {violations}")

    t0 = time.time()
    blocks = [audit_seed(s, full=True) for s in args.seeds]

    payload = {
        "phase": "Q3",
        "description": (
            "Collision-provenance audit: the earliest pipeline stage at which "
            "distinct UNSW payload rows become indistinguishable, and how much "
            "of the strict-100%-purity capture failure that explains."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_block(),
        "definitions": {
            "repeated_member_fraction": "rows in a class of size >= 2, / all rows",
            "redundancy_fraction": "(n_rows - n_unique_classes) / n_rows",
            "coordinate_class_obstruction":
                "poison rows sharing an exact final 60-vector with >= 1 clean row, / all poison rows",
            "earliest_merger_stage":
                "the first stage in the chain at which a row's exact equivalence "
                "class stops being a singleton; well-defined and mutually "
                "exclusive because each stage is a deterministic function of the "
                "previous one (monotonicity asserted, violations must be 0)",
            "diagram_canonicalisation":
                "valid points are those with death > birth; sorted by "
                "(homology_dimension, birth, death); giotto-tda diagonal padding "
                "excluded; array point order carries no information",
        },
        "scope": {
            "pipeline": "legacy single-threshold 0.4, 30x50 raster, 60 features",
            "frame": "R60 / malicious_random_attack, n_swaps=60, poison_rate=0.10",
            "clustering": OPTICS_PARAMS,
            "not_done": [
                "no row filtering", "no deduplication", "no metadata added",
                "no binarization change", "no filtration change",
                "no OPTICS tuning", "no Phase Q attack families",
            ],
        },
        "chain_stages": list(CHAIN_STAGES),
        "seeds": {str(b["seed"]): b for b in blocks},
        "five_seed_summary": five_seed_summary(blocks) if len(blocks) > 1 else None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(args.out, payload)
    print(f"\nTotal wall-clock: {payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
