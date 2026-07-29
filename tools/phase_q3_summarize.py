"""Read-only validator and summarizer for the Phase Q3 collision audit.

Recomputes every internal consistency relation the audit claims, prints the
tables that go into the report, and exits nonzero if any check fails.  It reads
``results/phase_q3_collision_audit.json`` and writes nothing.

Usage:
    python tools/phase_q3_summarize.py
    python tools/phase_q3_summarize.py --json results/phase_q3_collision_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "results" / "phase_q3_collision_audit.json"
EXPECTED_SEEDS = [42, 123, 456, 789, 1024]

failures = []


def check(condition, message):
    if condition:
        return True
    failures.append(message)
    print(f"  FAIL: {message}")
    return False


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT))
    args = ap.parse_args()

    path = Path(args.json)
    if not path.exists():
        print(f"missing artifact: {path}")
        sys.exit(1)
    with open(path) as fh:
        doc = json.load(fh)

    seeds = sorted(int(k) for k in doc["seeds"])
    rule("Phase Q3 — structural validation")
    check(seeds == EXPECTED_SEEDS, f"seed keys {seeds} != {EXPECTED_SEEDS}")

    for key in map(str, seeds):
        b = doc["seeds"][key]
        r = b["realization"]
        check(r["n_clean"] == 5000, f"seed {key}: n_clean {r['n_clean']} != 5000")
        check(r["n_poison"] == 500, f"seed {key}: n_poison {r['n_poison']} != 500")
        check(tuple(b["equality_check"]["shape"]) == (5500, 60),
              f"seed {key}: feature shape {b['equality_check']['shape']}")
        check(b["equality_check"]["instrumented_equals_production_bitwise"],
              f"seed {key}: instrumented != production bitwise")

        for st in b["Q3_C_population_attribution"]["stage_class_stats"]:
            tag = f"seed {key}/{st['stage']}"
            check(st["n_rows"] == 5500, f"{tag}: n_rows {st['n_rows']} != 5500")
            check(st["n_redundant_rows"] == st["n_rows"] - st["n_unique_classes"],
                  f"{tag}: redundancy count inconsistent")
            check(abs(st["redundancy_fraction"]
                      - st["n_redundant_rows"] / st["n_rows"]) < 1e-12,
                  f"{tag}: redundancy_fraction inconsistent")
            check(abs(st["repeated_member_fraction"]
                      - st["n_repeated_member_rows"] / st["n_rows"]) < 1e-12,
                  f"{tag}: repeated_member_fraction inconsistent")
            check(st["repeated_member_fraction"] >= st["redundancy_fraction"],
                  f"{tag}: repeated-member below redundancy (impossible)")

        att = b["Q3_C_population_attribution"]["earliest_merger_attribution"]
        check(att["monotonicity_violations"] == 0,
              f"seed {key}: {att['monotonicity_violations']} monotonicity violations")
        check(att["attribution_sums_to_final_repeated_mass"],
              f"seed {key}: attribution does not sum")
        check(sum(r_["member_rows"] for r_ in att["rows"])
              == att["n_final_repeated_member_rows"],
              f"seed {key}: attribution row sum mismatch")
        check(abs(sum(r_["share_of_final_repeated_member_mass"]
                      for r_ in att["rows"]) - 1.0) < 1e-9,
              f"seed {key}: attribution shares do not sum to 1")
        for r_ in att["rows"]:
            check(r_["member_rows"] == r_["clean_rows"] + r_["poison_rows"],
                  f"seed {key}/{r_['stage']}: clean+poison != member rows")

        d = b["Q3_D_strict_purity_failure_decomposition"]
        check(d["categories_sum_to_all_poison"], f"seed {key}: categories do not sum")
        check(sum(v["n"] for v in d["decomposition"].values()) == d["n_poison"],
              f"seed {key}: decomposition sum != n_poison")
        check(d["residual_is_empty"], f"seed {key}: residual category non-empty")

        t = b["Q3_B_largest_class_trace"]
        a = b["Q3_A_definition_and_reproduction"]["largest_final_class"]
        check(t["size"] == a["size"] and t["signature"] == a["signature"],
              f"seed {key}: Q3-B/Q3-A largest class disagree")
        check(t["n_clean"] + t["n_poison"] == t["size"],
              f"seed {key}: largest class composition inconsistent")
        counts = [p["n_distinct_upstream_signatures"]
                  for p in t["per_stage_distinct_signatures"]]
        check(counts == sorted(counts, reverse=True),
              f"seed {key}: upstream signature counts not monotone non-increasing")
        check(sum(t["earliest_merger_stage_histogram"].values()) == t["size"],
              f"seed {key}: largest-class earliest-merger histogram does not sum")

    q = doc["seeds"]["42"]["Q3_A_definition_and_reproduction"]["q2_statistic_resolution"]
    check(q["input_hash_matches_q2"], "seed 42 input hash does not match the Q2 frame")
    check(q["poison_mask_hash_matches_q2"], "seed 42 poison mask hash mismatch")
    check(q["feature_hash_matches_q2"], "seed 42 feature hash mismatch")
    cap42 = (doc["seeds"]["42"]["Q3_D_strict_purity_failure_decomposition"]
             ["comparisons"]["observed_exact_purity_capture_pct"])
    check(abs(cap42 - 2.2000) < 1e-4, f"seed 42 capture {cap42} != 2.2000%")

    s = doc["five_seed_summary"]
    for key, val in s.items():
        if isinstance(val, dict) and "per_seed" in val:
            arr = np.array(val["per_seed"], dtype=float)
            check(len(arr) == 5, f"{key}: {len(arr)} seeds")
            check(abs(val["mean"] - arr.mean()) < 1e-9, f"{key}: mean mismatch")
            check(abs(val["sd_pop"] - arr.std(ddof=0)) < 1e-9,
                  f"{key}: sd is not population SD (ddof=0)")

    # ---------------------------------------------------------------- tables
    b42 = doc["seeds"]["42"]
    rule("Q3-A  What the Q2 '51.8%' statistic measured (seed 42)")
    print(f"  Q2 reported                    : {q['q2_reported_n_rows']}/5500 "
          f"= {q['q2_reported_value']:.4f}")
    print(f"  redundancy fraction            : {q['reproduced_n_redundant_rows']}/5500 "
          f"= {q['reproduced_redundancy_fraction']:.4f}   "
          f"MATCH={q['q2_51_8_pct_is_redundancy_fraction']}")
    print(f"  repeated-member fraction       : {q['reproduced_n_repeated_member_rows']}/5500 "
          f"= {q['reproduced_repeated_member_fraction']:.4f}   "
          f"MATCH={q['q2_51_8_pct_is_repeated_member_fraction']}")
    lc = b42["Q3_A_definition_and_reproduction"]["largest_final_class"]
    print(f"  largest final class            : {lc['size']} "
          f"({lc['share_of_all_rows']:.4f})  reproduces 1043 = {lc['reproduces_q2_1043']}")
    print(f"                                   {lc['n_clean']} clean + {lc['n_poison']} poison")

    rule("Q3-C  Stage collision statistics (seed 42)")
    print(f"  {'stage':<26}{'unique':>8}{'repClass':>10}"
          f"{'repMemFrac':>12}{'redundFrac':>12}{'largest':>9}")
    for st in b42["Q3_C_population_attribution"]["stage_class_stats"]:
        print(f"  {st['stage']:<26}{st['n_unique_classes']:>8}"
              f"{st['n_repeated_classes']:>10}{st['repeated_member_fraction']:>12.4f}"
              f"{st['redundancy_fraction']:>12.4f}{st['largest_class_size']:>9}")

    rule("Q3-C  Earliest-merger attribution (seed 42, mutually exclusive)")
    att = b42["Q3_C_population_attribution"]["earliest_merger_attribution"]
    print(f"  {'earliest stage':<26}{'classes':>9}{'rows':>7}{'clean':>7}"
          f"{'poison':>8}{'mixPois':>9}{'share':>9}")
    for r_ in att["rows"]:
        print(f"  {r_['stage']:<26}{r_['n_newly_merged_classes']:>9}"
              f"{r_['member_rows']:>7}{r_['clean_rows']:>7}{r_['poison_rows']:>8}"
              f"{r_['mixed_class_poison_rows']:>9}"
              f"{r_['share_of_final_repeated_member_mass']:>9.4f}")
    print(f"  {'TOTAL':<26}{'':>9}{att['n_final_repeated_member_rows']:>7}")

    rule("Q3-D  Strict-100%-purity failure decomposition (seed 42)")
    d = b42["Q3_D_strict_purity_failure_decomposition"]
    for k, v in d["decomposition"].items():
        print(f"  {k:<58}{v['n']:>5}{v['pct_of_poison']:>8.2f}%")
    ob = d["coordinate_class_obstruction"]
    print(f"\n  coordinate-class obstruction   : "
          f"{ob['numerator_poison_sharing_exact_vector_with_clean']}/"
          f"{ob['denominator_all_poison']} = {ob['obstruction_fraction']:.4f}")
    print(f"  captured rows also in a mixed exact class: "
          f"{ob['n_captured_rows_also_in_a_mixed_exact_vector_class']} "
          f"(obstruction binds = {ob['obstruction_binds_under_this_fit']})")

    rule("Q3-E  Five-seed confirmation (mean +/- population SD)")
    for key, val in s.items():
        if isinstance(val, dict) and "per_seed" in val:
            print(f"  {key:<44}{val['mean']:>10.4f} +/- {val['sd_pop']:<9.4f}"
                  f"  {[round(x, 4) for x in val['per_seed']]}")
    print("\n  failure decomposition, % of poison:")
    for key, val in s["failure_decomposition_pct"].items():
        print(f"    {key:<58}{val['mean']:>7.2f} +/- {val['sd_pop']:.2f}")

    rule("Result")
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  All structural checks passed.")


if __name__ == "__main__":
    main()
