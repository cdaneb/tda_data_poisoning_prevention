"""Read-only structural validator and concise summarizer for Phase Q4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "results" / "phase_q4_dedup_mechanism.json"
EXPECTED_SEEDS = [42, 123, 456, 789, 1024]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT))
    args = ap.parse_args()
    with open(args.json) as fh:
        doc = json.load(fh)

    failures = []
    def check(ok, message):
        if not ok:
            failures.append(message)

    seeds = sorted(int(k) for k in doc["seeds"])
    check(seeds == EXPECTED_SEEDS, f"seed keys {seeds}")
    check(doc["preregistration"]["recorded_before_q4_results"],
          "preregistration flag is false")
    check(doc.get("requested_seeds") == EXPECTED_SEEDS,
          f"requested seeds {doc.get('requested_seeds')}")
    check(doc.get("complete") is True, "artifact is not marked complete")

    for key in map(str, seeds):
        b = doc["seeds"][key]
        s = b["standing_probe"]
        d = b["deduplicated_realization"]
        st = b["deduplicated_clean_raw_class_stats"]
        check(s["replay_matches_q3_input"], f"seed {key}: control input mismatch")
        check(s["replay_matches_q3_poison_mask"], f"seed {key}: poison mask mismatch")
        check(d["deduplication"]["n_before"] == 5000,
              f"seed {key}: pre-dedup frame is not 5000")
        check(d["n_clean"] == d["deduplication"]["n_after"],
              f"seed {key}: post-dedup count mismatch")
        check(d["n_poison"] == int(0.10 * d["n_clean"]),
              f"seed {key}: poison rate mismatch")
        check(st["n_repeated_member_rows"] == 0,
              f"seed {key}: repeated clean payload remains")
        check(b["equality_check"]["instrumented_equals_production_bitwise"],
              f"seed {key}: instrumented pipeline mismatch")
        q3d = b["deduplicated_Q3_D_failure_decomposition"]
        check(q3d["categories_sum_to_all_poison"],
              f"seed {key}: failure categories do not sum")
        check(q3d["residual_is_empty"], f"seed {key}: residual is nonempty")
        for metric, delta in b["deduplicated_minus_control"].items():
            expected = b["deduplicated_metrics"][metric] - b["control_metrics_from_q3"][metric]
            check(abs(delta - expected) < 1e-12,
                  f"seed {key}/{metric}: delta mismatch")

    summary = doc["five_seed_summary"]
    def validate_stats(obj, path=""):
        if isinstance(obj, dict):
            if {"per_seed", "mean", "sd_pop"} <= set(obj):
                a = np.asarray(obj["per_seed"], dtype=float)
                check(len(a) == 5, f"{path}: not five values")
                check(abs(obj["mean"] - a.mean()) < 1e-12, f"{path}: mean mismatch")
                check(abs(obj["sd_pop"] - a.std(ddof=0)) < 1e-12,
                      f"{path}: SD is not population SD")
            else:
                for k, v in obj.items():
                    validate_stats(v, f"{path}/{k}")
    validate_stats(summary)

    def validate_finite(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                validate_finite(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                validate_finite(v, f"{path}[{i}]")
        elif isinstance(obj, float):
            check(np.isfinite(obj), f"{path}: non-finite value")
    validate_finite(doc)

    print("\nPhase Q4: exact-payload clean-frame deduplication")
    print("=" * 78)
    print(f"{'seed':>6}{'clean':>8}{'removed':>9}{'raw noop':>11}"
          f"{'raw poison':>12}{'obstruction':>13}{'capture %':>11}")
    for key in map(str, seeds):
        b = doc["seeds"][key]
        d = b["deduplicated_realization"]
        m = b["deduplicated_metrics"]
        print(f"{key:>6}{d['n_clean']:>8}{d['deduplication']['n_removed']:>9}"
              f"{d['attack_diagnostics']['raw_noop_fraction']:>11.4f}"
              f"{m['raw_first_merger_poison_rate']:>12.4f}"
              f"{m['coordinate_class_obstruction_fraction']:>13.4f}"
              f"{m['exact_purity_capture_pct']:>11.2f}")

    print("\nFive-seed control -> deduplicated (mean +/- population SD)")
    wanted = [
        "raw_first_merger_poison_rate",
        "raw_first_merger_share_of_collision_mass",
        "coordinate_class_obstruction_fraction",
        "poison_unclustered_fraction",
        "clean_unclustered_fraction",
        "exact_purity_capture_pct",
    ]
    for metric in wanted:
        x = summary["metrics"][metric]
        print(f"  {metric:<44}"
              f"{x['control']['mean']:.4f} +/- {x['control']['sd_pop']:.4f} -> "
              f"{x['deduplicated']['mean']:.4f} +/- {x['deduplicated']['sd_pop']:.4f} "
              f"(delta {x['delta']['mean']:+.4f} +/- {x['delta']['sd_pop']:.4f})")

    print("\nResult")
    print("=" * 78)
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        sys.exit(1)
    print("All structural checks passed.")


if __name__ == "__main__":
    main()
