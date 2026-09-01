"""Phase M8: re-score the persisted M7 OPTICS clusters at relaxed purity levels.

This is a read-only reanalysis of ``results/phase_m_m7_capture.json``.  It does
not rerun attacks, feature extraction, or clustering.  The M7 artifact persists
each cluster's poisoned/clean counts, which makes a purity sweep possible.

The key A20 rule is explicit in ``captured``: label -1 is an unclustered point,
not a cluster, and is never captured at any purity level.  Exact-feature
duplicates and their union with -1 bound 100%-pure capture only; the only ceiling
plotted across relaxed thresholds is the unclustered ceiling.
"""

import json
from pathlib import Path

import numpy as np

from programs.paths import RESULTS_DIR


INPUT_NAME = "phase_m_m7_capture.json"
OUTPUT_NAME = "phase_m_m8_purity_sweep.json"
SEEDS = ["42", "123", "456", "789", "1024"]
FAMILIES = ["block_reversal", "block_swap", "transpositions", "cyclic_shift", "noise"]
ALGORITHM = "OPTICS"
# 100% is exact Red; relaxed thresholds are strict, matching the project's Pink
# convention (>80%) and the addendum's wording "down to >50%".
PURITY_LEVELS = [(1.00, "100%"), (0.95, ">95%"), (0.90, ">90%"),
                 (0.80, ">80%"), (0.50, ">50%")]

PREREGISTRATION = {
    "scope": "M7 OPTICS cluster records only; no new measurement is performed.",
    "ceiling_rule": {
        "unclustered": "Applies at every purity threshold: label -1 is never captured.",
        "duplicate": "Applies only at 100% purity; it dissolves below 100%.",
        "union": "Reported only as the 100%-purity annotated point; never extended across the sweep.",
    },
    "forty_percent_interpretation": {
        "block_reversal": "decisive: unclustered ceiling leaves >40% reachable.",
        "block_swap": "decisive: unclustered ceiling leaves >40% reachable.",
        "transpositions": "nominally reachable but practically uninformative: the 40% margin lies within ceiling variation.",
        "cyclic_shift": "structurally impossible: unclustered ceiling is below 40%.",
        "noise": "structurally impossible: unclustered ceiling is below 40%.",
    },
}


def captured(cluster, threshold):
    """Whether this cluster contributes poisoned samples at a given threshold."""
    if cluster["cluster_id"] == -1:
        return False
    fraction = float(cluster["poison_fraction"])
    return fraction == 1.0 if threshold == 1.0 else fraction > threshold


def population_summary(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "population_sd": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in a]}


def main():
    in_path = RESULTS_DIR / INPUT_NAME
    out_path = RESULTS_DIR / OUTPUT_NAME
    with open(in_path, encoding="utf-8") as fh:
        source = json.load(fh)

    output = {
        "_meta": {"source": INPUT_NAME, "algorithm": ALGORITHM, "seeds": SEEDS,
                  "purity_levels": [{"threshold": t, "label": label} for t, label in PURITY_LEVELS]},
        "_preregistration": PREREGISTRATION,
        "families": {},
    }
    all_minus_one_never_captured = True

    for family in FAMILIES:
        per_threshold = {label: [] for _, label in PURITY_LEVELS}
        unclustered_ceiling = []
        union_at_100 = []
        capture_at_100_recorded = []
        seed_audit = {}

        for seed in SEEDS:
            record = source[family][seed]["per_algo"][ALGORITHM]
            clusters = record["clusters"]
            minus_one = [c for c in clusters if c["cluster_id"] == -1]
            if len(minus_one) != 1 or minus_one[0]["color"] != "Noise":
                raise AssertionError(f"{family}/{seed}: -1 is not exactly one Noise record")
            total_poison = int(record["n_poison_total_this_pass"])
            ceilings = record["ceilings"]
            unclustered_ceiling.append(100 * float(ceilings["unclustered_ceiling"]))
            union_at_100.append(100 * float(ceilings["union_ceiling"]))
            capture_at_100_recorded.append(float(record["red_poison_capture_pct"]))

            seed_values = {}
            for threshold, label in PURITY_LEVELS:
                captured_poison = sum(int(c["n_poisoned"]) for c in clusters if captured(c, threshold))
                value = 100 * captured_poison / total_poison
                seed_values[label] = value
                per_threshold[label].append(value)
                # A20 blocking check: no poisoned -1 point is ever captured.
                if captured(minus_one[0], threshold):
                    all_minus_one_never_captured = False
            if abs(seed_values["100%"] - capture_at_100_recorded[-1]) > 1e-12:
                raise AssertionError(f"{family}/{seed}: 100% rescore disagrees with M7 capture")
            seed_audit[seed] = seed_values

        family_result = {
            "capture_curve": {label: population_summary(values) for label, values in per_threshold.items()},
            "unclustered_ceiling": population_summary(unclustered_ceiling),
            "union_ceiling_at_100pct_only": population_summary(union_at_100),
            "seed_capture": seed_audit,
        }
        max_capture = max(v["mean"] for v in family_result["capture_curve"].values())
        ceiling = family_result["unclustered_ceiling"]["mean"]
        if ceiling < 40:
            verdict = "structurally impossible to reach 40% because the unclustered ceiling is below 40%"
        elif family == "transpositions":
            verdict = "nominally reachable but practically uninformative near the ceiling"
        else:
            verdict = "decisive family for the purity-criterion hypothesis"
        family_result["forty_percent_score"] = {
            "max_observed_capture": max_capture,
            "crossed_40pct": bool(max_capture >= 40),
            "interpretation": verdict,
        }
        output["families"][family] = family_result

    output["_a20_minus_one_check"] = {
        "passed": all_minus_one_never_captured,
        "statement": "Every -1 record is color Noise and contributes zero captured poison at every tested threshold.",
    }
    if not all_minus_one_never_captured:
        raise AssertionError("A20 failed: a -1 record was scored as captured")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Wrote {out_path}")
    print("A20 -1 check: PASS")
    for family, result in output["families"].items():
        curve = result["capture_curve"]
        values = ", ".join(f"{label}={curve[label]['mean']:.2f}%" for _, label in PURITY_LEVELS)
        print(f"{family:<16} {values}; {result['forty_percent_score']['interpretation']}")


if __name__ == "__main__":
    main()
