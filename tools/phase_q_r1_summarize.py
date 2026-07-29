"""Read-only summarizer for the Phase Q R1 artifact.

Prints per-family, five-seed summaries (population SD, ddof=0) of the frozen
metrics and answers the preregistered interpretation questions.  Performs no
tuning, resampling, or selection — pure aggregation of the committed JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARTIFACT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "results/phase_q_r1_multithreshold_capture.json"
)
FAMILIES = ("transpositions", "block_reversal", "block_swap", "cyclic_shift")
SEEDS = ("42", "123", "456", "789", "1024")
PURITY_LABELS = ("1.0", ">0.95", ">0.90", ">0.80", ">0.50")


def ms(values):
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.4f} +/- {a.std(ddof=0):.4f}"


def validate(doc):
    problems = []
    runs = doc.get("runs", {})
    for fam in FAMILIES:
        if fam not in runs:
            problems.append(f"missing family {fam}")
            continue
        for seed in SEEDS:
            if seed not in runs[fam]:
                problems.append(f"missing {fam}/{seed}")
                continue
            r = runs[fam][seed]
            if r.get("n_clean") != 5000:
                problems.append(f"{fam}/{seed} n_clean={r.get('n_clean')}")
            if r.get("n_poison") != 500:
                problems.append(f"{fam}/{seed} n_poison={r.get('n_poison')}")
            if r.get("raw_noop_count") != 0:
                problems.append(f"{fam}/{seed} raw_noop_count={r.get('raw_noop_count')}")
            if r["control"]["n_features"] != 60:
                problems.append(f"{fam}/{seed} control n_features={r['control']['n_features']}")
            if r["repair"]["n_features"] != 540:
                problems.append(f"{fam}/{seed} repair n_features={r['repair']['n_features']}")
            if len(r["control"]["removal_curve"]) != 5:
                problems.append(f"{fam}/{seed} control curve len {len(r['control']['removal_curve'])}")
            if len(r["repair"]["removal_curve"]) != 5:
                problems.append(f"{fam}/{seed} repair curve len {len(r['repair']['removal_curve'])}")
            if len(r["matched_clean_cost"]) != 5:
                problems.append(f"{fam}/{seed} matched len {len(r['matched_clean_cost'])}")
            if "env" not in r:
                problems.append(f"{fam}/{seed} missing env block")
            # NaN / inf sweep over the interpretation-relevant numbers
            for arm in ("control", "repair"):
                for pt in r[arm]["removal_curve"]:
                    for k, v in pt.items():
                        if isinstance(v, float) and not np.isfinite(v):
                            problems.append(f"{fam}/{seed} {arm} {k}={v}")
    return problems


def main():
    doc = json.loads(ARTIFACT.read_text())
    problems = validate(doc)
    print(f"=== VALIDATION: {'PASS' if not problems else 'FAIL'} ===")
    for p in problems:
        print("  -", p)
    if problems:
        return

    runs = doc["runs"]
    print("\n=== Per-family five-seed summary (population SD, ddof=0) ===")
    for fam in FAMILIES:
        cells = [runs[fam][s] for s in SEEDS]
        # exact-purity (index 0) removal rates
        ctl_rm = [c["control"]["removal_curve"][0]["poison_removal_rate"] for c in cells]
        rep_rm = [c["repair"]["removal_curve"][0]["poison_removal_rate"] for c in cells]
        ctl_prec = [c["control"]["removal_curve"][0]["removal_precision"] or 0.0 for c in cells]
        rep_prec = [c["repair"]["removal_curve"][0]["removal_precision"] or 0.0 for c in cells]
        ctl_pun = [c["control"]["removal_curve"][0]["poison_unclustered_fraction"] for c in cells]
        rep_pun = [c["repair"]["removal_curve"][0]["poison_unclustered_fraction"] for c in cells]
        ctl_cun = [c["control"]["removal_curve"][0]["clean_unclustered_fraction"] for c in cells]
        rep_cun = [c["repair"]["removal_curve"][0]["clean_unclustered_fraction"] for c in cells]
        ctl_nc = [c["control"]["n_clusters"] for c in cells]
        rep_nc = [c["repair"]["n_clusters"] for c in cells]
        ctl_dup = [c["control"]["duplicate_with_any_clean_fraction"] for c in cells]
        rep_dup = [c["repair"]["duplicate_with_any_clean_fraction"] for c in cells]
        # matched-clean-cost delta at exact purity budget (index 0)
        mcc0 = [c["matched_clean_cost"][0]["poison_removal_rate_delta"] for c in cells]
        mcc0 = [x for x in mcc0 if x is not None]

        print(f"\n--- {fam} ---")
        print(f"  exact-purity poison removal   control {ms(ctl_rm)} | repair {ms(rep_rm)}")
        print(f"  per-seed control removal       {['%.4f' % x for x in ctl_rm]}")
        print(f"  per-seed repair  removal       {['%.4f' % x for x in rep_rm]}")
        print(f"  removal precision (exact)      control {ms(ctl_prec)} | repair {ms(rep_prec)}")
        print(f"  poison unclustered frac        control {ms(ctl_pun)} | repair {ms(rep_pun)}")
        print(f"  clean  unclustered frac        control {ms(ctl_cun)} | repair {ms(rep_cun)}")
        print(f"  n_clusters                     control {ms(ctl_nc)} | repair {ms(rep_nc)}")
        print(f"  exact-dup-with-clean frac      control {ms(ctl_dup)} | repair {ms(rep_dup)}")
        if mcc0:
            print(f"  matched-cost removal delta(P1) {ms(mcc0)}  per-seed {['%.4f' % x for x in mcc0]}")
        else:
            print(f"  matched-cost removal delta(P1) all None (no feasible repair point)")

        # clean false-removal at each frozen purity point (both arms)
        print("  clean false-removal by purity point:")
        for i, lab in enumerate(PURITY_LABELS):
            cfr_ctl = [c["control"]["removal_curve"][i]["clean_false_removal_rate"] for c in cells]
            cfr_rep = [c["repair"]["removal_curve"][i]["clean_false_removal_rate"] for c in cells]
            print(f"    {lab:>6}: control {ms(cfr_ctl)} | repair {ms(cfr_rep)}")


if __name__ == "__main__":
    main()
