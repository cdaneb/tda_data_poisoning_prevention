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
import time
import json
import numpy as np

from data_loader import load_unsw
from tda_pipeline import extract_tda_features
from clustering import run_all_clustering, classify_clusters
from results_io import convert_for_json
from adversarial_attack import (
    malicious_random_attack, block_reversal_attack, block_swap_attack, cyclic_shift_attack,
)
from paths import RESULTS_DIR

SEEDS = [42, 123, 456, 789, 1024]
MAX_SAMPLES = 5000
POISON_RATE = 0.10
THRESHOLD = 0.4

FAMILIES = {
    "transpositions": (malicious_random_attack, {"n_swaps": 60}),
    "block_reversal": (block_reversal_attack, {"k": 120}),
    "block_swap": (block_swap_attack, {"k": 60}),
    "cyclic_shift": (cyclic_shift_attack, {}),
}

# Pre-committed reference values (recorded, OPTICS, threshold 0.4), written
# before this script runs anything.
REFERENCE_CAPTURE = {
    "transpositions": {"mean": 1.80, "std": 0.51,
                        "per_seed": {"42": 2.2, "123": 2.2, "456": 2.2, "789": 1.0, "1024": 1.4}},
    "block_reversal": {"mean": 0.00, "std": 0.00},
    "block_swap": {"mean": 0.00, "std": 0.00},
    "cyclic_shift": {"mean": 6.28, "std": 1.31},
}


def subsample_for_seed(X_full, y_full, seed):
    """Matches run_lens4_baseline.py::subsample_for_seed exactly."""
    rng = np.random.RandomState(seed)
    if len(X_full) > MAX_SAMPLES:
        idx = rng.choice(len(X_full), size=MAX_SAMPLES, replace=False)
        return X_full[idx], y_full[idx]
    return X_full.copy(), y_full.copy()


def run_single_pass(X_combined, is_poisoned, threshold=THRESHOLD):
    t0 = time.time()
    X_tda, _ = extract_tda_features(X_combined, threshold=threshold)
    tda_time = time.time() - t0

    t0 = time.time()
    results = run_all_clustering(X_tda)
    cluster_time = time.time() - t0

    per_algo = {}
    for algo_name, labels in results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned)
        per_algo[algo_name] = {
            "red_poison_capture_pct": summary["red_poison_capture_pct"],
            "colors": summary["colors"],
        }
    return {"tda_time_s": tda_time, "cluster_time_s": cluster_time, "per_algo": per_algo}


def main():
    print("Loading full UNSW-NB15 dataset (once, for all seeds)...")
    X_all, y_all = load_unsw(max_samples=None)

    out_path = RESULTS_DIR / "test_b_permutation_families.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {family: {} for family in FAMILIES}
    all_results["_reference"] = REFERENCE_CAPTURE
    all_results["_meta"] = {"seeds": SEEDS, "max_samples": MAX_SAMPLES,
                             "poison_rate": POISON_RATE, "threshold": THRESHOLD}

    t_start = time.time()
    for family, (fn, kwargs) in FAMILIES.items():
        for seed in SEEDS:
            print(f"\n--- {family}, seed {seed} ---")
            X, y = subsample_for_seed(X_all, y_all, seed)
            t0 = time.time()
            Xc, yc, ip, log = fn(X, y, poison_rate=POISON_RATE, random_state=seed, **kwargs)
            gen_time = time.time() - t0
            validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
            baseline = run_single_pass(Xc, ip)
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

    print("\n=== SUMMARY (OPTICS, population std) ===")
    print(f"{'Family':<15} {'Observed':>18}   {'Recorded':>18}   {'Per-seed'}")
    for family in FAMILIES:
        captures = [all_results[family][str(seed)]["per_algo"]["OPTICS"]["red_poison_capture_pct"]
                    for seed in SEEDS]
        mean = float(np.mean(captures))
        std = float(np.std(captures))  # population (ddof=0), matching project convention
        ref = REFERENCE_CAPTURE[family]
        print(f"{family:<15} {mean:>7.2f}% +/- {std:<6.2f}%   "
              f"{ref['mean']:>7.2f}% +/- {ref['std']:<6.2f}%   {captures}")

    all_results["_summary"] = {
        family: {
            "mean": float(np.mean([all_results[family][str(s)]["per_algo"]["OPTICS"]["red_poison_capture_pct"]
                                    for s in SEEDS])),
            "std": float(np.std([all_results[family][str(s)]["per_algo"]["OPTICS"]["red_poison_capture_pct"]
                                  for s in SEEDS])),
            "per_seed": {str(s): all_results[family][str(s)]["per_algo"]["OPTICS"]["red_poison_capture_pct"]
                         for s in SEEDS},
        }
        for family in FAMILIES
    }
    all_results["_total_wall_clock_s"] = total_time
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=convert_for_json)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
