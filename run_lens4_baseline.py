"""
Lens 4 Phase 3 — single-pass Monkam-style baseline runner.

Mirrors run_baseline.py's TDA-extraction + clustering step (no raw-byte
diagnostic pass, no iteration) against the attribution-ladder attack
variants built in adversarial_attack.py, so the resulting capture rate can
be decomposed into perturbation magnitude, target selection, and
adversarial-guidance components before it is compared to Monkam's reported
range.

Does not modify run_baseline.py / run_iterative.py / run_multi_seed.py /
poison.py — those stay intact and runnable.
"""
import time
import json
import joblib
import numpy as np
from pathlib import Path

from data_loader import load_unsw
from tda_pipeline import extract_tda_features
from clustering import run_all_clustering, classify_clusters
from results_io import convert_for_json
from adversarial_attack import (
    random_swap_attack, malicious_random_attack, chale_ga_attack, train_surrogates,
)

RESULTS_DIR = Path(r"C:\TDA\results")
MODELS_DIR = Path(r"C:\TDA\models")
SEEDS = [42, 123, 456, 789, 1024]
MAX_SAMPLES = 5000
POISON_RATE = 0.10


def run_single_pass_baseline(X_combined, is_poisoned):
    """One TDA extraction, one clustering pass per algorithm. No iteration."""
    t0 = time.time()
    X_tda, _ = extract_tda_features(X_combined)
    tda_time = time.time() - t0

    t0 = time.time()
    results = run_all_clustering(X_tda)
    cluster_time = time.time() - t0

    per_algo = {}
    for algo_name, labels in results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned)

        n_fp = sum(c["n_poisoned"] for c in cluster_info if c["color"] == "Green")
        n_green_total = sum(c["size"] for c in cluster_info if c["color"] == "Green")
        sanitized_purity = (1 - n_fp / n_green_total) * 100 if n_green_total > 0 else float("nan")

        n_red_total = sum(c["size"] for c in cluster_info if c["color"] == "Red")
        n_red_true = sum(c["n_poisoned"] for c in cluster_info if c["color"] == "Red")
        poisoned_pool_precision = (n_red_true / n_red_total * 100) if n_red_total > 0 else float("nan")

        per_algo[algo_name] = {
            "red_poison_capture_pct": summary["red_poison_capture_pct"],
            "green_pct": summary["green_pct"],
            "red_pct": summary["red_pct"],
            "n_clusters": summary["n_clusters"],
            "colors": summary["colors"],
            "sanitized_purity": sanitized_purity,
            "poisoned_pool_precision": poisoned_pool_precision,
        }

    return {"tda_time_s": tda_time, "cluster_time_s": cluster_time, "per_algo": per_algo}


def load_surrogates():
    mlp = joblib.load(MODELS_DIR / "surrogate_mlp.joblib")
    rf = joblib.load(MODELS_DIR / "surrogate_rf.joblib")
    scale_fn = lambda Xraw: Xraw.astype(np.float64) / 255.0
    return mlp, rf, scale_fn


def generate_variant(variant, X, y, seed, mlp, rf, scale_fn, verbose=True):
    """Returns (X_combined, y_combined, is_poisoned, gen_time_s, validity_pct)."""
    t0 = time.time()
    if variant == "L":
        Xc, yc, ip = random_swap_attack(X, y, poison_rate=POISON_RATE, random_state=seed)
        validity_pct = float("nan")  # not logged by the legacy attack; range/multiset preserved by construction
    elif variant == "R60":
        Xc, yc, ip, log = malicious_random_attack(X, y, poison_rate=POISON_RATE, random_state=seed, n_swaps=60)
        validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
    elif variant == "G60-MLP":
        Xc, yc, ip, log = chale_ga_attack(X, y, mlp, scale_fn, poison_rate=POISON_RATE, random_state=seed,
                                           n_swaps=60, population_size=50, n_generations=100,
                                           early_stop_benign_proba=0.5, verbose=verbose)
        validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
    elif variant == "G60-RF":
        Xc, yc, ip, log = chale_ga_attack(X, y, rf, scale_fn, poison_rate=POISON_RATE, random_state=seed,
                                           n_swaps=60, population_size=50, n_generations=100,
                                           early_stop_benign_proba=0.5, verbose=verbose)
        validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    gen_time = time.time() - t0
    return Xc, yc, ip, gen_time, validity_pct


def subsample_for_seed(X_full, y_full, seed):
    """Matches run_multi_seed.py's per-seed subsampling convention exactly —
    data_loader.load_unsw()'s own subsample is hardcoded to random_state=42,
    so per-seed variation has to happen here, not by re-calling the loader."""
    rng = np.random.RandomState(seed)
    if len(X_full) > MAX_SAMPLES:
        idx = rng.choice(len(X_full), size=MAX_SAMPLES, replace=False)
        return X_full[idx], y_full[idx]
    return X_full.copy(), y_full.copy()


def main():
    mlp, rf, scale_fn = load_surrogates()

    all_results = {"seed42_ladder": {}, "multi_seed": {}}

    # Load the full dataset once; every seed subsamples from this in memory
    # (matches run_multi_seed.py's convention — avoids 5x CSV re-reads).
    print("Loading full UNSW-NB15 dataset (once, for all seeds)...")
    X_all, y_all = load_unsw(max_samples=None)

    # --- Step 1+2+3: seed-42 attribution ladder (L, R60, G60-MLP, G60-RF) ---
    print("=" * 70)
    print("SEED 42 ATTRIBUTION LADDER: L, R60, G60-MLP, G60-RF")
    print("=" * 70)
    X_full, y_full = subsample_for_seed(X_all, y_all, 42)

    for variant in ["L", "R60", "G60-MLP", "G60-RF"]:
        print(f"\n--- Variant {variant} (seed 42) ---")
        Xc, yc, ip, gen_time, validity_pct = generate_variant(variant, X_full, y_full, 42, mlp, rf, scale_fn)
        print(f"  Attack generation: {gen_time:.1f}s, validity: {validity_pct:.1f}%, n_poison={ip.sum()}")
        baseline = run_single_pass_baseline(Xc, ip)
        print(f"  TDA: {baseline['tda_time_s']:.1f}s, clustering: {baseline['cluster_time_s']:.1f}s")
        for algo, m in baseline["per_algo"].items():
            print(f"    {algo:<10} capture={m['red_poison_capture_pct']:.2f}%  "
                  f"purity={m['sanitized_purity']:.1f}%  precision={m['poisoned_pool_precision']:.1f}%")
        all_results["seed42_ladder"][variant] = {
            "gen_time_s": gen_time, "validity_pct": validity_pct, "n_poison": int(ip.sum()),
            **baseline,
        }
        with open(RESULTS_DIR / "lens4_baseline_seed42_ladder.json", "w") as f:
            json.dump(all_results["seed42_ladder"], f, indent=2, default=convert_for_json)

    # --- Step 5: multi-seed confirmation for R60 and G60-MLP ---
    print("\n" + "=" * 70)
    print("MULTI-SEED CONFIRMATION: R60 and G60-MLP across all 5 seeds")
    print("=" * 70)
    for variant in ["R60", "G60-MLP"]:
        all_results["multi_seed"][variant] = {}
        for seed in SEEDS:
            if seed == 42:
                # already computed above; reuse
                all_results["multi_seed"][variant][seed] = all_results["seed42_ladder"][variant]["per_algo"]
                continue
            print(f"\n--- Variant {variant}, seed {seed} ---")
            X_s, y_s = subsample_for_seed(X_all, y_all, seed)
            Xc, yc, ip, gen_time, validity_pct = generate_variant(variant, X_s, y_s, seed, mlp, rf, scale_fn)
            print(f"  Attack generation: {gen_time:.1f}s, validity: {validity_pct:.1f}%")
            baseline = run_single_pass_baseline(Xc, ip)
            for algo, m in baseline["per_algo"].items():
                print(f"    {algo:<10} capture={m['red_poison_capture_pct']:.2f}%")
            all_results["multi_seed"][variant][seed] = baseline["per_algo"]
            with open(RESULTS_DIR / "lens4_baseline_multiseed.json", "w") as f:
                json.dump(all_results["multi_seed"], f, indent=2, default=convert_for_json)

    with open(RESULTS_DIR / "lens4_baseline_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print("\n" + "=" * 70)
    print("LENS 4 PHASE 3 RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
