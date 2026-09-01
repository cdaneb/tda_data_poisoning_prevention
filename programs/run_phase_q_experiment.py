"""Phase Q R1: controlled single-threshold vs multithreshold comparison.

The 0.4 control is extracted as one block of the nine-threshold representation,
so both arms use the same poisoned batch and the same implementation.  Only
the representation differs: 60 features at threshold 0.4 versus the
concatenated 540-feature stack.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.cluster import OPTICS

from programs.data_loader import load_unsw
from programs.phase_m_env import env_block
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs.phase_q_metrics import matched_clean_cost, removal_curve
from programs.phase_q_pipeline import CONTROL_THRESHOLD, THRESHOLD_STACK, extract_multithreshold_features
from programs.paths import RESULTS_DIR
from programs.results_io import convert_for_json
from programs.run_test_b_capture import SEEDS, subsample_for_seed


OUTPUT_NAME = "phase_q_r1_multithreshold_capture.json"
MAX_SAMPLES = 5000
POISON_RATE = 0.10
OPTICS_PARAMS = {"min_samples": 5, "max_eps": 2.0}

PREREGISTRATION = {
    "recorded_before_results": True,
    "attack_substrate": (
        "same support-restricted, guaranteed-nontrivial attack realization for both arms"
    ),
    "control": {"thresholds": [0.4], "n_features": 60},
    "repair": {"thresholds": list(THRESHOLD_STACK), "n_features": 540,
               "global_scale": "1/sqrt(9), fixed dimension compensation"},
    "held_fixed": [
        "5000-row seed-specific UNSW sample",
        "500 appended poison rows",
        "30x50 raster",
        "two Height and three Radial filtrations",
        "CubicalPersistence and Scaler",
        "PersistenceEntropy plus five Amplitude summaries",
        "OPTICS(min_samples=5, max_eps=2.0)",
    ],
    "distance_scale_control": (
        "the concatenated stack is divided by sqrt(9), so nine identical blocks would preserve "
        "the control Euclidean distance scale under fixed OPTICS parameters"
    ),
    "primary_evaluation": (
        "poison removal at matched clean false-removal cost; exact-purity capture is also reported"
    ),
    "prediction": (
        "the stack reduces binary/feature identity and may improve poison removal, but an improvement "
        "counts only when it does not buy recall by increasing clean removal"
    ),
    "oracle_warning": (
        "cluster poison purity uses ground truth for retrospective evaluation and is not a deployable "
        "cluster-labeling rule"
    ),
}


def _fit_optics(X):
    return OPTICS(**OPTICS_PARAMS, n_jobs=-1).fit_predict(X)


def _duplicate_fraction(features, is_poisoned):
    n_clean = int((~is_poisoned).sum())
    clean = np.ascontiguousarray(features[:n_clean], dtype=np.float64)
    poison = np.ascontiguousarray(features[n_clean:], dtype=np.float64)
    clean_keys = {row.tobytes() for row in clean}
    return float(np.mean([row.tobytes() in clean_keys for row in poison]))


def _cluster_record(features, is_poisoned):
    start = time.time()
    labels = _fit_optics(features)
    elapsed = time.time() - start
    non_noise = set(labels) - {-1}
    return {
        "n_features": int(features.shape[1]),
        "cluster_time_s": elapsed,
        "n_clusters": int(len(non_noise)),
        "n_unclustered": int(np.count_nonzero(labels == -1)),
        "duplicate_with_any_clean_fraction": _duplicate_fraction(features, is_poisoned),
        "removal_curve": removal_curve(labels, is_poisoned),
    }


def run_one(X_full, y_full, family, seed):
    X, y = subsample_for_seed(X_full, y_full, seed)
    fn, kwargs = SUPPORTED_FAMILIES[family]
    Xc, _, is_poisoned, log = fn(
        X, y, poison_rate=POISON_RATE, random_state=seed, **kwargs
    )
    if any(entry["raw_noop"] for entry in log):
        raise AssertionError("Phase Q attack substrate emitted a raw no-op")

    start = time.time()
    stacked, blocks, _ = extract_multithreshold_features(Xc, return_blocks=True)
    extraction_time = time.time() - start
    control_features = blocks[CONTROL_THRESHOLD]

    control = _cluster_record(control_features, is_poisoned)
    repair = _cluster_record(stacked, is_poisoned)
    return {
        "seed": int(seed),
        "family": family,
        "n_clean": int((~is_poisoned).sum()),
        "n_poison": int(is_poisoned.sum()),
        "raw_noop_count": 0,
        "positions_changed": {
            "mean": float(np.mean([entry["positions_changed"] for entry in log])),
            "median": float(np.median([entry["positions_changed"] for entry in log])),
            "min": int(min(entry["positions_changed"] for entry in log)),
            "max": int(max(entry["positions_changed"] for entry in log)),
        },
        "feature_extraction_time_s": extraction_time,
        "control": control,
        "repair": repair,
        "matched_clean_cost": matched_clean_cost(
            control["removal_curve"], repair["removal_curve"]
        ),
        "env": env_block(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(SUPPORTED_FAMILIES) + ("all",),
                        default="block_reversal")
    parser.add_argument("--seed", choices=[str(s) for s in SEEDS] + ["all"], default="42")
    parser.add_argument("--output", default=OUTPUT_NAME)
    args = parser.parse_args()

    families = tuple(SUPPORTED_FAMILIES) if args.family == "all" else (args.family,)
    seeds = tuple(SEEDS) if args.seed == "all" else (int(args.seed),)
    X_full, y_full = load_unsw(max_samples=None)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / args.output
    if out.exists():
        with open(out, "r", encoding="utf-8") as stream:
            output = json.load(stream)
        if output.get("preregistration") != PREREGISTRATION:
            raise ValueError("Existing artifact has a different preregistration; refusing to merge")
    else:
        output = {
            "phase": "Q-R1",
            "preregistration": PREREGISTRATION,
            "runs": {},
            "artifact_created_env": env_block(),
        }
    for family in families:
        output["runs"].setdefault(family, {})
        for seed in seeds:
            print(f"\n=== {family}, seed {seed} ===")
            if str(seed) in output["runs"][family]:
                print("  Existing run found; skipping")
                continue
            output["runs"][family][str(seed)] = run_one(X_full, y_full, family, seed)
            with open(out, "w", encoding="utf-8") as stream:
                json.dump(output, stream, indent=2, default=convert_for_json)
    if not out.exists():
        with open(out, "w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2, default=convert_for_json)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
