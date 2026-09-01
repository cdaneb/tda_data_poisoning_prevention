"""Phase Q D1-D4 loss-localization diagnostics.

For genuinely altered, support-restricted attacks, measure where information is
lost along raw bytes -> binary images -> per-filtration cubical diagrams ->
summary vectors.  Clean/perturbed observations are always fitted together.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from gtda.diagrams import PairwiseDistance
from gtda.images import Binarizer

from programs.data_loader import load_unsw
from programs.phase_m_env import env_block
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs.phase_q_pipeline import (
    CONTROL_THRESHOLD,
    THRESHOLD_STACK,
    extract_multithreshold_features,
    extract_unscaled_diagrams,
)
from programs.paths import RESULTS_DIR
from programs.results_io import convert_for_json
from programs.run_test_b_capture import subsample_for_seed
from programs.tda_pipeline import extract_tda_features, reshape_for_tda


OUTPUT_NAME = "phase_q_d1_d4_diagnostics.json"


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_zero": float(np.mean(values == 0)),
    }


def _displacement_summary(clean, perturbed, reference_partner):
    attack = np.linalg.norm(perturbed - clean, axis=1)
    reference = np.linalg.norm(reference_partner - clean, axis=1)
    q95 = float(np.quantile(reference, 0.95))
    return {
        "attack": _summary(attack),
        "clean_clean_reference": _summary(reference),
        "fraction_attack_above_clean_q95": float(np.mean(attack > q95)),
        "median_ratio_attack_to_clean": (
            float(np.median(attack) / np.median(reference))
            if np.median(reference) > 0 else None
        ),
    }


def _diagram_distance(a, b):
    pair = np.stack([a, b])
    distances = PairwiseDistance(
        metric="bottleneck", order=np.inf, n_jobs=1
    ).fit_transform(pair)
    return float(distances[0, 1])


def _diagram_summary(diagrams, n):
    clean = diagrams[:n]
    perturbed = diagrams[n:]
    reference = np.roll(clean, -1, axis=0)
    attack_dist = np.array([_diagram_distance(a, b) for a, b in zip(clean, perturbed)])
    clean_dist = np.array([_diagram_distance(a, b) for a, b in zip(clean, reference)])
    q95 = float(np.quantile(clean_dist, 0.95))
    exact = np.array([np.array_equal(a, b) for a, b in zip(clean, perturbed)])
    return {
        "exact_diagram_identity_fraction": float(exact.mean()),
        "attack_bottleneck": _summary(attack_dist),
        "clean_clean_bottleneck": _summary(clean_dist),
        "fraction_attack_above_clean_q95": float(np.mean(attack_dist > q95)),
    }


def _paired_data(fn, kwargs, X, y, n_diag, seed):
    rate = n_diag / len(X)
    Xc, _, _, log = fn(X, y, poison_rate=rate, random_state=seed, **kwargs)
    targets = np.array([entry["target_index"] for entry in log])
    clean = X[targets]
    perturbed = Xc[len(X):]
    changed = np.count_nonzero(clean != perturbed, axis=1)
    if np.any(changed == 0):
        raise AssertionError("Phase Q generator emitted a raw no-op")
    return clean, perturbed, log


def run_family(name, fn, kwargs, X, y, n_diag, seed, include_stack):
    clean, perturbed, log = _paired_data(fn, kwargs, X, y, n_diag, seed)
    both = np.vstack([clean, perturbed])
    n = len(clean)
    images = reshape_for_tda(both)

    per_threshold_identity = {}
    stack_equal = np.ones(n, dtype=bool)
    for threshold in THRESHOLD_STACK:
        binary = Binarizer(threshold=threshold, n_jobs=-1).fit_transform(images)
        equal = np.all(binary[:n].reshape(n, -1) == binary[n:].reshape(n, -1), axis=1)
        per_threshold_identity[str(threshold)] = float(equal.mean())
        stack_equal &= equal

    diagrams, _ = extract_unscaled_diagrams(both, threshold=CONTROL_THRESHOLD)
    diagram_results = {
        filtration: _diagram_summary(values, n)
        for filtration, values in diagrams.items()
    }

    features, _ = extract_tda_features(both, threshold=CONTROL_THRESHOLD)
    clean_features, perturbed_features = features[:n], features[n:]
    baseline_displacement = _displacement_summary(
        clean_features, perturbed_features, np.roll(clean_features, -1, axis=0)
    )
    baseline_displacement["exact_feature_identity_fraction"] = float(np.mean(np.all(
        clean_features == perturbed_features, axis=1
    )))

    result = {
        "n_pairs": n,
        "positions_changed": _summary([entry["positions_changed"] for entry in log]),
        "binary_identity": {
            "per_threshold": per_threshold_identity,
            "whole_stack_identity_fraction": float(stack_equal.mean()),
        },
        "control_threshold_diagrams": diagram_results,
        "control_60_vector": baseline_displacement,
    }

    if include_stack:
        stacked, blocks, _ = extract_multithreshold_features(both, return_blocks=True)
        stack_clean, stack_perturbed = stacked[:n], stacked[n:]
        stack_result = _displacement_summary(
            stack_clean, stack_perturbed, np.roll(stack_clean, -1, axis=0)
        )
        stack_result["exact_feature_identity_fraction"] = float(np.mean(np.all(
            stack_clean == stack_perturbed, axis=1
        )))
        stack_result["n_features"] = int(stacked.shape[1])
        np.testing.assert_array_equal(blocks[CONTROL_THRESHOLD], features)
        result["stack_540_vector"] = stack_result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-diag", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=tuple(SUPPORTED_FAMILIES) + ("all",), default="all")
    parser.add_argument("--include-stack", action="store_true")
    parser.add_argument("--output", default=OUTPUT_NAME)
    args = parser.parse_args()

    X_full, y_full = load_unsw(max_samples=None)
    X, y = subsample_for_seed(X_full, y_full, args.seed)
    selected = SUPPORTED_FAMILIES.items() if args.family == "all" else (
        (args.family, SUPPORTED_FAMILIES[args.family]),
    )
    results = {
        "phase": "Q-D1-D4",
        "seed": args.seed,
        "n_diag": args.n_diag,
        "control_threshold": CONTROL_THRESHOLD,
        "threshold_stack": THRESHOLD_STACK,
        "stack_global_scale": float(1.0 / np.sqrt(len(THRESHOLD_STACK))),
        "include_stack": args.include_stack,
        "clean_clean_reference": "cyclic next-pair matching among the same clean target rows",
        "families": {},
        "env": env_block(),
    }
    for name, (fn, kwargs) in selected:
        print(f"\n=== {name} ===")
        results["families"][name] = run_family(
            name, fn, kwargs, X, y, args.n_diag, args.seed, args.include_stack
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / args.output
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, default=convert_for_json)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
