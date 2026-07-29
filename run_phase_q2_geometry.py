"""Phase Q2-B: the source-printed image geometry against the operational one.

One conceptual factor moves: the raster and the radial centers, which are the
same coordinate system and therefore move together.

  legacy  (30, 50)   centers [[0,50],  [0,25],  [30,0]]     -- what we have run
  source  (1, 1500)  centers [[0,1500],[0,750], [1500,0]]   -- what page 6 and
                                                               Algorithm 1 print

Everything else is pinned: the same R60/seed-42 combined byte matrix and poison
mask, the same Binarizer threshold, the same five filtrations, the same
CubicalPersistence, Scaler, entropy and amplitude metrics, the same library
versions, and the same OPTICS configuration.

The threshold arm (0.4 vs 0.3, both on the source geometry) is a SEPARATE
one-factor comparison and is only run when asked for with --threshold-arm.  It
is never mixed into the geometry comparison.

Seed 42 alone decides nothing here.  This project has twice been burned by a
seed-42 artifact that did not survive replication (CLAUDE.md section 7), so any
geometry effect large enough to matter is re-run on all five seeds with
population SD before it is described as an effect at all.

Usage:
    python run_phase_q2_geometry.py                  # geometry arm at 0.4, seed 42
    python run_phase_q2_geometry.py --threshold-arm  # additionally 0.4 vs 0.3
    python run_phase_q2_geometry.py --threshold-arm --confirm-seeds   # + 5 seeds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import RESULTS_DIR
from tda_pipeline import extract_tda_features
from phase_q_metrics import removal_curve
from phase_q2_source_pipeline import (
    GEOMETRIES, diagram_diagnostics, extract_geometry_features, geometry_spec,
)
from tools.phase_q2_common import (
    CONFIRMATION_SEEDS, DISCOVERY_SEED, accounting_views, build_realization,
    cached_features, cluster_structure, environment_block, feature_diagnostics,
    labeled_distance_diagnostics, optics_internals, realization_provenance,
    write_json,
)

BASELINE_OPTICS = {"min_samples": 5, "max_eps": 2.0}
FIXTURE_SEED = 20260729


def fixture_probe(threshold=0.4, n=8):
    """Tiny deterministic probe, run before the full arms.

    Answers the questions that decide whether the full arm is even meaningful:
    does the pinned library accept the printed shape and centers, what feature
    dimension comes out, which homology dimensions actually carry non-trivial
    points, and are there NaNs or dead columns.  No center is clipped or
    rescaled to make anything work; a failure is preserved verbatim.
    """
    rng = np.random.RandomState(FIXTURE_SEED)
    X = rng.randint(0, 256, size=(n, 1500)).astype(np.uint8)
    out = {}
    for geometry in ("legacy", "source"):
        spec = geometry_spec(geometry)
        entry = {
            "image_shape": list(spec["image_shape"]),
            "centers": [c.tolist() for c in spec["centers"]],
            "provenance": spec["provenance"],
            "accepted_by_giotto_tda": None,
        }
        try:
            X_tda, _ = extract_geometry_features(X, geometry, threshold, verbose=False)
            variances = X_tda.var(axis=0)
            entry.update({
                "accepted_by_giotto_tda": True,
                "exception": None,
                "output_feature_count": int(X_tda.shape[1]),
                "n_nan": int(np.isnan(X_tda).sum()),
                "n_inf": int(np.isinf(X_tda).sum()),
                "n_zero_variance_features": int((variances == 0).sum()),
                "per_filtration_diagrams": diagram_diagnostics(X, geometry, threshold),
            })
        except Exception as exc:  # preserved verbatim; this arm then stops
            entry.update({
                "accepted_by_giotto_tda": False,
                "exception": f"{type(exc).__name__}: {exc}",
            })
        out[geometry] = entry
    return out


def run_arm(name, X_tda, real, optics_params=BASELINE_OPTICS):
    """Cluster one feature matrix and score it with the frozen Phase Q metrics."""
    t0 = time.time()
    model = OPTICS(**optics_params, n_jobs=-1)
    labels = model.fit_predict(X_tda)
    fit_seconds = time.time() - t0

    structure = cluster_structure(labels)
    views = accounting_views(labels, real["is_poisoned"])
    print(f"  [{name}] {structure['n_clusters']} clusters, "
          f"unclustered {structure['unclustered_fraction']:.1%}, "
          f"largest share {structure['largest_cluster_share']:.2%}, "
          f"capture {views['true_poison_capture_pct']:.4f}%")
    return {
        "feature_diagnostics": feature_diagnostics(X_tda),
        "distance_diagnostics": labeled_distance_diagnostics(X_tda, real["is_poisoned"]),
        "optics": {
            "requested_params": dict(optics_params),
            "fit_seconds": fit_seconds,
            "internals": optics_internals(model, optics_params["max_eps"]),
            "structure": structure,
        },
        "accounting": {k: v for k, v in views.items() if k != "clusters"},
        "removal_curve": removal_curve(labels, real["is_poisoned"]),
    }


def compare(a, b, label_a, label_b):
    """Deltas that matter, b minus a, on the quantities the handoff names."""
    def cap(x):
        return x["accounting"]["true_poison_capture_pct"]

    def unc(x, who):
        return x["accounting"][f"{who}_unclustered_fraction"]

    exact_a = next(p for p in a["removal_curve"] if p["purity_threshold"] == 1.0)
    exact_b = next(p for p in b["removal_curve"] if p["purity_threshold"] == 1.0)
    return {
        "arms": [label_a, label_b],
        "true_capture_pct": [cap(a), cap(b)],
        "true_capture_pct_delta": cap(b) - cap(a),
        "poison_unclustered_fraction": [unc(a, "poison"), unc(b, "poison")],
        "clean_unclustered_fraction": [unc(a, "clean"), unc(b, "clean")],
        "n_clusters": [a["optics"]["structure"]["n_clusters"],
                       b["optics"]["structure"]["n_clusters"]],
        "largest_cluster_share": [a["optics"]["structure"]["largest_cluster_share"],
                                  b["optics"]["structure"]["largest_cluster_share"]],
        "n_zero_variance_features": [a["feature_diagnostics"]["n_zero_variance_features"],
                                     b["feature_diagnostics"]["n_zero_variance_features"]],
        "n_exact_duplicate_rows": [a["feature_diagnostics"]["n_exact_duplicate_rows"],
                                   b["feature_diagnostics"]["n_exact_duplicate_rows"]],
        "fraction_core_distance_within_max_eps": [
            a["optics"]["internals"]["fraction_core_distance_within_max_eps"],
            b["optics"]["internals"]["fraction_core_distance_within_max_eps"],
        ],
        "exact_purity_poison_removal_rate": [exact_a["poison_removal_rate"],
                                             exact_b["poison_removal_rate"]],
        "exact_purity_clean_false_removal_rate": [exact_a["clean_false_removal_rate"],
                                                  exact_b["clean_false_removal_rate"]],
        "median_poison_over_clean_distance_ratio": [
            a["distance_diagnostics"]["median_ratio_poison_over_clean"],
            b["distance_diagnostics"]["median_ratio_poison_over_clean"],
        ],
    }


ARM_BUILDERS = {
    "legacy_30x50_t04": lambda real: extract_tda_features(real["X_combined"])[0],
    "source_1x1500_t04": lambda real: extract_geometry_features(
        real["X_combined"], "source", 0.4)[0],
    "source_1x1500_t03": lambda real: extract_geometry_features(
        real["X_combined"], "source", 0.3)[0],
}
ARM_CACHE_TAG = {
    "legacy_30x50_t04": "legacy_t04",
    "source_1x1500_t04": "source_t04",
    "source_1x1500_t03": "source_t03",
}


def _ms(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "sd_pop": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in values]}


def confirm_seeds(arm_names, seeds):
    """Re-run the named arms on every seed. Population SD, no seed dropped."""
    per_seed = {name: [] for name in arm_names}
    for seed in seeds:
        print(f"  --- seed {seed} ---")
        real = build_realization(seed)
        for name in arm_names:
            X_tda, _ = cached_features(
                ARM_CACHE_TAG[name], seed, lambda n=name, r=real: ARM_BUILDERS[n](r))
            labels, model = OPTICS(**BASELINE_OPTICS, n_jobs=-1), None
            lab = labels.fit_predict(X_tda)
            structure = cluster_structure(lab)
            views = accounting_views(lab, real["is_poisoned"])
            curve = removal_curve(lab, real["is_poisoned"])
            exact = next(p for p in curve if p["purity_threshold"] == 1.0)
            per_seed[name].append({
                "seed": seed,
                "realization": realization_provenance(real),
                "feature_hash": None,
                "structure": structure,
                "true_poison_capture_pct": views["true_poison_capture_pct"],
                "poison_unclustered_fraction": views["poison_unclustered_fraction"],
                "clean_unclustered_fraction": views["clean_unclustered_fraction"],
                "color_shares_pct_all_sample_denominator":
                    views["views"]["all_sample_denominator"]["color_shares_pct"],
                "exact_purity": exact,
                "removal_curve": curve,
            })
            print(f"    {name:<20} capture={views['true_poison_capture_pct']:.4f}% "
                  f"clusters={structure['n_clusters']:<4} "
                  f"unclustered={structure['unclustered_fraction']:.3f} "
                  f"clean_removed={exact['clean_false_removal_rate']:.5f}")

    summary = {}
    for name, rows in per_seed.items():
        summary[name] = {
            "true_poison_capture_pct": _ms([r["true_poison_capture_pct"] for r in rows]),
            "n_clusters": _ms([r["structure"]["n_clusters"] for r in rows]),
            "unclustered_fraction": _ms([r["structure"]["unclustered_fraction"] for r in rows]),
            "largest_cluster_share": _ms([r["structure"]["largest_cluster_share"] for r in rows]),
            "poison_unclustered_fraction": _ms([r["poison_unclustered_fraction"] for r in rows]),
            "clean_unclustered_fraction": _ms([r["clean_unclustered_fraction"] for r in rows]),
            "exact_clean_false_removal_rate": _ms(
                [r["exact_purity"]["clean_false_removal_rate"] for r in rows]),
            "red_share_of_all_samples_pct": _ms(
                [r["color_shares_pct_all_sample_denominator"]["Red"] for r in rows]),
        }

    deltas = {}
    for a, b in (("legacy_30x50_t04", "source_1x1500_t04"),
                 ("source_1x1500_t04", "source_1x1500_t03")):
        if a in per_seed and b in per_seed:
            d = [per_seed[b][i]["true_poison_capture_pct"]
                 - per_seed[a][i]["true_poison_capture_pct"] for i in range(len(seeds))]
            deltas[f"{b}_minus_{a}"] = {
                **_ms(d),
                "n_seeds_positive": int(sum(1 for v in d if v > 0)),
                "n_seeds": len(d),
                "sign_stable": all(v > 0 for v in d) or all(v < 0 for v in d),
            }

    return {"seeds": list(seeds), "per_seed": per_seed,
            "summary": summary, "capture_deltas": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-arm", action="store_true",
                        help="additionally run the separate 0.4-vs-0.3 comparison "
                             "on the source geometry")
    parser.add_argument("--confirm-seeds", action="store_true",
                        help="re-run every arm on all five seeds with population SD")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print("PHASE Q2-B -- source-printed geometry vs operational geometry")
    print("=" * 72)

    print("\n[1] Deterministic fixture probe (before any full run)")
    probe = fixture_probe()
    for geometry, entry in probe.items():
        if not entry["accepted_by_giotto_tda"]:
            print(f"  {geometry}: REJECTED -- {entry['exception']}")
            continue
        print(f"  {geometry}: shape {tuple(entry['image_shape'])}, "
              f"{entry['output_feature_count']} features, "
              f"{entry['n_zero_variance_features']} zero-variance, "
              f"nan={entry['n_nan']} inf={entry['n_inf']}")
        for f in entry["per_filtration_diagrams"]:
            nt = f["nontrivial_points_by_dim"]
            print(f"      {f['filtration']:<16} filt-values unique="
                  f"{f['filtration_value_n_unique']:<5} nontrivial H0={nt.get('0', 0)} "
                  f"H1={nt.get('1', 0)}")

    if not probe["source"]["accepted_by_giotto_tda"]:
        print("\nSource geometry rejected by the pinned library. Arm stops here, "
              "exception preserved in the artifact.")
        write_json(Path(RESULTS_DIR) / "phase_q2_geometry.json", {
            "phase": "Q2-B",
            "status": "source_geometry_rejected",
            "environment": environment_block(),
            "fixture_probe": probe,
        })
        return

    print("\n[2] Matched R60/seed-42 realization")
    real = build_realization(DISCOVERY_SEED)
    print(f"  input_hash={real['input_hash']} poison_mask_hash={real['poison_mask_hash']}")

    print("\n[3] Feature extraction, both geometries, threshold 0.4")
    X_legacy, t_legacy = cached_features(
        "legacy_t04", DISCOVERY_SEED,
        lambda: extract_tda_features(real["X_combined"])[0])
    X_source, t_source = cached_features(
        "source_t04", DISCOVERY_SEED,
        lambda: extract_geometry_features(real["X_combined"], "source", 0.4)[0])

    print("\n[4] Clustering both arms with the identical OPTICS configuration")
    arms = {
        "legacy_30x50_t04": run_arm("legacy_30x50_t04", X_legacy, real),
        "source_1x1500_t04": run_arm("source_1x1500_t04", X_source, real),
    }
    for key, seconds in (("legacy_30x50_t04", t_legacy), ("source_1x1500_t04", t_source)):
        arms[key]["extraction_seconds"] = seconds

    geometry_comparison = compare(arms["legacy_30x50_t04"], arms["source_1x1500_t04"],
                                  "legacy_30x50_t04", "source_1x1500_t04")

    payload = {
        "phase": "Q2-B",
        "description": "Source-printed (1,1500)+printed centers vs operational (30,50)+"
                       "rescaled centers, one factor, matched data and matched OPTICS.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_block(),
        "geometries": {g: {"image_shape": list(s["image_shape"]),
                           "centers": [c.tolist() for c in s["centers"]],
                           "provenance": s["provenance"]}
                       for g, s in GEOMETRIES.items()},
        "optics_config": BASELINE_OPTICS,
        "realization": realization_provenance(real),
        "fixture_probe": probe,
        "arms": arms,
        "geometry_comparison": geometry_comparison,
    }

    if args.threshold_arm:
        print("\n[5] SEPARATE one-factor threshold arm on the source geometry: 0.4 vs 0.3")
        X_source_03, t_03 = cached_features(
            "source_t03", DISCOVERY_SEED,
            lambda: extract_geometry_features(real["X_combined"], "source", 0.3)[0])
        arms["source_1x1500_t03"] = run_arm("source_1x1500_t03", X_source_03, real)
        arms["source_1x1500_t03"]["extraction_seconds"] = t_03
        payload["threshold_comparison"] = compare(
            arms["source_1x1500_t04"], arms["source_1x1500_t03"],
            "source_1x1500_t04", "source_1x1500_t03")
        payload["threshold_comparison"]["note"] = (
            "Independent one-factor comparison. Binarizer threshold only; geometry "
            "and OPTICS held fixed. Not combined with the geometry comparison."
        )

    if args.confirm_seeds:
        arm_names = ["legacy_30x50_t04", "source_1x1500_t04"]
        if args.threshold_arm:
            arm_names.append("source_1x1500_t03")
        print(f"\n[6] Five-seed confirmation of every arm ({', '.join(arm_names)})")
        payload["seed_confirmation"] = confirm_seeds(arm_names, CONFIRMATION_SEEDS)
        print("\n  five-seed summary (mean +/- population SD):")
        for name, s in payload["seed_confirmation"]["summary"].items():
            cap = s["true_poison_capture_pct"]
            unc = s["unclustered_fraction"]
            print(f"    {name:<20} capture {cap['mean']:.2f} +/- {cap['sd_pop']:.2f}%   "
                  f"unclustered {unc['mean']:.3f} +/- {unc['sd_pop']:.3f}")
        for name, d in payload["seed_confirmation"]["capture_deltas"].items():
            print(f"    delta {name}: {d['mean']:+.2f} +/- {d['sd_pop']:.2f} pp, "
                  f"{d['n_seeds_positive']}/{d['n_seeds']} seeds positive, "
                  f"sign stable: {d['sign_stable']}")

    payload["elapsed_seconds"] = time.time() - t_start
    write_json(Path(RESULTS_DIR) / "phase_q2_geometry.json", payload)
    print(f"\nDone in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
