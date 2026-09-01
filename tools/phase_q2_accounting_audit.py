"""Phase Q2-A: OPTICS noise-accounting audit.

Motivating observation: this reconstruction assigns a large share of samples to
OPTICS label ``-1``, while the source paper's Figure 14 cluster rows sum to
approximately 100%.  Two explanations are compatible with that: the authors'
fitted model produced almost no noise, or noise was omitted from the display and
the remaining shares renormalized.

This stage does not refit anything to test that.  It fits the standing
configuration ONCE on the standing R60/seed-42 realization, dumps enough OPTICS
internals to prove provenance and to explain why points became ``-1``, and then
recomputes three display conventions from that single labelling.

The audit's whole point is the invariant: the denominator for true capture is
the full poisoned population in every view.  A display convention that changes
how the table reads while leaving capture untouched is an accounting
reconciliation, not detector reproduction.

Usage:
    python tools/phase_q2_accounting_audit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from programs.paths import RESULTS_DIR
from programs.tda_pipeline import extract_tda_features
from programs.phase_q_metrics import removal_curve
from tools.phase_q2_common import (
    DISCOVERY_SEED, accounting_views, build_realization, cached_features,
    cluster_structure, environment_block, feature_diagnostics,
    labeled_distance_diagnostics, optics_internals, realization_provenance,
    write_json,
)

# The standing configuration, verbatim from clustering.run_all_clustering.
BASELINE_OPTICS = {"min_samples": 5, "max_eps": 2.0}

# Source page-level facts this audit checks its own arithmetic against.
# Transcribed from the journal PDF by Christian; no local copy was available on
# WIRE during Q2, so these are treated as reported values, not as re-verified
# primary text. See docs/PHASE_Q2_RECONCILIATION_REPORT.md section 2.
SOURCE_FIGURE14 = {
    "prose_section": "6.5",
    "prose_capture_pct": 60.3,
    "prose_color_shares_pct": [47.54, 45.83, 6.03, 0.59],
    "fig14_126feat_n_clusters": 7,
    "fig14_126feat_red_share_of_poison_pct": 56.1,
    "fig14_72feat_n_entries": 8,
    "fig14_72feat_red_share_of_poison_pct": 64.1,
    "note": "one cluster identifier is duplicated in the 72-feature row of Figure 14",
}


def audit_source_arithmetic():
    """Check the paper's own prose against itself, before touching our data.

    Section 6.5 reports a capture summary and four color shares.  If the shares
    are shares of ALL samples and the poison rate is 10%, then a 100%-pure Red
    band holding a fraction c of the poisoned population must occupy 0.10 * c of
    all samples.  Testing that identity against the printed numbers is what tells
    us which printed share is Red and what denominator the figure used.
    """
    shares = SOURCE_FIGURE14["prose_color_shares_pct"]
    capture = SOURCE_FIGURE14["prose_capture_pct"]
    total = float(sum(shares))

    implied = {}
    for poison_rate in (0.05, 0.10, 0.20, 0.30):
        implied[f"poison_rate_{poison_rate:.2f}"] = {
            "implied_red_share_of_all_samples_pct": 100.0 * poison_rate * capture / 100.0,
            "matching_printed_share": next(
                (s for s in shares
                 if abs(s - 100.0 * poison_rate * capture / 100.0) < 0.02),
                None,
            ),
        }

    return {
        "printed_shares_sum_pct": total,
        "printed_shares_sum_is_100_within_0.05": abs(total - 100.0) < 0.05,
        "red_share_identity": {
            "definition": "red_share_of_all_samples = poison_rate * capture_fraction, "
                          "valid when Red clusters are exactly 100% poisoned",
            "by_poison_rate": implied,
        },
        "fig14_vs_prose_capture": {
            "prose_pct": capture,
            "fig14_126feat_pct": SOURCE_FIGURE14["fig14_126feat_red_share_of_poison_pct"],
            "fig14_72feat_pct": SOURCE_FIGURE14["fig14_72feat_red_share_of_poison_pct"],
            "prose_equals_either_fig14_row": False,
            "prose_within_fig14_range": (
                SOURCE_FIGURE14["fig14_126feat_red_share_of_poison_pct"]
                <= capture
                <= SOURCE_FIGURE14["fig14_72feat_red_share_of_poison_pct"]
            ),
        },
    }


def main():
    t_start = time.time()
    print("=" * 72)
    print("PHASE Q2-A -- OPTICS noise-accounting audit")
    print("=" * 72)

    print("\n[1] Source arithmetic audit (paper against itself)")
    source_audit = audit_source_arithmetic()
    print(f"  printed color shares sum to {source_audit['printed_shares_sum_pct']:.2f}%")
    for rate, block in source_audit["red_share_identity"]["by_poison_rate"].items():
        print(f"  {rate}: implied Red share of all samples = "
              f"{block['implied_red_share_of_all_samples_pct']:.2f}%  "
              f"matches printed share {block['matching_printed_share']}")

    print("\n[2] Rebuilding the standing R60/seed-42 realization")
    real = build_realization(DISCOVERY_SEED)
    print(f"  n_total={real['n_total']} n_clean={real['n_clean']} "
          f"n_poison={real['n_poison']} input_hash={real['input_hash']}")

    print("\n[3] Legacy 30x50 / threshold 0.4 feature matrix")
    X_tda, tda_time = cached_features(
        "legacy_t04", DISCOVERY_SEED,
        lambda: extract_tda_features(real["X_combined"])[0],
    )
    if X_tda.shape != (5500, 60):
        raise AssertionError(f"expected (5500, 60), got {X_tda.shape}")

    print("\n[4] Fitting the standing OPTICS configuration once")
    t0 = time.time()
    model = OPTICS(**BASELINE_OPTICS, n_jobs=-1)
    labels = model.fit_predict(X_tda)
    cluster_time = time.time() - t0
    structure = cluster_structure(labels)
    print(f"  {structure['n_clusters']} clusters, "
          f"{structure['n_unclustered']} unclustered "
          f"({structure['unclustered_fraction']:.1%}), "
          f"largest cluster share {structure['largest_cluster_share']:.2%}")

    internals = optics_internals(model, BASELINE_OPTICS["max_eps"])
    print(f"  core distances <= max_eps: "
          f"{internals['fraction_core_distance_within_max_eps']:.1%}; "
          f"infinite reachability: {internals['n_reachability_infinite']}")

    print("\n[5] Three accounting views over that one labelling")
    views = accounting_views(labels, real["is_poisoned"])
    for name, block in views["views"].items():
        shares = ", ".join(f"{c} {v:.2f}%" for c, v in block["color_shares_pct"].items())
        print(f"  {name:<28} sum={block['sum_pct']:.2f}%  [{shares}]")
    print(f"  TRUE capture (poison in Red / all poison) = "
          f"{views['true_poison_capture_pct']:.4f}%  -- identical in all three views")

    print("\n[6] Comparison to the source display")
    source_red = SOURCE_FIGURE14["prose_color_shares_pct"][2]
    ours_all = views["views"]["all_sample_denominator"]["color_shares_pct"]["Red"]
    ours_clustered = views["views"]["clustered_only_denominator"]["color_shares_pct"]["Red"]
    comparison = {
        "source_prose_red_share_of_all_samples_pct": source_red,
        "source_prose_capture_pct": SOURCE_FIGURE14["prose_capture_pct"],
        "ours_red_share_all_sample_denominator_pct": ours_all,
        "ours_red_share_clustered_only_denominator_pct": ours_clustered,
        "ours_true_capture_pct": views["true_poison_capture_pct"],
        "renormalization_gain_factor": (
            ours_clustered / ours_all if ours_all else float("nan")
        ),
        "red_share_gap_factor_vs_source": (
            source_red / ours_clustered if ours_clustered else float("inf")
        ),
        "capture_gap_factor_vs_source": (
            SOURCE_FIGURE14["prose_capture_pct"] / views["true_poison_capture_pct"]
            if views["true_poison_capture_pct"] else float("inf")
        ),
        "does_denominator_choice_explain_100pct_sum": (
            abs(views["views"]["clustered_only_denominator"]["sum_pct"] - 100.0) < 1e-6
        ),
        "does_denominator_choice_change_true_capture": False,
    }
    print(f"  ours Red share: {ours_all:.4f}% (all-sample) -> "
          f"{ours_clustered:.4f}% (clustered-only renormalized)")
    print(f"  source Red share: {source_red:.2f}% of all samples")
    print(f"  remaining Red-share gap after renormalization: "
          f"{comparison['red_share_gap_factor_vs_source']:.1f}x")

    print("\n[7] Label-free feature diagnostics + frozen removal curve")
    feats = feature_diagnostics(X_tda)
    dists = labeled_distance_diagnostics(X_tda, real["is_poisoned"])
    curve = removal_curve(labels, real["is_poisoned"])
    print(f"  zero-variance features: {feats['n_zero_variance_features']}/60; "
          f"exact duplicate rows: {feats['n_exact_duplicate_rows']}")
    for point in curve:
        print(f"    purity>{point['purity_threshold']:.2f}: "
              f"poison removed {point['poison_removal_rate']:.4f}, "
              f"clean removed {point['clean_false_removal_rate']:.4f}")

    payload = {
        "phase": "Q2-A",
        "description": "OPTICS noise-accounting audit on the standing R60/seed-42 arm. "
                       "One fit, three display conventions, no parameter search.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_block(),
        "source_facts": SOURCE_FIGURE14,
        "source_arithmetic_audit": source_audit,
        "realization": realization_provenance(real),
        "feature_map": {
            "geometry": "legacy_30x50",
            "centers": [[0, 50], [0, 25], [30, 0]],
            "threshold": 0.4,
            "shape": list(X_tda.shape),
            "feature_hash": feats["feature_hash"],
            "extraction_seconds": tda_time,
        },
        "optics": {
            "requested_params": BASELINE_OPTICS,
            "fit_seconds": cluster_time,
            "internals": internals,
            "structure": structure,
        },
        "feature_diagnostics": feats,
        "distance_diagnostics": dists,
        "accounting": views,
        "source_comparison": comparison,
        "removal_curve": curve,
        "elapsed_seconds": time.time() - t_start,
    }

    out = Path(RESULTS_DIR) / "phase_q2_accounting_audit.json"
    write_json(out, payload)
    print(f"\nDone in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
