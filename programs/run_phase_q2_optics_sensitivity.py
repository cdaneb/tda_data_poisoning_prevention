"""Phase Q2-C: provenance-first OPTICS sensitivity analysis.

The publication gives only broad ranges (section 7, page 14): ``min_samples`` in
[2, 300], epsilon in (0, 2], ``min_cluster_size`` in [2, 400].  It does not print
the fitted configuration, the metric, the Xi value, or the extraction rule.  No
author code or supplementary material was recoverable.  This stage is therefore
a SENSITIVITY ANALYSIS, not a source reproduction, and it is structured so that
it cannot quietly become one.

Two protocols, strictly separated:

  Discovery    -- one seed (42), one fixed feature map, one-factor-at-a-time
                  sweeps.  ``discover_candidates`` is a pure function of
                  LABEL-FREE cluster-structure records.  It never receives the
                  poison mask; ``tools/test_phase_q2.py`` asserts this by
                  passing it a record containing a poison field and checking the
                  selection is unchanged.

  Confirmation -- the locked candidates, unchanged, on all five seeds, where
                  poison metrics are computed for the first time.

The selection rule below is fixed in source before any cell is run.

Usage:
    python run_phase_q2_optics_sensitivity.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from programs.paths import RESULTS_DIR
from programs.tda_pipeline import extract_tda_features
from programs.phase_q_metrics import removal_curve
from tools.phase_q2_common import (
    CONFIRMATION_SEEDS, DISCOVERY_SEED, accounting_views, build_realization,
    cached_features, cluster_structure, environment_block, environment_block as _env,
    optics_internals, realization_provenance, write_json,
)

BASELINE = {"min_samples": 5, "max_eps": 2.0, "cluster_method": "xi", "min_cluster_size": None}

# One factor at a time, anchored on the ranges the paper actually prints.
SWEEPS = {
    "max_eps": [0.25, 0.5, 1.0, 2.0],
    "min_samples": [2, 5, 10, 25, 50, 100, 300],
    "min_cluster_size": [None, 2, 5, 10, 25, 50, 100, 300, 400],
}
# Outside the paper's stated epsilon range (0, 2]; included only because it is
# the sklearn default, and labeled as such wherever it appears.
OUT_OF_RANGE_CELLS = [{"max_eps": np.inf}]

# ---------------------------------------------------------------------------
# LABEL-FREE SELECTION RULE -- fixed before any cell was run.
#
# What Figure 14 shows, stripped of any ground truth: a small number of clusters
# whose sample shares account for essentially the whole population, with no
# displayed noise row.  The label-free analogue is "lowest unclustered fraction",
# guarded against the two degenerate ways to achieve it.
#
#   feasible   : n_clusters >= MIN_CLUSTERS and
#                largest_cluster_share <= MAX_LARGEST_SHARE
#   rank       : unclustered_fraction ascending
#   tie-break  : |n_clusters - FIG14_TARGET_CLUSTERS| ascending
#   lock       : top N_CANDIDATES
#
# No poison label, purity, capture or color share enters this rule.
# ---------------------------------------------------------------------------
MIN_CLUSTERS = 2
MAX_LARGEST_SHARE = 0.95
FIG14_TARGET_CLUSTERS = 7.5  # Figure 14 displays 7 (126-feat) and 8 (72-feat) entries
N_CANDIDATES = 3

SELECTION_RULE_TEXT = (
    "feasible = {cells : n_clusters >= 2 and largest_cluster_share <= 0.95}; "
    "rank feasible by unclustered_fraction ascending, tie-break by "
    "|n_clusters - 7.5| ascending; lock the top 3. Label-free: the rule reads "
    "only n_clusters, largest_cluster_share and unclustered_fraction."
)

LABEL_FREE_KEYS = ("n_clusters", "largest_cluster_share", "unclustered_fraction")


def discover_candidates(records, n=N_CANDIDATES):
    """Select candidates from label-free cluster-structure records only.

    ``records`` is a list of dicts, each with a ``config`` and a ``structure``.
    Only ``LABEL_FREE_KEYS`` of ``structure`` are read.  Any other field present
    on a record -- including a poison metric accidentally attached upstream -- is
    ignored, which is what makes the label-blindness testable.
    """
    feasible = []
    for rec in records:
        s = {k: rec["structure"][k] for k in LABEL_FREE_KEYS}
        if s["n_clusters"] < MIN_CLUSTERS:
            continue
        if s["largest_cluster_share"] > MAX_LARGEST_SHARE:
            continue
        feasible.append((s["unclustered_fraction"],
                         abs(s["n_clusters"] - FIG14_TARGET_CLUSTERS),
                         rec["name"], rec["config"]))
    feasible.sort(key=lambda t: (t[0], t[1], t[2]))
    return [{"name": name, "config": config,
             "unclustered_fraction": unc, "cluster_count_distance": dist}
            for unc, dist, name, config in feasible[:n]]


def _optics_kwargs(config):
    kw = {k: v for k, v in config.items() if v is not None or k != "min_cluster_size"}
    kw = {k: v for k, v in kw.items() if not (k == "min_cluster_size" and v is None)}
    return kw


def fit_optics(X, config):
    kw = _optics_kwargs(config)
    t0 = time.time()
    model = OPTICS(**kw, n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model, time.time() - t0


def run_discovery(X, verbose=True):
    """One-factor-at-a-time sweeps. Returns label-free records only."""
    cells = []
    for factor, values in SWEEPS.items():
        for value in values:
            config = dict(BASELINE)
            config[factor] = value
            name = f"{factor}={value}"
            cells.append({"name": name, "factor": factor, "value": value,
                          "config": config, "in_paper_range": True})
    for extra in OUT_OF_RANGE_CELLS:
        config = dict(BASELINE)
        config.update(extra)
        key, value = next(iter(extra.items()))
        cells.append({"name": f"{key}={value} [outside paper range]",
                      "factor": key, "value": float(value),
                      "config": config, "in_paper_range": False})

    seen = {}
    records = []
    for cell in cells:
        key = tuple(sorted((k, str(v)) for k, v in cell["config"].items()))
        if key in seen:
            if verbose:
                print(f"  {cell['name']:<40} duplicate of {seen[key]}; reusing")
            base = next(r for r in records if r["name"] == seen[key])
            records.append({**cell, "structure": base["structure"],
                            "optics_internals": base["optics_internals"],
                            "fit_seconds": 0.0, "duplicate_of": seen[key]})
            continue
        seen[key] = cell["name"]
        labels, model, secs = fit_optics(X, cell["config"])
        structure = cluster_structure(labels)
        max_eps = cell["config"]["max_eps"]
        rec = {**cell,
               "structure": structure,
               "optics_internals": optics_internals(model, max_eps),
               "fit_seconds": secs,
               "duplicate_of": None}
        records.append(rec)
        if verbose:
            print(f"  {cell['name']:<40} clusters={structure['n_clusters']:<5} "
                  f"unclustered={structure['unclustered_fraction']:.3f} "
                  f"largest={structure['largest_cluster_share']:.3f} ({secs:.0f}s)")
    return records


def confirm(candidate, seeds=CONFIRMATION_SEEDS):
    """Evaluate one locked candidate, unchanged, across seeds. Labels used here."""
    per_seed = []
    for seed in seeds:
        real = build_realization(seed)
        X_tda, _ = cached_features(
            "legacy_t04", seed, lambda r=real: extract_tda_features(r["X_combined"])[0])
        labels, model, secs = fit_optics(X_tda, candidate["config"])
        views = accounting_views(labels, real["is_poisoned"])
        curve = removal_curve(labels, real["is_poisoned"])
        exact = next(p for p in curve if p["purity_threshold"] == 1.0)
        loose = next(p for p in curve if p["purity_threshold"] == 0.50)
        structure = cluster_structure(labels)
        per_seed.append({
            "seed": seed,
            "realization": realization_provenance(real),
            "fit_seconds": secs,
            "structure": structure,
            "color_shares_pct_all_sample_denominator":
                views["views"]["all_sample_denominator"]["color_shares_pct"],
            "true_poison_capture_pct": views["true_poison_capture_pct"],
            "poison_unclustered_fraction": views["poison_unclustered_fraction"],
            "clean_unclustered_fraction": views["clean_unclustered_fraction"],
            "exact_purity": exact,
            "relaxed_purity_gt50": loose,
            "removal_curve": curve,
        })
        print(f"    seed {seed}: capture={views['true_poison_capture_pct']:.4f}% "
              f"clusters={structure['n_clusters']} "
              f"unclustered={structure['unclustered_fraction']:.3f} "
              f"clean_removed={exact['clean_false_removal_rate']:.4f}")
    return {"candidate": candidate, "per_seed": per_seed,
            "summary": summarize(per_seed)}


def _ms(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "sd_pop": None, "per_seed": list(values)}
    return {"mean": float(a.mean()), "sd_pop": float(a.std(ddof=0)),
            "per_seed": [None if v is None else float(v) for v in values]}


def summarize(per_seed):
    """Mean +/- population SD (ddof=0), per house convention."""
    return {
        "true_poison_capture_pct": _ms([s["true_poison_capture_pct"] for s in per_seed]),
        "n_clusters": _ms([s["structure"]["n_clusters"] for s in per_seed]),
        "unclustered_fraction": _ms([s["structure"]["unclustered_fraction"] for s in per_seed]),
        "largest_cluster_share": _ms([s["structure"]["largest_cluster_share"] for s in per_seed]),
        "poison_unclustered_fraction": _ms([s["poison_unclustered_fraction"] for s in per_seed]),
        "clean_unclustered_fraction": _ms([s["clean_unclustered_fraction"] for s in per_seed]),
        "exact_poison_removal_rate": _ms([s["exact_purity"]["poison_removal_rate"] for s in per_seed]),
        "exact_clean_false_removal_rate": _ms([s["exact_purity"]["clean_false_removal_rate"] for s in per_seed]),
        "exact_removal_precision": _ms([s["exact_purity"]["removal_precision"] for s in per_seed]),
        "gt50_poison_removal_rate": _ms([s["relaxed_purity_gt50"]["poison_removal_rate"] for s in per_seed]),
        "gt50_clean_false_removal_rate": _ms([s["relaxed_purity_gt50"]["clean_false_removal_rate"] for s in per_seed]),
        "red_share_of_all_samples_pct": _ms(
            [s["color_shares_pct_all_sample_denominator"]["Red"] for s in per_seed]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-only", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print("PHASE Q2-C -- provenance-first OPTICS sensitivity analysis")
    print("=" * 72)
    print(f"\nSelection rule (fixed in source before any run):\n  {SELECTION_RULE_TEXT}")

    print(f"\n[1] Discovery matrix: R60 seed {DISCOVERY_SEED}, legacy 30x50, threshold 0.4")
    real = build_realization(DISCOVERY_SEED)
    X_tda, _ = cached_features(
        "legacy_t04", DISCOVERY_SEED,
        lambda: extract_tda_features(real["X_combined"])[0])
    print(f"  features {X_tda.shape}, input_hash={real['input_hash']}")

    print("\n[2] One-factor-at-a-time sweeps (label-free records only)")
    records = run_discovery(X_tda)

    print("\n[3] Locking candidates by the label-free rule")
    candidates = discover_candidates(records)
    for i, c in enumerate(candidates, 1):
        print(f"  candidate {i}: {c['name']}  unclustered={c['unclustered_fraction']:.3f}")

    payload = {
        "phase": "Q2-C",
        "description": "OPTICS sensitivity analysis. Not a source reproduction: the "
                       "publication prints only parameter ranges, and no author code "
                       "or metadata was recoverable.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_block(),
        "baseline_config": {k: (None if v is None else v) for k, v in BASELINE.items()},
        "paper_stated_ranges": {
            "min_samples": [2, 300], "epsilon": "(0, 2]", "min_cluster_size": [2, 400],
            "source": "section 7, page 14; no fitted configuration, metric, Xi value "
                      "or extraction rule is printed",
        },
        "extraction_method_status": "UNRESOLVED -- the paper's pseudocode builds clusters "
                                    "from epsilon neighborhoods, which does not identify "
                                    "sklearn cluster_method='xi' vs 'dbscan'. No "
                                    "extraction-method arm was run, because no direct "
                                    "evidence supports one over the other.",
        "selection_rule": {
            "text": SELECTION_RULE_TEXT,
            "label_free_keys_read": list(LABEL_FREE_KEYS),
            "min_clusters": MIN_CLUSTERS,
            "max_largest_cluster_share": MAX_LARGEST_SHARE,
            "tie_break_target_cluster_count": FIG14_TARGET_CLUSTERS,
            "n_candidates": N_CANDIDATES,
            "fixed_before_any_cell_was_run": True,
        },
        "discovery": {
            "seed": DISCOVERY_SEED,
            "feature_map": "legacy_30x50 threshold 0.4",
            "feature_shape": list(X_tda.shape),
            "realization": realization_provenance(real),
            "records": records,
        },
        "locked_candidates": candidates,
    }

    if not args.discovery_only:
        print(f"\n[4] Confirmation on seeds {list(CONFIRMATION_SEEDS)}")
        confirmations = []
        for i, c in enumerate(candidates, 1):
            print(f"  candidate {i}: {c['name']}")
            confirmations.append(confirm(c))
        payload["confirmation"] = {
            "seeds": list(CONFIRMATION_SEEDS),
            "note": "Candidates locked by the label-free discovery rule and evaluated "
                    "unchanged. Poison metrics computed here for the first time.",
            "results": confirmations,
        }

    payload["elapsed_seconds"] = time.time() - t_start
    write_json(Path(RESULTS_DIR) / "phase_q2_optics_sensitivity.json", payload)
    print(f"\nDone in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
