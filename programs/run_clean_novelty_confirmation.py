"""Frozen, resumable independent confirmation of the Step 3 novelty result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import gtda
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from programs.data_loader import LABEL_COLUMN, PAYLOAD_COLUMNS
from programs.monkam_representation import stable_hash
from programs.novelty_detectors import HigherIsAnomaly, KNNDistance, fit_calibrate_evaluate
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs.phase_q_pipeline import CONTROL_THRESHOLD, THRESHOLD_STACK, extract_multithreshold_features
from programs.resource_usage import peak_rss_kib
from sklearn.ensemble import IsolationForest

SEEDS = (2026, 2027, 2028, 2029, 2030)
EXPLORATORY_SEEDS = (42, 123, 456, 789, 1024)
FAMILIES = tuple(SUPPORTED_FAMILIES)
REPRESENTATIONS = ("control", "stack")
DETECTORS = ("isolation_forest", "knn_distance")
BUDGETS = (0.05, 0.01)
WORKERS = 8
BOOTSTRAP_REPS = 100_000
RESULTS = ROOT / "results"
PRE = RESULTS / "clean_novelty_confirmation_preregistration.json"
CELLS = RESULTS / "clean_novelty_confirmation_cells"
OUT = RESULTS / "clean_novelty_confirmation.json"
CSVOUT = RESULTS / "clean_novelty_confirmation_summary.csv"
REPORT = ROOT / "docs/CLEAN_NOVELTY_CONFIRMATION_REPORT.md"
CACHE = ROOT / ".confirmation_cache"
DATASETS = {
    "unsw": ROOT / "data/Payload_data_UNSW.csv",
    "cicids": ROOT / "data/Payload_data_CICIDS2017.csv",
}
POPULATIONS = {
    "unsw_matched": {"dataset": "unsw", "clean": 5000, "poison": 500, "families": FAMILIES},
    "cicids_matched": {"dataset": "cicids", "clean": 5000, "poison": 500, "families": FAMILIES},
    "cicids_scale": {"dataset": "cicids", "clean": 50000, "poison": 5000, "families": ("transpositions",)},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_list(values):
    return stable_hash([int(v) for v in values])


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def raw_hashes(X):
    return [hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest() for row in X]


def split_indices(X, seed):
    groups = np.asarray(raw_hashes(X))
    idx = np.arange(len(X))
    train, rest = next(GroupShuffleSplit(1, test_size=.4, random_state=seed).split(idx, groups=groups))
    cal_rel, eval_rel = next(GroupShuffleSplit(1, test_size=.5, random_state=seed + 10000).split(rest, groups=groups[rest]))
    cal, evaluation = rest[cal_rel], rest[eval_rel]
    sets = [set(groups[x]) for x in (train, cal, evaluation)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("raw-payload group crossed clean partitions")
    return train, cal, evaluation, groups.tolist()


def row_count(path):
    with open(path, "rb") as f:
        return sum(block.count(b"\n") for block in iter(lambda: f.read(16 << 20), b"")) - 1


def selected_rows(path, wanted):
    """Read a union of literal zero-based data-row indices in one CSV pass."""
    wanted = np.asarray(sorted(set(map(int, wanted))), dtype=np.int64)
    found = {}
    offset = 0
    for chunk in pd.read_csv(path, usecols=PAYLOAD_COLUMNS + [LABEL_COLUMN], chunksize=10000):
        stop = offset + len(chunk)
        lo, hi = np.searchsorted(wanted, [offset, stop])
        for global_i in wanted[lo:hi]:
            row = chunk.iloc[int(global_i - offset)]
            found[int(global_i)] = (row[PAYLOAD_COLUMNS].to_numpy(dtype=np.uint8), row[LABEL_COLUMN])
        offset = stop
    if len(found) != len(wanted):
        raise RuntimeError(f"materialized {len(found)} of {len(wanted)} requested rows from {path}")
    return found


def base_cache_path(population, seed):
    return CACHE / "prepared" / f"{population}_seed{seed}.npz"


def preregistration_content():
    return {
        "experiment": "clean_novelty_independent_confirmation",
        "frozen_before_outcomes": True,
        "seeds": list(SEEDS),
        "excluded_exploratory_seeds": list(EXPLORATORY_SEEDS),
        "families": list(FAMILIES),
        "populations": POPULATIONS,
        "clean_split": {"group": "exact raw-payload SHA-256", "train": .6, "calibration": .2, "evaluation": .2},
        "representations": {
            "control": {"thresholds": [CONTROL_THRESHOLD], "features": 60},
            "stack": {"thresholds": list(THRESHOLD_STACK), "features": 540, "scale": "1/sqrt(9)"},
        },
        "detectors": {
            "isolation_forest": {"n_estimators": 200, "random_state": "cell seed", "n_jobs": WORKERS},
            "knn_distance": {"n_neighbors": 10, "n_jobs": WORKERS},
        },
        "preprocessing": "StandardScaler fitted on detector-training trusted clean only",
        "budgets": list(BUDGETS),
        "threshold": "higher empirical quantile of trusted-clean calibration scores only",
        "success_criteria": {
            "unsw_primary": [
                "mean IF540 held-out clean removal at 5% is in [0.04,0.06]",
                "mean paired IF540-IF60 poison capture is positive and paired 95% bootstrap CI excludes zero",
                "mean IF540 poison capture is >=0.12",
            ],
            "cicids_matched": [
                "mean IF540 held-out clean removal at 5% is in [0.04,0.06]",
                "mean paired IF540-IF60 poison capture is positive",
            ],
            "cicids_scale_calibration": "mean IF540 held-out clean removal at 5% is in [0.04,0.06]",
            "secondary_cannot_rescue_primary": True,
        },
        "allowed_sensitivity_analysis": "CICIDS transpositions, 50000 clean plus 5000 poison, both representations/detectors/budgets",
        "bootstrap": {"paired_cells": True, "repetitions": BOOTSTRAP_REPS, "seed": 20260901, "percentiles": [2.5, 97.5]},
        "parallelism": {"single_level": "detector internals", "effective_workers": WORKERS, "nested_parallelism": False},
        "software_at_preparation": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "giotto_tda": gtda.__version__},
        "platform_at_preparation": platform.platform(),
    }


def prepare():
    if PRE.exists():
        raise RuntimeError(f"refusing to rewrite existing preregistration: {PRE}")
    CACHE.joinpath("prepared").mkdir(parents=True, exist_ok=True)
    design = preregistration_content()
    counts = {name: row_count(path) for name, path in DATASETS.items()}
    design["inputs"] = {name: {"path": str(path.resolve()), "rows": counts[name], "sha256": sha256_file(path)} for name, path in DATASETS.items()}
    sampled = {}
    wanted_by_dataset = {name: [] for name in DATASETS}
    for population, spec in POPULATIONS.items():
        n = spec["clean"]
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            rows = rng.choice(counts[spec["dataset"]], size=n, replace=False).astype(int).tolist()
            sampled[(population, seed)] = rows
            wanted_by_dataset[spec["dataset"]].extend(rows)
    materialized = {name: selected_rows(DATASETS[name], wanted) for name, wanted in wanted_by_dataset.items()}
    realizations = {}
    for population, spec in POPULATIONS.items():
        realizations[population] = {}
        for seed in SEEDS:
            rows = sampled[(population, seed)]
            X = np.vstack([materialized[spec["dataset"]][i][0] for i in rows])
            y = np.asarray([materialized[spec["dataset"]][i][1] for i in rows])
            train, cal, evaluation, groups = split_indices(X, seed)
            cache_path = base_cache_path(population, seed)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, X=X, y=y)
            fam = {}
            for family in spec["families"]:
                fn, kwargs = SUPPORTED_FAMILIES[family]
                Xc, _, poisoned, log = fn(X, y, poison_rate=.1, random_state=seed, **kwargs)
                if len(log) != spec["poison"] or any(x["raw_noop"] for x in log):
                    raise AssertionError("attack realization violates poison count or zero-no-op rule")
                parents = [int(x["target_index"]) for x in log]
                fam[family] = {
                    "attack_seed": seed, "function": fn.__name__, "configuration": kwargs | {"poison_rate": .1},
                    "poison_source_parent_indices": parents,
                    "poison_source_parent_dataset_row_indices": [rows[i] for i in parents],
                    "poison_source_parent_indices_hash": hash_list(parents),
                    "poison_source_parent_dataset_row_indices_hash": hash_list([rows[i] for i in parents]),
                    "attack_log": log, "attack_log_hash": stable_hash(log),
                    "combined_raw_hash": stable_hash(Xc), "poison_mask_hash": stable_hash(poisoned),
                }
            realizations[population][str(seed)] = {
                "sampled_dataset_row_indices": rows, "sampled_dataset_row_indices_hash": hash_list(rows),
                "clean_training_indices": train.tolist(), "clean_training_indices_hash": hash_list(train),
                "calibration_indices": cal.tolist(), "calibration_indices_hash": hash_list(cal),
                "heldout_clean_evaluation_indices": evaluation.tolist(), "heldout_clean_evaluation_indices_hash": hash_list(evaluation),
                "clean_training_dataset_row_indices": [rows[i] for i in train],
                "calibration_dataset_row_indices": [rows[i] for i in cal],
                "heldout_clean_evaluation_dataset_row_indices": [rows[i] for i in evaluation],
                "raw_payload_group_hashes": groups, "raw_payload_group_hashes_hash": stable_hash(groups),
                "prepared_raw_hash": stable_hash(X), "prepared_labels_hash": stable_hash(y.tolist()),
                "prepared_cache": str(cache_path.relative_to(ROOT)), "families": fam,
            }
    design["realizations"] = realizations
    design["content_hash"] = stable_hash(design)
    atomic_json(PRE, design)
    print(f"{PRE}\ncontent_hash={design['content_hash']}")


def load_locked():
    design = json.load(open(PRE))
    claimed = design.pop("content_hash")
    actual = stable_hash(design)
    design["content_hash"] = claimed
    if claimed != actual:
        raise RuntimeError("preregistration content hash mismatch")
    return design


def feature_cache(population, seed, family):
    return CACHE / "features" / f"{population}_{family}_seed{seed}.npz"


def make_features(design, population, seed, family):
    path = feature_cache(population, seed, family)
    realization = design["realizations"][population][str(seed)]
    base = np.load(ROOT / realization["prepared_cache"], allow_pickle=True)
    X, y = base["X"], base["y"]
    attack = realization["families"][family]
    fn, kwargs = SUPPORTED_FAMILIES[family]
    Xc, _, poisoned, log = fn(X, y, poison_rate=.1, random_state=seed, **kwargs)
    input_hash = stable_hash({"raw": stable_hash(Xc), "attack": attack["attack_log_hash"], "representation": design["representations"]})
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = json.load(open(meta_path)); z = np.load(path)
        if meta["input_hash"] != input_hash or meta["control_hash"] != stable_hash(z["control"]) or meta["stack_hash"] != stable_hash(z["stack"]):
            raise RuntimeError("per-cell feature cache hash mismatch")
        return z["control"], z["stack"], meta
    if stable_hash(Xc) != attack["combined_raw_hash"] or stable_hash(log) != attack["attack_log_hash"] or any(x["raw_noop"] for x in log):
        raise RuntimeError("literal preregistered attack realization did not reproduce")
    started = time.time()
    stack, blocks, _ = extract_multithreshold_features(Xc, return_blocks=True)
    control = blocks[CONTROL_THRESHOLD]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, control=control, stack=stack)
    os.replace(tmp, path)
    meta = {"input_hash": input_hash, "control_hash": stable_hash(control), "stack_hash": stable_hash(stack), "raw_hash": stable_hash(Xc), "poison_mask_hash": stable_hash(poisoned), "runtime_seconds": time.time()-started}
    atomic_json(meta_path, meta)
    return control, stack, meta


def detector_factory(name, seed):
    if name == "isolation_forest":
        return lambda: HigherIsAnomaly(IsolationForest(n_estimators=200, random_state=seed, n_jobs=WORKERS))
    if name == "knn_distance":
        return lambda: KNNDistance(n_neighbors=10, n_jobs=WORKERS)
    raise KeyError(name)


def cell_path(population, seed, family, representation):
    return CELLS / f"{population}_{family}_{representation}_seed{seed}.json"


def run_cell(design, population, seed, family, representation):
    path = cell_path(population, seed, family, representation)
    if path.exists():
        return json.load(open(path))
    control, stack, cache_meta = make_features(design, population, seed, family)
    features = control if representation == "control" else stack
    locked = design["realizations"][population][str(seed)]
    train = np.asarray(locked["clean_training_indices"]); cal = np.asarray(locked["calibration_indices"]); ev = np.asarray(locked["heldout_clean_evaluation_indices"])
    nclean = design["populations"][population]["clean"]
    poison = np.arange(nclean, len(features))
    scaler = StandardScaler().fit(features[train])
    Ztrain, Zcal, Zev, Zpoison = (scaler.transform(features[x]) for x in (train, cal, ev, poison))
    labels = np.r_[np.zeros(len(Zev)), np.ones(len(Zpoison))]
    detectors = {}
    for name in DETECTORS:
        started = time.time()
        _, metrics, scores = fit_calibrate_evaluate(detector_factory(name, seed), Ztrain, Zcal, Zev, Zpoison, BUDGETS)
        combined = np.r_[scores[1], scores[2]]
        detectors[name] = {"budgets": metrics, "auroc": roc_auc_score(labels, combined), "auprc": average_precision_score(labels, combined), "runtime_seconds": time.time()-started}
    rec = {"population": population, "dataset": design["populations"][population]["dataset"], "seed": seed, "family": family, "representation": representation,
           "preregistration_hash": design["content_hash"], "feature_hash": stable_hash(features), "feature_cache_input_hash": cache_meta["input_hash"],
           "split_hashes": {k: locked[k] for k in ("clean_training_indices_hash", "calibration_indices_hash", "heldout_clean_evaluation_indices_hash", "raw_payload_group_hashes_hash")},
           "preprocessing": {"fit_indices_hash": locked["clean_training_indices_hash"], "mean_hash": stable_hash(scaler.mean_), "scale_hash": stable_hash(scaler.scale_)}, "detectors": detectors}
    atomic_json(path, rec)
    print(path)
    return rec


def expected_keys(design):
    return {(p, s, f, r) for p, spec in design["populations"].items() for s in SEEDS for f in spec["families"] for r in REPRESENTATIONS}


def paired_bootstrap(values):
    values = np.asarray(values, float); rng = np.random.default_rng(20260901)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def mean_ci(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "ci95": paired_bootstrap(values), "values": values.tolist()}


def metric_value(record, detector, budget, metric):
    if metric in ("auroc", "auprc"):
        return record["detectors"][detector][metric]
    return record["detectors"][detector]["budgets"][str(budget)][metric]


def detailed_analysis(records):
    metrics = ("poison_capture", "clean_removal_rate", "clean_removed", "poison_removed", "precision", "auroc", "auprc")
    aggregates, effects, families = [], {}, []
    for population, spec in POPULATIONS.items():
        for detector in DETECTORS:
            for rep in REPRESENTATIONS:
                rr = [r for r in records if r["population"] == population and r["representation"] == rep]
                for budget in BUDGETS:
                    row = {"population": population, "detector": detector, "representation": rep,
                           "budget": budget, "n_cells": len(rr)}
                    for metric in metrics:
                        row[metric] = mean_ci([metric_value(r, detector, budget, metric) for r in rr])
                    aggregates.append(row)
            for budget in BUDGETS:
                key = f"{population}|{detector}|{budget}"
                effects[key] = {"n_pairs": len(SEEDS) * len(spec["families"]), "metrics": {}}
                for metric in metrics:
                    diffs = []
                    for seed in SEEDS:
                        for family in spec["families"]:
                            pair = {r["representation"]: r for r in records if r["population"] == population and r["seed"] == seed and r["family"] == family}
                            diffs.append(metric_value(pair["stack"], detector, budget, metric) - metric_value(pair["control"], detector, budget, metric))
                    effects[key]["metrics"][metric] = mean_ci(diffs)
        for family in spec["families"]:
            for detector in DETECTORS:
                for budget in BUDGETS:
                    rr = [r for r in records if r["population"] == population and r["family"] == family]
                    diffs = [metric_value(next(r for r in rr if r["seed"] == seed and r["representation"] == "stack"), detector, budget, "poison_capture") - metric_value(next(r for r in rr if r["seed"] == seed and r["representation"] == "control"), detector, budget, "poison_capture") for seed in SEEDS]
                    families.append({"population": population, "family": family, "detector": detector, "budget": budget,
                                     "n_pairs": len(diffs), "stack_minus_control_poison_capture": mean_ci(diffs),
                                     "positive_seeds": sum(x > 0 for x in diffs), "negative_seeds": sum(x < 0 for x in diffs), "zero_seeds": sum(x == 0 for x in diffs)})
    return {"aggregates": aggregates, "paired_effects": effects, "per_family_consistency": families}


def validate_integrity(design, records, verify_source=True, verify_scores=True):
    """Run the frozen confirmation integrity gates without rewriting any cache/cell."""
    failures = []
    def gate(name, ok, detail=""):
        if not ok: failures.append(f"{name}: {detail}")
        return {"passed": bool(ok), "detail": detail}
    claimed = design["content_hash"]
    unhashed = dict(design); unhashed.pop("content_hash")
    gates = {"preregistration_semantic_hash": gate("preregistration_semantic_hash", stable_hash(unhashed) == claimed)}
    literal_ok = group_ok = attacks_ok = noops_ok = prepared_ok = features_ok = scores_ok = params_ok = clean_only_ok = atomic_ok = coverage_ok = True
    source_rows = {}
    if verify_source:
        for dataset, info in design["inputs"].items():
            if sha256_file(info["path"]) != info["sha256"]: literal_ok = False
            wanted = [i for pop, spec in POPULATIONS.items() if spec["dataset"] == dataset for r in design["realizations"][pop].values() for i in r["sampled_dataset_row_indices"]]
            source_rows[dataset] = selected_rows(info["path"], wanted)
    for population, spec in POPULATIONS.items():
        for seed in SEEDS:
            locked = design["realizations"][population][str(seed)]
            for key in ("sampled_dataset_row_indices", "clean_training_indices", "calibration_indices", "heldout_clean_evaluation_indices"):
                literal_ok &= hash_list(locked[key]) == locked[key + "_hash"]
            base = np.load(ROOT / locked["prepared_cache"], allow_pickle=True); X, y = base["X"], base["y"]
            prepared_ok &= stable_hash(X) == locked["prepared_raw_hash"] and stable_hash(y.tolist()) == locked["prepared_labels_hash"]
            if verify_source:
                material = source_rows[spec["dataset"]]
                literal_ok &= all(np.array_equal(X[j], material[i][0]) and y[j] == material[i][1] for j, i in enumerate(locked["sampled_dataset_row_indices"]))
            groups = locked["raw_payload_group_hashes"]
            group_ok &= stable_hash(groups) == locked["raw_payload_group_hashes_hash"]
            sets = [{groups[i] for i in locked[k]} for k in ("clean_training_indices", "calibration_indices", "heldout_clean_evaluation_indices")]
            group_ok &= not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            for family in spec["families"]:
                attack = locked["families"][family]; fn, kwargs = SUPPORTED_FAMILIES[family]
                Xc, _, poisoned, log = fn(X, y, poison_rate=.1, random_state=seed, **kwargs)
                attacks_ok &= stable_hash(Xc) == attack["combined_raw_hash"] and stable_hash(poisoned) == attack["poison_mask_hash"] and stable_hash(log) == attack["attack_log_hash"]
                attacks_ok &= [x["target_index"] for x in log] == attack["poison_source_parent_indices"]
                noops_ok &= len(log) == spec["poison"] and not any(x["raw_noop"] for x in log)
                z = np.load(feature_cache(population, seed, family)); meta = json.load(open(feature_cache(population, seed, family).with_suffix(".json")))
                features_ok &= z["control"].shape == (spec["clean"] + spec["poison"], 60) and z["stack"].shape == (spec["clean"] + spec["poison"], 540)
                features_ok &= np.isfinite(z["control"]).all() and np.isfinite(z["stack"]).all() and stable_hash(z["control"]) == meta["control_hash"] and stable_hash(z["stack"]) == meta["stack_hash"]
    keys = [(r.get("population"), r.get("seed"), r.get("family"), r.get("representation")) for r in records]
    coverage_ok = len(keys) == len(set(keys)) == 90 and set(keys) == expected_keys(design) and Counter(k[0] for k in keys) == Counter({"unsw_matched": 40, "cicids_matched": 40, "cicids_scale": 10})
    for r in records:
        locked = design["realizations"][r["population"]][str(r["seed"])]
        atomic_ok &= set(r) >= {"population", "dataset", "seed", "family", "representation", "preregistration_hash", "feature_hash", "split_hashes", "preprocessing", "detectors"}
        clean_only_ok &= r["preprocessing"]["fit_indices_hash"] == locked["clean_training_indices_hash"]
        params_ok &= set(r["detectors"]) == set(DETECTORS) and design["detectors"] == preregistration_content()["detectors"]
        for detector in DETECTORS:
            vals = [r["detectors"][detector][x] for x in ("auroc", "auprc")]
            for b in BUDGETS: vals += list(r["detectors"][detector]["budgets"][str(b)].values())
            scores_ok &= np.isfinite(np.asarray(vals, float)).all()
        if verify_scores:
            z = np.load(feature_cache(r["population"], r["seed"], r["family"]))
            features = z["control" if r["representation"] == "control" else "stack"]
            train = np.asarray(locked["clean_training_indices"]); cal = np.asarray(locked["calibration_indices"]); ev = np.asarray(locked["heldout_clean_evaluation_indices"])
            poison = np.arange(design["populations"][r["population"]]["clean"], len(features))
            scaler = StandardScaler().fit(features[train])
            transformed = [scaler.transform(features[x]) for x in (train, cal, ev, poison)]
            for detector in DETECTORS:
                model = detector_factory(detector, r["seed"])().fit(transformed[0])
                scores_ok &= all(np.isfinite(model.score_samples(x)).all() for x in transformed[1:])
    gates.update({
        "literal_source_rows_and_index_hashes": gate("literal_source_rows_and_index_hashes", literal_ok),
        "raw_group_split_disjointness": gate("raw_group_split_disjointness", group_ok),
        "attack_parent_reproduction": gate("attack_parent_reproduction", attacks_ok),
        "zero_raw_noops": gate("zero_raw_noops", noops_ok),
        "prepared_cache_hashes": gate("prepared_cache_hashes", prepared_ok),
        "finite_60_and_540_features": gate("finite_60_and_540_features", features_ok),
        "finite_detector_scores_and_metrics": gate("finite_detector_scores_and_metrics", scores_ok, "recomputed calibration, held-out-clean, and poison scores plus stored metrics"),
        "frozen_detector_parameters": gate("frozen_detector_parameters", params_ok),
        "clean_only_training_and_calibration": gate("clean_only_training_and_calibration", clean_only_ok, "fit API accepts no labels; scaler fit hash is clean-training split; thresholds derive only from calibration scores"),
        "atomic_cell_completeness": gate("atomic_cell_completeness", atomic_ok),
        "expected_grid_coverage": gate("expected_grid_coverage", coverage_ok),
    })
    return {"passed": not failures, "gates": gates, "failures": failures}


def merge(design, require_complete=True, integrity=None):
    records = [json.load(open(p)) for p in sorted(CELLS.glob("*.json"))]
    keys = [(r["population"], r["seed"], r["family"], r["representation"]) for r in records]
    expected = expected_keys(design)
    if len(keys) != len(set(keys)):
        raise RuntimeError("result merger rejects duplicate cells")
    extras = set(keys) - expected; missing = expected - set(keys)
    if extras or (require_complete and missing):
        raise RuntimeError(f"cell grid mismatch: missing={len(missing)} extras={len(extras)}")
    summary = []
    for population in POPULATIONS:
        for detector in DETECTORS:
            for rep in REPRESENTATIONS:
                for budget in BUDGETS:
                    rr = [r for r in records if r["population"] == population and r["representation"] == rep]
                    if not rr: continue
                    capture = [r["detectors"][detector]["budgets"][str(budget)]["poison_capture"] for r in rr]
                    clean = [r["detectors"][detector]["budgets"][str(budget)]["clean_removal_rate"] for r in rr]
                    summary.append({"population": population, "detector": detector, "representation": rep, "budget": budget, "n_cells": len(rr), "poison_capture_mean": float(np.mean(capture)), "clean_removal_mean": float(np.mean(clean))})
    effects = {}
    for population in POPULATIONS:
        for detector in DETECTORS:
            for budget in BUDGETS:
                diffs = []
                for seed in SEEDS:
                    for family in POPULATIONS[population]["families"]:
                        pair = {(r["representation"]): r for r in records if r["population"] == population and r["seed"] == seed and r["family"] == family}
                        if set(pair) == set(REPRESENTATIONS):
                            diffs.append(pair["stack"]["detectors"][detector]["budgets"][str(budget)]["poison_capture"] - pair["control"]["detectors"][detector]["budgets"][str(budget)]["poison_capture"])
                if diffs:
                    effects[f"{population}|{detector}|{budget}"] = {"mean": float(np.mean(diffs)), "paired_bootstrap_ci95": paired_bootstrap(diffs), "n_pairs": len(diffs), "differences": diffs}
    def sm(pop, rep): return next(x for x in summary if x["population"] == pop and x["detector"] == "isolation_forest" and x["representation"] == rep and x["budget"] == .05)
    gates = {}
    if not missing:
        u, ue = sm("unsw_matched", "stack"), effects["unsw_matched|isolation_forest|0.05"]
        gates["unsw_primary"] = {"calibration": .04 <= u["clean_removal_mean"] <= .06, "positive_ci_excludes_zero": ue["mean"] > 0 and ue["paired_bootstrap_ci95"][0] > 0, "capture_at_least_12pct": u["poison_capture_mean"] >= .12}
        gates["unsw_primary"]["passed"] = all(gates["unsw_primary"].values())
        c, ce = sm("cicids_matched", "stack"), effects["cicids_matched|isolation_forest|0.05"]
        gates["cicids_matched"] = {"calibration": .04 <= c["clean_removal_mean"] <= .06, "positive_effect": ce["mean"] > 0}
        gates["cicids_matched"]["supported"] = all(gates["cicids_matched"].values())
        z = sm("cicids_scale", "stack")
        gates["cicids_scale_calibration"] = {"passed": .04 <= z["clean_removal_mean"] <= .06}
    runtime = {"feature_seconds": sum(json.load(open(p))["runtime_seconds"] for p in CACHE.joinpath("features").glob("*.json")), "detector_seconds": sum(d["runtime_seconds"] for r in records for d in r["detectors"].values()), "workers": WORKERS, "platform": platform.platform(), "peak_rss_kib": peak_rss_kib()}
    analysis = detailed_analysis(records) if not missing else {}
    result = {"experiment": design["experiment"], "preregistration_hash": design["content_hash"], "complete": not missing, "missing_cells": [list(x) for x in sorted(missing)], "records": records, "summary": summary, "paired_effects": effects, "analysis": analysis, "integrity": integrity, "success": gates, "runtime": runtime, "versions": design["software_at_preparation"]}
    atomic_json(OUT, result)
    if analysis:
        csv_rows = []
        for row in analysis["aggregates"]:
            flat = {k: row[k] for k in ("population", "detector", "representation", "budget", "n_cells")}
            for metric in ("poison_capture", "clean_removal_rate", "clean_removed", "poison_removed", "precision", "auroc", "auprc"):
                flat[metric + "_mean"] = row[metric]["mean"]
                flat[metric + "_ci95_low"] = row[metric]["ci95"][0]
                flat[metric + "_ci95_high"] = row[metric]["ci95"][1]
            csv_rows.append(flat)
        with open(CSVOUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(csv_rows)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--audit", action="store_true", help="run all frozen integrity gates and write them into the merged artifact")
    ap.add_argument("--allow-incomplete-merge", action="store_true")
    ap.add_argument("--population", choices=list(POPULATIONS) + ["all"], default="all")
    ap.add_argument("--seed", choices=[str(x) for x in SEEDS] + ["all"], default="all")
    ap.add_argument("--family", choices=list(FAMILIES) + ["all"], default="all")
    ap.add_argument("--representation", choices=list(REPRESENTATIONS) + ["all"], default="all")
    args = ap.parse_args()
    if args.prepare_only:
        prepare(); return
    design = load_locked()
    if args.merge or args.audit:
        integrity = None
        if args.audit:
            records = [json.load(open(p)) for p in sorted(CELLS.glob("*.json"))]
            integrity = validate_integrity(design, records)
            if not integrity["passed"]:
                raise RuntimeError("integrity gates failed: " + "; ".join(integrity["failures"]))
        merge(design, require_complete=not args.allow_incomplete_merge, integrity=integrity); print(OUT); return
    populations = list(POPULATIONS) if args.population == "all" else [args.population]
    seeds = list(SEEDS) if args.seed == "all" else [int(args.seed)]
    reps = list(REPRESENTATIONS) if args.representation == "all" else [args.representation]
    for population in populations:
        families = list(POPULATIONS[population]["families"])
        if args.family != "all": families = [args.family] if args.family in families else []
        for seed in seeds:
            for family in families:
                for rep in reps: run_cell(design, population, seed, family, rep)
    merge(design, require_complete=False)


if __name__ == "__main__":
    main()
