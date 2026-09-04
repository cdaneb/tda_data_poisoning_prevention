"""Preregistered downstream-classifier extension of the clean-novelty study.

This program deliberately separates design finalization from outcome
generation.  ``--prepare-only`` freezes literal row selections and corrected
malicious-only attack realizations, but it cannot extract TDA features, fit an
anomaly detector, or train a classifier.  Outcome modes require an external
registration receipt whose hash matches the frozen manifest.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import gtda
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from programs.adversarial_attack import payload_label_to_binary as label_to_binary
from programs.data_loader import LABEL_COLUMN, PAYLOAD_COLUMNS
from programs.monkam_representation import stable_hash
from programs.novelty_detectors import HigherIsAnomaly, clean_threshold
from programs.paths import DATA_DIR
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs.phase_q_pipeline import (
    CONTROL_THRESHOLD,
    THRESHOLD_STACK,
    extract_multithreshold_features,
)
from programs.resource_usage import peak_rss_kib


SEEDS = (2026, 2027, 2028, 2029, 2030)
FAMILIES = tuple(SUPPORTED_FAMILIES)
REPRESENTATIONS = ("control", "stack")
BUDGETS = (0.05, 0.01)
WORKERS = 8
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260904
TEST_SELECTION_SEED_BASE = 2_000_000
TEST_ATTACK_SEED_BASE = 1_000_000
TEST_CANDIDATE_FACTOR = 5

RESULTS = ROOT / "results"
DESIGN = RESULTS / "downstream_classifier_design.json"
PRE = RESULTS / "downstream_classifier_preregistration.json"
RECEIPT = RESULTS / "downstream_classifier_registration_receipt.json"
CELLS = RESULTS / "downstream_classifier_cells"
OUT = RESULTS / "downstream_classifier_results.json"
CSVOUT = RESULTS / "downstream_classifier_summary.csv"
CELLCSV = RESULTS / "downstream_classifier_cell_metrics.csv"
REPORT = ROOT / "docs" / "DOWNSTREAM_CLASSIFIER_RESULTS.md"
CACHE = ROOT / ".downstream_cache"
CONFIRMATION_CACHE = ROOT / ".confirmation_cache"
CONFIRMATION_PRE = RESULTS / "clean_novelty_confirmation_preregistration.json"

DATASET_FILENAMES = {
    "unsw": "Payload_data_UNSW.csv",
    "cicids": "Payload_data_CICIDS2017.csv",
}
POPULATIONS = {
    "unsw_matched": {
        "dataset": "unsw", "clean": 5000, "poison": 500,
        "families": list(FAMILIES),
    },
    "cicids_matched": {
        "dataset": "cicids", "clean": 5000, "poison": 500,
        "families": list(FAMILIES),
    },
    "cicids_scale": {
        "dataset": "cicids", "clean": 50000, "poison": 5000,
        "families": ["transpositions"],
    },
}
POPULATION_OFFSETS = {
    "unsw_matched": 10_000,
    "cicids_matched": 20_000,
    "cicids_scale": 30_000,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def software_versions():
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "giotto_tda": gtda.__version__,
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_list(values):
    return stable_hash([int(value) for value in values])


def content_hash(document):
    body = copy.deepcopy(document)
    body.pop("content_hash", None)
    return stable_hash(body)


def atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_hashed_json(path):
    document = json.loads(path.read_text())
    if document.get("content_hash") != content_hash(document):
        raise RuntimeError(f"content hash mismatch: {path}")
    return document


def raw_hashes(X):
    return [
        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
        for row in X
    ]


def row_count(path):
    with open(path, "rb") as stream:
        return sum(
            block.count(b"\n")
            for block in iter(lambda: stream.read(16 << 20), b"")
        ) - 1


def selected_rows(path, wanted):
    """Read a union of literal zero-based data-row indices in one CSV pass."""
    wanted = np.asarray(sorted(set(map(int, wanted))), dtype=np.int64)
    found = {}
    offset = 0
    for chunk in pd.read_csv(
        path, usecols=PAYLOAD_COLUMNS + [LABEL_COLUMN], chunksize=10_000
    ):
        stop = offset + len(chunk)
        lower, upper = np.searchsorted(wanted, [offset, stop])
        global_indices = wanted[lower:upper]
        if len(global_indices):
            local_indices = global_indices - offset
            selected = chunk.iloc[local_indices]
            payloads = selected[PAYLOAD_COLUMNS].to_numpy(dtype=np.uint8)
            labels = selected[LABEL_COLUMN].to_numpy()
            found.update({
                int(global_index): (payloads[position].copy(), labels[position])
                for position, global_index in enumerate(global_indices)
            })
        offset = stop
    if len(found) != len(wanted):
        raise RuntimeError(
            f"materialized {len(found)} of {len(wanted)} requested rows from {path}"
        )
    return found


def design_content():
    """Return the version-controlled design, with no data-dependent fields."""
    return {
        "experiment": "preregistered_downstream_classifier_extension",
        "design_version": 1,
        "prospective_status": {
            "downstream_outcomes": "not examined before external registration",
            "prior_information": (
                "The completed clean-novelty confirmation results are known and "
                "motivated this extension; this is not an independent replication."
            ),
        },
        "protocol_correction": {
            "issue": (
                "The prior binary-label helper recognized UNSW 'normal' but not "
                "CICIDS 'BENIGN', so prior CICIDS poison targets were not guaranteed "
                "to be malicious."
            ),
            "locked_fix": (
                "After strip/lower normalization, 'normal' and 'benign' map to 0; "
                "every other label maps to 1. New CICIDS attacks are generated "
                "prospectively and no prior CICIDS attack outcome is reused."
            ),
            "unsw_reproduction_required": True,
            "cicids_difference_from_prior_required": True,
        },
        "seeds": list(SEEDS),
        "populations": copy.deepcopy(POPULATIONS),
        "training_partition": {
            "source": str(CONFIRMATION_PRE.relative_to(ROOT)).replace("\\", "/"),
            "reuse": "literal clean rows and exact raw-hash-disjoint 60/20/20 splits",
            "trusted_core": "detector-training plus calibration clean (80 percent)",
            "suspect_batch": "held-out clean (20 percent) plus appended poison",
            "filter_scope": "suspect batch only; trusted core is never filtered",
        },
        "external_test": {
            "size": "equal to the clean training population",
            "selection": (
                "fixed seeded row-random candidate order; take the first rows whose "
                "exact raw-payload SHA-256 is absent from every clean or poisoned "
                "training arm"
            ),
            "candidate_factor": TEST_CANDIDATE_FACTOR,
            "selection_seed": f"{TEST_SELECTION_SEED_BASE} + cell seed + population offset",
            "locked_secondary_stress_test": (
                "same-family transformed copies of 10 percent of test rows, selected "
                "only from truly malicious common-attackable examples"
            ),
            "attack_seed": f"{TEST_ATTACK_SEED_BASE} + cell seed",
        },
        "poisoning": {
            "rate": 0.10,
            "target": "truly malicious common-attackable rows only",
            "observed_training_label": 0,
            "true_label": 1,
            "threat_model": "dirty-label malicious-to-benign injection",
            "claim_boundary": (
                "Byte permutations preserve the byte multiset; network or application "
                "functionality is not claimed."
            ),
            "families_and_parameters": {
                family: kwargs | {"poison_rate": 0.10}
                for family, (_, kwargs) in SUPPORTED_FAMILIES.items()
            },
        },
        "representations": {
            "control": {"thresholds": [CONTROL_THRESHOLD], "features": 60},
            "stack": {
                "thresholds": list(THRESHOLD_STACK),
                "features": 540,
                "scale": "1/sqrt(9)",
            },
        },
        "sanitizer": {
            "preprocessing": "StandardScaler fit on detector-training trusted clean only",
            "detector": {
                "class": "IsolationForest",
                "n_estimators": 200,
                "random_state": "cell seed",
                "n_jobs": WORKERS,
                "other_parameters": "scikit-learn defaults",
            },
            "threshold": "higher empirical quantile of calibration-clean anomaly scores",
            "removal_rule": "score >= threshold",
            "budgets": list(BUDGETS),
            "primary_budget": 0.05,
            "secondary_budget": 0.01,
        },
        "arms": {
            "clean": "all sampled clean rows, correctly labeled",
            "poisoned": "all clean rows plus malicious poisons observed as benign",
            "random_cost_matched": (
                "fixed random suspect ordering; shortest prefix removing exactly the "
                "same number of unmodified suspect rows as its paired sanitizer"
            ),
            "if60": "Isolation Forest on the 60-feature control representation",
            "if540": "Isolation Forest on the 540-feature stack representation",
            "oracle": "all poison removed and all clean retained; identical to clean",
        },
        "classifier": {
            "input": "raw 1500-byte payload vector",
            "task": "benign=0 versus malicious=1",
            "class": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": "cell seed",
            "n_jobs": WORKERS,
            "other_parameters": "scikit-learn defaults",
            "preprocessing": "none",
        },
        "endpoints": {
            "primary": "malicious recall on the identity-safe unmodified test set",
            "guardrail": "benign false-positive rate on that test set",
            "secondary": [
                "balanced accuracy", "macro F1", "accuracy", "AUROC", "AUPRC",
                "attacked-malicious recall", "attacked-malicious mean probability",
                "training size", "poison retained", "per-class clean removal",
            ],
        },
        "primary_analysis": {
            "population": "unsw_matched",
            "representation": "control",
            "budget": 0.05,
            "gates": [
                "clean minus poisoned recall mean >= 0.02 and CI lower > 0",
                "IF60 minus poisoned recall mean >= 0.01 and CI lower > 0",
                "IF60 minus random-cost-matched recall mean > 0 and CI lower > 0",
                "IF60 minus clean benign-FPR CI upper <= 0.01",
            ],
            "decision": "all four gates are required for an operational-benefit claim",
            "non_rescue": (
                "IF540, the 1 percent budget, CICIDS populations, secondary endpoints, "
                "and the attacked-test stress test cannot rescue the primary claim."
            ),
            "recovery_fraction": (
                "reported only when clean minus poisoned malicious recall is positive"
            ),
        },
        "inference": {
            "bootstrap": "hierarchical paired nonparametric bootstrap",
            "hierarchy": "resample five seeds, then attack families within each seed",
            "repetitions": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95 percent",
            "estimand": "equal-weight mean over seeds and families",
        },
        "execution_policy": {
            "prepare_only_forbidden_operations": [
                "TDA feature extraction", "sanitizer fitting", "classifier fitting",
                "downstream metric computation",
            ],
            "outcomes_require": (
                "matching external OSF registration receipt and registered code commit"
            ),
            "deviations": "report all deviations; do not retune or selectively rerun cells",
            "parallelism": (
                "TDA extraction is sequential because its internal pipelines use all cores; "
                "classifier and Isolation Forest internals use eight workers."
            ),
            "feature_reuse": (
                "A prior UNSW TDA feature cache may be reused only after its input, "
                "raw, poison-mask, 60-feature, and 540-feature hashes reproduce. "
                "Prior CICIDS feature caches are forbidden."
            ),
        },
    }


def write_design():
    document = design_content()
    document["content_hash"] = content_hash(document)
    if DESIGN.exists():
        existing = load_hashed_json(DESIGN)
        if existing != document:
            raise RuntimeError(f"refusing to overwrite a different design: {DESIGN}")
        print(f"Design already current: {DESIGN}")
        return
    atomic_json(DESIGN, document)
    print(f"Wrote design: {DESIGN}")
    print(f"Design content hash: {document['content_hash']}")


def load_design():
    if not DESIGN.exists():
        raise RuntimeError(f"missing static design; run --write-design: {DESIGN}")
    design = load_hashed_json(DESIGN)
    expected = design_content()
    expected["content_hash"] = content_hash(expected)
    if design != expected:
        raise RuntimeError("static design does not match the executable protocol")
    return design


def load_confirmation():
    confirmation = load_hashed_json(CONFIRMATION_PRE)
    if confirmation.get("experiment") != "clean_novelty_independent_confirmation":
        raise RuntimeError("unexpected parent confirmation manifest")
    if tuple(confirmation["seeds"]) != SEEDS:
        raise RuntimeError("parent confirmation seeds no longer match")
    return confirmation


def resolve_datasets(confirmation):
    resolved = {}
    for name, filename in DATASET_FILENAMES.items():
        candidates = [
            DATA_DIR / filename,
            Path(confirmation["inputs"][name]["path"]),
            ROOT / "data" / filename,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"cannot locate {name}; set TDA_DATA_DIR or restore the WIRE mount"
            )
        expected = confirmation["inputs"][name]
        actual_rows = row_count(path)
        actual_sha = sha256_file(path)
        if actual_rows != expected["rows"] or actual_sha != expected["sha256"]:
            raise RuntimeError(
                f"{name} dataset identity mismatch: rows={actual_rows}, sha256={actual_sha}"
            )
        resolved[name] = path
    return resolved


def raw_cache_path(population, seed):
    return CACHE / "prepared" / f"{population}_seed{seed}.npz"


def feature_cache_path(population, seed, family):
    return CACHE / "features" / f"{population}_seed{seed}_{family}.npz"


def test_selection_candidates(total_rows, population, seed, sample_size):
    count = min(total_rows, TEST_CANDIDATE_FACTOR * sample_size)
    random = np.random.default_rng(
        TEST_SELECTION_SEED_BASE + POPULATION_OFFSETS[population] + seed
    )
    return random.choice(total_rows, size=count, replace=False).astype(int).tolist()


def attack_realization(X, y, family, seed):
    function, kwargs = SUPPORTED_FAMILIES[family]
    combined, _, poison_mask, log = function(
        X, y, poison_rate=0.10, random_state=seed,
        label_mapper=label_to_binary, **kwargs
    )
    parents = np.asarray([entry["target_index"] for entry in log], dtype=int)
    if len(parents) != int(0.10 * len(X)):
        raise AssertionError("attack did not produce the locked poison count")
    if not np.all(label_to_binary(y[parents]) == 1):
        raise AssertionError("attack selected a non-malicious parent")
    if any(entry["raw_noop"] for entry in log):
        raise AssertionError("attack produced a raw no-op")
    return combined, poison_mask, log, parents


def attack_manifest(X, y, family, seed, sampled_dataset_rows=None):
    combined, poison_mask, log, parents = attack_realization(X, y, family, seed)
    result = {
        "attack_seed": int(seed),
        "function": SUPPORTED_FAMILIES[family][0].__name__,
        "configuration": SUPPORTED_FAMILIES[family][1] | {"poison_rate": 0.10},
        "binary_label_mapper": "payload_label_to_binary_v1",
        "poison_source_parent_indices": parents.astype(int).tolist(),
        "poison_source_parent_indices_hash": hash_list(parents),
        "poison_source_labels_hash": stable_hash(y[parents].tolist()),
        "poison_source_binary_labels_hash": stable_hash(label_to_binary(y[parents])),
        "attack_log": log,
        "attack_log_hash": stable_hash(log),
        "combined_raw_hash": stable_hash(combined),
        "poison_mask_hash": stable_hash(poison_mask),
        "poison_raw_hash": stable_hash(combined[poison_mask]),
    }
    if sampled_dataset_rows is not None:
        parent_rows = [int(sampled_dataset_rows[index]) for index in parents]
        result["poison_source_parent_dataset_row_indices"] = parent_rows
        result["poison_source_parent_dataset_row_indices_hash"] = hash_list(parent_rows)
    return result


def materialize_matrix(materialized, rows):
    X = np.vstack([materialized[int(index)][0] for index in rows])
    y = np.asarray([materialized[int(index)][1] for index in rows])
    return X, y


def prepare():
    """Freeze inputs only.  This function contains no feature/model code."""
    if PRE.exists():
        raise RuntimeError(f"refusing to rewrite frozen preregistration: {PRE}")
    forbidden = [RECEIPT, OUT, CSVOUT, CELLCSV, REPORT]
    forbidden.extend(CELLS.glob("*.json") if CELLS.exists() else [])
    feature_directory = CACHE / "features"
    forbidden.extend(feature_directory.glob("*") if feature_directory.exists() else [])
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "prepare-only refuses pre-existing registration/outcome artifacts: "
            + ", ".join(existing)
        )
    design = load_design()
    confirmation = load_confirmation()
    datasets = resolve_datasets(confirmation)

    candidates = {}
    wanted_by_dataset = defaultdict(list)
    for population, specification in POPULATIONS.items():
        dataset = specification["dataset"]
        total_rows = confirmation["inputs"][dataset]["rows"]
        for seed in SEEDS:
            locked = confirmation["realizations"][population][str(seed)]
            wanted_by_dataset[dataset].extend(locked["sampled_dataset_row_indices"])
            key = (population, seed)
            candidates[key] = test_selection_candidates(
                total_rows, population, seed, specification["clean"]
            )
            wanted_by_dataset[dataset].extend(candidates[key])

    materialized = {
        name: selected_rows(datasets[name], rows)
        for name, rows in wanted_by_dataset.items()
    }
    realizations = {}
    correction_summary = Counter()
    for population, specification in POPULATIONS.items():
        dataset = specification["dataset"]
        realizations[population] = {}
        for seed in SEEDS:
            prior = confirmation["realizations"][population][str(seed)]
            sampled_rows = prior["sampled_dataset_row_indices"]
            X, y = materialize_matrix(materialized[dataset], sampled_rows)
            if stable_hash(X) != prior["prepared_raw_hash"]:
                raise AssertionError("parent clean matrix does not reproduce")
            if stable_hash(y.tolist()) != prior["prepared_labels_hash"]:
                raise AssertionError("parent clean labels do not reproduce")

            training_hashes = set(raw_hashes(X))
            all_training_hashes = set(training_hashes)
            family_training_hashes = {}
            family_manifests = {}
            for family in specification["families"]:
                training = attack_manifest(X, y, family, seed, sampled_rows)
                training_combined, _, _, _ = attack_realization(X, y, family, seed)
                family_training_hashes[family] = set(raw_hashes(training_combined))
                all_training_hashes.update(family_training_hashes[family])
                legacy = prior["families"][family]
                matches_legacy = all(
                    training[key] == legacy[key]
                    for key in ("combined_raw_hash", "poison_mask_hash", "attack_log_hash")
                )
                if dataset == "unsw" and not matches_legacy:
                    raise AssertionError("UNSW attack changed under the label correction")
                if dataset == "cicids" and matches_legacy:
                    raise AssertionError("corrected CICIDS attack unexpectedly matches legacy attack")
                training["legacy_confirmation_attack_match"] = bool(matches_legacy)
                correction_summary[f"{dataset}_matches_legacy_{matches_legacy}"] += 1
                family_manifests[family] = training

            chosen_test_rows = []
            chosen_test_X = []
            chosen_test_y = []
            for candidate in candidates[(population, seed)]:
                row, label = materialized[dataset][candidate]
                row_hash = hashlib.sha256(
                    np.ascontiguousarray(row).view(np.uint8)
                ).hexdigest()
                if row_hash in all_training_hashes:
                    continue
                chosen_test_rows.append(int(candidate))
                chosen_test_X.append(row)
                chosen_test_y.append(label)
                if len(chosen_test_rows) == specification["clean"]:
                    break
            if len(chosen_test_rows) != specification["clean"]:
                raise RuntimeError(
                    f"candidate factor {TEST_CANDIDATE_FACTOR} was insufficient for "
                    f"{population} seed {seed}; change requires a protocol revision"
                )
            X_test = np.vstack(chosen_test_X)
            y_test = np.asarray(chosen_test_y)
            binary_test = label_to_binary(y_test)
            if set(np.unique(binary_test)) != {0, 1}:
                raise AssertionError("external test set must contain both classes")
            unmodified_test_hashes = set(raw_hashes(X_test))
            if all_training_hashes & unmodified_test_hashes:
                raise AssertionError("external test payload identity overlaps a training arm")

            atomic_npz(raw_cache_path(population, seed), X=X, y=y, X_test=X_test, y_test=y_test)
            test_family_manifests = {}
            for family in specification["families"]:
                test_seed = TEST_ATTACK_SEED_BASE + seed
                test_attack = attack_manifest(
                    X_test, y_test, family, test_seed, chosen_test_rows
                )
                X_test_combined, test_poison_mask, _, _ = attack_realization(
                    X_test, y_test, family, test_seed
                )
                attacked = X_test_combined[test_poison_mask]
                attacked_hashes = set(raw_hashes(attacked))
                if attacked_hashes & family_training_hashes[family]:
                    raise AssertionError("attacked test payload identity overlaps its training arm")
                if attacked_hashes & unmodified_test_hashes:
                    raise AssertionError("attacked test payload identity overlaps unmodified test")
                test_attack["attacked_test_raw_hashes_hash"] = stable_hash(sorted(attacked_hashes))
                test_family_manifests[family] = test_attack

            realizations[population][str(seed)] = {
                "sampled_dataset_row_indices": sampled_rows,
                "sampled_dataset_row_indices_hash": prior["sampled_dataset_row_indices_hash"],
                "prepared_raw_hash": prior["prepared_raw_hash"],
                "prepared_labels_hash": prior["prepared_labels_hash"],
                "raw_payload_group_hashes_hash": prior["raw_payload_group_hashes_hash"],
                "clean_training_indices": prior["clean_training_indices"],
                "clean_training_indices_hash": prior["clean_training_indices_hash"],
                "calibration_indices": prior["calibration_indices"],
                "calibration_indices_hash": prior["calibration_indices_hash"],
                "heldout_clean_evaluation_indices": prior["heldout_clean_evaluation_indices"],
                "heldout_clean_evaluation_indices_hash": prior["heldout_clean_evaluation_indices_hash"],
                "training_attacks": family_manifests,
                "test": {
                    "selection_seed": TEST_SELECTION_SEED_BASE + POPULATION_OFFSETS[population] + seed,
                    "candidate_count": len(candidates[(population, seed)]),
                    "dataset_row_indices": chosen_test_rows,
                    "dataset_row_indices_hash": hash_list(chosen_test_rows),
                    "raw_hash": stable_hash(X_test),
                    "labels_hash": stable_hash(y_test.tolist()),
                    "binary_label_counts": {
                        "benign": int(np.sum(binary_test == 0)),
                        "malicious": int(np.sum(binary_test == 1)),
                    },
                    "unique_raw_payload_groups": len(unmodified_test_hashes),
                    "raw_payload_group_hashes_hash": stable_hash(raw_hashes(X_test)),
                    "attacks": test_family_manifests,
                },
                "prepared_cache": str(raw_cache_path(population, seed).relative_to(ROOT)).replace("\\", "/"),
            }

    preregistration = {
        "experiment": "preregistered_downstream_classifier_extension",
        "frozen_before_outcomes": True,
        "prepared_at_utc": utc_now(),
        "design_hash": design["content_hash"],
        "parent_confirmation_hash": confirmation["content_hash"],
        "inputs": copy.deepcopy(confirmation["inputs"]),
        "software_at_preparation": software_versions(),
        "platform_at_preparation": platform.platform(),
        "protocol_correction_audit": dict(sorted(correction_summary.items())),
        "realizations": realizations,
    }
    preregistration["content_hash"] = content_hash(preregistration)
    atomic_json(PRE, preregistration)
    print(f"Wrote frozen preregistration: {PRE}")
    print(f"Preregistration content hash: {preregistration['content_hash']}")
    print("No TDA feature, sanitizer, classifier, or downstream outcome was computed.")


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def record_registration(url, registered_at_utc, registered_code_commit, visibility):
    if RECEIPT.exists():
        raise RuntimeError(f"refusing to rewrite registration receipt: {RECEIPT}")
    preregistration = load_hashed_json(PRE)
    if not re.fullmatch(r"https://osf\.io/[A-Za-z0-9]+/?", url):
        raise ValueError("registration URL must be a direct https://osf.io/<id>/ URL")
    parse_timestamp(registered_at_utc)
    if not re.fullmatch(r"[0-9a-f]{40}", registered_code_commit):
        raise ValueError("registered code commit must be a full 40-character Git SHA")
    validate_registered_commit(registered_code_commit)
    receipt = {
        "experiment": preregistration["experiment"],
        "execution_authorized": True,
        "external_registry": "OSF Registries",
        "registration_url": url,
        "visibility": visibility,
        "registered_at_utc": registered_at_utc,
        "registered_code_commit": registered_code_commit,
        "preregistration_content_hash": preregistration["content_hash"],
        "design_hash": preregistration["design_hash"],
        "receipt_recorded_at_utc": utc_now(),
    }
    receipt["content_hash"] = content_hash(receipt)
    atomic_json(RECEIPT, receipt)
    print(f"Wrote registration receipt: {RECEIPT}")


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def validate_execution_lock():
    design = load_design()
    preregistration = load_hashed_json(PRE)
    receipt = load_hashed_json(RECEIPT)
    if preregistration["design_hash"] != design["content_hash"]:
        raise RuntimeError("preregistration does not match the executable design")
    if receipt.get("execution_authorized") is not True:
        raise RuntimeError("registration receipt does not authorize execution")
    if receipt.get("preregistration_content_hash") != preregistration["content_hash"]:
        raise RuntimeError("receipt does not match the frozen preregistration")
    if receipt.get("design_hash") != design["content_hash"]:
        raise RuntimeError("receipt design hash mismatch")
    if preregistration.get("software_at_preparation") != software_versions():
        raise RuntimeError("execution software differs from the prepared environment")
    if not re.fullmatch(r"https://osf\.io/[A-Za-z0-9]+/?", receipt["registration_url"]):
        raise RuntimeError("receipt does not contain a direct OSF registration URL")
    parse_timestamp(receipt["registered_at_utc"])
    validate_registered_commit(receipt["registered_code_commit"])
    return design, preregistration, receipt


def validate_registered_commit(commit):
    """Require current protocol files to be byte-identical to the OSF commit."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("registered code commit is not a full Git SHA")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT, check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "registered code commit is missing or is not an ancestor of HEAD"
        ) from error
    locked_paths = (
        "programs/run_downstream_classifier_preregistered.py",
        "programs/adversarial_attack.py",
        "programs/phase_q_attacks.py",
        "programs/phase_q_pipeline.py",
        "programs/novelty_detectors.py",
        "programs/monkam_representation.py",
        "programs/tda_pipeline.py",
        "programs/data_loader.py",
        "requirements.lock.txt",
        "results/downstream_classifier_design.json",
        "results/downstream_classifier_preregistration.json",
    )
    for relative in locked_paths:
        try:
            registered = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"], cwd=ROOT
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"registered commit does not contain required file: {relative}"
            ) from error
        current = ROOT.joinpath(relative).read_bytes()
        if current != registered:
            raise RuntimeError(
                f"current {relative} differs from registered commit; do not run outcomes"
            )


def ensure_raw_caches(preregistration, datasets):
    missing = [
        (population, seed)
        for population in POPULATIONS
        for seed in SEEDS
        if not raw_cache_path(population, seed).exists()
    ]
    if missing:
        wanted_by_dataset = defaultdict(list)
        for population, seed in missing:
            realization = preregistration["realizations"][population][str(seed)]
            dataset = POPULATIONS[population]["dataset"]
            wanted_by_dataset[dataset].extend(realization["sampled_dataset_row_indices"])
            wanted_by_dataset[dataset].extend(realization["test"]["dataset_row_indices"])
        materialized = {
            name: selected_rows(datasets[name], rows)
            for name, rows in wanted_by_dataset.items()
        }
        for population, seed in missing:
            realization = preregistration["realizations"][population][str(seed)]
            dataset = POPULATIONS[population]["dataset"]
            X, y = materialize_matrix(
                materialized[dataset], realization["sampled_dataset_row_indices"]
            )
            X_test, y_test = materialize_matrix(
                materialized[dataset], realization["test"]["dataset_row_indices"]
            )
            atomic_npz(raw_cache_path(population, seed), X=X, y=y, X_test=X_test, y_test=y_test)

    for population in POPULATIONS:
        for seed in SEEDS:
            realization = preregistration["realizations"][population][str(seed)]
            with np.load(raw_cache_path(population, seed), allow_pickle=True) as cache:
                checks = {
                    "prepared raw": stable_hash(cache["X"]) == realization["prepared_raw_hash"],
                    "prepared labels": stable_hash(cache["y"].tolist()) == realization["prepared_labels_hash"],
                    "test raw": stable_hash(cache["X_test"]) == realization["test"]["raw_hash"],
                    "test labels": stable_hash(cache["y_test"].tolist()) == realization["test"]["labels_hash"],
                }
            if not all(checks.values()):
                raise RuntimeError(f"raw cache integrity failure {population} seed {seed}: {checks}")


def reproduce_attack(X, y, family, locked):
    combined, poison_mask, log, parents = attack_realization(
        X, y, family, locked["attack_seed"]
    )
    checks = {
        "combined_raw_hash": stable_hash(combined) == locked["combined_raw_hash"],
        "poison_mask_hash": stable_hash(poison_mask) == locked["poison_mask_hash"],
        "attack_log_hash": stable_hash(log) == locked["attack_log_hash"],
        "parents": parents.astype(int).tolist() == locked["poison_source_parent_indices"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"attack reproduction failure: {checks}")
    return combined, poison_mask


def make_features(preregistration, population, seed, family):
    path = feature_cache_path(population, seed, family)
    metadata_path = path.with_suffix(".json")
    realization = preregistration["realizations"][population][str(seed)]
    locked_attack = realization["training_attacks"][family]
    with np.load(raw_cache_path(population, seed), allow_pickle=True) as cache:
        X, y = cache["X"], cache["y"]
    combined, poison_mask = reproduce_attack(X, y, family, locked_attack)
    input_hash = stable_hash({
        "raw": stable_hash(combined),
        "attack": locked_attack["attack_log_hash"],
        "representations": design_content()["representations"],
    })
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        with np.load(path) as cache:
            if (
                metadata.get("input_hash") == input_hash
                and metadata.get("control_hash") == stable_hash(cache["control"])
                and metadata.get("stack_hash") == stable_hash(cache["stack"])
            ):
                return cache["control"].copy(), cache["stack"].copy(), metadata
        raise RuntimeError(f"stale or corrupt feature cache: {path}")

    if POPULATIONS[population]["dataset"] == "unsw":
        prior_path = (
            CONFIRMATION_CACHE / "features"
            / f"{population}_{family}_seed{seed}.npz"
        )
        prior_metadata_path = prior_path.with_suffix(".json")
        if prior_path.exists() != prior_metadata_path.exists():
            raise RuntimeError("parent confirmation feature cache is incomplete")
        if prior_path.exists():
            confirmation = load_confirmation()
            expected_parent_input_hash = stable_hash({
                "raw": stable_hash(combined),
                "attack": locked_attack["attack_log_hash"],
                "representation": confirmation["representations"],
            })
            prior_metadata = json.loads(prior_metadata_path.read_text())
            with np.load(prior_path) as prior_cache:
                control = prior_cache["control"].copy()
                stack = prior_cache["stack"].copy()
            checks = {
                "input": prior_metadata.get("input_hash") == expected_parent_input_hash,
                "raw": prior_metadata.get("raw_hash") == stable_hash(combined),
                "poison_mask": prior_metadata.get("poison_mask_hash") == stable_hash(poison_mask),
                "control": prior_metadata.get("control_hash") == stable_hash(control),
                "stack": prior_metadata.get("stack_hash") == stable_hash(stack),
                "control_shape": control.shape == (len(combined), 60),
                "stack_shape": stack.shape == (len(combined), 540),
                "finite": bool(np.isfinite(control).all() and np.isfinite(stack).all()),
            }
            if not all(checks.values()):
                raise RuntimeError(f"parent UNSW feature cache failed reuse checks: {checks}")
            metadata = {
                "input_hash": input_hash,
                "control_hash": stable_hash(control),
                "stack_hash": stable_hash(stack),
                "runtime_seconds": 0.0,
                "peak_rss_kib": peak_rss_kib(),
                "source": "verified_parent_confirmation_cache",
                "parent_input_hash": expected_parent_input_hash,
            }
            atomic_npz(path, control=control, stack=stack)
            atomic_json(metadata_path, metadata)
            return control, stack, metadata

    started = time.time()
    stack, blocks, _ = extract_multithreshold_features(combined, return_blocks=True)
    control = np.asarray(blocks[CONTROL_THRESHOLD])
    if control.shape != (len(combined), 60) or stack.shape != (len(combined), 540):
        raise AssertionError("unexpected TDA feature shape")
    if not np.isfinite(control).all() or not np.isfinite(stack).all():
        raise ValueError("TDA extraction produced non-finite features")
    metadata = {
        "input_hash": input_hash,
        "control_hash": stable_hash(control),
        "stack_hash": stable_hash(stack),
        "runtime_seconds": time.time() - started,
        "peak_rss_kib": peak_rss_kib(),
        "source": "fresh_registered_extraction",
    }
    atomic_npz(path, control=control, stack=stack)
    atomic_json(metadata_path, metadata)
    return control, stack, metadata


def sanitizer_removal(features, realization, seed, budget, poison_count):
    train = np.asarray(realization["clean_training_indices"], dtype=int)
    calibration = np.asarray(realization["calibration_indices"], dtype=int)
    evaluation = np.asarray(realization["heldout_clean_evaluation_indices"], dtype=int)
    clean_count = len(features) - int(poison_count)
    poison = np.arange(clean_count, len(features), dtype=int)
    suspect = np.concatenate([evaluation, poison])

    scaler = StandardScaler().fit(features[train])
    detector = HigherIsAnomaly(
        IsolationForest(n_estimators=200, random_state=seed, n_jobs=WORKERS)
    ).fit(scaler.transform(features[train]))
    calibration_scores = detector.score_samples(scaler.transform(features[calibration]))
    suspect_scores = detector.score_samples(scaler.transform(features[suspect]))
    threshold = clean_threshold(calibration_scores, budget)
    removal = np.asarray(suspect_scores >= threshold, dtype=bool)
    return suspect, removal, {
        "threshold": float(threshold),
        "calibration_scores_hash": stable_hash(calibration_scores),
        "suspect_scores_hash": stable_hash(suspect_scores),
        "clean_removed": int(removal[:len(evaluation)].sum()),
        "poison_removed": int(removal[len(evaluation):].sum()),
        "clean_evaluation_count": int(len(evaluation)),
        "poison_count": int(len(poison)),
    }
def random_order(population, seed, family, size):
    token = f"{population}|{seed}|{family}|random-cost-matched-v1"
    random_seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(random_seed).permutation(size)


def random_cost_removal(order, clean_suspect_count, target_clean_removed):
    removal = np.zeros(len(order), dtype=bool)
    if target_clean_removed == 0:
        return removal
    clean_removed = 0
    for index in order:
        removal[index] = True
        if index < clean_suspect_count:
            clean_removed += 1
            if clean_removed == target_clean_removed:
                return removal
    raise AssertionError("random ordering could not match sanitizer clean cost")


def classifier_metrics(model, X_test, y_test, X_attacked):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_.tolist()
    if classes != [0, 1]:
        raise AssertionError(f"classifier training lost a class: {classes}")
    malicious_probability = probabilities[:, 1]
    benign = y_test == 0
    malicious = y_test == 1
    attacked_predictions = model.predict(X_attacked)
    attacked_probabilities = model.predict_proba(X_attacked)[:, 1]
    return {
        "malicious_recall": float(np.mean(predictions[malicious] == 1)),
        "benign_false_positive_rate": float(np.mean(predictions[benign] == 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "auroc": float(roc_auc_score(y_test, malicious_probability)),
        "auprc": float(average_precision_score(y_test, malicious_probability)),
        "attacked_malicious_recall": float(np.mean(attacked_predictions == 1)),
        "attacked_malicious_mean_probability": float(np.mean(attacked_probabilities)),
        "prediction_hash": stable_hash(predictions),
        "probability_hash": stable_hash(probabilities),
        "attacked_prediction_hash": stable_hash(attacked_predictions),
        "attacked_probability_hash": stable_hash(attacked_probabilities),
    }


def fit_arm(X, observed_y, keep, X_test, y_test, X_attacked, seed, poison_mask):
    if keep.dtype != bool or len(keep) != len(X):
        raise ValueError("training keep mask is invalid")
    started = time.time()
    model = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=WORKERS
    ).fit(X[keep], observed_y[keep])
    result = classifier_metrics(model, X_test, y_test, X_attacked)
    result["training"] = {
        "rows": int(keep.sum()),
        "benign_labels": int(np.sum(observed_y[keep] == 0)),
        "malicious_labels": int(np.sum(observed_y[keep] == 1)),
        "poison_retained": int(np.sum(keep & poison_mask)),
        "raw_hash": stable_hash(X[keep]),
        "observed_labels_hash": stable_hash(observed_y[keep]),
        "keep_mask_hash": stable_hash(keep),
    }
    result["runtime_seconds"] = time.time() - started
    return result


def cell_path(population, seed, family):
    return CELLS / f"{population}_seed{seed}_{family}.json"


def run_cell(design, preregistration, receipt, population, seed, family, overwrite=False):
    path = cell_path(population, seed, family)
    if path.exists() and not overwrite:
        print(f"Skip complete cell: {path.name}")
        return
    specification = POPULATIONS[population]
    if family not in specification["families"]:
        raise ValueError(f"{family} is not in the {population} design")
    realization = preregistration["realizations"][population][str(seed)]
    with np.load(raw_cache_path(population, seed), allow_pickle=True) as cache:
        X, y, X_test, y_test = (
            cache["X"], cache["y"], cache["X_test"], cache["y_test"]
        )
    combined, poison_mask = reproduce_attack(
        X, y, family, realization["training_attacks"][family]
    )
    test_combined, test_poison_mask = reproduce_attack(
        X_test, y_test, family, realization["test"]["attacks"][family]
    )
    X_attacked = test_combined[test_poison_mask]
    binary_clean = label_to_binary(y)
    binary_test = label_to_binary(y_test)
    observed_y = np.concatenate([
        binary_clean,
        np.zeros(int(poison_mask.sum()), dtype=int),
    ])
    if not np.all(label_to_binary(y[[entry["target_index"] for entry in realization["training_attacks"][family]["attack_log"]]]) == 1):
        raise AssertionError("frozen training poison parents are not all malicious")

    started_at = utc_now()
    control, stack, feature_metadata = make_features(
        preregistration, population, seed, family
    )
    feature_sets = {"control": control, "stack": stack}
    all_keep = np.ones(len(combined), dtype=bool)
    clean_keep = np.ones(len(X), dtype=bool)
    clean_poison_mask = np.zeros(len(X), dtype=bool)
    outcomes = {
        "clean": fit_arm(
            X, binary_clean, clean_keep, X_test, binary_test, X_attacked,
            seed, clean_poison_mask,
        ),
        "poisoned": fit_arm(
            combined, observed_y, all_keep, X_test, binary_test, X_attacked,
            seed, poison_mask,
        ),
    }
    outcomes["oracle"] = copy.deepcopy(outcomes["clean"])
    sanitizers = {}
    evaluation = np.asarray(realization["heldout_clean_evaluation_indices"], dtype=int)
    order = random_order(
        population, seed, family, len(evaluation) + int(poison_mask.sum())
    )
    for representation in REPRESENTATIONS:
        for budget in BUDGETS:
            suspect, filter_removal, sanitizer = sanitizer_removal(
                feature_sets[representation], realization, seed, budget,
                int(poison_mask.sum()),
            )
            filter_keep = np.ones(len(combined), dtype=bool)
            filter_keep[suspect[filter_removal]] = False
            random_removal = random_cost_removal(
                order, len(evaluation), sanitizer["clean_removed"]
            )
            random_keep = np.ones(len(combined), dtype=bool)
            random_keep[suspect[random_removal]] = False
            budget_key = str(budget)
            filter_key = f"filter|{representation}|{budget_key}"
            random_key = f"random|{representation}|{budget_key}"
            sanitizer.update({
                "representation": representation,
                "budget": budget,
                "filter_removal_mask_hash": stable_hash(filter_removal),
                "random_removal_mask_hash": stable_hash(random_removal),
                "random_clean_removed": int(random_removal[:len(evaluation)].sum()),
                "random_poison_removed": int(random_removal[len(evaluation):].sum()),
                "random_order_hash": stable_hash(order),
                "filter_benign_clean_removed": int(np.sum(
                    filter_removal[:len(evaluation)] & (binary_clean[evaluation] == 0)
                )),
                "filter_malicious_clean_removed": int(np.sum(
                    filter_removal[:len(evaluation)] & (binary_clean[evaluation] == 1)
                )),
                "random_benign_clean_removed": int(np.sum(
                    random_removal[:len(evaluation)] & (binary_clean[evaluation] == 0)
                )),
                "random_malicious_clean_removed": int(np.sum(
                    random_removal[:len(evaluation)] & (binary_clean[evaluation] == 1)
                )),
            })
            if sanitizer["random_clean_removed"] != sanitizer["clean_removed"]:
                raise AssertionError("random arm failed exact clean-cost matching")
            sanitizers[f"{representation}|{budget_key}"] = sanitizer
            outcomes[filter_key] = fit_arm(
                combined, observed_y, filter_keep, X_test, binary_test,
                X_attacked, seed, poison_mask,
            )
            outcomes[random_key] = fit_arm(
                combined, observed_y, random_keep, X_test, binary_test,
                X_attacked, seed, poison_mask,
            )

    record = {
        "experiment": preregistration["experiment"],
        "population": population,
        "dataset": specification["dataset"],
        "seed": int(seed),
        "family": family,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "preregistration_hash": preregistration["content_hash"],
        "registration_receipt_hash": receipt["content_hash"],
        "registered_code_commit": receipt["registered_code_commit"],
        "execution_git_head": git_head(),
        "feature_metadata": feature_metadata,
        "test": {
            "rows": int(len(X_test)),
            "benign": int(np.sum(binary_test == 0)),
            "malicious": int(np.sum(binary_test == 1)),
            "attacked_malicious_rows": int(len(X_attacked)),
            "raw_hash": stable_hash(X_test),
            "labels_hash": stable_hash(binary_test),
            "attacked_raw_hash": stable_hash(X_attacked),
        },
        "sanitizers": sanitizers,
        "outcomes": outcomes,
        "peak_rss_kib": peak_rss_kib(),
    }
    record["content_hash"] = content_hash(record)
    atomic_json(path, record)
    print(f"Completed cell: {path.name}")


def expected_cells():
    return {
        (population, seed, family)
        for population, specification in POPULATIONS.items()
        for seed in SEEDS
        for family in specification["families"]
    }


def load_cells(preregistration, receipt, require_complete=True):
    records = [load_hashed_json(path) for path in sorted(CELLS.glob("*.json"))]
    keys = [(row["population"], row["seed"], row["family"]) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate result cells")
    expected = expected_cells()
    extras = set(keys) - expected
    missing = expected - set(keys)
    if extras or (require_complete and missing):
        raise RuntimeError(f"cell grid mismatch: missing={len(missing)} extras={len(extras)}")
    for row in records:
        if row["preregistration_hash"] != preregistration["content_hash"]:
            raise RuntimeError("cell preregistration hash mismatch")
        if row["registration_receipt_hash"] != receipt["content_hash"]:
            raise RuntimeError("cell registration receipt hash mismatch")
    return records


def hierarchical_bootstrap(records, left, right, metric, population):
    by_seed = defaultdict(list)
    for row in records:
        if row["population"] == population:
            difference = row["outcomes"][left][metric] - row["outcomes"][right][metric]
            by_seed[int(row["seed"])].append(float(difference))
    if set(by_seed) != set(SEEDS):
        raise RuntimeError("hierarchical bootstrap is missing a seed")
    family_counts = {len(values) for values in by_seed.values()}
    if len(family_counts) != 1:
        raise RuntimeError("hierarchical bootstrap family counts differ by seed")
    values = np.asarray([by_seed[seed] for seed in SEEDS], dtype=float)
    point = float(np.mean(values))
    random = np.random.default_rng(BOOTSTRAP_SEED)
    seed_draws = random.integers(
        0, len(SEEDS), size=(BOOTSTRAP_REPS, len(SEEDS), 1)
    )
    family_draws = random.integers(
        0, values.shape[1],
        size=(BOOTSTRAP_REPS, len(SEEDS), values.shape[1]),
    )
    draws = values[seed_draws, family_draws].mean(axis=(1, 2))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": point,
        "ci95": [float(lower), float(upper)],
        "differences_by_seed": {str(key): values for key, values in sorted(by_seed.items())},
    }


def merge(preregistration, receipt):
    records = load_cells(preregistration, receipt, require_complete=True)
    primary_filter = "filter|control|0.05"
    primary_random = "random|control|0.05"
    effects = {
        "attack_harm": hierarchical_bootstrap(
            records, "clean", "poisoned", "malicious_recall", "unsw_matched"
        ),
        "if60_recovery": hierarchical_bootstrap(
            records, primary_filter, "poisoned", "malicious_recall", "unsw_matched"
        ),
        "if60_vs_random": hierarchical_bootstrap(
            records, primary_filter, primary_random, "malicious_recall", "unsw_matched"
        ),
        "if60_fpr_inflation": hierarchical_bootstrap(
            records, primary_filter, "clean", "benign_false_positive_rate", "unsw_matched"
        ),
    }
    gates = {
        "attack_harm": (
            effects["attack_harm"]["mean"] >= 0.02
            and effects["attack_harm"]["ci95"][0] > 0
        ),
        "if60_recovery": (
            effects["if60_recovery"]["mean"] >= 0.01
            and effects["if60_recovery"]["ci95"][0] > 0
        ),
        "if60_vs_random": (
            effects["if60_vs_random"]["mean"] > 0
            and effects["if60_vs_random"]["ci95"][0] > 0
        ),
        "if60_fpr_guardrail": effects["if60_fpr_inflation"]["ci95"][1] <= 0.01,
    }
    harm = effects["attack_harm"]["mean"]
    recovery_fraction = (
        effects["if60_recovery"]["mean"] / harm if harm > 0 else None
    )
    secondary_effects = {}
    for population in POPULATIONS:
        for representation in REPRESENTATIONS:
            for budget in BUDGETS:
                suffix = f"{representation}|{budget}"
                filter_arm = f"filter|{suffix}"
                random_arm = f"random|{suffix}"
                secondary_effects[f"{population}|{suffix}"] = {
                    "filter_minus_poisoned_malicious_recall": hierarchical_bootstrap(
                        records, filter_arm, "poisoned", "malicious_recall", population
                    ),
                    "filter_minus_random_malicious_recall": hierarchical_bootstrap(
                        records, filter_arm, random_arm, "malicious_recall", population
                    ),
                    "filter_minus_clean_benign_fpr": hierarchical_bootstrap(
                        records, filter_arm, "clean", "benign_false_positive_rate", population
                    ),
                    "filter_minus_poisoned_attacked_recall": hierarchical_bootstrap(
                        records, filter_arm, "poisoned", "attacked_malicious_recall", population
                    ),
                }
        for budget in BUDGETS:
            stack_arm = f"filter|stack|{budget}"
            control_arm = f"filter|control|{budget}"
            secondary_effects[f"{population}|stack-minus-control|{budget}"] = {
                "malicious_recall": hierarchical_bootstrap(
                    records, stack_arm, control_arm, "malicious_recall", population
                ),
                "benign_false_positive_rate": hierarchical_bootstrap(
                    records, stack_arm, control_arm,
                    "benign_false_positive_rate", population,
                ),
                "attacked_malicious_recall": hierarchical_bootstrap(
                    records, stack_arm, control_arm,
                    "attacked_malicious_recall", population,
                ),
            }

    summary = []
    arm_keys = list(records[0]["outcomes"])
    metrics = [
        "malicious_recall", "benign_false_positive_rate", "balanced_accuracy",
        "macro_f1", "accuracy", "auroc", "auprc", "attacked_malicious_recall",
        "attacked_malicious_mean_probability",
    ]
    for population in POPULATIONS:
        selected = [row for row in records if row["population"] == population]
        for arm in arm_keys:
            summary.append({
                "population": population,
                "family": "ALL",
                "arm": arm,
                "n_cells": len(selected),
                **{
                    f"{metric}_mean": float(np.mean([
                        row["outcomes"][arm][metric] for row in selected
                    ]))
                    for metric in metrics
                },
            })
        for family in POPULATIONS[population]["families"]:
            family_rows = [row for row in selected if row["family"] == family]
            for arm in arm_keys:
                summary.append({
                    "population": population,
                    "family": family,
                    "arm": arm,
                    "n_cells": len(family_rows),
                    **{
                        f"{metric}_mean": float(np.mean([
                            row["outcomes"][arm][metric] for row in family_rows
                        ]))
                        for metric in metrics
                    },
                })
    cell_rows = []
    for row in records:
        for arm, outcome in row["outcomes"].items():
            cell_rows.append({
                "population": row["population"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "family": row["family"],
                "arm": arm,
                **{metric: outcome[metric] for metric in metrics},
                "training_rows": outcome["training"]["rows"],
                "training_benign_labels": outcome["training"]["benign_labels"],
                "training_malicious_labels": outcome["training"]["malicious_labels"],
                "poison_retained": outcome["training"]["poison_retained"],
            })
    result = {
        "experiment": preregistration["experiment"],
        "preregistration_hash": preregistration["content_hash"],
        "registration": {
            "url": receipt["registration_url"],
            "registered_at_utc": receipt["registered_at_utc"],
            "registered_code_commit": receipt["registered_code_commit"],
        },
        "merged_at_utc": utc_now(),
        "cell_count": len(records),
        "primary": {
            "effects": effects,
            "gates": gates,
            "all_gates_pass": bool(all(gates.values())),
            "recovery_fraction": recovery_fraction,
        },
        "secondary_effects": secondary_effects,
        "summary": summary,
        "cell_metric_rows": len(cell_rows),
    }
    result["content_hash"] = content_hash(result)
    atomic_json(OUT, result)
    CSVOUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSVOUT, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(CELLCSV, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    write_results_report(result)
    print(f"Wrote merged results: {OUT}")
    print(f"Primary all-gates decision: {result['primary']['all_gates_pass']}")


def write_results_report(result):
    effects = result["primary"]["effects"]
    gates = result["primary"]["gates"]
    lines = [
        "# Preregistered downstream-classifier results", "",
        f"Registration: {result['registration']['url']}", "",
        f"Frozen preregistration hash: `{result['preregistration_hash']}`", "",
        "## Primary UNSW analysis", "",
        "| Contrast | Mean | 95% hierarchical bootstrap CI | Gate |", "|---|---:|---:|:---:|",
    ]
    labels = {
        "attack_harm": "Clean - poisoned malicious recall",
        "if60_recovery": "IF60 - poisoned malicious recall",
        "if60_vs_random": "IF60 - random malicious recall",
        "if60_fpr_inflation": "IF60 - clean benign FPR",
    }
    gate_names = {
        "attack_harm": "attack_harm",
        "if60_recovery": "if60_recovery",
        "if60_vs_random": "if60_vs_random",
        "if60_fpr_inflation": "if60_fpr_guardrail",
    }
    for key in labels:
        effect = effects[key]
        lines.append(
            f"| {labels[key]} | {effect['mean']:.6f} | "
            f"[{effect['ci95'][0]:.6f}, {effect['ci95'][1]:.6f}] | "
            f"{'PASS' if gates[gate_names[key]] else 'FAIL'} |"
        )
    lines.extend([
        "", f"All four primary gates: **{'PASS' if all(gates.values()) else 'FAIL'}**.", "",
        "The registered decision rule does not permit secondary analyses to rescue a failed primary claim.", "",
    ])
    REPORT.write_text("\n".join(lines))


def finite_metrics(document):
    for key, value in document.items():
        if isinstance(value, dict):
            if not finite_metrics(value):
                return False
        elif isinstance(value, float) and not np.isfinite(value):
            return False
    return True


def audit_preregistration(preregistration, datasets):
    ensure_raw_caches(preregistration, datasets)
    design = load_design()
    confirmation = load_confirmation()
    failures = []
    if preregistration.get("design_hash") != design["content_hash"]:
        failures.append("design hash mismatch")
    if preregistration.get("parent_confirmation_hash") != confirmation["content_hash"]:
        failures.append("parent confirmation hash mismatch")
    if preregistration.get("frozen_before_outcomes") is not True:
        failures.append("manifest is not marked frozen before outcomes")
    if any(path.exists() for path in CACHE.joinpath("features").glob("*.npz")):
        failures.append("feature cache exists during preregistration audit")
    if CELLS.exists() and any(CELLS.glob("*.json")):
        failures.append("outcome cells exist during preregistration audit")
    for population, specification in POPULATIONS.items():
        for seed in SEEDS:
            realization = preregistration["realizations"][population][str(seed)]
            with np.load(raw_cache_path(population, seed), allow_pickle=True) as cache:
                X, y, X_test, y_test = cache["X"], cache["y"], cache["X_test"], cache["y_test"]
            if set(raw_hashes(X)) & set(raw_hashes(X_test)):
                failures.append(f"test identity overlap: {population} seed {seed}")
            if set(np.unique(label_to_binary(y_test))) != {0, 1}:
                failures.append(f"test class missing: {population} seed {seed}")
            for family in specification["families"]:
                locked = realization["training_attacks"][family]
                training_combined, _ = reproduce_attack(X, y, family, locked)
                test_combined, test_poison_mask = reproduce_attack(
                    X_test, y_test, family,
                    realization["test"]["attacks"][family],
                )
                training_identities = set(raw_hashes(training_combined))
                unmodified_test_identities = set(raw_hashes(X_test))
                attacked_test_identities = set(raw_hashes(test_combined[test_poison_mask]))
                if training_identities & unmodified_test_identities:
                    failures.append(f"training/test overlap: {population} seed {seed} {family}")
                if training_identities & attacked_test_identities:
                    failures.append(f"training/attacked-test overlap: {population} seed {seed} {family}")
                if unmodified_test_identities & attacked_test_identities:
                    failures.append(f"unmodified/attacked-test overlap: {population} seed {seed} {family}")
                legacy_match = locked["legacy_confirmation_attack_match"]
                if specification["dataset"] == "unsw" and not legacy_match:
                    failures.append(f"UNSW did not reproduce: {population} seed {seed} {family}")
                if specification["dataset"] == "cicids" and legacy_match:
                    failures.append(f"CICIDS correction absent: {population} seed {seed} {family}")
                prior = confirmation["realizations"][population][str(seed)]["families"][family]
                if specification["dataset"] == "unsw" and locked["attack_log_hash"] != prior["attack_log_hash"]:
                    failures.append(f"UNSW legacy hash mismatch: {population} seed {seed} {family}")
    report = {
        "passed": not failures,
        "failures": failures,
        "checked_at_utc": utc_now(),
        "preregistration_hash": preregistration["content_hash"],
        "no_outcomes_computed": True,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("preregistration audit failed")


def audit_results(preregistration, receipt):
    records = load_cells(preregistration, receipt, require_complete=True)
    registered_at = parse_timestamp(receipt["registered_at_utc"])
    failures = []
    for row in records:
        identity = f"{row['population']} seed {row['seed']} {row['family']}"
        if parse_timestamp(row["started_at_utc"]) < registered_at:
            failures.append(f"cell predates registration: {identity}")
        if row["outcomes"]["clean"] != row["outcomes"]["oracle"]:
            failures.append(f"clean/oracle mismatch: {identity}")
        if not finite_metrics(row["outcomes"]):
            failures.append(f"non-finite metric: {identity}")
        for key, sanitizer in row["sanitizers"].items():
            if sanitizer["clean_removed"] != sanitizer["random_clean_removed"]:
                failures.append(f"random cost mismatch: {identity} {key}")
            if sanitizer["poison_count"] != POPULATIONS[row["population"]]["poison"]:
                failures.append(f"poison count mismatch: {identity} {key}")
        if row["test"]["raw_hash"] != preregistration["realizations"][row["population"]][str(row["seed"])]["test"]["raw_hash"]:
            failures.append(f"test hash mismatch: {identity}")
    if not OUT.exists():
        failures.append("merged result is missing")
    else:
        merged = load_hashed_json(OUT)
        if merged["cell_count"] != len(expected_cells()):
            failures.append("merged cell count mismatch")
        expected_metric_rows = len(expected_cells()) * len(records[0]["outcomes"])
        if merged.get("cell_metric_rows") != expected_metric_rows:
            failures.append("cell-metric row count mismatch")
    if not CSVOUT.exists() or not CELLCSV.exists() or not REPORT.exists():
        failures.append("one or more merged report artifacts are missing")
    report = {
        "passed": not failures,
        "expected_cells": len(expected_cells()),
        "observed_cells": len(records),
        "failures": failures,
        "checked_at_utc": utc_now(),
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("result audit failed")


def choose_cells(args):
    cells = sorted(expected_cells())
    if args.population:
        cells = [cell for cell in cells if cell[0] == args.population]
    if args.seed is not None:
        cells = [cell for cell in cells if cell[1] == args.seed]
    if args.family:
        cells = [cell for cell in cells if cell[2] == args.family]
    if not cells:
        raise ValueError("cell selection is empty")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write-design", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--record-registration", action="store_true")
    modes.add_argument("--audit-preregistration", action="store_true")
    modes.add_argument("--list-cells", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--merge", action="store_true")
    modes.add_argument("--audit", action="store_true")
    parser.add_argument("--registration-url")
    parser.add_argument("--registered-at-utc")
    parser.add_argument("--registered-code-commit")
    parser.add_argument("--visibility", choices=("public", "embargoed"))
    parser.add_argument("--population", choices=tuple(POPULATIONS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.write_design:
        write_design()
        return
    if args.prepare_only:
        prepare()
        return
    if args.record_registration:
        required = (
            args.registration_url, args.registered_at_utc,
            args.registered_code_commit, args.visibility,
        )
        if not all(required):
            parser.error(
                "--record-registration requires --registration-url, "
                "--registered-at-utc, --registered-code-commit, and --visibility"
            )
        record_registration(
            args.registration_url, args.registered_at_utc,
            args.registered_code_commit, args.visibility,
        )
        return
    if args.audit_preregistration:
        preregistration = load_hashed_json(PRE)
        datasets = resolve_datasets(load_confirmation())
        audit_preregistration(preregistration, datasets)
        return
    if args.list_cells:
        for population, seed, family in sorted(expected_cells()):
            print(f"{population}\t{seed}\t{family}")
        return

    design, preregistration, receipt = validate_execution_lock()
    if args.run:
        datasets = resolve_datasets(load_confirmation())
        ensure_raw_caches(preregistration, datasets)
        for population, seed, family in choose_cells(args):
            run_cell(
                design, preregistration, receipt, population, seed, family,
                overwrite=args.overwrite,
            )
    elif args.merge:
        merge(preregistration, receipt)
    elif args.audit:
        audit_results(preregistration, receipt)


if __name__ == "__main__":
    main()
