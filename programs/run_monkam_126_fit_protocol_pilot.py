"""Controlled pilot of separate-fit versus shared-fit Monkam 126 features.

This experiment changes one methodological variable: whether the learned
``Binarizer`` and ``Scaler`` states are fitted separately on the unmodified and
poison batches (as in the supplied notebooks) or once on their concatenation.

The complete 1,000-poison corpus was not supplied.  The pilot therefore uses
all 18 delivered poison examples and scales the supplied notebook's population
composition exactly to 576 observations:

* 378 normal, unmodified payloads (21/31 of 558);
* 180 attack-category, unmodified payloads (10/31 of 558), including every
  unique source row named by a poison filename; and
* 18 poisoned payloads (18/576 = 1,000/32,000 = 3.125%).

Run ``--prepare-only`` before any feature extraction to lock the observation
indices, hashes, feature map, split, parameters, and confirmation criterion.
The result-producing run refuses to proceed if its reconstructed design does
not match that preregistration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from gtda.diagrams import Amplitude, PersistenceEntropy, Scaler
from gtda.homology import CubicalPersistence
from gtda.images import Binarizer, HeightFiltration, RadialFiltration
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "Payload_data_UNSW.csv"
MATERIALS_DIR = REPO_ROOT / "monkam_files"
RESULTS_DIR = REPO_ROOT / "results"

N_DATA_ROWS = 79_881
NORMAL_START = 50_138
NORMAL_STOP = 71_138
N_UNMODIFIED_NORMAL = 378
N_UNMODIFIED_ATTACK = 180
N_POISON = 18
THRESHOLD = 0.3
TEST_SIZE = 0.5
DBSCAN_EPS = 200.0
DBSCAN_MIN_SAMPLES = 9
PURITY_THRESHOLDS = (1.0, 0.95, 0.90, 0.80, 0.50)

BRANCH_SPECS: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("height_0_1", "height", (0, 1)),
    ("height_1_0", "height", (1, 0)),
    ("radial_0_1500", "radial", (0, 1500)),
    ("radial_0_600", "radial", (0, 600)),
    ("radial_0_750", "radial", (0, 750)),
    ("radial_1500_0", "radial", (1500, 0)),
    ("radial_600_0", "radial", (600, 0)),
)

METRIC_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("bottleneck", {}),
    ("wasserstein", {"p": 1}),
    ("landscape_l1", {"p": 1, "n_layers": 1, "n_bins": 81}),
    ("landscape_l2", {"p": 1, "n_layers": 2, "n_bins": 81}),
    ("betti_p1", {"p": 1, "n_bins": 81}),
    ("betti_p2", {"p": 2, "n_bins": 81}),
    ("heat_s1_6", {"p": 1, "sigma": 1.6, "n_bins": 81}),
    ("heat_s3_2", {"p": 1, "sigma": 3.2, "n_bins": 81}),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def poison_source_index(filename: str) -> int:
    stem = filename.removeprefix("final_payload_")
    return int(stem.split("-", 1)[0])


def poison_paths() -> list[Path]:
    paths = sorted(MATERIALS_DIR.glob("final_payload_*.csv"))
    if len(paths) != N_POISON:
        raise ValueError(f"Expected {N_POISON} poison CSVs, found {len(paths)}")
    return paths


def select_unmodified_indices(seed: int, poison_files: list[Path]) -> list[int]:
    """Scale the notebook's 21k normal + 10k attack population to 558 rows."""
    parent_indices = sorted({poison_source_index(path.name) for path in poison_files})
    if len(parent_indices) >= N_UNMODIFIED_ATTACK:
        raise ValueError("Too many mandatory parent rows for the attack sample")

    attack_candidates = np.concatenate(
        (np.arange(0, NORMAL_START), np.arange(NORMAL_STOP, N_DATA_ROWS))
    )
    attack_candidates = np.setdiff1d(
        attack_candidates, np.asarray(parent_indices, dtype=int), assume_unique=True
    )
    rng = np.random.default_rng(seed)
    n_other_attack = N_UNMODIFIED_ATTACK - len(parent_indices)
    other_attack = rng.choice(attack_candidates, size=n_other_attack, replace=False)
    normal = rng.choice(
        np.arange(NORMAL_START, NORMAL_STOP),
        size=N_UNMODIFIED_NORMAL,
        replace=False,
    )

    # Keep attack-category observations first, mirroring the supplied notebook's
    # random_rows + include_df concatenation.  Exact order is locked in JSON.
    return other_attack.tolist() + parent_indices + normal.tolist()


def load_selected_payloads(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    positions = {row_index: position for position, row_index in enumerate(indices)}
    if len(positions) != len(indices):
        raise ValueError("Unmodified row indices must be unique")
    X = np.empty((len(indices), 1500), dtype=np.uint8)
    labels = np.empty(len(indices), dtype=object)
    found: set[int] = set()

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        label_index = header.index("label")
        for row_index, row in enumerate(reader):
            position = positions.get(row_index)
            if position is None:
                continue
            X[position] = np.fromiter(
                (int(float(value)) for value in row[:1500]),
                dtype=np.uint8,
                count=1500,
            )
            labels[position] = row[label_index]
            found.add(row_index)
            if len(found) == len(indices):
                break
    missing = sorted(set(indices) - found)
    if missing:
        raise ValueError(f"Selected dataset rows not found: {missing[:10]}")
    return X, labels


def load_poison_payloads(paths: list[Path]) -> np.ndarray:
    rows: list[list[int]] = []
    for path in paths:
        values: list[int] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                values.append(round(float(stripped) * 255))
        if len(values) != 1500:
            raise ValueError(f"{path.name} has {len(values)} values, expected 1500")
        if min(values) < 0 or max(values) > 255:
            raise ValueError(f"{path.name} contains a byte outside [0, 255]")
        rows.append(values)
    return np.asarray(rows, dtype=np.uint8)


def _feature_union(n_jobs: int) -> FeatureUnion:
    extractors: list[tuple[str, Any]] = [
        ("entropy", PersistenceEntropy(nan_fill_value=-1, n_jobs=n_jobs))
    ]
    for name, params in METRIC_SPECS:
        if name.startswith("landscape"):
            metric = "landscape"
        elif name.startswith("betti"):
            metric = "betti"
        elif name.startswith("heat"):
            metric = "heat"
        else:
            metric = name
        extractors.append(
            (
                f"amplitude_{name}",
                Amplitude(metric=metric, metric_params=params, n_jobs=n_jobs),
            )
        )
    return FeatureUnion(extractors, n_jobs=n_jobs)


def build_126_pipeline(threshold: float = THRESHOLD, n_jobs: int = -1) -> FeatureUnion:
    branches: list[tuple[str, Pipeline]] = []
    for name, kind, vector in BRANCH_SPECS:
        if kind == "height":
            filtration = HeightFiltration(direction=np.asarray(vector), n_jobs=n_jobs)
        else:
            filtration = RadialFiltration(center=np.asarray(vector), n_jobs=n_jobs)
        branches.append(
            (
                name,
                Pipeline(
                    [
                        ("binarizer", Binarizer(threshold=threshold, n_jobs=n_jobs)),
                        ("filtration", filtration),
                        ("persistence", CubicalPersistence(n_jobs=n_jobs)),
                        ("scaler", Scaler(n_jobs=n_jobs)),
                        ("features", _feature_union(n_jobs)),
                    ]
                ),
            )
        )
    return FeatureUnion(branches, n_jobs=n_jobs)


def learned_state(pipeline: FeatureUnion) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {}
    for name, branch in pipeline.transformer_list:
        binarizer = branch.named_steps["binarizer"]
        scaler = branch.named_steps["scaler"]
        state[name] = {
            "binarizer_max_value": float(binarizer.max_value_),
            "effective_byte_cut": float(binarizer.max_value_ * binarizer.threshold),
            "scaler_scale": float(scaler.scale_),
        }
    return state


def build_design(seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    paths = poison_paths()
    unmodified_indices = select_unmodified_indices(seed, paths)
    X_unmodified, unmodified_labels = load_selected_payloads(unmodified_indices)
    X_poison = load_poison_payloads(paths)
    X_all = np.vstack((X_unmodified, X_poison))
    is_poisoned = np.concatenate(
        (np.zeros(len(X_unmodified), dtype=bool), np.ones(len(X_poison), dtype=bool))
    )
    all_indices = np.arange(len(X_all))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=TEST_SIZE, random_state=seed
    )

    label_counts = Counter(str(label) for label in unmodified_labels)
    design_core = {
        "experiment": "monkam_126_separate_fit_vs_shared_fit_pilot",
        "seed": seed,
        "diagnostic_only": True,
        "missing_full_poison_corpus": True,
        "population": {
            "n_unmodified": len(X_unmodified),
            "n_unmodified_normal": int(label_counts.get("normal", 0)),
            "n_unmodified_attack": int(len(X_unmodified) - label_counts.get("normal", 0)),
            "n_poison": len(X_poison),
            "n_combined": len(X_all),
            "poison_fraction": float(is_poisoned.mean()),
            "unmodified_label_counts": dict(sorted(label_counts.items())),
        },
        "observations": {
            "unmodified_row_indices": unmodified_indices,
            "unmodified_payload_sha256": sha256_bytes(X_unmodified.tobytes()),
            "poison_filenames": [path.name for path in paths],
            "poison_file_sha256": {path.name: sha256_file(path) for path in paths},
            "poison_payload_sha256": sha256_bytes(X_poison.tobytes()),
            "combined_payload_sha256": sha256_bytes(X_all.tobytes()),
            "mandatory_parent_row_indices": sorted(
                {poison_source_index(path.name) for path in paths}
            ),
        },
        "feature_map": {
            "input_shape": [1, 1500],
            "threshold": THRESHOLD,
            "branches": [list(spec) for spec in BRANCH_SPECS],
            "metrics": [[name, params] for name, params in METRIC_SPECS],
            "nan_fill_value": -1,
            "expected_features": 126,
        },
        "fit_protocol_arms": {
            "separate_fit": "fit one pipeline on unmodified and another on poison, then concatenate",
            "shared_fit": "fit one pipeline once on the combined population",
        },
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "test_size": TEST_SIZE,
            "random_state": seed,
            "stratify": None,
            "clustered_partition": "train",
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_index_sha256": sha256_bytes(train_indices.astype("<i8").tobytes()),
            "test_index_sha256": sha256_bytes(test_indices.astype("<i8").tobytes()),
            "train_poison_count": int(is_poisoned[train_indices].sum()),
            "test_poison_count": int(is_poisoned[test_indices].sum()),
        },
        "dbscan": {
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_SAMPLES,
            "min_samples_scaling_note": "round(500 * 288 / 16000) = 9",
        },
        "removal_purity_thresholds": list(PURITY_THRESHOLDS),
        "confirmation_criterion": {
            "state_divergence": (
                "separate poison versus unmodified effective byte cut differs by at least "
                "1.0, or any branch scaler differs by at least 5%"
            ),
            "material_clustering_effect": (
                "absolute silhouette change >= 0.10, or adjusted Rand index <= 0.90, "
                "or absolute poison-removal-rate change >= 0.10 at purity 1.0 or 0.8"
            ),
            "confirmed_when": "both state_divergence and material_clustering_effect are true",
            "multi_seed_gate": "do not run additional seeds unless confirmed_when is satisfied",
        },
    }
    design_core["design_hash"] = stable_json_hash(design_core)
    return design_core, X_unmodified, X_poison, is_poisoned


def cluster_evaluation(
    features: np.ndarray, is_poisoned: np.ndarray, train_indices: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    X_train = features[train_indices]
    poison_train = is_poisoned[train_indices]
    labels = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1
    ).fit_predict(X_train)

    unique_labels = sorted(int(value) for value in np.unique(labels))
    silhouette_all = None
    if 1 < len(unique_labels) < len(labels):
        silhouette_all = float(silhouette_score(X_train, labels))

    nonnoise = labels != -1
    nonnoise_unique = np.unique(labels[nonnoise])
    silhouette_nonnoise = None
    if nonnoise.sum() > 1 and 1 < len(nonnoise_unique) < nonnoise.sum():
        silhouette_nonnoise = float(silhouette_score(X_train[nonnoise], labels[nonnoise]))

    clusters: list[dict[str, Any]] = []
    for label in unique_labels:
        mask = labels == label
        n_poison = int(poison_train[mask].sum())
        n_unmodified = int(mask.sum() - n_poison)
        fraction = n_poison / int(mask.sum())
        if label == -1:
            color = "Noise"
        elif fraction == 0:
            color = "Green"
        elif fraction == 1:
            color = "Red"
        elif fraction > 0.8:
            color = "Pink"
        else:
            color = "Yellow"
        clusters.append(
            {
                "cluster_id": label,
                "color": color,
                "size": int(mask.sum()),
                "n_poison": n_poison,
                "n_unmodified": n_unmodified,
                "poison_fraction": fraction,
            }
        )

    removal_curve: list[dict[str, Any]] = []
    n_poison_total = int(poison_train.sum())
    n_unmodified_total = int((~poison_train).sum())
    for threshold in PURITY_THRESHOLDS:
        selected_clusters = {
            cluster["cluster_id"]
            for cluster in clusters
            if cluster["cluster_id"] != -1
            and cluster["poison_fraction"] >= threshold
        }
        removed = np.isin(labels, list(selected_clusters))
        true_poison = int(poison_train[removed].sum())
        clean_removed = int(removed.sum() - true_poison)
        removal_curve.append(
            {
                "purity_threshold": threshold,
                "selected_clusters": sorted(selected_clusters),
                "n_removed": int(removed.sum()),
                "true_poison_removed": true_poison,
                "unmodified_removed": clean_removed,
                "poison_removal_rate": (
                    true_poison / n_poison_total if n_poison_total else None
                ),
                "unmodified_removal_rate": (
                    clean_removed / n_unmodified_total if n_unmodified_total else None
                ),
                "removal_precision": (
                    true_poison / int(removed.sum()) if removed.any() else None
                ),
            }
        )

    green = [cluster for cluster in clusters if cluster["color"] == "Green"]
    red = [cluster for cluster in clusters if cluster["color"] == "Red"]
    noise = next((cluster for cluster in clusters if cluster["cluster_id"] == -1), None)
    evaluation = {
        "n_train": len(train_indices),
        "n_train_poison": n_poison_total,
        "n_train_unmodified": n_unmodified_total,
        "n_clusters_excluding_noise": sum(label != -1 for label in unique_labels),
        "n_noise": int((labels == -1).sum()),
        "silhouette_including_noise_as_label": silhouette_all,
        "silhouette_excluding_noise": silhouette_nonnoise,
        "clusters": clusters,
        "green_pool": {
            "n_clusters": len(green),
            "n_unmodified": sum(cluster["n_unmodified"] for cluster in green),
            "n_poison": sum(cluster["n_poison"] for cluster in green),
        },
        "red_pool": {
            "n_clusters": len(red),
            "n_unmodified": sum(cluster["n_unmodified"] for cluster in red),
            "n_poison": sum(cluster["n_poison"] for cluster in red),
        },
        "noise_pool": noise,
        "removal_curve": removal_curve,
        "labels_sha256": sha256_bytes(labels.astype("<i8").tobytes()),
    }
    return evaluation, labels


def feature_delta_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"Feature shapes differ: {left.shape} versus {right.shape}")
    delta = left - right
    row_l2 = np.linalg.norm(delta, axis=1)
    changed = np.any(left != right, axis=1)
    return {
        "shape": list(left.shape),
        "rows_changed_exactly": int(changed.sum()),
        "rows_unchanged_exactly": int((~changed).sum()),
        "changed_row_fraction": float(changed.mean()),
        "mean_row_l2": float(row_l2.mean()),
        "median_row_l2": float(np.median(row_l2)),
        "max_row_l2": float(row_l2.max(initial=0.0)),
        "mean_absolute_coordinate_delta": float(np.mean(np.abs(delta))),
        "max_absolute_coordinate_delta": float(np.max(np.abs(delta), initial=0.0)),
    }


def fit_arms(
    X_unmodified: np.ndarray, X_poison: np.ndarray, n_jobs: int
) -> dict[str, dict[str, Any]]:
    unmodified_images = X_unmodified.reshape(-1, 1, 1500)
    poison_images = X_poison.reshape(-1, 1, 1500)
    combined_images = np.vstack((unmodified_images, poison_images))

    start = time.perf_counter()
    unmodified_pipeline = build_126_pipeline(n_jobs=n_jobs)
    X_unmodified_separate = unmodified_pipeline.fit_transform(unmodified_images)
    poison_pipeline = build_126_pipeline(n_jobs=n_jobs)
    X_poison_separate = poison_pipeline.fit_transform(poison_images)
    X_separate = np.vstack((X_unmodified_separate, X_poison_separate))
    separate_seconds = time.perf_counter() - start

    start = time.perf_counter()
    shared_pipeline = build_126_pipeline(n_jobs=n_jobs)
    X_shared = shared_pipeline.fit_transform(combined_images)
    shared_seconds = time.perf_counter() - start

    if X_separate.shape != (len(combined_images), 126):
        raise AssertionError(f"Separate-fit shape is {X_separate.shape}, expected (*, 126)")
    if X_shared.shape != X_separate.shape:
        raise AssertionError(
            f"Shared-fit shape {X_shared.shape} differs from separate {X_separate.shape}"
        )
    if not np.isfinite(X_separate).all() or not np.isfinite(X_shared).all():
        raise AssertionError("Nonfinite TDA feature encountered")

    n_unmodified = len(X_unmodified)
    displacement = {
        "all_rows": feature_delta_summary(X_separate, X_shared),
        "unmodified_rows": feature_delta_summary(
            X_separate[:n_unmodified], X_shared[:n_unmodified]
        ),
        "poison_rows": feature_delta_summary(
            X_separate[n_unmodified:], X_shared[n_unmodified:]
        ),
    }

    return {
        "separate_fit": {
            "features": X_separate,
            "feature_shape": list(X_separate.shape),
            "feature_sha256": sha256_bytes(X_separate.astype("<f8").tobytes()),
            "fit_seconds": separate_seconds,
            "feature_displacement_from_shared_fit": displacement,
            "learned_state": {
                "unmodified_pipeline": learned_state(unmodified_pipeline),
                "poison_pipeline": learned_state(poison_pipeline),
            },
        },
        "shared_fit": {
            "features": X_shared,
            "feature_shape": list(X_shared.shape),
            "feature_sha256": sha256_bytes(X_shared.astype("<f8").tobytes()),
            "fit_seconds": shared_seconds,
            "learned_state": {"combined_pipeline": learned_state(shared_pipeline)},
        },
    }


def removal_rate(evaluation: dict[str, Any], threshold: float) -> float:
    row = next(
        item
        for item in evaluation["removal_curve"]
        if item["purity_threshold"] == threshold
    )
    return float(row["poison_removal_rate"] or 0.0)


def compare_results(
    arms: dict[str, dict[str, Any]], labels: dict[str, np.ndarray]
) -> dict[str, Any]:
    separate_state = arms["separate_fit"]["learned_state"]
    unmodified_state = separate_state["unmodified_pipeline"]
    poison_state = separate_state["poison_pipeline"]

    state_comparison: dict[str, Any] = {}
    max_cut_difference = 0.0
    max_scaler_relative_difference = 0.0
    for branch in unmodified_state:
        unmodified = unmodified_state[branch]
        poison = poison_state[branch]
        cut_difference = abs(
            unmodified["effective_byte_cut"] - poison["effective_byte_cut"]
        )
        denominator = max(
            abs(unmodified["scaler_scale"]), abs(poison["scaler_scale"]), 1e-12
        )
        scaler_relative_difference = abs(
            unmodified["scaler_scale"] - poison["scaler_scale"]
        ) / denominator
        max_cut_difference = max(max_cut_difference, cut_difference)
        max_scaler_relative_difference = max(
            max_scaler_relative_difference, scaler_relative_difference
        )
        state_comparison[branch] = {
            "effective_byte_cut_difference": cut_difference,
            "scaler_relative_difference": scaler_relative_difference,
        }

    sep_eval = arms["separate_fit"]["clustering"]
    shared_eval = arms["shared_fit"]["clustering"]
    sep_silhouette = sep_eval["silhouette_including_noise_as_label"]
    shared_silhouette = shared_eval["silhouette_including_noise_as_label"]
    silhouette_delta = None
    if sep_silhouette is not None and shared_silhouette is not None:
        silhouette_delta = float(sep_silhouette - shared_silhouette)
    ari = float(adjusted_rand_score(labels["separate_fit"], labels["shared_fit"]))
    poison_delta_pure = removal_rate(sep_eval, 1.0) - removal_rate(shared_eval, 1.0)
    poison_delta_80 = removal_rate(sep_eval, 0.8) - removal_rate(shared_eval, 0.8)

    state_divergence = (
        max_cut_difference >= 1.0 or max_scaler_relative_difference >= 0.05
    )
    clustering_effect = (
        (silhouette_delta is not None and abs(silhouette_delta) >= 0.10)
        or ari <= 0.90
        or abs(poison_delta_pure) >= 0.10
        or abs(poison_delta_80) >= 0.10
    )
    return {
        "learned_state_comparison": state_comparison,
        "max_effective_byte_cut_difference": max_cut_difference,
        "max_scaler_relative_difference": max_scaler_relative_difference,
        "silhouette_separate_minus_shared": silhouette_delta,
        "adjusted_rand_index_between_arm_clusterings": ari,
        "poison_removal_rate_delta_at_purity_1_0": poison_delta_pure,
        "poison_removal_rate_delta_at_purity_0_8": poison_delta_80,
        "state_divergence_criterion_met": state_divergence,
        "material_clustering_effect_criterion_met": clustering_effect,
        "single_seed_mechanism_confirmed": state_divergence and clustering_effect,
        "multi_seed_gate_open": state_divergence and clustering_effect,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--prereg-path",
        type=Path,
        default=RESULTS_DIR / "monkam_126_fit_protocol_pilot_preregistration.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "monkam_126_fit_protocol_pilot_seed60.json",
    )
    args = parser.parse_args()

    design, X_unmodified, X_poison, is_poisoned = build_design(args.seed)

    if args.prepare_only:
        if args.prereg_path.exists():
            raise FileExistsError(
                f"Preregistration already exists; refusing to overwrite: {args.prereg_path}"
            )
        preregistration = {
            "locked_before_results": True,
            "locked_at_utc": utc_now(),
            "design": design,
        }
        write_json(args.prereg_path, preregistration)
        print(f"Locked preregistration: {args.prereg_path}")
        print(f"Design hash: {design['design_hash']}")
        return 0

    if not args.prereg_path.exists():
        raise FileNotFoundError(
            f"Preregistration missing. Run this program with --prepare-only first: "
            f"{args.prereg_path}"
        )
    preregistration = json.loads(args.prereg_path.read_text(encoding="utf-8"))
    prereg_design = preregistration["design"]
    # JSON round-tripping converts tuples in the in-memory feature specification
    # to lists.  Compare canonical serialisations so this type-only change does
    # not reject an otherwise byte-identical locked design.
    if (
        prereg_design["design_hash"] != design["design_hash"]
        or stable_json_hash(prereg_design) != stable_json_hash(design)
    ):
        raise AssertionError("Current design does not exactly match locked preregistration")

    print("Fitting separate-fit and shared-fit 126-feature arms...")
    arms = fit_arms(X_unmodified, X_poison, args.n_jobs)
    train_indices = np.asarray(design["split"]["train_indices"], dtype=int)
    labels: dict[str, np.ndarray] = {}
    for name, arm in arms.items():
        evaluation, arm_labels = cluster_evaluation(
            arm["features"], is_poisoned, train_indices
        )
        arm["clustering"] = evaluation
        labels[name] = arm_labels
        del arm["features"]

    comparison = compare_results(arms, labels)
    result = {
        "completed_at_utc": utc_now(),
        "preregistration_path": str(args.prereg_path.resolve()),
        "preregistration_sha256": sha256_file(args.prereg_path),
        "design": design,
        "arms": arms,
        "comparison": comparison,
        "interpretation_boundary": (
            "Diagnostic miniature using all 18 delivered poison examples; not a reproduction "
            "of the unpublished 1,000-poison experiment."
        ),
    }
    write_json(args.output, result)
    print(f"Result: {args.output}")
    print(json.dumps(json_ready(comparison), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
