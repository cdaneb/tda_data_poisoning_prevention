"""Evidence-locked Monkam representations with inspectable fitted state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from gtda.diagrams import Amplitude, PersistenceEntropy, Scaler
from gtda.homology import CubicalPersistence
from gtda.images import Binarizer, HeightFiltration, RadialFiltration
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    image_shape: tuple[int, int]
    threshold: float
    filtrations: tuple[tuple[str, str, tuple[int, int]], ...]
    metrics: tuple[tuple[str, dict[str, Any]], ...]
    homology_dimensions: tuple[int, ...] = (0, 1)

    @property
    def extractors_per_dimension(self) -> int:
        return 1 + len(self.metrics)

    @property
    def expected_features(self) -> int:
        return len(self.filtrations) * len(self.homology_dimensions) * self.extractors_per_dimension

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "image_shape": list(self.image_shape),
                "threshold": self.threshold, "filtrations": list(self.filtrations),
                "metrics": list(self.metrics), "homology_dimensions": list(self.homology_dimensions),
                "expected_features": self.expected_features}


METRICS_60 = (("bottleneck", {}), ("wasserstein", {"p": 1}),
              ("landscape", {"p": 1, "n_layers": 1, "n_bins": 64}),
              ("betti", {"p": 1, "n_bins": 64}),
              ("heat", {"p": 1, "sigma": 1.6, "n_bins": 64}))
METRICS_126 = (("bottleneck", {}), ("wasserstein", {"p": 1}),
 ("landscape", {"p": 1, "n_layers": 1, "n_bins": 81}),
 ("landscape", {"p": 1, "n_layers": 2, "n_bins": 81}),
 ("betti", {"p": 1, "n_bins": 81}), ("betti", {"p": 2, "n_bins": 81}),
 ("heat", {"p": 1, "sigma": 1.6, "n_bins": 81}),
 ("heat", {"p": 1, "sigma": 3.2, "n_bins": 81}))
FILT_30 = (("height_0_1", "height", (0, 1)), ("height_1_0", "height", (1, 0)),
 ("radial_0_50", "radial", (0, 50)), ("radial_0_25", "radial", (0, 25)),
 ("radial_30_0", "radial", (30, 0)))
FILT_126 = (("height_0_1", "height", (0, 1)), ("height_1_0", "height", (1, 0)),
 ("radial_0_1500", "radial", (0, 1500)), ("radial_0_600", "radial", (0, 600)),
 ("radial_0_750", "radial", (0, 750)), ("radial_1500_0", "radial", (1500, 0)),
 ("radial_600_0", "radial", (600, 0)))

SPECS = {
 "notebook_1x1500_t03": RepresentationSpec("notebook_1x1500_t03", (1,1500), .3, FILT_126, METRICS_126),
 "algorithm1_1x1500_t04": RepresentationSpec("algorithm1_1x1500_t04", (1,1500), .4, FILT_126, METRICS_126),
 "project_30x50_t04": RepresentationSpec("project_30x50_t04", (30,50), .4, FILT_30, METRICS_60),
 "supplied_126": RepresentationSpec("supplied_126", (1,1500), .3, FILT_126, METRICS_126),
}

# Recovered from the workbook-producing notebook cell.  Unlike the other
# supported configurations it omits Scaler, so the supplied workbook is
# audited directly rather than forced through the Scaler-bearing builder.
SUPPLIED_280_DEFINITION = {
    "image_shape": [1, 1500], "threshold": 0.3, "scaler": False,
    "filtrations": [["height", [1, 0]], ["height", [-1, 0]],
                    ["height", [-1, 1]], ["radial", [600, 0]],
                    ["radial", [1500, 0]], ["radial", [20, 13]],
                    ["radial", [6, 20]]],
    "homology_dimensions": [0, 1], "extractors_per_dimension": 20,
    "expected_features": 280,
}


def stable_hash(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _features(metrics: Iterable[tuple[str, dict[str, Any]]], n_jobs: int) -> FeatureUnion:
    items = [("entropy", PersistenceEntropy(nan_fill_value=-1, n_jobs=n_jobs))]
    for i, (metric, params) in enumerate(metrics):
        items.append((f"amplitude_{i}_{metric}", Amplitude(metric=metric, metric_params=params, n_jobs=n_jobs)))
    return FeatureUnion(items, n_jobs=n_jobs)


def build_pipeline(spec: RepresentationSpec, n_jobs: int = 1) -> FeatureUnion:
    branches = []
    for name, kind, vector in spec.filtrations:
        filtration = (HeightFiltration(direction=np.asarray(vector), n_jobs=n_jobs) if kind == "height"
                      else RadialFiltration(center=np.asarray(vector), n_jobs=n_jobs))
        branches.append((name, Pipeline([
            ("binarizer", Binarizer(threshold=spec.threshold, n_jobs=n_jobs)),
            ("filtration", filtration),
            ("persistence", CubicalPersistence(homology_dimensions=spec.homology_dimensions, n_jobs=n_jobs)),
            ("scaler", Scaler(n_jobs=n_jobs)), ("features", _features(spec.metrics, n_jobs))])))
    return FeatureUnion(branches, n_jobs=n_jobs)


def reshape_payloads(X: np.ndarray, spec: RepresentationSpec) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 1500:
        raise ValueError(f"expected (n, 1500), got {X.shape}")
    return X.reshape((len(X),) + spec.image_shape)


def fit_shared(X: np.ndarray, spec: RepresentationSpec, n_jobs: int = 1):
    pipeline = build_pipeline(spec, n_jobs=n_jobs)
    vectors = pipeline.fit_transform(reshape_payloads(X, spec))
    if vectors.shape[1] != spec.expected_features:
        raise AssertionError((vectors.shape, spec.expected_features))
    return vectors, pipeline


def learned_state(pipeline: FeatureUnion) -> dict[str, Any]:
    out = {}
    for name, branch in pipeline.transformer_list:
        b, s = branch.named_steps["binarizer"], branch.named_steps["scaler"]
        out[name] = {"binarizer_max_value": float(b.max_value_),
                     "effective_cut": float(b.max_value_ * b.threshold),
                     "scaler_scale": float(s.scale_)}
    return out


def feature_blocks(spec: RepresentationSpec) -> dict[str, Any]:
    """Return exact FeatureUnion output indices.

    Ordering is filtration-major, then extractor-major, then homology-major.
    In particular H0/H1 are interleaved across extractors; they are not two
    contiguous halves of a filtration block.
    """
    width = len(spec.homology_dimensions) * spec.extractors_per_dimension
    extractor_names = ["entropy"] + [f"amplitude_{i}_{metric}"
                                      for i, (metric, _) in enumerate(spec.metrics)]
    blocks = {}
    for branch_i, (name, _, _) in enumerate(spec.filtrations):
        start = branch_i * width
        extractors = {}
        homology = {f"h{dimension}": [] for dimension in spec.homology_dimensions}
        for extractor_i, extractor_name in enumerate(extractor_names):
            extractor_start = start + extractor_i * len(spec.homology_dimensions)
            indices = list(range(extractor_start,
                                 extractor_start + len(spec.homology_dimensions)))
            extractors[extractor_name] = {
                "start": extractor_start,
                "stop": extractor_start + len(spec.homology_dimensions),
                "indices_by_homology": dict(zip(
                    [f"h{d}" for d in spec.homology_dimensions], indices
                )),
            }
            for dimension, index in zip(spec.homology_dimensions, indices):
                homology[f"h{dimension}"].append(index)
        blocks[name] = {"start": start, "stop": start + width,
                        "extractors": extractors, "homology_indices": homology}
    return blocks


def equivalence_profile(X: np.ndarray, labels: np.ndarray | None = None) -> dict[str, Any]:
    X = np.ascontiguousarray(X)
    keys = [hashlib.sha256(row.view(np.uint8)).digest() for row in X]
    groups: dict[bytes, list[int]] = {}
    for i, key in enumerate(keys): groups.setdefault(key, []).append(i)
    repeated = [g for g in groups.values() if len(g) > 1]
    conflicts = [] if labels is None else [g for g in repeated if len(set(map(str, labels[g]))) > 1]
    return {"n_rows": len(X), "n_unique": len(groups), "n_repeated_classes": len(repeated),
            "repeated_member_rate": sum(map(len,repeated))/len(X),
            "redundancy_rate": (len(X)-len(groups))/len(X),
            "conflicting_label_member_rate": sum(map(len,conflicts))/len(X),
            "largest_class": max(map(len, groups.values()), default=0)}
