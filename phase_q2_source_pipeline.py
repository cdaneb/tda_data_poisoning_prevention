"""Geometry-parameterized TDA feature map for Phase Q2.

Phase Q2 audits an explicit source/code mismatch.  Monkam, De Lucia and Bastian
(2024) print, on page 6, that payloads become one-dimensional images of size
``1 x 1500``, and Algorithm 1 prints radial centers ``[[0, 1500], [0, 750],
[1500, 0]]``.  The captions to Figures 8 and 9 on page 8 instead say the
``30 x 50`` format is used "for illustration purposes".  This reconstruction has
always run the operational ``30 x 50`` raster with centers rescaled to
``[[0, 50], [0, 25], [30, 0]]``.

This module makes the raster and the centers a single named argument so the two
readings can be run against each other with one factor changed.  It is purely
additive: ``tda_pipeline.build_tda_pipeline`` is imported, not edited, and the
``legacy`` geometry is required to reproduce it bit-for-bit (see
``tools/test_phase_q2.py``).

Nothing here selects a geometry.  The caller does.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline, make_union
from gtda.images import Binarizer, HeightFiltration, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude


# Directions are printed identically in both readings; only the raster and the
# centers move, and they move together because they name the same coordinates.
DIRECTION_LIST = (np.array([0, 1]), np.array([1, 0]))

GEOMETRIES = {
    # What this repository has always run.  Centers rescaled from the printed
    # values by the reconstruction, on the reading that the printed centers were
    # written in payload-length units (1500) rather than image units.
    "legacy": {
        "image_shape": (30, 50),
        "centers": (np.array([0, 50]), np.array([0, 25]), np.array([30, 0])),
        "provenance": "reconstruction; Figures 8-9 raster with centers rescaled to image units",
    },
    # What the paper literally prints: page 6 image size, Algorithm 1 centers,
    # neither adjusted.
    "source": {
        "image_shape": (1, 1500),
        "centers": (np.array([0, 1500]), np.array([0, 750]), np.array([1500, 0])),
        "provenance": "source-printed; page 6 image size + Algorithm 1 centers verbatim",
    },
}

METRIC_LIST = (
    {"metric": "bottleneck", "metric_params": {}},
    {"metric": "wasserstein", "metric_params": {"p": 1}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 64}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 64}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 64}},
)


def geometry_spec(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; expected one of {sorted(GEOMETRIES)}")
    return GEOMETRIES[geometry]


def reshape_for_geometry(X, geometry):
    """Reshape (N, 1500) payload bytes into the raster named by ``geometry``.

    No clipping, padding or rescaling is applied.  Both supported rasters are
    exact factorizations of 1500, so every payload byte survives in both arms.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 1500:
        raise ValueError(f"expected (N, 1500) input, got {X.shape}")
    shape = geometry_spec(geometry)["image_shape"]
    if shape[0] * shape[1] != 1500:
        raise AssertionError(f"geometry {geometry!r} raster {shape} does not hold 1500 bytes")
    return X.reshape(X.shape[0], *shape)


def build_geometry_pipeline(geometry, threshold=0.4):
    """Algorithm 1's feature map with the raster/centers named by ``geometry``.

    Everything other than the raster and the centers is held at the values this
    project has used since the original reconstruction: the same two directions,
    the same Binarizer semantics, the same CubicalPersistence defaults
    (``homology_dimensions=(0, 1)``, ``coeff=2``), the same Scaler, the same
    PersistenceEntropy, and the same five Amplitude metrics.
    """
    spec = geometry_spec(geometry)

    filtration_list = (
        [HeightFiltration(direction=d, n_jobs=-1) for d in DIRECTION_LIST]
        + [RadialFiltration(center=c, n_jobs=-1) for c in spec["centers"]]
    )

    diagram_steps = [
        [
            Binarizer(threshold=threshold, n_jobs=-1),
            filtration,
            CubicalPersistence(n_jobs=-1),
            Scaler(n_jobs=-1),
        ]
        for filtration in filtration_list
    ]

    feature_union = make_union(
        PersistenceEntropy(nan_fill_value=-1),
        *[Amplitude(**m, n_jobs=-1) for m in METRIC_LIST],
    )

    return make_union(
        *[make_pipeline(*step, feature_union) for step in diagram_steps],
        n_jobs=-1,
    )


def extract_geometry_features(X, geometry, threshold=0.4, verbose=True):
    """Reshape + fit_transform in one combined batch.

    One combined fit is mandatory here for the same reason it is everywhere else
    in this project: ``Scaler`` normalizes per batch, so fitting clean and
    perturbed samples separately manufactures differences that are not in the
    data (CLAUDE.md section 9, item 8).
    """
    images = reshape_for_geometry(X, geometry)
    if verbose:
        print(f"  [{geometry}] reshaped {X.shape} -> {images.shape} (threshold={threshold})")
    pipeline = build_geometry_pipeline(geometry, threshold=threshold)
    X_tda = pipeline.fit_transform(images)
    if verbose:
        print(f"  [{geometry}] features: {X_tda.shape}")
    return X_tda, pipeline


def filtration_specs(geometry):
    """Named constructors for the five filtrations of a geometry, in order."""
    spec = geometry_spec(geometry)
    names = ["height_d01", "height_d10"] + [
        "radial_" + "_".join(str(int(v)) for v in c) for c in spec["centers"]
    ]
    makers = [lambda d=d: HeightFiltration(direction=d, n_jobs=-1) for d in DIRECTION_LIST] + [
        lambda c=c: RadialFiltration(center=c, n_jobs=-1) for c in spec["centers"]
    ]
    return tuple(zip(names, makers))


def diagram_diagnostics(X, geometry, threshold=0.4):
    """Per-filtration diagram structure, before Scaler and before vectorization.

    Reports which homology dimensions actually carry non-trivial (birth != death)
    points.  This is the measurement that decides whether a raster can support
    the homology dimensions Algorithm 1 asks for, independent of any clustering.
    """
    images = reshape_for_geometry(X, geometry)
    out = []
    for name, make_filtration in filtration_specs(geometry):
        pipe = make_pipeline(
            Binarizer(threshold=threshold, n_jobs=-1),
            make_filtration(),
            CubicalPersistence(n_jobs=-1),
        )
        diagrams = pipe.fit_transform(images)
        filt_vals = make_pipeline(
            Binarizer(threshold=threshold, n_jobs=-1), make_filtration()
        ).fit_transform(images)
        entry = {
            "filtration": name,
            "diagram_shape": list(diagrams.shape),
            "filtration_value_min": float(filt_vals.min()),
            "filtration_value_max": float(filt_vals.max()),
            "filtration_value_n_unique": int(np.unique(filt_vals).size),
            "nontrivial_points_by_dim": {},
            "samples_with_nontrivial_by_dim": {},
        }
        dims = np.unique(diagrams[:, :, 2])
        for dim in dims:
            sel = diagrams[:, :, 2] == dim
            nontrivial = sel & (diagrams[:, :, 0] != diagrams[:, :, 1])
            entry["nontrivial_points_by_dim"][str(int(dim))] = int(nontrivial.sum())
            entry["samples_with_nontrivial_by_dim"][str(int(dim))] = int(
                np.count_nonzero(nontrivial.any(axis=1))
            )
        out.append(entry)
    return out
