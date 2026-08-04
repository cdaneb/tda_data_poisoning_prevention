"""Shared Phase Q2 machinery: fixed data realization, hashing, OPTICS internals.

Every Phase Q2 arm must run on the *same* data realization as the standing
regression gate (``tools/repro_check.py``), so that any difference reported is
attributable to the factor under test and not to a different draw.  The
realization builder here reproduces that gate's construction exactly:

    load_unsw(max_samples=None)
    RandomState(42).choice(len(X_all), size=5000, replace=False)
    malicious_random_attack(..., poison_rate=0.10, random_state=seed, n_swaps=60)

Nothing in this module reads ``is_poisoned`` except the explicitly retrospective
evaluation helpers.  Discovery code must not import those.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT))

from data_loader import load_unsw
from adversarial_attack import malicious_random_attack

MAX_SAMPLES = 5000
POISON_RATE = 0.10
N_SWAPS = 60
DISCOVERY_SEED = 42
CONFIRMATION_SEEDS = (42, 123, 456, 789, 1024)

CACHE_DIR = ROOT / ".q2_cache"


# --------------------------------------------------------------------------
# hashing / provenance
# --------------------------------------------------------------------------

def array_hash(a):
    """Stable content hash of an array, independent of memory layout."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def float_array_hash(a, decimals=9):
    """Content hash of a float array, rounded so it survives benign last-bit noise."""
    return array_hash(np.round(np.asarray(a, dtype=np.float64), decimals))


def environment_block():
    import sklearn
    import gtda
    import scipy
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "giotto_tda": gtda.__version__,
    }


# --------------------------------------------------------------------------
# fixed data realization
# --------------------------------------------------------------------------

_UNSW_CACHE = {}


def load_unsw_once():
    if "X" not in _UNSW_CACHE:
        X_all, y_all = load_unsw(max_samples=None)
        _UNSW_CACHE["X"] = X_all
        _UNSW_CACHE["y"] = y_all
    return _UNSW_CACHE["X"], _UNSW_CACHE["y"]


def build_realization(seed=DISCOVERY_SEED):
    """The R60 realization for ``seed``, identical to the standing gate at seed 42.

    Returns a dict with the combined byte matrix, the poison mask, and hashes
    sufficient to prove two arms saw the same input without shipping the data.
    """
    X_all, y_all = load_unsw_once()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_all), size=MAX_SAMPLES, replace=False)
    X_full, y_full = X_all[idx], y_all[idx]

    Xc, yc, is_poisoned, log = malicious_random_attack(
        X_full, y_full, poison_rate=POISON_RATE, random_state=seed, n_swaps=N_SWAPS
    )
    validity_pct = 100.0 * sum(entry["valid"] for entry in log) / len(log)

    return {
        "seed": int(seed),
        "dataset": "UNSW-NB15 Payload-Byte",
        "attack": "malicious_random_attack",
        "n_swaps": N_SWAPS,
        "poison_rate": POISON_RATE,
        "max_samples": MAX_SAMPLES,
        "n_total": int(len(is_poisoned)),
        "n_clean": int((~is_poisoned).sum()),
        "n_poison": int(is_poisoned.sum()),
        "attack_validity_pct": validity_pct,
        "X_combined": Xc,
        "y_combined": yc,
        "is_poisoned": is_poisoned,
        "subsample_index_hash": array_hash(idx),
        "input_hash": array_hash(Xc),
        "poison_mask_hash": array_hash(is_poisoned),
    }


def realization_provenance(real):
    """The JSON-safe half of a realization (no bulk arrays)."""
    return {k: v for k, v in real.items()
            if k not in {"X_combined", "y_combined", "is_poisoned"}}


# --------------------------------------------------------------------------
# cached feature extraction
# --------------------------------------------------------------------------

def cached_features(tag, seed, builder):
    """Machine-local feature cache.

    The cache is regenerable and gitignored.  Every number that depends on it is
    published with the feature hash, so an auditor can rebuild and compare
    without the cache file.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{tag}_seed{seed}.npy"
    if path.exists():
        X_tda = np.load(path)
        print(f"  [cache hit] {path.name} {X_tda.shape}")
        return X_tda, 0.0
    t0 = time.time()
    X_tda = builder()
    elapsed = time.time() - t0
    np.save(path, X_tda)
    print(f"  [cache write] {path.name} {X_tda.shape} ({elapsed:.1f}s)")
    return X_tda, elapsed


# --------------------------------------------------------------------------
# label-free feature diagnostics
# --------------------------------------------------------------------------

def feature_diagnostics(X_tda, rng_seed=20260729, n_sample=400):
    """Structure of a feature matrix. Uses no labels."""
    X = np.asarray(X_tda, dtype=np.float64)
    variances = X.var(axis=0)
    _, counts = np.unique(X, axis=0, return_counts=True)
    rng = np.random.RandomState(rng_seed)
    sub = X[rng.choice(len(X), size=min(n_sample, len(X)), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(sub), k=1)
    pair = d[iu]
    np.fill_diagonal(d, np.inf)
    nn1 = d.min(axis=1)
    nn5 = np.partition(d, 4, axis=1)[:, 4]
    q = [0, 5, 25, 50, 75, 95, 100]
    return {
        "shape": list(X.shape),
        "feature_hash": float_array_hash(X),
        "n_zero_variance_features": int((variances == 0).sum()),
        "variance_quantiles": {str(p): float(np.percentile(variances, p)) for p in q},
        "n_exact_duplicate_rows": int(len(X) - len(counts)),
        "max_duplicate_multiplicity": int(counts.max()),
        "diagnostic_sample_size": int(len(sub)),
        "pairwise_distance_quantiles": {str(p): float(np.percentile(pair, p)) for p in q},
        "nn1_distance_quantiles": {str(p): float(np.percentile(nn1, p)) for p in q},
        "nn5_distance_quantiles": {str(p): float(np.percentile(nn5, p)) for p in q},
        "n_nonfinite": int((~np.isfinite(X)).sum()),
    }


def labeled_distance_diagnostics(X_tda, is_poisoned, rng_seed=20260729, n_sample=300):
    """Clean-clean / clean-poison distance quantiles. Retrospective evaluation only."""
    X = np.asarray(X_tda, dtype=np.float64)
    poisoned = np.asarray(is_poisoned, dtype=bool)
    rng = np.random.RandomState(rng_seed)
    ci = np.flatnonzero(~poisoned)
    pi = np.flatnonzero(poisoned)
    cs = rng.choice(ci, size=min(n_sample, len(ci)), replace=False)
    ps = rng.choice(pi, size=min(n_sample, len(pi)), replace=False)
    cc = np.sqrt(((X[cs][:, None] - X[cs][None]) ** 2).sum(-1))
    cp = np.sqrt(((X[cs][:, None] - X[ps][None]) ** 2).sum(-1))
    iu = np.triu_indices(len(cs), k=1)
    q = [5, 25, 50, 75, 95]
    return {
        "clean_clean_quantiles": {str(p): float(np.percentile(cc[iu], p)) for p in q},
        "clean_poison_quantiles": {str(p): float(np.percentile(cp, p)) for p in q},
        "median_ratio_poison_over_clean": float(
            np.median(cp) / np.median(cc[iu]) if np.median(cc[iu]) else float("nan")
        ),
    }


# --------------------------------------------------------------------------
# OPTICS internals
# --------------------------------------------------------------------------

def optics_internals(model, max_eps):
    """Everything needed to explain why points became -1, plus provenance hashes."""
    reach = np.asarray(model.reachability_, dtype=np.float64)
    core = np.asarray(model.core_distances_, dtype=np.float64)
    finite_reach = reach[np.isfinite(reach)]
    finite_core = core[np.isfinite(core)]
    q = [5, 25, 50, 75, 95]
    return {
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in model.get_params(deep=True).items()},
        "ordering_hash": array_hash(np.asarray(model.ordering_)),
        "reachability_hash": float_array_hash(reach),
        "core_distances_hash": float_array_hash(core),
        "predecessor_hash": array_hash(np.asarray(model.predecessor_)),
        "labels_hash": array_hash(np.asarray(model.labels_)),
        "n_reachability_infinite": int((~np.isfinite(reach)).sum()),
        "n_reachability_finite": int(np.isfinite(reach).sum()),
        "reachability_finite_quantiles": (
            {str(p): float(np.percentile(finite_reach, p)) for p in q} if finite_reach.size else {}
        ),
        "n_core_distance_infinite": int((~np.isfinite(core)).sum()),
        "n_core_distance_finite": int(np.isfinite(core).sum()),
        "core_distance_finite_quantiles": (
            {str(p): float(np.percentile(finite_core, p)) for p in q} if finite_core.size else {}
        ),
        "n_core_distance_within_max_eps": int(np.count_nonzero(core <= max_eps)),
        "fraction_core_distance_within_max_eps": float(np.mean(core <= max_eps)),
    }


def cluster_structure(labels):
    """Label-free cluster shape summary: the only thing discovery may rank on."""
    labels = np.asarray(labels)
    n = len(labels)
    ids, sizes = np.unique(labels[labels != -1], return_counts=True)
    return {
        "n_clusters": int(len(ids)),
        "n_unclustered": int((labels == -1).sum()),
        "unclustered_fraction": float((labels == -1).mean()),
        "largest_cluster_size": int(sizes.max()) if sizes.size else 0,
        "largest_cluster_share": float(sizes.max() / n) if sizes.size else 0.0,
        "median_cluster_size": float(np.median(sizes)) if sizes.size else 0.0,
        "min_cluster_size_observed": int(sizes.min()) if sizes.size else 0,
        "cluster_size_quantiles": (
            {str(p): float(np.percentile(sizes, p)) for p in [5, 25, 50, 75, 95]}
            if sizes.size else {}
        ),
        "labels_hash": array_hash(labels),
    }


# --------------------------------------------------------------------------
# accounting views  (retrospective; uses labels)
# --------------------------------------------------------------------------

COLOR_ORDER = ("Green", "Red", "Pink", "Yellow", "Noise")


def color_cluster_table(labels, is_poisoned):
    """Per-cluster color assignment using the project's hardcoded purity literals.

    Mirrors ``clustering.classify_clusters`` exactly (Green 0%, Red 100%,
    Pink >80%, Yellow otherwise, label -1 -> Noise) without importing it, so the
    audit can attach the extra fields it needs without touching that module.
    """
    labels = np.asarray(labels)
    poisoned = np.asarray(is_poisoned, dtype=bool)
    n_total = len(labels)
    n_poison_total = int(poisoned.sum())
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        size = int(mask.sum())
        n_pois = int((poisoned & mask).sum())
        frac = n_pois / size if size else 0.0
        if label == -1:
            color = "Noise"
        elif frac == 0:
            color = "Green"
        elif frac == 1.0:
            color = "Red"
        elif frac > 0.80:
            color = "Pink"
        else:
            color = "Yellow"
        rows.append({
            "cluster_id": int(label),
            "color": color,
            "size": size,
            "n_poisoned": n_pois,
            "n_clean": size - n_pois,
            "poison_fraction": frac,
            "share_of_all_samples_pct": 100.0 * size / n_total,
            "share_of_all_poison_pct": (100.0 * n_pois / n_poison_total) if n_poison_total else 0.0,
        })
    return rows


def accounting_views(labels, is_poisoned):
    """Three display conventions over ONE fitted labelling. No refit.

    The denominator for true capture is always the full poisoned population, in
    every view.  That is the point of the audit: a display convention may change
    how the table looks, but it must not be allowed to change what was caught.
    """
    rows = color_cluster_table(labels, is_poisoned)
    labels = np.asarray(labels)
    poisoned = np.asarray(is_poisoned, dtype=bool)
    n_total = len(labels)
    n_poison_total = int(poisoned.sum())
    n_clustered = int((labels != -1).sum())

    def share(color, subset):
        return sum(r["size"] for r in subset if r["color"] == color)

    clustered_rows = [r for r in rows if r["cluster_id"] != -1]
    noise_size = sum(r["size"] for r in rows if r["cluster_id"] == -1)

    # View 1 -- all-sample denominator, Noise is its own displayed category.
    view_all = {c: 100.0 * share(c, rows) / n_total for c in COLOR_ORDER}

    # View 2 -- clustered-only denominator, -1 dropped and the rest renormalized.
    view_clustered = {
        c: (100.0 * share(c, clustered_rows) / n_clustered if n_clustered else 0.0)
        for c in COLOR_ORDER if c != "Noise"
    }

    # View 3 -- -1 folded into the mixed/unknown display bucket. Never captured.
    view_noise_as_yellow = {c: 100.0 * share(c, clustered_rows) / n_total
                            for c in COLOR_ORDER if c != "Noise"}
    view_noise_as_yellow["Yellow"] += 100.0 * noise_size / n_total

    red_poison = sum(r["n_poisoned"] for r in rows if r["color"] == "Red")
    true_capture = 100.0 * red_poison / n_poison_total if n_poison_total else 0.0

    return {
        "n_total": n_total,
        "n_poison_total": n_poison_total,
        "n_clustered": n_clustered,
        "n_unclustered": n_total - n_clustered,
        "views": {
            "all_sample_denominator": {
                "color_shares_pct": view_all,
                "sum_pct": float(sum(view_all.values())),
            },
            "clustered_only_denominator": {
                "color_shares_pct": view_clustered,
                "sum_pct": float(sum(view_clustered.values())),
            },
            "noise_as_yellow_display": {
                "color_shares_pct": view_noise_as_yellow,
                "sum_pct": float(sum(view_noise_as_yellow.values())),
            },
        },
        # Invariant across all three views by construction. Asserted in tests.
        "true_poison_capture_pct": true_capture,
        "n_poison_in_red_clusters": red_poison,
        "poison_unclustered_fraction": float((labels == -1)[poisoned].mean()) if n_poison_total else 0.0,
        "clean_unclustered_fraction": float((labels == -1)[~poisoned].mean()),
        "clusters": rows,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)
    print(f"  wrote {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
