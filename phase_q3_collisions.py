"""Phase Q3: exact-collision provenance machinery.

Phase Q2 reported that "51.8% of the feature matrix is exact duplicate rows"
with one block of 1043 identical 60-vectors.  That sentence conflates two
different statistics and says nothing about *where* in the pipeline distinct
packets stopped being distinguishable.  This module supplies the vocabulary and
the arithmetic for answering both questions exactly.

Design rules this module enforces (from the Q3 preregistration):

* Exact equality means exact equality.  Nothing is rounded before a collision is
  declared.  A tolerance-based view may be computed separately and is never
  substituted for the exact one.
* Persistence diagrams are multisets.  A diagram signature is built from the
  valid (off-diagonal) points, canonically ordered by homology dimension, birth
  and death, so that giotto-tda's array point order and its diagonal padding
  never masquerade as topology.
* Signed zero is normalised and NaN is canonicalised before any float array is
  hashed, so that two numerically identical arrays cannot receive different
  hashes for representation reasons.
* Collisions are found with stable hashes and equivalence classes, never with an
  O(N^2) pairwise matrix.
* No function that builds a stage signature accepts ground-truth labels.  Labels
  enter only in the retrospective composition and attribution helpers, which are
  named accordingly.
"""
from __future__ import annotations

import hashlib
import inspect
from collections import defaultdict

import numpy as np

# Quantiles reported for every repeated-class size distribution.
SIZE_QUANTILES = (5, 25, 50, 75, 95, 100)


# ---------------------------------------------------------------------------
# exact hashing primitives  (label-free)
# ---------------------------------------------------------------------------

def normalize_float_array(a):
    """Return a float64 copy with signed zero and NaN canonicalised.

    ``-0.0`` and ``+0.0`` compare equal under ``==`` but have different bit
    patterns, and NaN has many bit patterns.  Both would produce spurious hash
    differences between arrays that are numerically identical.  Nothing else is
    touched: no rounding, no clipping, and infinities are preserved as they are
    so that a non-finite entry stays visible to the auditor.
    """
    out = np.array(a, dtype=np.float64, copy=True)
    if out.size:
        np.copyto(out, 0.0, where=(out == 0.0))
        np.copyto(out, np.nan, where=np.isnan(out))
    return out


def exact_hash(a):
    """Stable exact content hash of an array: dtype, shape and raw bytes.

    Float arrays are normalised first (signed zero / NaN only).  Integer and
    boolean arrays are hashed byte-for-byte with no transformation at all.
    """
    a = np.asarray(a)
    if a.dtype.kind == "f":
        a = normalize_float_array(a)
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:32]


def row_hashes(arr):
    """Per-row exact hashes of an array whose first axis indexes samples."""
    arr = np.asarray(arr)
    return np.array([exact_hash(row) for row in arr], dtype=object)


def combine_hashes(hash_arrays):
    """Hash a tuple of per-row hashes into one per-row signature.

    Used to turn "the five filtration images of this sample" into a single
    stage signature without ever concatenating the underlying float arrays.
    """
    hash_arrays = list(hash_arrays)
    n = len(hash_arrays[0])
    out = np.empty(n, dtype=object)
    for i in range(n):
        h = hashlib.sha256()
        for column in hash_arrays:
            h.update(column[i].encode())
            h.update(b"|")
        out[i] = h.hexdigest()[:32]
    return out


def pack_binary_mask(mask):
    """Lossless packed representation of a boolean mask.

    ``np.packbits`` is a storage/hash convenience only.  ``unpack_binary_mask``
    inverts it exactly, which the unit tests assert; the packed form is never
    treated as a different object from the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    return np.packbits(mask.reshape(mask.shape[0], -1), axis=1), mask.shape


def unpack_binary_mask(packed, shape):
    n_bits = int(np.prod(shape[1:]))
    flat = np.unpackbits(packed, axis=1)[:, :n_bits]
    return flat.reshape(shape).astype(bool)


# ---------------------------------------------------------------------------
# persistence diagram canonicalisation  (label-free)
# ---------------------------------------------------------------------------

def canonical_diagram_points(diagram):
    """Canonical (n_valid, 3) point array for one giotto-tda diagram.

    ``diagram`` is the ``(n_points, 3)`` slice for a single sample, with columns
    ``(birth, death, homology_dimension)``.  giotto-tda pads every sample up to a
    common point count using diagonal points, so a point with ``death == birth``
    carries no persistence and is indistinguishable from padding; those points
    are dropped.  This is stated rather than hidden: a genuine zero-persistence
    point would also be dropped, and it could not be told apart from padding by
    any consumer of the diagram.

    Everything else is kept, including the homology dimension, which stays part
    of the signature.  Points are sorted lexicographically by
    ``(homology_dimension, birth, death)`` so that array order carries no
    information -- the diagram is treated as the multiset it is.
    """
    d = normalize_float_array(diagram)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"expected a (n_points, 3) diagram, got {d.shape}")
    birth, death = d[:, 0], d[:, 1]
    # NaN entries can never satisfy death > birth, so they would be silently
    # dropped.  Surface them instead of losing them.
    if np.isnan(d).any():
        raise ValueError("diagram contains NaN; refusing to canonicalise silently")
    valid = death > birth
    pts = d[valid]
    order = np.lexsort((pts[:, 1], pts[:, 0], pts[:, 2]))
    return pts[order]


def diagram_row_hashes(diagrams):
    """Per-sample canonical signatures for a ``(n_samples, n_points, 3)`` stack."""
    diagrams = np.asarray(diagrams)
    return np.array([exact_hash(canonical_diagram_points(d)) for d in diagrams],
                    dtype=object)


def diagram_point_counts(diagrams):
    """Per-sample count of valid (off-diagonal) points, split by homology dim."""
    diagrams = np.asarray(diagrams)
    total, by_dim = [], defaultdict(list)
    for d in diagrams:
        pts = canonical_diagram_points(d)
        total.append(len(pts))
        dims, counts = np.unique(pts[:, 2], return_counts=True) if len(pts) else ([], [])
        seen = dict(zip([int(x) for x in dims], [int(c) for c in counts]))
        for dim in (0, 1):
            by_dim[dim].append(seen.get(dim, 0))
    return np.array(total), {k: np.array(v) for k, v in by_dim.items()}


# ---------------------------------------------------------------------------
# equivalence classes and the two collision statistics
# ---------------------------------------------------------------------------

def equivalence_classes(hashes):
    """Map each distinct signature to the array of row indices carrying it."""
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    return {h: np.array(v, dtype=int) for h, v in groups.items()}


def class_size_array(hashes):
    """Per-row size of the equivalence class that row belongs to."""
    classes = equivalence_classes(hashes)
    sizes = np.empty(len(hashes), dtype=int)
    for idx in classes.values():
        sizes[idx] = len(idx)
    return sizes


def class_stats(hashes, is_poisoned, stage_name):
    """Full collision statistics for one stage.

    Reports *both* headline statistics, because they differ and the Q2 sentence
    did not say which one it meant:

    ``repeated_member_fraction``
        rows belonging to a class of size >= 2, divided by all rows.
    ``redundancy_fraction``
        ``(n_rows - n_unique_classes) / n_rows`` -- the number of rows that
        could be deleted without losing a distinct value, divided by all rows.

    The bare phrase "duplicate fraction" is deliberately never produced.
    """
    hashes = np.asarray(hashes, dtype=object)
    poisoned = np.asarray(is_poisoned, dtype=bool)
    n_rows = len(hashes)
    classes = equivalence_classes(hashes)
    sizes = np.array([len(v) for v in classes.values()])
    repeated = {h: v for h, v in classes.items() if len(v) >= 2}
    repeated_sizes = np.array([len(v) for v in repeated.values()]) if repeated else np.array([])
    n_repeated_members = int(repeated_sizes.sum()) if repeated_sizes.size else 0

    clean_only = poison_only = mixed = 0
    clean_only_rows = poison_only_rows = mixed_rows = 0
    mixed_poison_rows = mixed_clean_rows = 0
    r_clean_only = r_poison_only = r_mixed = 0
    r_clean_only_rows = r_poison_only_rows = r_mixed_rows = 0
    largest_h, largest_n = None, 0

    for h, idx in classes.items():
        n_p = int(poisoned[idx].sum())
        n_c = len(idx) - n_p
        big = len(idx) >= 2
        if len(idx) > largest_n:
            largest_h, largest_n = h, len(idx)
        if n_p == 0:
            clean_only += 1
            clean_only_rows += len(idx)
            if big:
                r_clean_only += 1
                r_clean_only_rows += len(idx)
        elif n_c == 0:
            poison_only += 1
            poison_only_rows += len(idx)
            if big:
                r_poison_only += 1
                r_poison_only_rows += len(idx)
        else:
            mixed += 1
            mixed_rows += len(idx)
            mixed_poison_rows += n_p
            mixed_clean_rows += n_c
            if big:
                r_mixed += 1
                r_mixed_rows += len(idx)

    n_poison = int(poisoned.sum())
    return {
        "stage": stage_name,
        "n_rows": int(n_rows),
        "n_unique_classes": int(len(classes)),
        "n_repeated_classes": int(len(repeated)),
        "n_repeated_member_rows": n_repeated_members,
        "repeated_member_fraction": float(n_repeated_members / n_rows),
        "redundancy_fraction": float((n_rows - len(classes)) / n_rows),
        "n_redundant_rows": int(n_rows - len(classes)),
        "largest_class_size": int(largest_n),
        "largest_class_share": float(largest_n / n_rows),
        "largest_class_signature": largest_h,
        "repeated_class_size_quantiles": (
            {str(q): float(np.percentile(repeated_sizes, q)) for q in SIZE_QUANTILES}
            if repeated_sizes.size else {}
        ),
        "mean_class_size": float(sizes.mean()),
        "all_classes": {
            "clean_only": int(clean_only), "poison_only": int(poison_only),
            "mixed": int(mixed),
            "clean_only_member_rows": int(clean_only_rows),
            "poison_only_member_rows": int(poison_only_rows),
            "mixed_member_rows": int(mixed_rows),
            "mixed_class_poison_rows": int(mixed_poison_rows),
            "mixed_class_clean_rows": int(mixed_clean_rows),
        },
        "repeated_classes": {
            "clean_only": int(r_clean_only), "poison_only": int(r_poison_only),
            "mixed": int(r_mixed),
            "clean_only_member_rows": int(r_clean_only_rows),
            "poison_only_member_rows": int(r_poison_only_rows),
            "mixed_member_rows": int(r_mixed_rows),
        },
        # The obstruction numerator: poison rows that share this stage's exact
        # signature with at least one clean row.
        "poison_rows_sharing_class_with_clean": int(mixed_poison_rows),
        "poison_obstruction_fraction": float(mixed_poison_rows / n_poison) if n_poison else 0.0,
    }


# ---------------------------------------------------------------------------
# earliest-merger attribution  (retrospective; uses labels only for composition)
# ---------------------------------------------------------------------------

def earliest_merger_attribution(stage_names, stage_hashes, is_poisoned):
    """Assign every finally-repeated row to the earliest stage where it merged.

    Each stage in the chain is a deterministic function of the one before it, so
    equality can only be created, never destroyed: if two rows are equal at
    stage *s* they are equal at every later stage.  Consequently every row has a
    well-defined *earliest* stage at which its equivalence class stopped being a
    singleton, and that assignment is mutually exclusive by construction.

    Monotonicity is asserted rather than assumed -- ``monotonicity_violations``
    must be zero, and a nonzero value invalidates the table rather than being
    quietly tolerated.
    """
    poisoned = np.asarray(is_poisoned, dtype=bool)
    n_rows = len(poisoned)
    if len(stage_names) != len(stage_hashes):
        raise ValueError("stage_names and stage_hashes must be the same length")

    repeated_flags = np.vstack([class_size_array(h) >= 2 for h in stage_hashes])

    violations = 0
    for s in range(1, len(stage_names)):
        violations += int(np.count_nonzero(repeated_flags[s - 1] & ~repeated_flags[s]))

    earliest = np.full(n_rows, -1, dtype=int)
    for s in range(len(stage_names) - 1, -1, -1):
        earliest[repeated_flags[s]] = s

    final_classes = equivalence_classes(stage_hashes[-1])
    final_mixed = np.zeros(n_rows, dtype=bool)
    for idx in final_classes.values():
        n_p = int(poisoned[idx].sum())
        if 0 < n_p < len(idx):
            final_mixed[idx] = True

    total_repeated = int(np.count_nonzero(earliest >= 0))
    rows = []
    for s, name in enumerate(stage_names):
        sel = earliest == s
        n_sel = int(np.count_nonzero(sel))
        stage_class_ids = set(np.asarray(stage_hashes[s], dtype=object)[sel].tolist())
        rows.append({
            "stage": name,
            "n_newly_merged_classes": len(stage_class_ids),
            "member_rows": n_sel,
            "clean_rows": int(np.count_nonzero(sel & ~poisoned)),
            "poison_rows": int(np.count_nonzero(sel & poisoned)),
            "mixed_class_poison_rows": int(np.count_nonzero(sel & poisoned & final_mixed)),
            "share_of_final_repeated_member_mass": (
                float(n_sel / total_repeated) if total_repeated else 0.0
            ),
        })

    return {
        "stage_order": list(stage_names),
        "n_final_repeated_member_rows": total_repeated,
        "monotonicity_violations": violations,
        "rows": rows,
        "attribution_sums_to_final_repeated_mass": (
            sum(r["member_rows"] for r in rows) == total_repeated
        ),
        "_earliest_stage_index": earliest,
    }


def transition_report(upstream_hashes, downstream_hashes, is_poisoned,
                      n_examples=3, rng_seed=20260729):
    """What one stage-to-stage transition does to the equivalence classes."""
    poisoned = np.asarray(is_poisoned, dtype=bool)
    up = np.asarray(upstream_hashes, dtype=object)
    down = np.asarray(downstream_hashes, dtype=object)
    down_classes = equivalence_classes(down)

    n_merging_classes = 0
    n_merged_rows = 0
    n_new_mixed = 0
    largest = {"size": 0, "n_upstream_classes": 0, "signature": None,
               "example_row_indices": [], "example_upstream_signatures": []}

    for h, idx in down_classes.items():
        up_sigs = set(up[idx].tolist())
        if len(up_sigs) <= 1:
            continue
        n_merging_classes += 1
        n_merged_rows += len(idx)
        n_p = int(poisoned[idx].sum())
        if 0 < n_p < len(idx):
            # Mixed here, and not already mixed inside any single upstream class.
            already = False
            for u in up_sigs:
                sub = idx[up[idx] == u]
                n_ps = int(poisoned[sub].sum())
                if 0 < n_ps < len(sub):
                    already = True
                    break
            if not already:
                n_new_mixed += 1
        if len(idx) > largest["size"]:
            ex = idx[:n_examples]
            largest = {
                "size": int(len(idx)),
                "n_upstream_classes": int(len(up_sigs)),
                "signature": h,
                "example_row_indices": [int(i) for i in ex],
                "example_upstream_signatures": [up[i] for i in ex],
            }

    return {
        "n_downstream_classes_merging_multiple_upstream_classes": n_merging_classes,
        "n_rows_in_newly_merged_classes": n_merged_rows,
        "n_mixed_classes_first_created_here": n_new_mixed,
        "largest_newly_merged_class": largest,
    }


# ---------------------------------------------------------------------------
# guardrail used by the test suite
# ---------------------------------------------------------------------------

LABEL_FREE_FUNCTIONS = (
    normalize_float_array, exact_hash, row_hashes, combine_hashes,
    pack_binary_mask, unpack_binary_mask, canonical_diagram_points,
    diagram_row_hashes, diagram_point_counts, equivalence_classes,
    class_size_array,
)

_LABEL_PARAM_NAMES = {"is_poisoned", "poisoned", "y", "labels", "y_true", "ground_truth"}


def label_free_signature_violations():
    """Names of stage-signature functions that accept a ground-truth argument.

    A stage signature must be constructible without knowing which rows are
    poisoned.  If this ever returns a nonempty list, a signature could have been
    conditioned on the answer.
    """
    bad = []
    for fn in LABEL_FREE_FUNCTIONS:
        params = set(inspect.signature(fn).parameters)
        if params & _LABEL_PARAM_NAMES:
            bad.append(fn.__name__)
    return bad
