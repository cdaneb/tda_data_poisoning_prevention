"""Phase Q3: matched extraction of every intermediate stage of the legacy pipeline.

This module is strictly additive.  It does not modify ``tda_pipeline.py`` and it
does not reimplement any transformer.  It calls the production
``extract_tda_features()`` exactly as the standing regression gate does, and then
replays the *already fitted* sub-pipelines step by step with ``.transform()`` to
recover the intermediates that ``fit_transform`` discards.

The final 60-vector rebuilt from those replayed blocks is required to be bitwise
equal to the production output.  That equality is asserted, not assumed, and it
is what licenses treating the recovered intermediates as the real ones.

Stage order (each stage is a deterministic function of the one before it):

    raw -> supported payload record -> binary mask -> filtration images
        -> unscaled diagrams -> scaled diagrams -> 12-feature blocks -> 60-vector

Nothing here accepts ground-truth poison labels.  Signatures are built from the
representation alone.
"""
from __future__ import annotations

import gc

import numpy as np

from phase_q3_collisions import (
    combine_hashes,
    diagram_point_counts,
    diagram_row_hashes,
    exact_hash,
    pack_binary_mask,
    row_hashes,
)
from phase_q_pipeline import filtration_specs
from tda_pipeline import extract_tda_features, reshape_for_tda

# Legacy filtration order, verbatim from ``build_tda_pipeline``.
FILTRATION_NAMES = tuple(name for name, _ in filtration_specs())

# The mutually exclusive earliest-merger chain.  Per-filtration stages are
# reported separately as an ablation diagnostic; only these combined stages form
# the attribution chain, because the final vector depends on all five.
CHAIN_STAGES = (
    "raw_payload",
    "supported_payload_record",
    "binary_mask",
    "filtration_images",
    "unscaled_diagrams",
    "scaled_diagrams",
    "feature_blocks",
    "final_60_vector",
)


# ---------------------------------------------------------------------------
# stage 1-2: raw bytes and the conservative support record
# ---------------------------------------------------------------------------

def support_records(X):
    """Conservative support boundary and padding profile for each payload row.

    ``support_end`` is one past the index of the last nonzero byte, so the row is
    exactly ``row[:support_end]`` followed by zeros.

    Two limitations are recorded rather than papered over:

    * Legitimate trailing zero bytes in the real transport payload cannot be
      recovered from the zero-padded 1500-byte array.  A payload that genuinely
      ended in ``0x00`` is indistinguishable from one that was padded there.
    * Payload-Byte's ``total_len`` is the IPv4 packet length, not a verified
      transport-payload length, so it is not used as the boundary here.

    Because the padding is zero by construction, the pair
    ``(support_end, row[:support_end])`` determines the whole row.  The support
    record is therefore a *bijection* with the raw row, and the audit asserts
    that it produces exactly the same equivalence classes.  It earns its place in
    the chain as a self-check and as the carrier of the padding profile, not as a
    stage that can merge anything on its own.
    """
    X = np.asarray(X, dtype=np.uint8)
    nonzero_counts = np.count_nonzero(X, axis=1)
    support_end = np.zeros(len(X), dtype=int)
    any_nonzero = nonzero_counts > 0
    # Last nonzero index via a reversed argmax on the boolean mask.
    rev = (X[any_nonzero] != 0)[:, ::-1]
    support_end[any_nonzero] = X.shape[1] - rev.argmax(axis=1)

    sigs = np.empty(len(X), dtype=object)
    for i in range(len(X)):
        end = support_end[i]
        sigs[i] = exact_hash(np.concatenate([
            np.array([end], dtype=np.int64).view(np.uint8),
            X[i, :end],
        ]))

    return {
        "signatures": sigs,
        "support_end": support_end,
        "nonzero_count": nonzero_counts,
        "padding_count": X.shape[1] - support_end,
        "all_zero": ~any_nonzero,
    }


# ---------------------------------------------------------------------------
# full instrumented extraction
# ---------------------------------------------------------------------------

def extract_all_stages(X, threshold=0.4, verbose=True):
    """Run the production pipeline and recover every intermediate stage.

    Returns a dict with per-row signature arrays for every stage, the fitted
    transformer state worth publishing, per-row descriptive counts, and the
    production feature matrix itself.
    """
    X = np.asarray(X, dtype=np.uint8)
    n = len(X)

    if verbose:
        print("  [q3] running the production pipeline (legacy call, unchanged)...")
    X_tda, pipeline = extract_tda_features(X, threshold=threshold)
    images = reshape_for_tda(X)

    stages = {}
    per_filtration = {}
    fitted = {"binarizer_max_value_": {}, "scaler_scale_": {}}
    descriptive = {}

    # --- stage 1: raw padded payload -------------------------------------
    if verbose:
        print("  [q3] stage: raw payload")
    stages["raw_payload"] = row_hashes(X)

    # --- stage 2: supported payload record --------------------------------
    if verbose:
        print("  [q3] stage: supported payload record")
    sup = support_records(X)
    stages["supported_payload_record"] = sup["signatures"]
    descriptive["support_end"] = sup["support_end"]
    descriptive["nonzero_count"] = sup["nonzero_count"]
    descriptive["padding_count"] = sup["padding_count"]
    descriptive["all_zero_payload"] = sup["all_zero"]

    # --- stage 3: threshold-0.4 binary mask -------------------------------
    # Every sub-pipeline holds its own fitted Binarizer.  They see the same
    # batch and the same threshold, so they must agree; that is asserted below
    # rather than assumed, and the mask is taken from the first one.
    if verbose:
        print("  [q3] stage: binary mask")
    masks = None
    for fname, (_, sub) in zip(FILTRATION_NAMES, pipeline.transformer_list):
        binarizer = sub.steps[0][1]
        fitted["binarizer_max_value_"][fname] = float(binarizer.max_value_)
        m = binarizer.transform(images)
        if masks is None:
            masks = m
        elif not np.array_equal(masks, m):
            raise AssertionError(
                f"binarizer disagreement between filtrations at {fname}; "
                "the five sub-pipelines must produce one identical mask"
            )
    packed, packed_shape = pack_binary_mask(masks)
    stages["binary_mask"] = row_hashes(packed)
    descriptive["foreground_count"] = masks.reshape(n, -1).sum(axis=1)
    descriptive["binary_mask_packed_shape"] = list(packed_shape)
    del packed
    gc.collect()

    # --- stages 4-7: per filtration ---------------------------------------
    filt_img_hashes, undiag_hashes, scdiag_hashes, block_hashes = [], [], [], []
    blocks = []
    for fname, (_, sub) in zip(FILTRATION_NAMES, pipeline.transformer_list):
        if verbose:
            print(f"  [q3] stage: filtration/diagram/block for {fname}")
        filt = sub.steps[1][1].transform(masks)
        filt_img_hashes.append(row_hashes(filt))

        undiag = sub.steps[2][1].transform(filt)
        del filt
        undiag_hashes.append(diagram_row_hashes(undiag))
        counts, by_dim = diagram_point_counts(undiag)
        descriptive[f"n_valid_points_{fname}"] = counts
        descriptive[f"n_valid_points_H0_{fname}"] = by_dim[0]
        descriptive[f"n_valid_points_H1_{fname}"] = by_dim[1]
        descriptive[f"n_nonfinite_diagram_entries_{fname}"] = int(
            (~np.isfinite(undiag)).sum())

        scaler = sub.steps[3][1]
        fitted["scaler_scale_"][fname] = float(np.asarray(scaler.scale_).reshape(-1)[0])
        scdiag = scaler.transform(undiag)
        del undiag
        scdiag_hashes.append(diagram_row_hashes(scdiag))

        block = sub.steps[4][1].transform(scdiag)
        del scdiag
        gc.collect()
        if block.shape[1] != 12:
            raise AssertionError(f"{fname} produced {block.shape[1]} features, expected 12")
        blocks.append(block)
        block_hashes.append(row_hashes(block))

        per_filtration[fname] = {
            "filtration_image": filt_img_hashes[-1],
            "unscaled_diagram": undiag_hashes[-1],
            "scaled_diagram": scdiag_hashes[-1],
            "feature_block": block_hashes[-1],
        }

    del masks
    gc.collect()

    stages["filtration_images"] = combine_hashes(filt_img_hashes)
    stages["unscaled_diagrams"] = combine_hashes(undiag_hashes)
    stages["scaled_diagrams"] = combine_hashes(scdiag_hashes)
    stages["feature_blocks"] = combine_hashes(block_hashes)

    # --- stage 8: the final 60-vector, and the equality that licenses it ----
    rebuilt = np.ascontiguousarray(np.concatenate(blocks, axis=1))
    production = np.ascontiguousarray(X_tda)
    if rebuilt.shape != production.shape:
        raise AssertionError(
            f"instrumented shape {rebuilt.shape} != production {production.shape}")
    exact_equal = bool(np.array_equal(rebuilt, production))
    bitwise_equal = bool(rebuilt.tobytes() == production.tobytes())
    if not exact_equal:
        raise AssertionError(
            "instrumented 60-vector is not exactly equal to the production "
            f"output; max abs diff {np.abs(rebuilt - production).max():.3e}. "
            "Refusing to relax to allclose."
        )
    stages["final_60_vector"] = row_hashes(production)

    return {
        "stages": stages,
        "chain_stages": list(CHAIN_STAGES),
        "per_filtration": per_filtration,
        "filtration_names": list(FILTRATION_NAMES),
        "fitted_state": fitted,
        "descriptive": descriptive,
        "X_tda": production,
        "equality_check": {
            "instrumented_equals_production_exactly": exact_equal,
            "instrumented_equals_production_bitwise": bitwise_equal,
            "production_feature_hash": exact_hash(production),
            "instrumented_feature_hash": exact_hash(rebuilt),
            "shape": list(production.shape),
        },
        "threshold": float(threshold),
        "effective_byte_cut": {
            k: float(threshold) * v for k, v in fitted["binarizer_max_value_"].items()
        },
    }
