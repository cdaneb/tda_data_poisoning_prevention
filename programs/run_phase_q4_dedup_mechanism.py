"""Phase Q4: exact-payload clean-frame deduplication mechanism test.

The control is the completed Q3 artifact.  Q4 rebuilds the same seeded
5,000-row clean draw, removes exact 1,500-byte duplicates without backfilling,
then runs the unchanged R60 attack, threshold-0.4 TDA pipeline, and OPTICS.

Usage:
    python run_phase_q4_dedup_mechanism.py
    python run_phase_q4_dedup_mechanism.py --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from programs.phase_q3_collisions import class_stats, earliest_merger_attribution, exact_hash
from programs.phase_q3_stage_pipeline import CHAIN_STAGES, extract_all_stages
from programs.phase_q4_frame import build_q4_frames, q4_realization_provenance
from programs.run_phase_q3_collision_audit import OPTICS_PARAMS, q3c, q3d, run_optics
from tools.phase_q2_common import CONFIRMATION_SEEDS, environment_block, write_json

Q3_PATH = ROOT / "results" / "phase_q3_collision_audit.json"
OUT_PATH = ROOT / "results" / "phase_q4_dedup_mechanism.json"

PREREGISTRATION = {
    "recorded_before_q4_results": True,
    "single_variable": (
        "after the seeded 5000-row clean draw and before poison generation, "
        "deduplicate exact 1500-byte payloads, retain first occurrence, preserve "
        "sample order, and do not backfill"
    ),
    "control": "completed Q3 standing R60 frame and artifact",
    "held_fixed": [
        "UNSW-NB15 Payload-Byte dataset",
        "seeded 5000-row pre-dedup draw",
        "malicious_random_attack with n_swaps=60",
        "poison_rate=0.10 applied after frame construction",
        "threshold=0.4 and 30x50 raster",
        "two Height and three Radial filtrations",
        "CubicalPersistence, Scaler, PersistenceEntropy, five Amplitude summaries",
        "OPTICS(min_samples=5, max_eps=2.0) and default xi extraction",
        "seeds 42, 123, 456, 789, 1024",
    ],
    "primary_mechanism_endpoints": [
        "raw first-merger poison rows divided by all poison rows",
        "poison coordinate-class obstruction fraction at the final 60-vector",
    ],
    "secondary_endpoints": [
        "raw first-merger share of final repeated-member collision mass",
        "binary and final repeated-member and redundancy fractions",
        "strict-100%-purity poison capture",
        "poison and clean unclustered fractions",
        "Q3-D exhaustive failure decomposition",
    ],
    "interpretation_lock": (
        "deduplication must eliminate exact duplicates among clean rows, but raw "
        "no-op poison remains identical to its retained source. Any capture change "
        "is a frame-composition mechanism result, not a reproduction improvement "
        "or a deployable defense claim"
    ),
    "prohibited": [
        "no threshold selection", "no threshold/block weighting",
        "no filtration or feature selection", "no OPTICS tuning",
        "no attack resampling", "no seed deletion", "no second dataset",
    ],
}


def _stage(block, name):
    return next(
        row for row in block["Q3_C_population_attribution"]["stage_class_stats"]
        if row["stage"] == name)


def metric_snapshot(block, n_clean, n_poison):
    c = block["Q3_C_population_attribution"]
    d = block["Q3_D_strict_purity_failure_decomposition"]
    raw_row = next(
        row for row in c["earliest_merger_attribution"]["rows"]
        if row["stage"] == "raw_payload")
    out = {
        "n_clean": int(n_clean),
        "n_poison": int(n_poison),
        "raw_repeated_member_fraction": _stage(block, "raw_payload")["repeated_member_fraction"],
        "raw_redundancy_fraction": _stage(block, "raw_payload")["redundancy_fraction"],
        "binary_repeated_member_fraction": _stage(block, "binary_mask")["repeated_member_fraction"],
        "binary_redundancy_fraction": _stage(block, "binary_mask")["redundancy_fraction"],
        "final_repeated_member_fraction": _stage(block, "final_60_vector")["repeated_member_fraction"],
        "final_redundancy_fraction": _stage(block, "final_60_vector")["redundancy_fraction"],
        "raw_first_merger_member_rows": int(raw_row["member_rows"]),
        "raw_first_merger_poison_rows": int(raw_row["poison_rows"]),
        "raw_first_merger_poison_rate": float(raw_row["poison_rows"] / n_poison),
        "raw_first_merger_share_of_collision_mass": raw_row[
            "share_of_final_repeated_member_mass"],
        "coordinate_class_obstruction_fraction": d[
            "coordinate_class_obstruction"]["obstruction_fraction"],
        "exact_purity_capture_pct": d["comparisons"]["observed_exact_purity_capture_pct"],
        "poison_unclustered_fraction": d["comparisons"]["poison_unclustered_fraction"],
        "clean_unclustered_fraction": d["comparisons"]["clean_unclustered_fraction"],
        "n_clusters": int(d["comparisons"]["n_clusters"]),
    }
    for name, item in d["decomposition"].items():
        out[f"failure_pct::{name}"] = item["pct_of_poison"]
    return out


def _control_block(q3_seed):
    return {
        "Q3_C_population_attribution": q3_seed["Q3_C_population_attribution"],
        "Q3_D_strict_purity_failure_decomposition": q3_seed[
            "Q3_D_strict_purity_failure_decomposition"],
    }


def audit_seed(seed, q3_doc):
    t0 = time.time()
    print(f"\n{'=' * 70}\n=== Phase Q4 deduplication mechanism, seed {seed} ===\n{'=' * 70}")
    standing, real = build_q4_frames(seed)
    q3_seed = q3_doc["seeds"][str(seed)]
    if standing["input_hash"] != q3_seed["realization"]["input_hash"]:
        raise AssertionError(f"seed {seed}: standing input hash differs from Q3")
    if standing["poison_mask_hash"] != q3_seed["realization"]["poison_mask_hash"]:
        raise AssertionError(f"seed {seed}: standing poison mask differs from Q3")

    print(
        f"  [q4] clean frame {real['deduplication']['n_before']} -> "
        f"{real['deduplication']['n_after']} rows; "
        f"poison={real['n_poison']}")
    extracted = extract_all_stages(real["X_combined"], threshold=0.4)
    print("  [q4] instrumented 60-vector equals production:",
          extracted["equality_check"]["instrumented_equals_production_bitwise"])
    labels, _ = run_optics(extracted["X_tda"])
    attribution = earliest_merger_attribution(
        list(CHAIN_STAGES),
        [extracted["stages"][name] for name in CHAIN_STAGES],
        real["is_poisoned"])
    dedup_q3c = q3c(real, extracted, attribution)
    dedup_q3d = q3d(real, extracted, labels)

    clean_raw_stats = class_stats(
        extracted["stages"]["raw_payload"][:real["n_clean"]],
        np.zeros(real["n_clean"], dtype=bool), "deduplicated_clean_raw_payload")
    if clean_raw_stats["n_repeated_member_rows"] != 0:
        raise AssertionError("deduplication left repeated exact payloads among clean rows")

    dedup_block = {
        "Q3_C_population_attribution": dedup_q3c,
        "Q3_D_strict_purity_failure_decomposition": dedup_q3d,
    }
    control_metrics = metric_snapshot(
        _control_block(q3_seed), q3_seed["realization"]["n_clean"],
        q3_seed["realization"]["n_poison"])
    dedup_metrics = metric_snapshot(dedup_block, real["n_clean"], real["n_poison"])
    deltas = {
        key: float(dedup_metrics[key] - control_metrics[key])
        for key in dedup_metrics
        if isinstance(dedup_metrics[key], (int, float))
        and key not in {"n_clean", "n_poison", "raw_first_merger_member_rows",
                        "raw_first_merger_poison_rows", "n_clusters"}
    }

    block = {
        "seed": int(seed),
        "standing_probe": standing,
        "deduplicated_realization": q4_realization_provenance(real),
        "deduplicated_clean_raw_class_stats": clean_raw_stats,
        "equality_check": extracted["equality_check"],
        "fitted_state": extracted["fitted_state"],
        "effective_byte_cut": extracted["effective_byte_cut"],
        "optics_params": OPTICS_PARAMS,
        "labels_hash": exact_hash(np.asarray(labels)),
        "control_metrics_from_q3": control_metrics,
        "deduplicated_metrics": dedup_metrics,
        "deduplicated_minus_control": deltas,
        "deduplicated_Q3_C_population_attribution": dedup_q3c,
        "deduplicated_Q3_D_failure_decomposition": dedup_q3d,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"  [q4] seed {seed} done in {block['elapsed_seconds']}s")
    return block


def five_seed_summary(blocks):
    metric_keys = sorted(blocks[0]["deduplicated_minus_control"])

    def stats(values):
        a = np.asarray(values, dtype=float)
        return {
            "per_seed": [float(v) for v in a],
            "mean": float(a.mean()),
            "sd_pop": float(a.std(ddof=0)),
        }

    summary = {
        "seeds": [int(b["seed"]) for b in blocks],
        "deduplication": {
            "n_clean_after": stats([
                b["deduplicated_realization"]["n_clean"] for b in blocks]),
            "n_clean_removed": stats([
                b["deduplicated_realization"]["deduplication"]["n_removed"]
                for b in blocks]),
            "n_label_conflict_payload_classes": stats([
                b["deduplicated_realization"]["deduplication"]
                ["n_label_conflict_payload_classes_before"] for b in blocks]),
            "raw_noop_fraction_control": stats([
                b["standing_probe"]["attack_diagnostics"]["raw_noop_fraction"]
                for b in blocks]),
            "raw_noop_fraction_deduplicated": stats([
                b["deduplicated_realization"]["attack_diagnostics"]["raw_noop_fraction"]
                for b in blocks]),
        },
        "metrics": {},
    }
    for key in metric_keys:
        summary["metrics"][key] = {
            "control": stats([b["control_metrics_from_q3"][key] for b in blocks]),
            "deduplicated": stats([b["deduplicated_metrics"][key] for b in blocks]),
            "delta": stats([b["deduplicated_minus_control"][key] for b in blocks]),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(CONFIRMATION_SEEDS))
    ap.add_argument("--q3", default=str(Q3_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    with open(args.q3) as fh:
        q3_doc = json.load(fh)
    missing = set(map(str, args.seeds)) - set(q3_doc["seeds"])
    if missing:
        raise ValueError(f"Q3 artifact lacks requested seeds: {sorted(missing)}")

    t0 = time.time()
    out_path = Path(args.out)
    completed = {}
    elapsed_before = 0.0
    if out_path.exists():
        with open(out_path) as fh:
            prior = json.load(fh)
        if prior.get("phase") != "Q4":
            raise ValueError(f"refusing to resume non-Q4 artifact: {out_path}")
        if prior.get("preregistration") != PREREGISTRATION:
            raise ValueError("refusing to resume artifact with a different preregistration")
        completed = dict(prior.get("seeds", {}))
        elapsed_before = float(prior.get("elapsed_seconds", 0.0))

    requested = [int(seed) for seed in args.seeds]

    def save():
        blocks = [completed[str(seed)] for seed in requested if str(seed) in completed]
        all_requested_complete = len(blocks) == len(requested)
        payload = {
            "phase": "Q4",
            "description": (
                "Single-variable mechanism test of whether exact duplicate clean "
                "payloads in the standing frame are causal for collision obstruction "
                "and strict-purity poison capture"
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment_block(),
            "preregistration": PREREGISTRATION,
            "control_artifact": str(Path(args.q3).resolve()),
            "chain_stages": list(CHAIN_STAGES),
            "requested_seeds": requested,
            "seeds": {str(b["seed"]): b for b in blocks},
            "five_seed_summary": (
                five_seed_summary(blocks)
                if all_requested_complete and len(blocks) == len(CONFIRMATION_SEEDS)
                else None
            ),
            "complete": all_requested_complete,
            "elapsed_seconds": round(elapsed_before + time.time() - t0, 1),
        }
        write_json(out_path, payload)
        return payload

    for seed in requested:
        if str(seed) in completed:
            print(f"  [resume] seed {seed} already present; skipping")
            continue
        block = audit_seed(seed, q3_doc)
        completed[str(seed)] = block
        save()

    payload = save()
    print(f"\nTotal recorded wall-clock: {payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
