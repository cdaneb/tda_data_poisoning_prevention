"""Structural validators for Phase Q2.

These are the assertions the handoff names: that the legacy behaviour is
untouched, that matched arms really saw matched inputs, that the accounting
views cannot change what was captured, that discovery never sees the poison
mask, and that confirmation only ever evaluates candidates the label-free rule
locked.

Fast unit tests run unconditionally.  Tests that read a Phase Q2 artifact skip
cleanly when the artifact has not been produced yet, so this file is runnable
before the long jobs finish.

    python -m unittest tools.test_phase_q2 -v
"""
from __future__ import annotations

import json
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT))

from paths import RESULTS_DIR
from tda_pipeline import build_tda_pipeline, reshape_for_tda
from phase_q2_source_pipeline import (
    GEOMETRIES, build_geometry_pipeline, extract_geometry_features,
    geometry_spec, reshape_for_geometry,
)
from tools.phase_q2_common import accounting_views, array_hash, color_cluster_table
import run_phase_q2_optics_sensitivity as sens

RESULTS = Path(RESULTS_DIR)
ACCOUNTING_JSON = RESULTS / "phase_q2_accounting_audit.json"
GEOMETRY_JSON = RESULTS / "phase_q2_geometry.json"
SENSITIVITY_JSON = RESULTS / "phase_q2_optics_sensitivity.json"


def load(path):
    if not path.exists():
        raise unittest.SkipTest(f"{path.name} not produced yet")
    with open(path) as fh:
        return json.load(fh)


def walk_numbers(obj, path="$"):
    """Yield (path, value) for every float/int in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield path, obj


class GeometryPipelineTests(unittest.TestCase):
    """The additive geometry module must not have changed the legacy arm."""

    def test_legacy_geometry_reproduces_legacy_pipeline_exactly(self):
        rng = np.random.RandomState(7)
        X = rng.randint(0, 256, size=(6, 1500)).astype(np.uint8)
        legacy = build_tda_pipeline(threshold=0.4).fit_transform(reshape_for_tda(X))
        via_geom = build_geometry_pipeline("legacy", threshold=0.4).fit_transform(
            reshape_for_geometry(X, "legacy"))
        np.testing.assert_array_equal(legacy, via_geom)
        self.assertEqual(legacy.shape, (6, 60))

    def test_both_rasters_preserve_every_payload_byte(self):
        rng = np.random.RandomState(8)
        X = rng.randint(0, 256, size=(4, 1500)).astype(np.uint8)
        for geometry in GEOMETRIES:
            shape = geometry_spec(geometry)["image_shape"]
            self.assertEqual(shape[0] * shape[1], 1500, geometry)
            img = reshape_for_geometry(X, geometry)
            np.testing.assert_array_equal(img.reshape(len(X), 1500), X)

    def test_reshape_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            reshape_for_geometry(np.zeros((3, 1400)), "source")

    def test_source_geometry_uses_the_printed_centers_unmodified(self):
        centers = [c.tolist() for c in geometry_spec("source")["centers"]]
        self.assertEqual(centers, [[0, 1500], [0, 750], [1500, 0]])
        self.assertEqual(list(geometry_spec("source")["image_shape"]), [1, 1500])


class AccountingInvariantTests(unittest.TestCase):
    """A display convention may not change what was captured."""

    def _fixture(self):
        labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2])
        poisoned = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        return labels, poisoned

    def test_all_three_views_sum_to_100(self):
        labels, poisoned = self._fixture()
        views = accounting_views(labels, poisoned)
        for name, block in views["views"].items():
            self.assertAlmostEqual(block["sum_pct"], 100.0, places=9, msg=name)

    def test_true_capture_is_view_independent(self):
        labels, poisoned = self._fixture()
        views = accounting_views(labels, poisoned)
        # 5 poisoned samples in total (indices 0, 2, 6, 7, 8). Cluster 1 is the
        # only 100%-poison cluster and holds 2 of them, so capture is 2/5.
        self.assertEqual(views["n_poison_total"], 5)
        self.assertEqual(views["n_poison_in_red_clusters"], 2)
        self.assertAlmostEqual(views["true_poison_capture_pct"], 40.0)

    def test_noise_is_never_counted_as_captured(self):
        """Label -1 holds poison here; folding it into a display bucket must not
        move capture, and it must never be colored Red."""
        labels, poisoned = self._fixture()
        views = accounting_views(labels, poisoned)
        baseline = views["true_poison_capture_pct"]
        noise_rows = [r for r in views["clusters"] if r["cluster_id"] == -1]
        self.assertEqual(len(noise_rows), 1)
        self.assertEqual(noise_rows[0]["color"], "Noise")
        self.assertGreater(noise_rows[0]["n_poisoned"], 0)
        yellow_view = views["views"]["noise_as_yellow_display"]["color_shares_pct"]
        self.assertNotIn("Noise", yellow_view)
        self.assertEqual(views["true_poison_capture_pct"], baseline)
        self.assertAlmostEqual(
            views["views"]["noise_as_yellow_display"]["color_shares_pct"]["Red"],
            views["views"]["all_sample_denominator"]["color_shares_pct"]["Red"])

    def test_color_table_matches_project_purity_literals(self):
        from clustering import classify_clusters
        labels, poisoned = self._fixture()
        ours = color_cluster_table(labels, poisoned)
        theirs, _ = classify_clusters(labels, poisoned)
        self.assertEqual([r["color"] for r in ours], [c["color"] for c in theirs])
        self.assertEqual([r["size"] for r in ours], [int(c["size"]) for c in theirs])


class DiscoveryLabelBlindnessTests(unittest.TestCase):
    """Discovery must be a pure function of label-free structure."""

    def _records(self):
        return [
            {"name": "a", "config": {"min_samples": 5},
             "structure": {"n_clusters": 7, "largest_cluster_share": 0.30,
                           "unclustered_fraction": 0.10}},
            {"name": "b", "config": {"min_samples": 2},
             "structure": {"n_clusters": 1, "largest_cluster_share": 0.99,
                           "unclustered_fraction": 0.00}},
            {"name": "c", "config": {"min_samples": 10},
             "structure": {"n_clusters": 40, "largest_cluster_share": 0.20,
                           "unclustered_fraction": 0.05}},
            {"name": "d", "config": {"min_samples": 25},
             "structure": {"n_clusters": 8, "largest_cluster_share": 0.50,
                           "unclustered_fraction": 0.40}},
        ]

    def test_selection_ignores_poison_fields(self):
        clean = self._records()
        poisoned = [dict(r) for r in clean]
        # Attach exactly the statistics that would tempt a label-aware rule,
        # inverted so a leak would change the answer.
        for i, r in enumerate(poisoned):
            r["true_poison_capture_pct"] = 90.0 - 30.0 * i
            r["is_poisoned"] = np.ones(10, dtype=bool)
            r["structure"] = dict(r["structure"], red_capture=99.0 - i)
        self.assertEqual(sens.discover_candidates(clean),
                         sens.discover_candidates(poisoned))

    def test_degenerate_single_cluster_is_rejected(self):
        picked = [c["name"] for c in sens.discover_candidates(self._records())]
        self.assertNotIn("b", picked)  # 1 cluster, 99% in it, 0% unclustered

    def test_rank_is_unclustered_fraction_then_cluster_count_distance(self):
        picked = [c["name"] for c in sens.discover_candidates(self._records())]
        self.assertEqual(picked, ["c", "a", "d"])

    def test_selection_rule_reads_only_declared_keys(self):
        records = self._records()
        for r in records:
            r["structure"] = {k: r["structure"][k] for k in sens.LABEL_FREE_KEYS}
        self.assertEqual(len(sens.discover_candidates(records)), 3)


class AccountingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.d = load(ACCOUNTING_JSON)

    def test_realization_is_fully_recorded(self):
        r = self.d["realization"]
        self.assertEqual(r["dataset"], "UNSW-NB15 Payload-Byte")
        self.assertEqual(r["seed"], 42)
        self.assertEqual((r["n_total"], r["n_clean"], r["n_poison"]), (5500, 5000, 500))
        for key in ("input_hash", "poison_mask_hash", "subsample_index_hash"):
            self.assertRegex(r[key], r"^[0-9a-f]{16}$")

    def test_legacy_behaviour_unchanged(self):
        self.assertEqual(self.d["feature_map"]["shape"], [5500, 60])
        self.assertAlmostEqual(
            self.d["accounting"]["true_poison_capture_pct"], 2.2000, places=4)

    def test_capture_identical_across_views_and_noise_excluded(self):
        acc = self.d["accounting"]
        for block in acc["views"].values():
            self.assertAlmostEqual(block["sum_pct"], 100.0, places=6)
        self.assertNotIn("Noise", acc["views"]["clustered_only_denominator"]["color_shares_pct"])
        self.assertNotIn("Noise", acc["views"]["noise_as_yellow_display"]["color_shares_pct"])
        self.assertFalse(self.d["source_comparison"]["does_denominator_choice_change_true_capture"])
        # Red must be exactly the poison the Red clusters hold, in every view.
        red = [c for c in acc["clusters"] if c["color"] == "Red"]
        self.assertEqual(sum(c["n_poisoned"] for c in red), acc["n_poison_in_red_clusters"])
        for c in red:
            self.assertEqual(c["poison_fraction"], 1.0)

    def test_no_unexplained_nonfinite_numbers(self):
        allowed = ("infinite", "reachability", "core_distance")
        for path, value in walk_numbers(self.d):
            if isinstance(value, float) and not math.isfinite(value):
                self.assertTrue(any(a in path for a in allowed),
                                f"unexplained non-finite at {path}: {value}")

    def test_optics_params_are_recorded(self):
        params = self.d["optics"]["internals"]["params"]
        self.assertEqual(params["min_samples"], 5)
        self.assertEqual(params["max_eps"], 2.0)
        self.assertIn("cluster_method", params)
        self.assertIn("metric", params)


class GeometryArtifactTests(unittest.TestCase):
    def setUp(self):
        self.d = load(GEOMETRY_JSON)
        if self.d.get("status") == "source_geometry_rejected":
            self.skipTest("source geometry rejected by the pinned library")

    def test_arms_share_one_input_and_one_poison_mask(self):
        r = self.d["realization"]
        self.assertEqual(r["seed"], 42)
        self.assertEqual(r["n_total"], 5500)
        # A single realization block is shared, so the arms cannot diverge; the
        # per-arm feature hashes must nonetheless differ, or nothing was tested.
        hashes = {k: v["feature_diagnostics"]["feature_hash"] for k, v in self.d["arms"].items()}
        self.assertEqual(len(set(hashes.values())), len(hashes), hashes)

    def test_geometries_are_the_printed_and_operational_ones(self):
        self.assertEqual(self.d["geometries"]["source"]["image_shape"], [1, 1500])
        self.assertEqual(self.d["geometries"]["source"]["centers"],
                         [[0, 1500], [0, 750], [1500, 0]])
        self.assertEqual(self.d["geometries"]["legacy"]["image_shape"], [30, 50])
        self.assertEqual(self.d["geometries"]["legacy"]["centers"],
                         [[0, 50], [0, 25], [30, 0]])

    def test_legacy_arm_still_reproduces_the_gate(self):
        legacy = self.d["arms"]["legacy_30x50_t04"]
        self.assertEqual(legacy["feature_diagnostics"]["shape"], [5500, 60])
        self.assertAlmostEqual(
            legacy["accounting"]["true_poison_capture_pct"], 2.2000, places=4)

    def test_one_factor_at_a_time(self):
        """The geometry comparison must not also move the threshold, and the
        threshold comparison must not also move the geometry."""
        gc = self.d["geometry_comparison"]
        self.assertEqual(gc["arms"], ["legacy_30x50_t04", "source_1x1500_t04"])
        for arm in gc["arms"]:
            self.assertTrue(arm.endswith("_t04"), arm)
        if "threshold_comparison" in self.d:
            tc = self.d["threshold_comparison"]
            self.assertTrue(all(a.startswith("source_1x1500") for a in tc["arms"]), tc["arms"])

    def test_optics_held_fixed_across_geometry_arms(self):
        configs = [self.d["arms"][a]["optics"]["requested_params"]
                   for a in self.d["geometry_comparison"]["arms"]]
        self.assertEqual(configs[0], configs[1])

    def test_fixture_probe_records_acceptance_and_homology(self):
        probe = self.d["fixture_probe"]
        for geometry in ("legacy", "source"):
            entry = probe[geometry]
            self.assertIn("accepted_by_giotto_tda", entry)
            if entry["accepted_by_giotto_tda"]:
                self.assertIn("output_feature_count", entry)
                self.assertEqual(len(entry["per_filtration_diagrams"]), 5)
                for f in entry["per_filtration_diagrams"]:
                    self.assertIn("nontrivial_points_by_dim", f)


class SensitivityArtifactTests(unittest.TestCase):
    def setUp(self):
        self.d = load(SENSITIVITY_JSON)

    def test_selection_rule_is_declared_and_label_free(self):
        rule = self.d["selection_rule"]
        self.assertTrue(rule["fixed_before_any_cell_was_run"])
        self.assertEqual(set(rule["label_free_keys_read"]),
                         {"n_clusters", "largest_cluster_share", "unclustered_fraction"})

    def test_discovery_records_carry_no_poison_statistics(self):
        """No discovery record may carry a ground-truth statistic.

        Matched on whole `_`/`.`-delimited tokens, not substrings: a naive
        substring check flags "n_unclustered" for containing "red".
        """
        forbidden = {"poison", "poisoned", "capture", "purity", "red", "pink",
                     "green", "yellow", "dpdc", "precision", "removal"}
        for rec in self.d["discovery"]["records"]:
            for path, _ in walk_numbers(rec):
                tokens = set(re.split(r"[._\[\]]+", path.lower()))
                leaked = tokens & forbidden
                self.assertFalse(leaked,
                                 f"poison statistic leaked into discovery at {path}: {leaked}")

    def test_discovery_is_single_seed(self):
        self.assertEqual(self.d["discovery"]["seed"], 42)
        self.assertEqual(self.d["discovery"]["realization"]["seed"], 42)

    def test_locked_candidates_are_exactly_what_the_rule_selects(self):
        recomputed = sens.discover_candidates(self.d["discovery"]["records"])
        self.assertEqual([c["name"] for c in recomputed],
                         [c["name"] for c in self.d["locked_candidates"]])

    def test_confirmation_only_evaluates_locked_candidates(self):
        if "confirmation" not in self.d:
            self.skipTest("confirmation stage not run")
        locked = {c["name"] for c in self.d["locked_candidates"]}
        for result in self.d["confirmation"]["results"]:
            self.assertIn(result["candidate"]["name"], locked)
            self.assertEqual(result["candidate"]["config"],
                             next(c["config"] for c in self.d["locked_candidates"]
                                  if c["name"] == result["candidate"]["name"]))

    def test_confirmation_uses_all_five_seeds_with_population_sd(self):
        if "confirmation" not in self.d:
            self.skipTest("confirmation stage not run")
        self.assertEqual(self.d["confirmation"]["seeds"], [42, 123, 456, 789, 1024])
        for result in self.d["confirmation"]["results"]:
            seeds = [s["seed"] for s in result["per_seed"]]
            self.assertEqual(seeds, [42, 123, 456, 789, 1024])
            for s in result["per_seed"]:
                self.assertEqual(
                    (s["realization"]["n_total"], s["realization"]["n_poison"]), (5500, 500))
            summary = result["summary"]["true_poison_capture_pct"]
            per_seed = np.asarray(summary["per_seed"], dtype=np.float64)
            self.assertAlmostEqual(summary["mean"], float(per_seed.mean()), places=9)
            self.assertAlmostEqual(summary["sd_pop"], float(per_seed.std(ddof=0)), places=9)

    def test_baseline_cell_still_reproduces_the_gate(self):
        """min_samples=5 / max_eps=2.0 appears in the sweep; it is the gate cell."""
        if "confirmation" not in self.d:
            self.skipTest("confirmation stage not run")
        for result in self.d["confirmation"]["results"]:
            cfg = result["candidate"]["config"]
            if cfg.get("min_samples") == 5 and cfg.get("max_eps") == 2.0 \
                    and cfg.get("min_cluster_size") is None:
                seed42 = next(s for s in result["per_seed"] if s["seed"] == 42)
                self.assertAlmostEqual(seed42["true_poison_capture_pct"], 2.2000, places=4)

    def test_out_of_range_cells_are_labeled(self):
        for rec in self.d["discovery"]["records"]:
            if not rec["in_paper_range"]:
                self.assertIn("outside paper range", rec["name"])

    def test_extraction_method_is_reported_unresolved(self):
        self.assertIn("UNRESOLVED", self.d["extraction_method_status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
