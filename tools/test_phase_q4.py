"""Unit and artifact-structure tests for Phase Q4."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from programs.phase_q4_frame import stable_exact_payload_deduplicate

ARTIFACT = ROOT / "results" / "phase_q4_dedup_mechanism.json"
SEEDS = [42, 123, 456, 789, 1024]


class DeduplicationUnitTests(unittest.TestCase):
    def test_exact_payload_dedup_is_stable_label_free_and_no_backfill(self):
        X = np.zeros((6, 1500), dtype=np.uint8)
        X[0, 0] = 10
        X[1, 0] = 20
        X[2] = X[0]
        X[3, 0] = 30
        X[4] = X[1]
        X[5] = X[1]
        y = np.array([7, 1, 9, 2, 1, 3])
        Xu, yu, keep, meta = stable_exact_payload_deduplicate(X, y)
        self.assertEqual(keep.tolist(), [0, 1, 3])
        self.assertTrue(np.array_equal(Xu, X[keep]))
        self.assertEqual(yu.tolist(), [7, 1, 2])
        self.assertEqual(meta["n_before"], 6)
        self.assertEqual(meta["n_after"], 3)
        self.assertEqual(meta["n_removed"], 3)
        self.assertEqual(meta["n_label_conflict_payload_classes_before"], 2)
        self.assertEqual(meta["n_rows_in_label_conflict_classes_before"], 5)

    def test_rejects_non_payload_shape(self):
        with self.assertRaises(ValueError):
            stable_exact_payload_deduplicate(np.zeros((3, 10)), np.zeros(3))


class ArtifactStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.exists():
            raise unittest.SkipTest(f"{ARTIFACT.name} not generated yet")
        with open(ARTIFACT) as fh:
            cls.doc = json.load(fh)

    def test_seed_keys_and_preregistration(self):
        self.assertEqual(sorted(int(k) for k in self.doc["seeds"]), SEEDS)
        self.assertEqual(self.doc["requested_seeds"], SEEDS)
        self.assertTrue(self.doc["complete"])
        self.assertTrue(self.doc["preregistration"]["recorded_before_q4_results"])
        self.assertIn("do not backfill", self.doc["preregistration"]["single_variable"])

    def test_only_frame_is_changed_and_control_matches_q3(self):
        for key, b in self.doc["seeds"].items():
            s = b["standing_probe"]
            d = b["deduplicated_realization"]
            self.assertTrue(s["replay_matches_q3_input"], key)
            self.assertTrue(s["replay_matches_q3_poison_mask"], key)
            self.assertEqual(s["max_samples"], 5000, key)
            self.assertEqual(d["deduplication"]["n_before"], 5000, key)
            self.assertEqual(d["n_clean"], d["deduplication"]["n_after"], key)
            self.assertEqual(d["n_poison"], int(d["n_clean"] * 0.10), key)
            self.assertEqual(d["n_total"], d["n_clean"] + d["n_poison"], key)
            self.assertEqual(d["attack"], "malicious_random_attack", key)
            self.assertEqual(d["n_swaps"], 60, key)
            self.assertEqual(d["poison_rate"], 0.10, key)
            self.assertEqual(b["optics_params"], {"min_samples": 5, "max_eps": 2.0})

    def test_clean_payloads_are_exactly_unique_after_dedup(self):
        for key, b in self.doc["seeds"].items():
            st = b["deduplicated_clean_raw_class_stats"]
            self.assertEqual(st["n_rows"], st["n_unique_classes"], key)
            self.assertEqual(st["n_repeated_classes"], 0, key)
            self.assertEqual(st["n_repeated_member_rows"], 0, key)
            self.assertEqual(st["n_redundant_rows"], 0, key)

    def test_instrumented_pipeline_and_failure_decomposition(self):
        for key, b in self.doc["seeds"].items():
            self.assertTrue(
                b["equality_check"]["instrumented_equals_production_bitwise"], key)
            d = b["deduplicated_Q3_D_failure_decomposition"]
            self.assertTrue(d["categories_sum_to_all_poison"], key)
            self.assertTrue(d["residual_is_empty"], key)

    def test_deltas_are_exact(self):
        for seed, b in self.doc["seeds"].items():
            for key, delta in b["deduplicated_minus_control"].items():
                expected = b["deduplicated_metrics"][key] - b["control_metrics_from_q3"][key]
                self.assertAlmostEqual(delta, expected, places=12, msg=f"{seed}/{key}")

    def test_summary_uses_population_sd(self):
        summary = self.doc["five_seed_summary"]
        self.assertEqual(summary["seeds"], SEEDS)

        def walk(obj, path=""):
            if isinstance(obj, dict):
                if set(("per_seed", "mean", "sd_pop")) <= set(obj):
                    a = np.asarray(obj["per_seed"], dtype=float)
                    self.assertEqual(len(a), 5, path)
                    self.assertAlmostEqual(obj["mean"], float(a.mean()), places=12)
                    self.assertAlmostEqual(obj["sd_pop"], float(a.std(ddof=0)), places=12)
                else:
                    for k, v in obj.items():
                        walk(v, f"{path}/{k}")
        walk(summary)

    def test_artifact_contains_no_nonfinite_numbers(self):
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    walk(value, f"{path}/{key}")
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    walk(value, f"{path}[{i}]")
            elif isinstance(obj, float):
                self.assertTrue(np.isfinite(obj), f"non-finite value at {path}")
        walk(self.doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
