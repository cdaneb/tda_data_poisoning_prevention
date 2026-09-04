"""Fast, data-free checks for the preregistered downstream experiment."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from programs.adversarial_attack import label_to_binary, payload_label_to_binary
from programs.monkam_representation import stable_hash
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs import run_downstream_classifier_preregistered as experiment


class DownstreamProtocolTests(unittest.TestCase):
    def test_cross_dataset_label_mapping(self):
        labels = np.asarray([
            "normal", "NORMAL", " normal ", "BENIGN", "benign", " Bot ",
            "DoS Hulk",
        ])
        np.testing.assert_array_equal(
            payload_label_to_binary(labels),
            np.asarray([0, 0, 0, 0, 0, 1, 1]),
        )
        np.testing.assert_array_equal(
            label_to_binary(np.asarray(["normal", "BENIGN"])),
            np.asarray([0, 1]),
        )

    def test_design_and_grid_are_stable(self):
        design = experiment.design_content()
        self.assertEqual(design["primary_analysis"]["population"], "unsw_matched")
        self.assertEqual(design["primary_analysis"]["representation"], "control")
        self.assertEqual(design["primary_analysis"]["budget"], 0.05)
        self.assertEqual(len(experiment.expected_cells()), 45)
        document = copy.deepcopy(design)
        document["content_hash"] = experiment.content_hash(document)
        self.assertEqual(document["content_hash"], experiment.content_hash(document))

    def test_all_attacks_select_only_malicious_parents(self):
        X = np.zeros((30, 1500), dtype=np.uint8)
        for row in range(len(X)):
            X[row, :180] = (np.arange(180, dtype=np.uint16) + row) % 251 + 1
        y = np.asarray(["BENIGN"] * 15 + ["Bot"] * 15)
        for family in SUPPORTED_FAMILIES:
            combined, poison_mask, log, parents = experiment.attack_realization(
                X, y, family, 2026
            )
            self.assertEqual(combined.shape, (33, 1500))
            self.assertEqual(int(poison_mask.sum()), 3)
            self.assertFalse(any(entry["raw_noop"] for entry in log))
            np.testing.assert_array_equal(
                payload_label_to_binary(y[parents]), np.ones(3, dtype=int)
            )

    def test_random_comparator_matches_clean_cost_exactly(self):
        order = np.asarray([5, 0, 6, 1, 7, 2, 3, 4])
        removal = experiment.random_cost_removal(
            order, clean_suspect_count=5, target_clean_removed=2
        )
        self.assertEqual(int(removal[:5].sum()), 2)
        self.assertEqual(int(removal[5:].sum()), 2)
        self.assertFalse(removal[2:].all())
        self.assertFalse(experiment.random_cost_removal(order, 5, 0).any())

    def test_hashed_json_rejects_tampering(self):
        document = {"experiment": "test", "value": 7}
        document["content_hash"] = experiment.content_hash(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.json"
            path.write_text(json.dumps(document))
            self.assertEqual(experiment.load_hashed_json(path), document)
            document["value"] = 8
            path.write_text(json.dumps(document))
            with self.assertRaises(RuntimeError):
                experiment.load_hashed_json(path)

    def test_random_order_is_reproducible_and_cell_specific(self):
        first = experiment.random_order("unsw_matched", 2026, "transpositions", 100)
        second = experiment.random_order("unsw_matched", 2026, "transpositions", 100)
        other = experiment.random_order("unsw_matched", 2026, "block_swap", 100)
        self.assertEqual(stable_hash(first), stable_hash(second))
        self.assertNotEqual(stable_hash(first), stable_hash(other))

    def test_sanitizer_and_classifier_helpers(self):
        random = np.random.default_rng(44)
        features = random.normal(size=(110, 8))
        features[100:] += 4.0
        realization = {
            "clean_training_indices": list(range(60)),
            "calibration_indices": list(range(60, 80)),
            "heldout_clean_evaluation_indices": list(range(80, 100)),
        }
        suspect, removal, metadata = experiment.sanitizer_removal(
            features, realization, seed=2026, budget=0.05, poison_count=10
        )
        self.assertEqual(len(suspect), 30)
        self.assertEqual(len(removal), 30)
        self.assertEqual(metadata["poison_count"], 10)
        self.assertTrue(np.isfinite(metadata["threshold"]))

        X = random.integers(0, 256, size=(50, 12), dtype=np.uint8)
        y = np.asarray([0] * 25 + [1] * 25)
        keep = np.ones(50, dtype=bool)
        poison = np.zeros(50, dtype=bool)
        result = experiment.fit_arm(
            X, y, keep, X[5:45], y[5:45], X[30:40], 2026, poison
        )
        self.assertEqual(result["training"]["rows"], 50)
        self.assertEqual(result["training"]["poison_retained"], 0)
        for metric in (
            "malicious_recall", "benign_false_positive_rate", "balanced_accuracy",
            "macro_f1", "accuracy", "auroc", "auprc",
            "attacked_malicious_recall",
        ):
            self.assertTrue(np.isfinite(result[metric]))

    def test_hierarchical_bootstrap_uses_seed_and_family_levels(self):
        records = []
        families = ("transpositions", "block_swap")
        expected = []
        for seed_index, seed in enumerate(experiment.SEEDS):
            for family_index, family in enumerate(families):
                difference = 0.01 * (seed_index + 1) + 0.001 * family_index
                expected.append(difference)
                records.append({
                    "population": "unsw_matched",
                    "seed": seed,
                    "family": family,
                    "outcomes": {
                        "left": {"malicious_recall": 0.5 + difference},
                        "right": {"malicious_recall": 0.5},
                    },
                })
        with patch.object(experiment, "BOOTSTRAP_REPS", 1000):
            result = experiment.hierarchical_bootstrap(
                records, "left", "right", "malicious_recall", "unsw_matched"
            )
        self.assertAlmostEqual(result["mean"], float(np.mean(expected)))
        self.assertLess(result["ci95"][0], result["mean"])
        self.assertGreater(result["ci95"][1], result["mean"])
        self.assertEqual(set(result["differences_by_seed"]), set(map(str, experiment.SEEDS)))

    def test_merger_emits_complete_registered_artifacts(self):
        arms = ["clean", "oracle", "poisoned"]
        for representation in experiment.REPRESENTATIONS:
            for budget in experiment.BUDGETS:
                arms.extend([
                    f"filter|{representation}|{budget}",
                    f"random|{representation}|{budget}",
                ])
        records = []
        for population, seed, family in sorted(experiment.expected_cells()):
            outcomes = {}
            for arm in arms:
                if arm in {"clean", "oracle"}:
                    recall = 0.80
                elif arm == "poisoned":
                    recall = 0.70
                elif arm.startswith("filter|control"):
                    recall = 0.76
                elif arm.startswith("filter|stack"):
                    recall = 0.77
                else:
                    recall = 0.73
                outcomes[arm] = {
                    "malicious_recall": recall,
                    "benign_false_positive_rate": 0.02,
                    "balanced_accuracy": 0.80,
                    "macro_f1": 0.80,
                    "accuracy": 0.80,
                    "auroc": 0.85,
                    "auprc": 0.84,
                    "attacked_malicious_recall": recall - 0.05,
                    "attacked_malicious_mean_probability": 0.65,
                    "training": {
                        "rows": 100, "benign_labels": 50,
                        "malicious_labels": 50, "poison_retained": 0,
                    },
                }
            records.append({
                "population": population,
                "dataset": experiment.POPULATIONS[population]["dataset"],
                "seed": seed,
                "family": family,
                "outcomes": outcomes,
            })
        preregistration = {"experiment": "test", "content_hash": "pre"}
        receipt = {
            "registration_url": "https://osf.io/abcde/",
            "registered_at_utc": "2026-09-04T00:00:00Z",
            "registered_code_commit": "0" * 40,
        }
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            with (
                patch.object(experiment, "BOOTSTRAP_REPS", 100),
                patch.object(experiment, "load_cells", return_value=records),
                patch.object(experiment, "OUT", temporary / "results.json"),
                patch.object(experiment, "CSVOUT", temporary / "summary.csv"),
                patch.object(experiment, "CELLCSV", temporary / "cells.csv"),
                patch.object(experiment, "REPORT", temporary / "report.md"),
            ):
                experiment.merge(preregistration, receipt)
                merged = experiment.load_hashed_json(temporary / "results.json")
            self.assertEqual(merged["cell_count"], 45)
            self.assertEqual(merged["cell_metric_rows"], 45 * len(arms))
            self.assertTrue((temporary / "summary.csv").exists())
            self.assertTrue((temporary / "cells.csv").exists())
            self.assertTrue((temporary / "report.md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
