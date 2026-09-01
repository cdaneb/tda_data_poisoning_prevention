"""Fast regression checks for Phase Q support-restricted attacks."""
import unittest

import numpy as np

from programs.phase_q_attacks import (
    SUPPORTED_FAMILIES,
    common_attackable_mask,
    conservative_support_lengths,
)
from programs.phase_q_metrics import matched_clean_cost, removal_curve
from programs.phase_q_pipeline import THRESHOLD_STACK, extract_multithreshold_features
from programs.tda_pipeline import extract_tda_features


class PhaseQAttackTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = np.zeros((40, 1500), dtype=np.uint8)
        self.X[:, :240] = rng.integers(1, 256, size=(40, 240), dtype=np.uint8)
        self.y = np.array(["normal"] * 10 + ["analysis"] * 30)

    def test_conservative_support(self):
        X = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [1, 2, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(conservative_support_lengths(X), [3, 0, 2])

    def test_common_attackability_rejects_constant_prefix(self):
        X = np.ones((2, 1500), dtype=np.uint8)
        X[1, 60:120] = 2
        lengths = np.array([120, 120])
        np.testing.assert_array_equal(common_attackable_mask(X, lengths), [False, True])

    def test_all_families_preserve_multisets_and_padding(self):
        target_sets = []
        for fn, kwargs in SUPPORTED_FAMILIES.values():
            Xc, _, poisoned, log = fn(
                self.X, self.y, poison_rate=0.25, random_state=42, **kwargs
            )
            targets = np.array([entry["target_index"] for entry in log])
            target_sets.append(targets)
            poison = Xc[len(self.X):]
            clean = self.X[targets]
            self.assertTrue(poisoned[-len(targets):].all())
            self.assertTrue((~poisoned[:len(self.X)]).all())
            for original, changed, entry in zip(clean, poison, log):
                np.testing.assert_array_equal(np.sort(original), np.sort(changed))
                support = entry["support_length"]
                np.testing.assert_array_equal(original[support:], changed[support:])
                self.assertEqual(entry["positions_changed"], int(np.count_nonzero(original != changed)))
                self.assertGreater(entry["positions_changed"], 0)
                self.assertGreaterEqual(entry["draw_attempts"], 1)
        for targets in target_sets[1:]:
            np.testing.assert_array_equal(target_sets[0], targets)

    def test_short_support_fails_explicitly(self):
        X = self.X.copy()
        X[:, 80:] = 0
        fn, kwargs = SUPPORTED_FAMILIES["block_reversal"]
        with self.assertRaisesRegex(ValueError, "support/attackability"):
            fn(X, self.y, poison_rate=0.25, random_state=42, **kwargs)

    def test_removal_metrics_charge_clean_cost(self):
        labels = np.array([0, 0, 1, 1, 2, 2, -1])
        poisoned = np.array([False, False, True, True, False, True, True])
        curve = removal_curve(labels, poisoned, thresholds=(1.0, 0.49))
        self.assertEqual(curve[0]["true_poison_removed"], 2)
        self.assertEqual(curve[0]["clean_removed"], 0)
        self.assertEqual(curve[1]["clean_removed"], 1)
        matches = matched_clean_cost(curve, curve)
        self.assertGreaterEqual(matches[0]["repair_best"]["poison_removal_rate"],
                                curve[0]["poison_removal_rate"])

    def test_multithreshold_embeds_exact_legacy_control(self):
        rng = np.random.default_rng(9)
        X = rng.integers(0, 256, size=(4, 1500), dtype=np.uint8)
        legacy, _ = extract_tda_features(X, threshold=0.4)
        stacked, blocks, _ = extract_multithreshold_features(
            X, thresholds=(0.4, 0.5), return_blocks=True
        )
        np.testing.assert_array_equal(blocks[0.4], legacy)
        self.assertEqual(stacked.shape, (4, 120))
        np.testing.assert_allclose(stacked[:, :60], blocks[0.4] / np.sqrt(2))
        self.assertIn(0.4, THRESHOLD_STACK)


if __name__ == "__main__":
    unittest.main()
