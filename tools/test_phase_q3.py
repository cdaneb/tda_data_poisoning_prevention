"""Deterministic unit and regression tests for the Phase Q3 collision audit.

These run before the full audit.  A failing equality test is a finding, not an
obstacle: none of them may be weakened to let the audit proceed.
"""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT))

from phase_q3_collisions import (  # noqa: E402
    canonical_diagram_points,
    class_size_array,
    class_stats,
    combine_hashes,
    diagram_row_hashes,
    earliest_merger_attribution,
    equivalence_classes,
    exact_hash,
    label_free_signature_violations,
    normalize_float_array,
    pack_binary_mask,
    row_hashes,
    transition_report,
    unpack_binary_mask,
)
from phase_q3_stage_pipeline import (  # noqa: E402
    CHAIN_STAGES,
    FILTRATION_NAMES,
    extract_all_stages,
    support_records,
)

ARTIFACT = ROOT / "results" / "phase_q3_collision_audit.json"


class HashingTests(unittest.TestCase):
    def test_distinct_raw_rows_hash_differently(self):
        a = np.zeros(1500, dtype=np.uint8)
        b = a.copy()
        b[7] = 1
        self.assertNotEqual(exact_hash(a), exact_hash(b))
        self.assertEqual(exact_hash(a), exact_hash(a.copy()))

    def test_signed_zero_normalizes(self):
        a = np.array([0.0, 1.0, -2.5])
        b = np.array([-0.0, 1.0, -2.5])
        self.assertEqual(exact_hash(a), exact_hash(b))
        self.assertEqual(normalize_float_array(b)[0].tobytes(),
                         np.float64(0.0).tobytes())

    def test_nan_canonicalizes_but_infinity_is_preserved(self):
        weird = np.array([np.nan], dtype=np.float64)
        weird_bits = weird.view(np.uint64).copy()
        weird_bits[0] |= np.uint64(0x7)  # a different NaN payload
        other = weird_bits.view(np.float64)
        self.assertTrue(np.isnan(other[0]))
        self.assertNotEqual(other.tobytes(), weird.tobytes())
        self.assertEqual(exact_hash(other), exact_hash(weird))
        self.assertNotEqual(exact_hash(np.array([np.inf])),
                            exact_hash(np.array([1e308])))

    def test_tiny_float_differences_are_not_rounded_away(self):
        a = np.array([1.0])
        b = np.array([1.0 + 2 ** -52])
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(exact_hash(a), exact_hash(b))

    def test_packed_binary_mask_is_lossless(self):
        rng = np.random.RandomState(0)
        mask = rng.rand(11, 30, 50) > 0.6
        packed, shape = pack_binary_mask(mask)
        self.assertTrue(np.array_equal(unpack_binary_mask(packed, shape), mask))
        self.assertEqual(shape, mask.shape)
        h = row_hashes(packed)
        self.assertEqual(len(set(h.tolist())), len(np.unique(
            mask.reshape(len(mask), -1), axis=0)))

    def test_combine_hashes_is_order_sensitive_across_columns(self):
        a = np.array(["aa", "bb"], dtype=object)
        b = np.array(["cc", "dd"], dtype=object)
        self.assertNotEqual(combine_hashes([a, b])[0], combine_hashes([b, a])[0])
        self.assertEqual(combine_hashes([a, b])[0], combine_hashes([a, b])[0])


class DiagramCanonicalisationTests(unittest.TestCase):
    def _diagram(self):
        return np.array([
            [0.0, 3.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.5, 0.5, 0.0],   # diagonal padding
            [0.0, 4.0, 1.0],
            [2.0, 2.0, 1.0],   # diagonal padding
        ])

    def test_point_order_does_not_change_signature(self):
        d = self._diagram()
        shuffled = d[[3, 1, 0, 4, 2]]
        self.assertEqual(exact_hash(canonical_diagram_points(d)),
                         exact_hash(canonical_diagram_points(shuffled)))

    def test_padding_excluded_without_dropping_valid_points(self):
        pts = canonical_diagram_points(self._diagram())
        self.assertEqual(len(pts), 3)
        self.assertTrue((pts[:, 1] > pts[:, 0]).all())
        # The three off-diagonal points survive exactly.
        self.assertEqual(sorted(map(tuple, pts.tolist())),
                         sorted([(0.0, 3.0, 0.0), (1.0, 2.0, 1.0), (0.0, 4.0, 1.0)]))

    def test_extra_padding_rows_do_not_change_signature(self):
        d = self._diagram()
        pad = np.array([[9.0, 9.0, 0.0], [9.0, 9.0, 1.0]])
        self.assertEqual(exact_hash(canonical_diagram_points(d)),
                         exact_hash(canonical_diagram_points(np.vstack([d, pad]))))

    def test_homology_dimension_stays_in_the_signature(self):
        a = np.array([[0.0, 1.0, 0.0]])
        b = np.array([[0.0, 1.0, 1.0]])
        self.assertNotEqual(exact_hash(canonical_diagram_points(a)),
                            exact_hash(canonical_diagram_points(b)))

    def test_nan_in_diagram_is_raised_not_silently_dropped(self):
        d = self._diagram()
        d[0, 1] = np.nan
        with self.assertRaises(ValueError):
            canonical_diagram_points(d)

    def test_diagram_row_hashes_group_identical_diagrams(self):
        d = self._diagram()
        stack = np.stack([d, d[[2, 0, 1, 3, 4]], np.zeros_like(d)])
        h = diagram_row_hashes(stack)
        self.assertEqual(h[0], h[1])
        self.assertNotEqual(h[0], h[2])


class CollisionStatisticTests(unittest.TestCase):
    def test_repeated_member_and_redundancy_differ_on_a_known_fixture(self):
        # 6 rows: one class of 3, one class of 2, one singleton.
        hashes = np.array(["a", "a", "a", "b", "b", "c"], dtype=object)
        poisoned = np.array([0, 0, 0, 0, 1, 0], dtype=bool)
        st = class_stats(hashes, poisoned, "fixture")
        self.assertEqual(st["n_rows"], 6)
        self.assertEqual(st["n_unique_classes"], 3)
        self.assertEqual(st["n_repeated_classes"], 2)
        self.assertEqual(st["n_repeated_member_rows"], 5)
        self.assertAlmostEqual(st["repeated_member_fraction"], 5 / 6)
        self.assertAlmostEqual(st["redundancy_fraction"], 3 / 6)
        self.assertNotAlmostEqual(st["repeated_member_fraction"],
                                  st["redundancy_fraction"])
        self.assertEqual(st["largest_class_size"], 3)

    def test_class_composition_is_exact(self):
        hashes = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
        poisoned = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
        st = class_stats(hashes, poisoned, "fixture")
        self.assertEqual(st["all_classes"]["clean_only"], 1)
        self.assertEqual(st["all_classes"]["poison_only"], 1)
        self.assertEqual(st["all_classes"]["mixed"], 1)
        self.assertEqual(st["all_classes"]["mixed_class_poison_rows"], 1)
        self.assertEqual(st["all_classes"]["mixed_class_clean_rows"], 1)
        self.assertEqual(st["poison_rows_sharing_class_with_clean"], 1)
        self.assertAlmostEqual(st["poison_obstruction_fraction"], 1 / 3)

    def test_class_size_array_and_equivalence_classes_agree(self):
        hashes = np.array(["x", "y", "x", "x"], dtype=object)
        self.assertTrue(np.array_equal(class_size_array(hashes), [3, 1, 3, 3]))
        self.assertEqual(sorted(equivalence_classes(hashes)["x"].tolist()), [0, 2, 3])


class AttributionTests(unittest.TestCase):
    def test_identifies_the_first_merger_stage(self):
        # rows 0,1 merge at stage 0; rows 2,3 merge only at stage 1.
        s0 = np.array(["a", "a", "c", "d"], dtype=object)
        s1 = np.array(["a", "a", "e", "e"], dtype=object)
        s2 = np.array(["a", "a", "e", "e"], dtype=object)
        poisoned = np.array([0, 1, 0, 0], dtype=bool)
        res = earliest_merger_attribution(["s0", "s1", "s2"], [s0, s1, s2], poisoned)
        self.assertEqual(res["monotonicity_violations"], 0)
        rows = {r["stage"]: r for r in res["rows"]}
        self.assertEqual(rows["s0"]["member_rows"], 2)
        self.assertEqual(rows["s1"]["member_rows"], 2)
        self.assertEqual(rows["s2"]["member_rows"], 0)
        self.assertTrue(res["attribution_sums_to_final_repeated_mass"])
        self.assertEqual(res["n_final_repeated_member_rows"], 4)
        self.assertEqual(rows["s0"]["poison_rows"], 1)
        self.assertEqual(rows["s0"]["mixed_class_poison_rows"], 1)

    def test_attribution_is_mutually_exclusive_and_exhaustive(self):
        rng = np.random.RandomState(3)
        n = 60
        s0 = np.array([f"r{i}" for i in range(n)], dtype=object)
        s1 = np.array([f"b{i // 2}" for i in range(n)], dtype=object)
        s2 = np.array([f"f{i // 4}" for i in range(n)], dtype=object)
        poisoned = rng.rand(n) > 0.8
        res = earliest_merger_attribution(["s0", "s1", "s2"], [s0, s1, s2], poisoned)
        self.assertEqual(res["monotonicity_violations"], 0)
        total = sum(r["member_rows"] for r in res["rows"])
        self.assertEqual(total, res["n_final_repeated_member_rows"])
        self.assertEqual(total, n)
        self.assertEqual({r["stage"]: r["member_rows"] for r in res["rows"]},
                         {"s0": 0, "s1": n, "s2": 0})

    def test_monotonicity_violation_is_detected_not_ignored(self):
        s0 = np.array(["a", "a"], dtype=object)
        s1 = np.array(["p", "q"], dtype=object)  # impossible split
        res = earliest_merger_attribution(["s0", "s1"], [s0, s1],
                                          np.array([0, 0], dtype=bool))
        self.assertEqual(res["monotonicity_violations"], 2)

    def test_transition_report_counts_merged_classes(self):
        up = np.array(["a", "b", "c", "c"], dtype=object)
        down = np.array(["z", "z", "c", "c"], dtype=object)
        poisoned = np.array([0, 1, 0, 0], dtype=bool)
        rep = transition_report(up, down, poisoned)
        self.assertEqual(rep["n_downstream_classes_merging_multiple_upstream_classes"], 1)
        self.assertEqual(rep["n_rows_in_newly_merged_classes"], 2)
        self.assertEqual(rep["n_mixed_classes_first_created_here"], 1)
        self.assertEqual(rep["largest_newly_merged_class"]["size"], 2)


class LabelFreedomTests(unittest.TestCase):
    def test_no_stage_signature_function_accepts_ground_truth(self):
        self.assertEqual(label_free_signature_violations(), [])

    def test_stage_extraction_entry_points_take_no_labels(self):
        for fn in (extract_all_stages, support_records):
            params = set(inspect.signature(fn).parameters)
            self.assertFalse(
                params & {"is_poisoned", "poisoned", "y", "labels", "y_true"},
                f"{fn.__name__} accepts a ground-truth argument")


class SupportRecordTests(unittest.TestCase):
    def test_support_end_is_one_past_the_last_nonzero_byte(self):
        X = np.zeros((3, 1500), dtype=np.uint8)
        X[0, 0] = 5
        X[1, 1499] = 1
        rec = support_records(X)
        self.assertTrue(np.array_equal(rec["support_end"], [1, 1500, 0]))
        self.assertTrue(np.array_equal(rec["padding_count"], [1499, 0, 1500]))
        self.assertTrue(np.array_equal(rec["all_zero"], [False, False, True]))

    def test_support_record_is_a_bijection_with_the_raw_row(self):
        rng = np.random.RandomState(1)
        X = np.zeros((40, 1500), dtype=np.uint8)
        for i in range(40):
            k = rng.randint(1, 60)
            X[i, :k] = rng.randint(1, 256, size=k)
        X[5] = X[4]
        raw = row_hashes(X)
        sup = support_records(X)["signatures"]
        self.assertEqual(class_size_array(raw).tolist(),
                         class_size_array(sup).tolist())


class InstrumentedPipelineTests(unittest.TestCase):
    """The load-bearing equality: instrumented stages must be the real ones."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(11)
        X = rng.randint(0, 256, size=(24, 1500)).astype(np.uint8)
        X[3] = X[2]                      # an exact raw duplicate
        X[7, :] = 0                      # an all-zero payload
        X[8, :] = 0
        cls.X = X
        cls.res = extract_all_stages(X, threshold=0.4, verbose=False)

    def test_instrumented_60_vector_equals_legacy_output(self):
        eq = self.res["equality_check"]
        self.assertTrue(eq["instrumented_equals_production_exactly"])
        self.assertTrue(eq["instrumented_equals_production_bitwise"])
        self.assertEqual(eq["production_feature_hash"], eq["instrumented_feature_hash"])
        self.assertEqual(tuple(eq["shape"]), (24, 60))

    def test_every_stage_covers_every_row(self):
        for name in CHAIN_STAGES:
            self.assertEqual(len(self.res["stages"][name]), len(self.X), name)
        for fname in FILTRATION_NAMES:
            for k, v in self.res["per_filtration"][fname].items():
                self.assertEqual(len(v), len(self.X), f"{fname}/{k}")

    def test_raw_duplicates_stay_merged_downstream(self):
        st = self.res["stages"]
        for name in CHAIN_STAGES:
            self.assertEqual(st[name][2], st[name][3], f"{name} split a raw duplicate")
            self.assertEqual(st[name][7], st[name][8], f"{name} split a zero payload")

    def test_chain_is_monotone_on_a_real_batch(self):
        res = earliest_merger_attribution(
            list(CHAIN_STAGES), [self.res["stages"][n] for n in CHAIN_STAGES],
            np.zeros(len(self.X), dtype=bool))
        self.assertEqual(res["monotonicity_violations"], 0)

    def test_binarizer_state_is_recorded_and_consistent(self):
        mv = self.res["fitted_state"]["binarizer_max_value_"]
        self.assertEqual(len(set(mv.values())), 1)
        cut = self.res["effective_byte_cut"]
        for fname in FILTRATION_NAMES:
            self.assertAlmostEqual(cut[fname], 0.4 * mv[fname])

    def test_feature_blocks_and_final_vector_induce_the_same_classes(self):
        st = self.res["stages"]
        self.assertTrue(np.array_equal(class_size_array(st["feature_blocks"]),
                                       class_size_array(st["final_60_vector"])))


class ArtifactStructureTests(unittest.TestCase):
    """Structural validation of the published artifact, when it exists."""

    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.exists():
            raise unittest.SkipTest(f"{ARTIFACT.name} not generated yet")
        with open(ARTIFACT) as fh:
            cls.doc = json.load(fh)

    def test_seed_keys_are_exact(self):
        self.assertEqual(sorted(int(k) for k in self.doc["seeds"]),
                         [42, 123, 456, 789, 1024])

    def test_frame_shape_and_hashes(self):
        for key, block in self.doc["seeds"].items():
            r = block["realization"]
            self.assertEqual(r["n_clean"], 5000, key)
            self.assertEqual(r["n_poison"], 500, key)
            self.assertEqual(r["n_total"], 5500, key)
            self.assertEqual(tuple(block["equality_check"]["shape"]), (5500, 60), key)
            self.assertTrue(
                block["equality_check"]["instrumented_equals_production_bitwise"], key)
            for h in ("input_hash", "poison_mask_hash", "subsample_index_hash"):
                self.assertRegex(r[h], r"^[0-9a-f]{16}$")

    def test_seed_42_matches_the_q2_frame(self):
        res = self.doc["seeds"]["42"]["Q3_A_definition_and_reproduction"]
        q = res["q2_statistic_resolution"]
        self.assertTrue(q["input_hash_matches_q2"])
        self.assertTrue(q["poison_mask_hash_matches_q2"])
        self.assertTrue(q["feature_hash_matches_q2"])

    def test_every_stage_has_all_rows_and_obeys_the_definitions(self):
        for key, block in self.doc["seeds"].items():
            for st in block["Q3_C_population_attribution"]["stage_class_stats"]:
                self.assertEqual(st["n_rows"], 5500, f"{key}/{st['stage']}")
                self.assertEqual(
                    st["n_redundant_rows"], st["n_rows"] - st["n_unique_classes"])
                self.assertAlmostEqual(
                    st["redundancy_fraction"], st["n_redundant_rows"] / st["n_rows"])
                self.assertAlmostEqual(
                    st["repeated_member_fraction"],
                    st["n_repeated_member_rows"] / st["n_rows"])
                self.assertGreaterEqual(
                    st["repeated_member_fraction"], st["redundancy_fraction"])

    def test_attribution_is_exclusive_and_sums(self):
        for key, block in self.doc["seeds"].items():
            att = block["Q3_C_population_attribution"]["earliest_merger_attribution"]
            self.assertEqual(att["monotonicity_violations"], 0, key)
            self.assertTrue(att["attribution_sums_to_final_repeated_mass"], key)
            self.assertEqual(sum(r["member_rows"] for r in att["rows"]),
                             att["n_final_repeated_member_rows"], key)
            self.assertAlmostEqual(
                sum(r["share_of_final_repeated_member_mass"] for r in att["rows"]),
                1.0, places=9)
            for r in att["rows"]:
                self.assertEqual(r["member_rows"], r["clean_rows"] + r["poison_rows"])

    def test_failure_decomposition_is_exclusive_and_exhaustive(self):
        for key, block in self.doc["seeds"].items():
            d = block["Q3_D_strict_purity_failure_decomposition"]
            self.assertTrue(d["categories_sum_to_all_poison"], key)
            self.assertEqual(sum(v["n"] for v in d["decomposition"].values()),
                             d["n_poison"], key)
            self.assertEqual(d["n_poison"], 500, key)
            self.assertTrue(d["residual_is_empty"], key)

    def test_largest_class_trace_is_internally_consistent(self):
        for key, block in self.doc["seeds"].items():
            b = block["Q3_B_largest_class_trace"]
            a = block["Q3_A_definition_and_reproduction"]["largest_final_class"]
            self.assertEqual(b["size"], a["size"], key)
            self.assertEqual(b["signature"], a["signature"], key)
            self.assertEqual(b["n_clean"] + b["n_poison"], b["size"], key)
            self.assertEqual(b["n_source_rows"] + b["n_appended_poison_rows"],
                             b["size"], key)
            self.assertEqual(b["n_poison"], b["n_appended_poison_rows"], key)
            self.assertEqual(sum(b["earliest_merger_stage_histogram"].values()),
                             b["size"], key)
            stages = [p["stage"] for p in b["per_stage_distinct_signatures"]]
            self.assertEqual(stages, self.doc["chain_stages"], key)
            counts = [p["n_distinct_upstream_signatures"]
                      for p in b["per_stage_distinct_signatures"]]
            self.assertEqual(counts, sorted(counts, reverse=True), key)
            self.assertEqual(counts[-1], 1, key)

    def test_no_required_numeric_field_is_nan_or_infinite(self):
        allowed = {"n_nonfinite_diagram_entries"}

        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k not in allowed:
                        walk(v, f"{path}/{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
            elif isinstance(o, float):
                self.assertTrue(np.isfinite(o), f"non-finite at {path}")

        walk(self.doc)

    def test_five_seed_summary_uses_population_sd(self):
        s = self.doc["five_seed_summary"]
        self.assertEqual(s["seeds"], [42, 123, 456, 789, 1024])
        for key, val in s.items():
            if isinstance(val, dict) and "per_seed" in val:
                a = np.array(val["per_seed"], dtype=float)
                self.assertEqual(len(a), 5, key)
                self.assertAlmostEqual(val["mean"], float(a.mean()), places=9)
                self.assertAlmostEqual(val["sd_pop"], float(a.std(ddof=0)), places=9)

    def test_legacy_capture_is_still_2_2_percent_at_seed_42(self):
        c = (self.doc["seeds"]["42"]["Q3_D_strict_purity_failure_decomposition"]
             ["comparisons"]["observed_exact_purity_capture_pct"])
        self.assertAlmostEqual(c, 2.2000, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
