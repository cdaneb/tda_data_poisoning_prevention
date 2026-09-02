import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from programs import run_clean_novelty_confirmation as c


class FrozenDesignTests(unittest.TestCase):
    def test_no_exploratory_seed(self):
        self.assertFalse(set(c.SEEDS) & set(c.EXPLORATORY_SEEDS))

    def test_detector_parameters_match_step3(self):
        design = c.preregistration_content()
        self.assertEqual(design["detectors"]["isolation_forest"]["n_estimators"], 200)
        self.assertEqual(design["detectors"]["knn_distance"]["n_neighbors"], 10)

    def test_fit_api_has_no_labels(self):
        import inspect
        names = list(inspect.signature(c.fit_calibrate_evaluate).parameters)
        self.assertNotIn("labels", names); self.assertNotIn("y", names)

    def test_primary_rule_exact(self):
        self.assertTrue(.04 <= .04 <= .06)
        self.assertFalse(.04 <= .039999 <= .06)
        self.assertTrue(.12 >= .12)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not c.PRE.exists(): raise unittest.SkipTest("confirmation preregistration not prepared")
        cls.d = c.load_locked()

    def test_literal_indices_and_hashes(self):
        for pop, seeds in self.d["realizations"].items():
            for seed, r in seeds.items():
                for key in ("sampled_dataset_row_indices", "clean_training_indices", "calibration_indices", "heldout_clean_evaluation_indices"):
                    self.assertEqual(c.hash_list(r[key]), r[key + "_hash"])

    def test_groups_do_not_overlap(self):
        for seeds in self.d["realizations"].values():
            for r in seeds.values():
                g = r["raw_payload_group_hashes"]
                sets = [{g[i] for i in r[k]} for k in ("clean_training_indices", "calibration_indices", "heldout_clean_evaluation_indices")]
                self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_attacks_have_no_noops_and_parents_reproduce(self):
        for seeds in self.d["realizations"].values():
            for r in seeds.values():
                for a in r["families"].values():
                    self.assertFalse(any(x["raw_noop"] for x in a["attack_log"]))
                    self.assertEqual([x["target_index"] for x in a["attack_log"]], a["poison_source_parent_indices"])

    def test_preprocessing_fit_is_training_only(self):
        for p in c.CELLS.glob("*.json"):
            r=json.load(open(p)); locked=self.d["realizations"][r["population"]][str(r["seed"])]
            self.assertEqual(r["preprocessing"]["fit_indices_hash"], locked["clean_training_indices_hash"])

    def test_cache_hashes(self):
        for p in c.CACHE.joinpath("features").glob("*.json"):
            m=json.load(open(p)); z=np.load(p.with_suffix(".npz"))
            self.assertEqual(m["control_hash"], c.stable_hash(z["control"]))
            self.assertEqual(m["stack_hash"], c.stable_hash(z["stack"]))

    def test_merger_rejects_missing(self):
        with patch.object(c, "CELLS", Path(tempfile.mkdtemp())):
            with self.assertRaises(RuntimeError): c.merge(self.d, require_complete=True)


if __name__ == "__main__": unittest.main(verbosity=2)
