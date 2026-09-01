import unittest
import inspect
import numpy as np
from programs.novelty_detectors import clean_threshold, calibrated_pvalues, fit_calibrate_evaluate, KNNDistance

class DetectorTests(unittest.TestCase):
    def test_calibration_is_label_free_and_deterministic(self):
        rng=np.random.default_rng(1); tr=rng.normal(size=(40,3)); cal=rng.normal(size=(20,3))
        clean=rng.normal(size=(10,3)); poison=rng.normal(4,size=(10,3))
        _,a,_=fit_calibrate_evaluate(lambda:KNNDistance(3,1),tr,cal,clean,poison,[.1])
        _,b,_=fit_calibrate_evaluate(lambda:KNNDistance(3,1),tr,cal,clean,poison[::-1],[.1])
        self.assertEqual(a['0.1']['threshold'],b['0.1']['threshold'])
    def test_conformal_range(self):
        p=calibrated_pvalues(np.arange(5.),np.array([-1.,9.])); self.assertTrue(np.all((p>0)&(p<=1)))
    def test_threshold_rejects_bad_budget(self):
        with self.assertRaises(ValueError): clean_threshold(np.arange(3.),0)
    def test_fit_and_calibration_api_cannot_accept_labels(self):
        params=inspect.signature(fit_calibrate_evaluate).parameters
        self.assertNotIn('y',params); self.assertNotIn('poison_labels',params)
if __name__ == '__main__': unittest.main()
