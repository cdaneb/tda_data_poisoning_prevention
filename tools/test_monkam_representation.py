import unittest
import numpy as np
from gtda.diagrams import Amplitude, PersistenceEntropy
from programs.monkam_representation import (METRICS_60, SPECS, _features,
    feature_blocks, fit_shared, learned_state, stable_hash)

class RepresentationTests(unittest.TestCase):
    def test_feature_arithmetic_and_boundaries(self):
        self.assertEqual(SPECS["project_30x50_t04"].expected_features, 60)
        self.assertEqual(SPECS["supplied_126"].expected_features, 126)
        block=feature_blocks(SPECS["supplied_126"])["height_0_1"]
        self.assertEqual(block["homology_indices"]["h0"], list(range(0,18,2)))
        self.assertEqual(block["homology_indices"]["h1"], list(range(1,18,2)))
    def test_index_map_matches_identifiable_branch_extractors(self):
        # Two deliberately unequal points per homology dimension make each
        # standalone extractor output identifiable.  Compare the union output
        # coordinate-by-coordinate with the documented extractor-major map.
        diagrams=np.array([[[0.,2.,0.],[0.,5.,0.],[1.,4.,1.],[2.,8.,1.]]])
        union=_features(METRICS_60,1).fit_transform(diagrams)
        block=feature_blocks(SPECS["project_30x50_t04"])["height_0_1"]
        standalone=[PersistenceEntropy(nan_fill_value=-1,n_jobs=1)]
        standalone += [Amplitude(metric=m,metric_params=p,n_jobs=1)
                       for m,p in METRICS_60]
        for (name, layout), extractor in zip(block["extractors"].items(), standalone):
            expected=extractor.fit_transform(diagrams)[0]
            observed=union[0,layout["start"]:layout["stop"]]
            np.testing.assert_allclose(observed,expected,err_msg=name)
    def test_shared_fit_and_state(self):
        rng=np.random.default_rng(3); X=rng.integers(0,256,(4,1500),dtype=np.uint8)
        V,p=fit_shared(X,SPECS["project_30x50_t04"],1)
        self.assertEqual(V.shape,(4,60)); self.assertEqual(len(learned_state(p)),5)
    def test_one_by_1500_h1_is_constant(self):
        rng=np.random.default_rng(4); X=rng.integers(0,256,(5,1500),dtype=np.uint8)
        V,_=fit_shared(X,SPECS["notebook_1x1500_t03"],1)
        for block in feature_blocks(SPECS["notebook_1x1500_t03"]).values():
            indices=block["homology_indices"]["h1"]
            self.assertTrue(np.all(np.ptp(V[:,indices],axis=0)==0))
    def test_hash_deterministic(self):
        X=np.arange(12,dtype=np.int16).reshape(3,4)
        self.assertEqual(stable_hash(X),stable_hash(X.copy()))

if __name__ == '__main__': unittest.main()
