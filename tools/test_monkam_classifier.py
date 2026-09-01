import unittest
import numpy as np
from programs.run_monkam_classifier_replication import make_split,row_keys

class ClassifierProtocolTests(unittest.TestCase):
 def test_group_split_has_no_identity_overlap(self):
  X=np.array([[1,2],[1,2],[3,4],[5,6],[5,6],[7,8]],dtype=np.uint8); groups=row_keys(X)
  tr,te=make_split(len(X),groups,.34,60,'group_raw_payload')
  self.assertFalse(set(groups[tr]) & set(groups[te]))
 def test_random_split_is_deterministic(self):
  a=make_split(20,None,.2,60,'random_row'); b=make_split(20,None,.2,60,'random_row')
  np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
if __name__=='__main__': unittest.main()
