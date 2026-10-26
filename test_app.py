import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_gradient(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(8,9));y=np.array([0.,1.]*4);p=init();_,g=loss_grad(p,x,y);v=p[0][0,0];eps=1e-5;p[0][0,0]=v+eps;hi=loss_grad(p,x,y)[0];p[0][0,0]=v-eps;lo=loss_grad(p,x,y)[0];self.assertAlmostEqual((hi-lo)/(2*eps),g[0][0,0],places=5)
    def test_patch_shape(self):self.assertEqual(patches(np.ones((24,24))).shape,(576,9))
    def test_perfect_mask(self):self.assertEqual(iou(np.array([1,0]),np.array([True,False])),1)
