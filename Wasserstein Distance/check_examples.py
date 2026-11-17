"""Mathematical cross-checks, independent of the figure generation."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, wasserstein_distance
from numpy.testing import assert_allclose
from ot_tools import (gaussian_distances, empirical_wp_equal, exact_ot,
                      sinkhorn_log, image_measure, mmd2_biased,
                      signature_level2, signature_level3)


class MathematicalChecks(unittest.TestCase):
    def test_gaussian_formula_against_integration(self):
        for mu, sigma in [(2, 1), (0, 2), (2, 2), (-1, .3), (0, 1)]:
            w1, w2 = gaussian_distances(0, 1, mu, sigma)
            integral1 = quad(lambda z: abs(mu+(sigma-1)*z)*norm.pdf(z),
                             -12, 12, epsabs=1e-9, points=[-mu/(sigma-1)]
                             if sigma != 1 and abs(mu/(sigma-1)) < 12 else None)[0]
            integral2 = quad(lambda z: (mu+(sigma-1)*z)**2*norm.pdf(z), -12, 12)[0]
            assert_allclose([w1, w2**2], [integral1, integral2], atol=1e-8)

    def test_hand_plan_and_weighted_scipy(self):
        x, y, a, b = [0, 2], [1, 3], [.5, .5], [.25, .75]
        w1, _, _ = exact_ot(x, y, a, b)
        w2, pi, _ = exact_ot(x, y, a, b, p=2)
        assert_allclose(w1, wasserstein_distance(x, y, a, b))
        assert_allclose(w2**2, 3)
        assert_allclose(pi, [[.25, .25], [0, .5]])

    def test_sorting_matches_lp_and_scipy(self):
        rng = np.random.default_rng(10)
        x, y = rng.normal(size=13), rng.normal(size=13)
        assert_allclose(empirical_wp_equal(x, y, 1), wasserstein_distance(x, y))
        assert_allclose(empirical_wp_equal(x, y, 2), exact_ot(x, y, p=2)[0])
        with self.assertRaises(ValueError):
            empirical_wp_equal([1, 2], [1])

    def test_image_translation_and_normalization(self):
        a, b = np.zeros((7, 7)), np.zeros((7, 7))
        a[1:3, 1:3] = 1; b[2:4, 3:5] = 10
        x, wa = image_measure(a); y, wb = image_measure(b)
        assert_allclose(wa, wb)
        for p in [1, 2]:
            assert_allclose(exact_ot(x, y, wa, wb, p=p)[0], np.sqrt(5))
        with self.assertRaises(ValueError):
            image_measure(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            image_measure(np.array([[1, -1]]))

    def test_sinkhorn_marginals_bias_and_symmetry(self):
        x = np.arange(4.)
        a, b = np.array([.4, .3, .2, .1]), np.array([.1, .2, .3, .4])
        exact, _, C = exact_ot(x, x, a, b, p=2)
        pi, ab = sinkhorn_log(a, b, C, epsilon=.5)
        _, ba = sinkhorn_log(b, a, C, epsilon=.5)
        _, aa = sinkhorn_log(a, a, C, epsilon=.5)
        _, bb = sinkhorn_log(b, b, C, epsilon=.5)
        assert_allclose(pi.sum(1), a, atol=1e-9)
        assert_allclose(pi.sum(0), b, atol=1e-9)
        self.assertGreaterEqual(ab['transport_cost'] + 1e-8, exact**2)
        self.assertGreater(aa['transport_cost'], 0)
        assert_allclose(ab['objective'], ba['objective'], atol=1e-8)
        self.assertGreater(ab['objective']-.5*(aa['objective']+bb['objective']), 0)
        with self.assertRaises(RuntimeError):
            sinkhorn_log(a, b, C, epsilon=.5, max_iter=1)

    def test_time_representations(self):
        a, b = np.array([0., 0., 1., 1.]), np.array([0., 1., 0., 1.])
        self.assertEqual(wasserstein_distance(a, b), 0)
        assert_allclose(exact_ot(np.c_[np.arange(4), a], np.c_[np.arange(4), b])[0], .5)
        wa = np.lib.stride_tricks.sliding_window_view(a, 2)
        wb = np.lib.stride_tricks.sliding_window_view(b, 2)
        assert_allclose(exact_ot(wa, wb)[0], 2/3)
        assert_allclose(mmd2_biased(a, b), 0, atol=1e-14)
        self.assertGreater(mmd2_biased(wa, wb), 0)

    def test_signature_collision_and_level_three(self):
        t = np.arange(4)/3
        pa = np.c_[t, [0, 0, 1, 1]]; pb = np.c_[t, [0, 1, 0, 1]]
        assert_allclose(signature_level2(pa), signature_level2(pb))
        sa, sb = signature_level3(pa), signature_level3(pb)
        assert_allclose([sa[7], sb[7]], [7/54, 13/54])
        # A straight segment has signature tensor powers / factorial.
        straight = signature_level3([[0, 0], [2, 3]])
        assert_allclose(straight[6:], np.einsum('i,j,k->ijk', [2,3], [2,3], [2,3]).ravel()/6)
        # Subdividing a straight segment must not change its signature.
        assert_allclose(straight, signature_level3([[0,0], [1,1.5], [2,3]]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
