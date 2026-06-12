"""
Soundness tests for LeakyReLU layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch LeakyReLU produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestLeakyReLUStarExactSoundness:
    """Soundness tests for exact LeakyReLU with Star sets."""

    def test_all_positive_input(self):
        """LeakyReLU with all-positive input: identity."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        result = reach_layer(layer, [star], method='exact')

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """LeakyReLU with all-negative input: scales by gamma."""
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        star = Star.from_bounds(lb, ub)
        gamma = 0.1
        layer = nn.LeakyReLU(negative_slope=gamma)

        result = reach_layer(layer, [star], method='exact')

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, gamma * lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, gamma * ub, atol=1e-6)

    def test_crossing_zero_exact_sampling(self):
        """LeakyReLU with crossing zero: sample 200 points, verify containment."""
        lb = np.array([[-1.0], [-0.5], [0.5]])
        ub = np.array([[1.0], [1.0], [2.0]])
        star = Star.from_bounds(lb, ub)
        gamma = 0.2
        layer = nn.LeakyReLU(negative_slope=gamma)

        result = reach_layer(layer, [star], method='exact')

        # Collect union of output bounds
        union_lb = np.ones((3, 1)) * np.inf
        union_ub = np.ones((3, 1)) * -np.inf
        for s in result:
            if not s.is_empty_set():
                s_lb, s_ub = s.estimate_ranges()
                union_lb = np.minimum(union_lb, s_lb)
                union_ub = np.maximum(union_ub, s_ub)

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()

            assert np.all(pt_output >= union_lb.flatten() - 1e-5), \
                f"Sample below lb: {(pt_output - union_lb.flatten()).min()}"
            assert np.all(pt_output <= union_ub.flatten() + 1e-5), \
                f"Sample above ub: {(pt_output - union_ub.flatten()).max()}"

    def test_different_negative_slopes(self):
        """LeakyReLU soundness across several negative_slope values."""
        lb = np.array([[-2.0], [-1.0]])
        ub = np.array([[1.0], [2.0]])

        for gamma in [0.01, 0.1, 0.3, 0.5]:
            star = Star.from_bounds(lb, ub)
            layer = nn.LeakyReLU(negative_slope=gamma)
            result = reach_layer(layer, [star], method='exact')

            union_lb = np.ones((2, 1)) * np.inf
            union_ub = np.ones((2, 1)) * -np.inf
            for s in result:
                if not s.is_empty_set():
                    s_lb, s_ub = s.estimate_ranges()
                    union_lb = np.minimum(union_lb, s_lb)
                    union_ub = np.maximum(union_ub, s_ub)

            np.random.seed(42)
            for _ in range(50):
                point = np.random.uniform(lb.flatten(), ub.flatten())
                pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pt_output = layer(pt_input).numpy().squeeze()
                assert np.all(pt_output >= union_lb.flatten() - 1e-5)
                assert np.all(pt_output <= union_ub.flatten() + 1e-5)


class TestLeakyReLUStarApproxSoundness:
    """Soundness tests for approximate LeakyReLU with Star sets."""

    def test_approx_contains_exact(self):
        """Approx LeakyReLU should over-approximate exact result."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        exact = reach_layer(layer, [star], method='exact')
        approx = reach_layer(layer, [star], method='approx')

        # Exact union bounds (LP-based)
        exact_lb = np.ones((2, 1)) * np.inf
        exact_ub = np.ones((2, 1)) * -np.inf
        for s in exact:
            if not s.is_empty_set():
                s_lb, s_ub = s.get_ranges()
                exact_lb = np.minimum(exact_lb, s_lb)
                exact_ub = np.maximum(exact_ub, s_ub)

        # Approx bounds
        approx_lb, approx_ub = approx[0].estimate_ranges()

        assert np.all(approx_lb <= exact_lb + 1e-6)
        assert np.all(exact_ub <= approx_ub + 1e-6)

    def test_approx_sampling(self):
        """Approx LeakyReLU: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.2)

        result = reach_layer(layer, [star], method='approx')
        out_lb, out_ub = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)


class TestLeakyReLUZonoSoundness:
    """Soundness tests for LeakyReLU with Zonotope sets."""

    def test_zono_sampling(self):
        """LeakyReLU Zono: all sampled outputs within bounds."""
        lb = np.array([[-1.0], [-2.0], [0.0]])
        ub = np.array([[1.0], [1.0], [2.0]])
        zono = Zono.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        result = reach_layer(layer, [zono], method='approx')
        out_lb, out_ub = result[0].get_bounds()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)


class TestLeakyReLUBoxSoundness:
    """Soundness tests for LeakyReLU with Box sets."""

    def test_box_sampling(self):
        """LeakyReLU Box: all sampled outputs within bounds."""
        lb = np.array([[-1.0], [-2.0], [0.5]])
        ub = np.array([[1.0], [1.0], [2.0]])
        box = Box(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.2)

        result = reach_layer(layer, [box], method='approx')
        out = result[0]

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out.lb.flatten() - 1e-5)
            assert np.all(pt_output <= out.ub.flatten() + 1e-5)


class TestLeakyReLUZonoJointSoundnessRegression:
    """Regression tests for issue #16: the Zonotope relaxation never added
    its error generator (the computed ``error`` was algebraically zero),
    pinning crossing dimensions exactly to the secant line. Per-dimension
    intervals looked right, but the JOINT affine relation was wrong:
    on the correlated zonotope {(a, -a)} the functional y1 + y2 collapsed
    to the single value 0.99 (true range [0, 0.99]) and the true output
    f(0, 0) = (0, 0) was excluded.
    """

    def test_correlated_zono_contains_kink_output(self):
        from n2v.nn.layer_ops.leakyrelu_reach import leakyrelu_zono_approx
        z = Zono(np.zeros((2, 1)), np.array([[1.0], [-1.0]]))
        out = leakyrelu_zono_approx([z], gamma=0.01)[0]
        # An error generator must have been added per crossing dim.
        assert out.V.shape[1] > z.V.shape[1], (
            "no error generator added (issue #16)")
        # The true output at alpha=0 must be contained.
        assert out.to_star().contains(np.zeros((2, 1))), (
            "true output f(0,0) = (0,0) excluded (issue #16)")

    def test_zono_pushforward_containment_sweep(self):
        """Monte-Carlo joint containment incl. gamma > 1."""
        from n2v.nn.layer_ops.leakyrelu_reach import leakyrelu_zono_approx
        rng = np.random.default_rng(1)
        for _ in range(15):
            n = int(rng.integers(1, 5))
            c = rng.uniform(-0.5, 0.5, (n, 1))
            G = rng.uniform(-1.0, 1.0, (n, int(rng.integers(1, 4))))
            for gamma in (0.0, 0.01, 0.3, 1.0, 2.0):
                out = leakyrelu_zono_approx(
                    [Zono(c.copy(), G.copy())], gamma=gamma)[0]
                out_star = out.to_star()
                for _ in range(8):
                    alpha = rng.uniform(-1.0, 1.0, (G.shape[1], 1))
                    x = c + G @ alpha
                    y = np.where(x >= 0, x, gamma * x)
                    assert out_star.contains(y), (
                        f"pushforward escaped LeakyReLU zono reach "
                        f"(gamma={gamma})")

    def test_zono_band_interval_matches_deepz_form(self):
        """The band's per-dim interval is [a*l + min(0,b_u), a*u + max(0,b_u)].

        Like classic DeepZ ReLU, the parallel band trades per-dimension
        interval tightness for a sound JOINT affine relation: its
        interval is wider than the exact image but must contain it.
        """
        from n2v.nn.layer_ops.leakyrelu_reach import leakyrelu_zono_approx
        gamma = 0.1
        z = Zono(np.array([[0.0]]), np.array([[1.0]]))  # x in [-1, 1]
        out = leakyrelu_zono_approx([z], gamma=gamma)[0]
        lo, hi = out.get_bounds()
        li, ui = -1.0, 1.0
        a = (ui - gamma * li) / (ui - li)
        b_u = gamma * li - a * li
        assert abs(float(lo[0, 0]) - (a * li + min(0.0, b_u))) < 1e-9
        assert abs(float(hi[0, 0]) - (a * ui + max(0.0, b_u))) < 1e-9
        # Sound: the exact image [-gamma, 1] lies inside the band interval.
        assert float(lo[0, 0]) <= -gamma + 1e-12
        assert float(hi[0, 0]) >= 1.0 - 1e-12
