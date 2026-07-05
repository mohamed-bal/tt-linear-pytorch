"""
Test suite for tt_linear. Run: pytest tests/ -v

These tests encode the correctness checks used to validate the
implementation before publication — including two real bugs caught
during that process (interleaving order for d>2 factors, and an axis
pairing bug in the efficient forward pass) — so they double as
regression tests against reintroducing either one.
"""
import math

import pytest
import torch
import torch.nn as nn

from tt_linear import TTLinear, TTMatrixConfig, tt_svd_decompose, tt_forward
from tt_linear.profiling import profile_layer_compressibility


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestTTMatrixConfig:
    def test_valid_config(self):
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(32,))
        assert cfg.out_features == 64
        assert cfg.in_features == 64
        assert cfg.max_ranks() == [64]

    def test_mismatched_factor_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8, 8), ranks=(4, 4))

    def test_wrong_rank_length_raises(self):
        with pytest.raises(ValueError, match="ranks must have length"):
            TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(4, 8))

    def test_zero_rank_raises(self):
        with pytest.raises(ValueError, match="ranks must be >= 1"):
            TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(0,))

    def test_validate_against_shape_mismatch_raises(self):
        cfg = TTMatrixConfig(out_factors=(4, 4), in_factors=(8, 8), ranks=(16,))
        with pytest.raises(ValueError, match="Config factorizes"):
            cfg.validate_against(64, 64)

    def test_validate_against_excessive_rank_raises(self):
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(100,))
        with pytest.raises(ValueError, match="exceeds theoretical max"):
            cfg.validate_against(64, 64)


class TestDecompositionCorrectness:
    """These reconstruct the dense matrix from cores and check error against
    the theoretical maximum rank — this is the test that caught the original
    interleaving bug for d > 2 (error stuck near 1.0 instead of -> 0 at full rank)."""

    @pytest.mark.parametrize(
        "out_factors,in_factors",
        [((8, 8), (8, 8)), ((4, 4, 4), (4, 4, 4)), ((16, 16), (16, 16))],
    )
    def test_full_rank_reconstructs_exactly(self, out_factors, in_factors):
        O, I = math.prod(out_factors), math.prod(in_factors)
        W = torch.randn(O, I)
        max_ranks = TTMatrixConfig(out_factors, in_factors, tuple(1 for _ in out_factors[:-1])).max_ranks()
        cfg = TTMatrixConfig(out_factors, in_factors, tuple(max_ranks))
        cores = tt_svd_decompose(W, cfg)

        # reconstruct via tt_forward against the identity matrix trick:
        # feed standard basis vectors and assemble the output columns.
        x = torch.eye(I)
        W_hat_T = tt_forward(x, cores, out_factors, in_factors)  # (I, O) == W^T applied to each basis vector
        W_hat = W_hat_T.T
        rel_error = (torch.linalg.norm(W - W_hat) / torch.linalg.norm(W)).item()
        assert rel_error < 1e-4, f"full-rank reconstruction error too high: {rel_error}"

    def test_truncated_rank_error_decreases_monotonically(self):
        out_factors, in_factors = (8, 8), (8, 8)
        O, I = 64, 64
        W = torch.randn(O, I)
        x = torch.eye(I)
        errors = []
        for r in [4, 8, 16, 32, 64]:
            cfg = TTMatrixConfig(out_factors, in_factors, (r,))
            cores = tt_svd_decompose(W, cfg)
            W_hat = tt_forward(x, cores, out_factors, in_factors).T
            errors.append((torch.linalg.norm(W - W_hat) / torch.linalg.norm(W)).item())
        assert all(a >= b - 1e-6 for a, b in zip(errors, errors[1:])), (
            f"error should decrease monotonically as rank increases, got {errors}"
        )
        assert errors[-1] < 1e-4


class TestTTLinear:
    def test_matches_dense_linear_at_full_rank(self):
        lin = nn.Linear(64, 64, bias=True)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(64,))
        tt = TTLinear.from_dense(lin, cfg)

        x = torch.randn(10, 64)
        y_dense = lin(x)
        y_tt = tt(x)
        rel_error = (torch.linalg.norm(y_dense - y_tt) / torch.linalg.norm(y_dense)).item()
        assert rel_error < 1e-4

    def test_truncated_rank_gives_valid_but_approximate_output(self):
        lin = nn.Linear(64, 64, bias=True)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(8,))
        tt = TTLinear.from_dense(lin, cfg)
        x = torch.randn(10, 64)
        y_tt = tt(x)
        assert y_tt.shape == (10, 64)
        assert torch.isfinite(y_tt).all()

    def test_from_dense_shape_mismatch_raises(self):
        lin = nn.Linear(64, 64)
        bad_cfg = TTMatrixConfig(out_factors=(4, 4), in_factors=(8, 8), ranks=(16,))
        with pytest.raises(ValueError, match="Config factorizes"):
            TTLinear.from_dense(lin, bad_cfg)

    def test_handles_3d_batch_input(self):
        """(batch, seq, features) inputs, as used by any real transformer block."""
        lin = nn.Linear(64, 64)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(64,))
        tt = TTLinear.from_dense(lin, cfg)
        x3d = torch.randn(3, 5, 64)
        y3d = tt(x3d)
        assert y3d.shape == (3, 5, 64)

    def test_compression_report_ratio(self):
        lin = nn.Linear(64, 64)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(8,))
        tt = TTLinear.from_dense(lin, cfg)
        report = tt.compression_report()
        assert report["dense_params"] == 64 * 64
        assert report["compression_ratio"] < 1.0  # meaningfully truncated -> smaller than dense

    def test_full_rank_costs_more_than_dense(self):
        """Honest property, not a bug: TT format at/near full rank costs MORE
        than dense storage due to factorization overhead. Compression only
        pays off where the layer's actual entropy is low."""
        lin = nn.Linear(64, 64)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(64,))
        tt = TTLinear.from_dense(lin, cfg)
        report = tt.compression_report()
        assert report["compression_ratio"] > 1.0


class TestProfiling:
    def test_profile_returns_functional_error(self):
        lin = nn.Linear(64, 64)
        cfg = TTMatrixConfig(out_factors=(8, 8), in_factors=(8, 8), ranks=(8,))
        x = torch.randn(16, 64)
        report = profile_layer_compressibility(lin, cfg, x)
        assert "functional_rel_error" in report
        assert report["functional_rel_error"] >= 0
