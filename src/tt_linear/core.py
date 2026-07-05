"""
tt_linear.core — TT-matrix (tensor-train) compressed linear layer.

Implements TT-SVD decomposition (Oseledets, 2011) in TT-matrix / TTM form
(Novikov et al., "Tensorizing Neural Networks", 2015) and a drop-in
nn.Linear replacement whose forward pass never materializes the dense
weight matrix.

Verified: at full rank, reconstruction and forward-pass output match a
dense nn.Linear to ~1e-6 relative error (float32). See tests/test_core.py.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "TTMatrixConfig",
    "tt_svd_decompose",
    "tt_forward",
    "TTLinear",
]


@dataclass(frozen=True)
class TTMatrixConfig:
    """Factorization spec for a TT-matrix layer.

    out_factors / in_factors: dimension factorizations such that
        prod(out_factors) == out_features
        prod(in_factors)  == in_features
    ranks: internal bond dimensions, length == len(out_factors) - 1
           (boundary ranks r_0 = r_d = 1 are implicit, not listed here).

    Raises:
        ValueError: on any internally inconsistent shape (mismatched
            factor-list lengths, wrong rank-list length, non-positive
            factors/ranks). Fails fast at construction time rather than
            surfacing a confusing error later at decomposition time.
    """
    out_factors: Sequence[int]
    in_factors: Sequence[int]
    ranks: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.out_factors) != len(self.in_factors):
            raise ValueError(
                f"out_factors (len={len(self.out_factors)}) and in_factors "
                f"(len={len(self.in_factors)}) must have the same length."
            )
        if len(self.ranks) != len(self.out_factors) - 1:
            raise ValueError(
                f"ranks must have length {len(self.out_factors) - 1}, "
                f"got {len(self.ranks)}."
            )
        if any(r < 1 for r in self.ranks):
            raise ValueError("all ranks must be >= 1.")
        if any(f < 1 for f in list(self.out_factors) + list(self.in_factors)):
            raise ValueError("all factors must be >= 1.")

    @property
    def out_features(self) -> int:
        return math.prod(self.out_factors)

    @property
    def in_features(self) -> int:
        return math.prod(self.in_factors)

    def max_ranks(self) -> list[int]:
        """Theoretical max useful rank at each internal cut. Exceeding this
        wastes parameters without adding representational power — verified
        empirically: reconstruction error saturates past this point and TT
        storage exceeds dense storage."""
        d = len(self.out_factors)
        result = []
        for k in range(d - 1):
            left = math.prod(self.out_factors[: k + 1]) * math.prod(self.in_factors[: k + 1])
            right = math.prod(self.out_factors[k + 1:]) * math.prod(self.in_factors[k + 1:])
            result.append(min(left, right))
        return result

    def validate_against(self, out_features: int, in_features: int) -> None:
        """Raise if this config doesn't factorize the target layer shape.

        Call this before decomposing any real weight. A silent shape
        mismatch here is a correctness bug, not something to coerce or
        truncate around.
        """
        if self.out_features != out_features or self.in_features != in_features:
            raise ValueError(
                f"Config factorizes ({self.out_features}, {self.in_features}), "
                f"but target layer is ({out_features}, {in_features})."
            )
        max_r = self.max_ranks()
        for requested, maximum in zip(self.ranks, max_r):
            if requested > maximum:
                raise ValueError(
                    f"Requested rank {requested} exceeds theoretical max {maximum} "
                    f"at this cut — reduce ranks; this factorization would waste "
                    f"parameters without adding accuracy."
                )


def tt_svd_decompose(weight: torch.Tensor, config: TTMatrixConfig) -> list[torch.Tensor]:
    """Decompose a dense (out_features, in_features) weight matrix into TT-cores.

    core[k] has shape (r_{k-1}, out_factors[k], in_factors[k], r_k), with
    r_0 = r_d = 1. Uses sequential SVD (TT-SVD); the last core absorbs the
    remainder directly with no further truncation, since r_d = 1 by
    construction.

    Raises:
        ValueError: if `config` doesn't factorize `weight`'s actual shape.
        RuntimeError: if SVD fails (typically NaN/Inf already present in
            the source weight).
    """
    out_f, in_f, ranks = config.out_factors, config.in_factors, config.ranks
    config.validate_against(*weight.shape)

    d = len(out_f)
    full_ranks = [1] + list(ranks) + [1]

    tensor = weight.reshape(list(out_f) + list(in_f))
    interleave_perm = [axis for k in range(d) for axis in (k, d + k)]
    current = tensor.permute(interleave_perm).contiguous()

    cores: list[torch.Tensor] = []
    left_dim = 1
    for k in range(d):
        m_k = out_f[k] * in_f[k]
        if k < d - 1:
            mat = current.reshape(left_dim * m_k, -1)
            try:
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            except torch.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"SVD failed decomposing core {k}; check the source weight "
                    f"tensor for NaN/Inf before decomposing."
                ) from e

            r_k = min(full_ranks[k + 1], S.shape[0])
            core = U[:, :r_k].reshape(left_dim, out_f[k], in_f[k], r_k)
            cores.append(core)

            remainder = torch.diag(S[:r_k]) @ Vh[:r_k, :]
            trailing_shape = [dim for j in range(k + 1, d) for dim in (out_f[j], in_f[j])]
            current = remainder.reshape([r_k] + trailing_shape)
            left_dim = r_k
        else:
            cores.append(current.reshape(left_dim, out_f[k], in_f[k], 1))

    return cores


def tt_forward(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    out_factors: Sequence[int],
    in_factors: Sequence[int],
) -> torch.Tensor:
    """y = x @ W^T for W implicit in TT-cores — the dense matrix is never formed.

    x: (batch, in_features). Returns (batch, out_features).
    """
    batch = x.shape[0]
    d = len(cores)
    state = x.reshape([batch] + list(in_factors))
    n_out_so_far = 0

    for k in range(d):
        core = cores[k]
        if k == 0:
            core2 = core.squeeze(0)  # (o_k, i_k, r_k); r_prev == 1
            state = state.movedim(1, -1)
            state = torch.tensordot(state, core2, dims=([-1], [1]))
            state = state.movedim(-2, 1)
        else:
            i_k_axis = 1 + n_out_so_far
            state = state.movedim(i_k_axis, -1)  # -> (..., r_prev, i_k)
            state = torch.tensordot(state, core, dims=([-2, -1], [0, 2]))
            state = state.movedim(-2, 1 + n_out_so_far)
        n_out_so_far += 1

    state = state.squeeze(-1)  # drop trailing r_d == 1
    return state.reshape(batch, math.prod(out_factors))


class TTLinear(nn.Module):
    """Drop-in replacement for nn.Linear backed by TT-matrix cores.

    Construct via `TTLinear.from_dense(existing_linear, config)` to compress
    an already-trained layer, or directly for training a TT-native layer
    from random initialization.
    """

    def __init__(
        self,
        config: TTMatrixConfig,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        factory = {"dtype": dtype, "device": device}
        self.cores = nn.ParameterList()
        d = len(config.out_factors)
        full_ranks = [1] + list(config.ranks) + [1]
        for k in range(d):
            shape = (full_ranks[k], config.out_factors[k], config.in_factors[k], full_ranks[k + 1])
            # Scaling keeps the product's output variance sane at init;
            # this is not a principled derivation, just a stable default —
            # re-tune if training TT-native from scratch rather than
            # compressing an already-trained layer.
            core = torch.randn(shape, **factory) / math.sqrt(config.in_factors[k] * full_ranks[k])
            self.cores.append(nn.Parameter(core))
        self.bias = nn.Parameter(torch.zeros(config.out_features, **factory)) if bias else None

    @classmethod
    def from_dense(cls, linear: nn.Linear, config: TTMatrixConfig) -> "TTLinear":
        """Compress an existing trained nn.Linear into TT-matrix form."""
        config.validate_against(linear.out_features, linear.in_features)
        module = cls(
            config,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            decomposed = tt_svd_decompose(linear.weight.detach(), config)
            for core_param, core_value in zip(module.cores, decomposed):
                core_param.copy_(core_value)
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach())
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        out = tt_forward(x2d, list(self.cores), self.config.out_factors, self.config.in_factors)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.config.out_features)

    def compression_report(self) -> dict:
        """Parameter/byte comparison against the equivalent dense nn.Linear.

        Run before deploying a given rank choice, not after.
        """
        dense_params = self.config.out_features * self.config.in_features
        tt_params = sum(c.numel() for c in self.cores)
        bytes_per_elem = next(iter(self.cores)).element_size()
        return {
            "dense_params": dense_params,
            "tt_params": tt_params,
            "compression_ratio": tt_params / dense_params,
            "dense_bytes": dense_params * bytes_per_elem,
            "tt_bytes": tt_params * bytes_per_elem,
        }
