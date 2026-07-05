"""
tt_linear.profiling — measure compressibility of a real, trained layer
on real activations before choosing a rank in production.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .core import TTLinear, TTMatrixConfig

__all__ = ["profile_layer_compressibility"]


def profile_layer_compressibility(
    linear: nn.Linear, config: TTMatrixConfig, sample_batch: torch.Tensor
) -> dict:
    """Measure reconstruction error AND functional error (output divergence
    on real activations, not just weight Frobenius norm) at the rank
    specified by `config`.

    Always pass real activations captured from your actual model, not
    synthetic noise — a random input has no relation to the low-entanglement
    structure that makes trained weights compressible in the first place,
    and will systematically under- or over-estimate true functional error.

    Raises:
        ValueError: propagated from TTMatrixConfig validation if `config`
            doesn't match `linear`'s shape.
    """
    tt_layer = TTLinear.from_dense(linear, config)
    with torch.no_grad():
        dense_out = linear(sample_batch)
        tt_out = tt_layer(sample_batch)
        functional_rel_error = (
            torch.linalg.norm(dense_out - tt_out) / torch.linalg.norm(dense_out)
        ).item()
    report = tt_layer.compression_report()
    report["functional_rel_error"] = functional_rel_error
    return report
