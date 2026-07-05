"""
tt_linear — TT-matrix (tensor-train) compressed linear layers for PyTorch.

    from tt_linear import TTLinear, TTMatrixConfig
    from tt_linear.profiling import profile_layer_compressibility

See README.md for usage and the accompanying article for the theory
(area law, entanglement entropy, bond dimension) behind why this works.
"""
from .core import TTMatrixConfig, tt_svd_decompose, tt_forward, TTLinear
from .profiling import profile_layer_compressibility

__version__ = "0.1.0"

__all__ = [
    "TTMatrixConfig",
    "tt_svd_decompose",
    "tt_forward",
    "TTLinear",
    "profile_layer_compressibility",
]
