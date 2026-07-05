"""
Example: compress an existing nn.Linear layer, profile it on real
activations, and compare against the dense baseline.

Run: python examples/compress_linear_layer.py
"""
import torch
import torch.nn as nn

from tt_linear import TTLinear, TTMatrixConfig
from tt_linear.profiling import profile_layer_compressibility

torch.manual_seed(0)

# Stand-in for a real trained layer — in practice, load this from your
# actual checkpoint (see README security notes on torch.load).
original_layer = nn.Linear(256, 256)

# 256 = 16 x 16 factorization. Pick factors that divide your real
# layer's dimensions; there is no single "right" factorization, and
# it's own a design choice worth sweeping.
config = TTMatrixConfig(
    out_factors=(16, 16),
    in_factors=(16, 16),
    ranks=(32,),  # try several of these and compare functional_rel_error
)

# Stand-in for a real batch of activations — replace with actual
# intermediate activations captured from your model, not synthetic noise.
sample_activations = torch.randn(32, 256)

report = profile_layer_compressibility(original_layer, config, sample_activations)
print("Compression report:")
for k, v in report.items():
    print(f"  {k}: {v}")

# Once you've picked a rank you're satisfied with, swap it in:
compressed_layer = TTLinear.from_dense(original_layer, config)

x = torch.randn(4, 10, 256)  # (batch, seq, features) — 3D input works too
y_dense = original_layer(x)
y_compressed = compressed_layer(x)
print("\nOutput shapes match:", y_dense.shape == y_compressed.shape)
