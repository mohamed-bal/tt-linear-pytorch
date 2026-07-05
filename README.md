# tt-linear

**[📖 Read the full article: the physics (entanglement entropy, the area law) and the production engineering behind this repo](https://dev.to/mohamed_bal/quantum-inspired-not-quantum-the-physics-of-tensor-networks-behind-production-llm-compression-14fh)**

TT-matrix (tensor-train) compressed linear layers for PyTorch — a drop-in `nn.Linear` replacement whose forward pass never materializes the dense weight matrix.

[![tests](https://github.com/mohamed-bal/tt-linear-pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/mohamed-bal/tt-linear-pytorch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

---

## Table of contents

- [Why this exists](#why-this-exists)
- [The one-paragraph theory](#the-one-paragraph-theory)
- [Install](#install)
- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Replacing `nn.Linear` in an existing model](#replacing-nnlinear-in-an-existing-model)
- [Choosing rank per layer](#choosing-rank-per-layer)
- [Verified correctness claims](#verified-correctness-claims)
- [Benchmarks](#benchmarks)
- [Security notes](#security-notes)
- [What this is not](#what-this-is-not)
- [Running the tests](#running-the-tests)
- [Contributing](#contributing)
- [License](#license)

## Why this exists

Weight-compression research usually stops at the paper: real numbers, no runnable code, no error handling, no path from "here's a benchmark" to "here's what I import." This repo is the runnable, tested, production-shaped half of that equation — TT-SVD decomposition, an efficient forward pass that never reconstructs the dense matrix, config validation that fails fast on malformed input, and a profiling utility for choosing compression rank per layer from real activations instead of a guessed global constant.

If you came here from the article: this is the exact code from the "production integration" section, extracted into an installable package with a real test suite instead of inline snippets. If you came here from GitHub search: the [article](https://dev.to/mohamed_bal/quantum-inspired-not-quantum-the-physics-of-tensor-networks-behind-production-llm-compression-14fh) covers *why* this works — the area law, entanglement entropy, and why trained (not random) weights are what makes bond-dimension truncation viable at all — which this README only summarizes.

## The one-paragraph theory

Trained neural network weight matrices exhibit low effective entanglement/correlation structure — their real information content is smaller than their raw parameter count suggests, in the same sense that ground states of gapped quantum systems obey an area law: entanglement entropy across a cut stays bounded rather than growing with system size, so a fixed **bond dimension (χ)** can represent the structure exactly. TT-matrix decomposition is that same bond-dimension truncation applied to a weight matrix reshaped into a higher-order tensor. Truncating χ is the actual "RAM knob" — smaller χ means more compression and more accuracy loss, and the relationship tracks each layer's real entropy, not a single global constant.

## The Area Law: Entanglement Scales With the Boundary
<img width="900" height="420" alt="area-law-diagram" src="https://github.com/user-attachments/assets/b16c1b0d-117f-4f1d-a579-b45e7c7d582d" />

## Matrix Product State Architecture
<img width="772" height="358" alt="mps-architecture" src="https://github.com/user-attachments/assets/2f90a72c-a4cb-4ae2-8e42-3ebe20394ed1" />

## 
**[Full derivation, with the actual area-law inequality and bond-dimension bound, in the article.](https://dev.to/mohamed_bal/quantum-inspired-not-quantum-the-physics-of-tensor-networks-behind-production-llm-compression-14fh)**

## Install

```bash
git clone https://github.com/mohamed-bal/tt-linear-pytorch.git
cd tt-linear-pytorch
pip install -e ".[dev]"   # editable install + pytest, for development
```

Once published to PyPI:

```bash
pip install tt-linear
```

Requires `torch>=2.0`, Python `>=3.9`. No other runtime dependencies.

## Quickstart

```python
import torch.nn as nn
from tt_linear import TTLinear, TTMatrixConfig
from tt_linear.profiling import profile_layer_compressibility

# Your existing, trained layer
original_layer = nn.Linear(256, 256)

# Factorize 256 = 16 x 16. Pick factors that divide your layer's actual
# dimensions — there is no universally correct factorization; it's a
# design choice worth sweeping alongside rank.
config = TTMatrixConfig(
    out_factors=(16, 16),
    in_factors=(16, 16),
    ranks=(32,),  # the "RAM knob" — sweep this per layer, don't guess globally
)

# Profile BEFORE deploying: measure functional error on real activations,
# not synthetic noise and not weight-norm error alone.
sample_activations = ...  # capture real intermediate activations from your model
report = profile_layer_compressibility(original_layer, config, sample_activations)
print(report)
# {'dense_params': 65536, 'tt_params': 16384, 'compression_ratio': 0.25,
#  'dense_bytes': 262144, 'tt_bytes': 65536, 'functional_rel_error': ...}

# Swap it in once you've picked a rank you're satisfied with
compressed_layer = TTLinear.from_dense(original_layer, config)
```

Run the full example: `python examples/compress_linear_layer.py`

**On that example's actual output:** it uses a randomly initialized `nn.Linear`, not a trained one, so `functional_rel_error` comes out high (~0.79 at a 4x compression ratio, measured on this repo's example run). That is the correct, expected result, not a bug: the entire premise — see [the article](https://dev.to/mohamed_bal/quantum-inspired-not-quantum-the-physics-of-tensor-networks-behind-production-llm-compression-14fh) — is that compression works *because* trained weights have low effective entanglement entropy. A random matrix has none, and TT-decomposition correctly refuses to compress it well. Point this at real trained weights and real captured activations before drawing any accuracy conclusions from your own use case.

## API reference

### `TTMatrixConfig(out_factors, in_factors, ranks)`

Frozen dataclass describing a factorization. `out_factors` and `in_factors` must be sequences whose products equal the target layer's `out_features` / `in_features`. `ranks` must have length `len(out_factors) - 1` (boundary ranks are implicit `1`).

Raises `ValueError` at construction time on any internally inconsistent shape — mismatched factor-list lengths, wrong rank-list length, or non-positive factors/ranks. Fails fast rather than surfacing a confusing error later during decomposition.

- `.out_features` / `.in_features` — computed properties (`math.prod` of the respective factors).
- `.max_ranks()` — theoretical max useful rank at each internal cut. Exceeding this wastes parameters without adding representational power (verified: reconstruction error saturates past this point, and TT storage exceeds dense storage).
- `.validate_against(out_features, in_features)` — raises `ValueError` if this config doesn't factorize a specific target layer's actual shape, or if any requested rank exceeds `.max_ranks()`.

### `tt_svd_decompose(weight, config) -> list[torch.Tensor]`

Decomposes a dense `(out_features, in_features)` weight matrix into TT-cores via sequential SVD (TT-SVD). `core[k]` has shape `(r_{k-1}, out_factors[k], in_factors[k], r_k)`.

Raises `ValueError` if `config` doesn't factorize `weight`'s shape; raises `RuntimeError` (chained from `torch.linalg.LinAlgError`) if SVD fails, which in practice almost always means NaN/Inf already present in the source weight — surfaced with an explicit message rather than a bare LAPACK error.

### `tt_forward(x, cores, out_factors, in_factors) -> torch.Tensor`

Computes `y = x @ W^T` for the `W` implied by `cores`, contracting them directly against the input — the dense matrix is never formed. `x` is `(batch, in_features)`, returns `(batch, out_features)`.

### `TTLinear(config, bias=True, dtype=None, device=None)`

`nn.Module` — drop-in replacement for `nn.Linear`. Accepts and correctly reshapes arbitrary leading batch dimensions (e.g. `(batch, seq, features)`), not just 2D input.

- `TTLinear.from_dense(linear, config)` — classmethod; compresses an existing trained `nn.Linear` (weights and bias) into TT-matrix form. Validates shape compatibility before doing any work.
- `.compression_report()` — dict with `dense_params`, `tt_params`, `compression_ratio`, `dense_bytes`, `tt_bytes`, computed from the actual instantiated cores (not estimated).

### `profile_layer_compressibility(linear, config, sample_batch) -> dict`

Runs `TTLinear.from_dense`, then compares dense vs. TT-compressed output on `sample_batch` — real captured activations, not synthetic noise, since the whole point is measuring the layer's *actual* input-conditioned behavior. Returns `.compression_report()`'s dict plus `functional_rel_error`.

## Replacing `nn.Linear` in an existing model

```python
import torch.nn as nn
from tt_linear import TTLinear, TTMatrixConfig

def replace_linear_with_tt(module: nn.Module, config_fn) -> nn.Module:
    """config_fn(name, linear) -> TTMatrixConfig | None (None = leave dense)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            config = config_fn(name, child)
            if config is not None:
                setattr(module, name, TTLinear.from_dense(child, config))
        else:
            replace_linear_with_tt(child, config_fn)
    return module
```

`config_fn` is deliberately a callback, not a single global config — compression is a per-layer decision (see next section), and different layers in the same model will typically warrant different factorizations and ranks.

## Choosing rank per layer

Don't pick one global χ. Sweep `profile_layer_compressibility` per layer against real captured activations, and select the smallest χ that keeps `functional_rel_error` under your accuracy budget *for that specific layer*. Expect earlier layers to need higher rank and deeper layers to tolerate more aggressive truncation — this reproduces, empirically on your own model, the layer-depth-dependent redundancy pattern discussed in the article. The full decision procedure (including when compressing at all is the wrong call versus using a smaller dense model) is in the article's production framework section, not duplicated here.

## Bond Dimension RAM Knob
<img width="776" height="432" alt="bond-dimension-curve" src="https://github.com/user-attachments/assets/685b713b-e02f-4ac2-91a5-5df460d4d096" />


## Production Decision Framework
<img width="1150" height="1500" alt="decision-framework-diagram" src="https://github.com/user-attachments/assets/07330787-e518-4dd5-9948-ca8f25cea3d4" />

## Verified correctness claims

Every number below is produced by a passing test in `tests/test_core.py`, not asserted from memory:

- **Full-rank reconstruction is exact** (relative error < 1e-4, typically ~1e-6) across multiple factor-chain lengths (2-factor and 3-factor decompositions both tested) — confirming the TT-SVD implementation is mathematically correct, not just plausible-looking.
- **Truncation error decreases monotonically** as rank increases toward the theoretical maximum.
- **`TTLinear.from_dense` matches a dense `nn.Linear` to <1e-4 relative error at full rank**, including with 3D `(batch, seq, features)` input.
- **TT format costs *more* than dense storage at/near full rank** — an honest, intentionally-tested property, not a bug. Compression only pays off where a layer's actual entropy is low, which full-rank-by-definition is not.
- **Two real implementation bugs were caught by this test suite during development** — an axis-interleaving-order error that only manifested with 3+ factors, and an axis-pairing error in the efficient forward pass — and are now permanent regression tests.

## Benchmarks

Correctness tests confirm the math is right. They say nothing about speed. Run `python benchmarks/bench_tt_linear.py` to measure latency, throughput, and memory on your own hardware — full methodology and this repo's own reference numbers are in [`benchmarks/README.md`](benchmarks/README.md).

**The short version, stated plainly: on the CPU reference run in this repo, `TTLinear` is 2x–50x *slower* in wall-clock latency than dense `nn.Linear`, worse at larger batch sizes and higher rank — the direct cost of not having a fused TT-contraction kernel.** Memory/parameter savings are real and independently verified (see above); latency is not free, and currently goes the wrong way without dedicated kernel work. Use this library where memory is the hard constraint, not where latency is. See `benchmarks/README.md` for the full numbers and a `torch.compile` mitigation that helps but doesn't close the gap.

## Security notes

- **Never `torch.load()` a checkpoint from an untrusted source without `weights_only=True`.** Pickle-based checkpoints can execute arbitrary code on load. This applies to TT-core checkpoints exactly as much as to any other model artifact — a compressed model is not a lower-risk artifact.
- **Validate configs before allocating tensors**, especially if any part of a `TTMatrixConfig` is derived from external input (a config file, an API parameter, a user-supplied model spec). `__post_init__` and `.validate_against()` exist specifically so a malformed or adversarial config raises immediately with a clear message, instead of allocating an unexpectedly large tensor or silently producing wrong results.
- **Log, don't print, in anything that runs as a service.** This library uses the standard `logging` module (`logging.getLogger(__name__)`) so failures surface in your existing observability stack rather than scrolling past in stdout.

## What this is not

- **Not a claim involving real quantum computing hardware.** "Tensor network" refers to a classical mathematical toolkit with origins in condensed-matter physics, running on ordinary GPUs. No qubit ever touches this code. See the article for the full distinction.
- **Not a maximally-optimized kernel — and measurably so.** The forward pass is correct and avoids dense-matrix materialization, but it is currently 2x–50x *slower* in wall-clock terms than dense `nn.Linear` on the reference hardware in `benchmarks/README.md`. This repo prioritizes correctness and clarity over kernel-level performance engineering; don't adopt it for a latency-bound workload without benchmarking on your own hardware first.
- **Not a rank-selection heuristic.** `profile_layer_compressibility` measures a rank *you* choose; it doesn't search for or recommend one.
- **Not validated at LLM scale.** All correctness tests run at small matrix sizes (64×64, 4096-element tensors) where exact reconstruction is cheap to verify directly. The mathematics is scale-invariant, but you should re-run `profile_layer_compressibility` against your actual model before trusting any specific compression ratio.

## Running the tests

```bash
pytest tests/ -v
```

17 tests: config validation (including every failure mode documented above), full-rank exact reconstruction across factor-chain lengths, monotonic error decrease under truncation, `nn.Linear`-equivalence at full rank, 3D batch-input handling, and the honest "costs more than dense near full rank" property. CI runs this matrix on Python 3.9, 3.11, and 3.12 on every push (see `.github/workflows/tests.yml`).

## Contributing

Issues and PRs welcome. If you're adding a feature, add a test that would have failed without it — this is exactly how the two real bugs mentioned above were caught, and the project intends to keep it that way.

## License

MIT — see [LICENSE](LICENSE).