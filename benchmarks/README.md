# Benchmarks

**Read this before deciding whether TT-decomposition helps your workload.** These are real, measured numbers from `bench_tt_linear.py`, not projections — including a finding that should change how you think about when to use this library.

## Methodology

- `time.perf_counter()` around the forward call only; `torch.cuda.synchronize()` immediately before and after the timed region on CUDA (required — without it you measure kernel *launch* time, not completion time, and will get numbers that look great and mean nothing).
- 10 warmup iterations discarded per configuration (JIT/cuDNN autotuning, allocator warmup), then 50 timed repetitions by default; every reported number is a mean ± stdev over those repetitions, never a single sample.
- Every configuration is wrapped in its own try/except — an OOM or shape error at configuration N is logged and skipped, not fatal to the rest of the sweep.
- Run it yourself: `python benchmarks/bench_tt_linear.py --device cuda --dtype float16` (defaults to CPU + float32 if no CUDA is available).

## Measured results (CPU, float32, this repo's reference machine)

No GPU was available on the machine these reference numbers were collected on — **if you're targeting GPU deployment, re-run the script yourself before drawing conclusions; the finding below is worth checking on your own hardware regardless, not assuming to be CPU-specific.**

| size | rank | batch | dense (ms) | TT (ms) | speedup | compression ratio |
|---|---|---|---|---|---|---|
| 256 | 8 | 1 | 0.020 | 0.175 | 0.11x | 0.062 |
| 256 | 8 | 32 | 0.065 | 0.310 | 0.21x | 0.062 |
| 256 | 128 | 32 | 0.071 | 2.124 | 0.03x | 1.000 |
| 512 | 8 | 1 | 0.064 | 0.151 | 0.42x | 0.039 |
| 512 | 32 | 32 | 0.263 | 1.449 | 0.18x | 0.156 |
| 512 | 128 | 32 | 0.270 | 13.967 | 0.02x | 0.625 |

Full sweep in the repo's CI artifact / your own `bench_results.csv` run.

## The honest finding

**TTLinear is slower than dense `nn.Linear` in every configuration measured — by 2x to over 50x, worse at larger batch sizes and higher rank.** This is not a bug; it's the direct, expected cost of the README's own disclaimer: *"Not a maximally-optimized kernel."* Dense `nn.Linear` dispatches to one heavily-optimized BLAS GEMM call with decades of engineering behind it. `TTLinear.forward` performs several smaller `tensordot`/`movedim` operations with real Python and dispatch overhead between them — the parameter-count and memory savings are real and verified (see the `compression_ratio` column and `tests/test_core.py`), but they do not currently translate into a wall-clock speedup.

We tested one mitigation: `torch.compile(tt_layer)` measurably helps (at size=512, rank=32, batch=32: **0.96 ms compiled vs 1.45 ms eager** — roughly 1.5x faster than eager TT) but still does not close the gap to dense (**0.28 ms** at the same configuration) — compiled TT remains ~3.4x slower than dense on this configuration. `torch.compile` is worth trying in your own deployment; it is not a fix for the underlying kernel-fusion gap, which would require a dedicated fused TT-contraction kernel — out of scope for this repo (see `README.md`'s "What this is not").

## When this tradeoff is worth taking

Given the measured numbers above, use this library when **memory, not latency, is the binding constraint** — you cannot fit the model in available VRAM/RAM at all, or you're capacity-planning around a hard memory ceiling — and you can tolerate slower per-call latency in exchange for fitting. If your bottleneck is throughput or per-request latency rather than a hard memory ceiling, dense (optionally quantized — INT8/FP8, which composes with TT-decomposition per the main README) is very likely the better choice today, pending a fused kernel.

This is exactly the kind of number a `profile_layer_compressibility()` + benchmark sweep is for: measure it on your actual model and hardware before committing, rather than assuming either direction.
