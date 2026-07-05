# Benchmarks

**Read this before deciding whether TT-decomposition helps your workload — the answer is scale-and-configuration-dependent, not universal, and an earlier version of this document got that wrong by only testing small matrices.** Everything below is measured, not projected.

## Methodology

- `time.perf_counter()` around the forward call only; `torch.cuda.synchronize()` immediately before and after the timed region on CUDA (required — without it you measure kernel *launch* time, not completion time).
- 8–10 warmup iterations discarded per configuration, then 30–50 timed repetitions; every number is mean ± stdev, never a single sample.
- Every configuration wrapped in its own try/except — an OOM or shape error at one configuration is logged and skipped, not fatal to the sweep.
- Run it yourself: `python benchmarks/bench_tt_linear.py --device cuda --dtype float16` (defaults to CPU + float32 if no CUDA is available). No GPU was available on the machine these reference numbers were collected on — if you're targeting GPU deployment, re-run yourself before concluding either direction.

## Two different experiments, two different pictures — read both

### Experiment 1: small square matrices (256–1024), swept across rank

At these sizes, **TT is consistently slower than dense — 2x to over 50x, worse at larger batch and higher rank.** Full numbers from `bench_tt_linear.py`:

| size | rank | batch | dense (ms) | TT (ms) | speedup | compression ratio |
|---|---|---|---|---|---|---|
| 256 | 8 | 1 | 0.020 | 0.175 | 0.11x | 0.062 |
| 256 | 128 | 32 | 0.071 | 2.124 | 0.03x | 1.000 |
| 512 | 8 | 1 | 0.064 | 0.151 | 0.42x | 0.039 |
| 512 | 128 | 32 | 0.270 | 13.967 | 0.02x | 0.625 |

At this scale, fixed per-operation Python/dispatch overhead (several `tensordot`/`movedim` calls vs. one dense GEMM) dominates regardless of rank — there isn't enough actual compute in a 256×256 or 512×512 matmul for the FLOP reduction to outweigh that overhead.

### Experiment 2: real Gemma-3-1B MLP dimensions (embed_dim=1152, MLP hidden=6912 — verified architecture, randomly initialized weights), swept across rank and batch

**This is where the picture changes completely.** At real model scale, low rank + low-to-moderate batch is a genuine speed *and* memory win — not just a memory/latency tradeoff:

| direction | rank | batch | dense (ms) | TT (ms) | speedup | compression ratio |
|---|---|---|---|---|---|---|
| up-proj (1152→6912) | 16 | 1 | 1.622 | 0.275 | **5.89x** | 0.012 |
| up-proj (1152→6912) | 16 | 8 | 5.234 | 1.064 | **4.92x** | 0.012 |
| up-proj (1152→6912) | 32 | 32 | 7.320 | 7.017 | 1.04x | 0.023 |
| up-proj (1152→6912) | 72 | 32 | 7.035 | 16.899 | 0.42x | 0.052 |
| down-proj (6912→1152) | 16 | 1 | 1.328 | 0.345 | **3.85x** | 0.012 |
| down-proj (6912→1152) | 72 | 32 | 6.003 | 25.961 | 0.23x | 0.052 |

Full sweep (ranks 16/32/48/72 × batch 1/8/32, both MLP directions) in `bench_real_model.py`.

## The actual crossover, stated precisely

There isn't a single "TT is faster" or "TT is slower" answer — there's a crossover surface in (matrix size, rank, batch) space:

- **Larger matrices favor TT** (more real FLOPs for the compression to actually save, relative to fixed per-op overhead).
- **Lower rank favors TT** (the FLOP reduction is bigger — and this is also the regime where compression is worth doing at all, per the entropy argument in the main article: aggressive truncation only works where a layer's real entropy is low).
- **Lower batch size favors TT** (dense GEMM's advantage grows with batch since it parallelizes a single large matmul better than TT's chain of smaller ones).
- **Higher rank and higher batch both favor dense** — as either increases, the crossover point moves toward dense being faster, consistent with both experiments above.

**The practically important consequence:** the exact regime where compression is *statistically justified* (low rank, because a layer's measured entropy is low) is also, at real model scale, frequently the regime where it's *faster*, not just smaller. The earlier, blanket "TT is always slower" statement was true for the toy sizes tested but does not generalize — this is exactly why `profile_layer_compressibility` measures on your actual layer shapes rather than assuming from a small-matrix benchmark, and why re-running these benchmarks on your specific model dimensions (not this repo's reference numbers) is the only way to know where you land.

## What we still can't verify from this repo alone

These are real Gemma-3-1B *architecture* dimensions, not real Gemma-3-1B *weights* — no Hugging Face network access was available when collecting these numbers, so every result above is on randomly initialized layers of the correct shape. That's sufficient for the latency/throughput question (which depends on tensor shapes and rank, not weight values), but **not** sufficient for the accuracy/compressibility question — that requires real trained weights and real captured activations, which only you can provide. See `bench_real_checkpoint.py` for a ready-to-run script against an actual loaded checkpoint.

## When this tradeoff is worth taking

Given both experiments: at small layers or high rank/batch, use this library only when memory is the hard constraint, independent of latency. At real LLM-scale layers with a validated low rank (which `profile_layer_compressibility` should confirm holds acceptable functional error), you may get a genuine speed win alongside the memory win — but confirm this on your own model's actual dimensions before planning capacity around it either way.