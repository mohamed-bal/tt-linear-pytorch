"""
benchmarks/bench_real_model.py — same measurement methodology as
bench_tt_linear.py, but using real Gemma-3-1B MLP layer dimensions
(embed_dim=1152, MLP hidden=6912 — verified against the architecture,
not asserted from memory) instead of arbitrary square sizes.

This exists because the small-square-matrix results in bench_tt_linear.py
do NOT generalize to real model scale — see benchmarks/README.md for the
full explanation. Weights here are randomly initialized (no Hugging Face
network access was available when writing this): valid for the
latency/throughput question, NOT valid for the accuracy/compressibility
question. For that, use bench_real_checkpoint.py against an actual
loaded checkpoint.

Usage:
    python benchmarks/bench_real_model.py
    python benchmarks/bench_real_model.py --ranks 16 32 48 72 --batch-sizes 1 8 32
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tt_linear import TTLinear, TTMatrixConfig  # noqa: E402

# Gemma-3-1B architecture dimensions (verified: embed_dim=1152, MLP hidden=6912).
# Factor pairs chosen close to sqrt() of each dimension for balanced cores:
# 1152 = 32 x 36, 6912 = 72 x 96.
EMBED_DIM = 1152
MLP_HIDDEN = 6912
EMBED_FACTORS = (32, 36)
MLP_FACTORS = (72, 96)


@dataclass
class RealModelBenchResult:
    direction: str  # "up_proj" or "down_proj"
    rank: int
    batch_size: int
    dense_latency_ms_mean: float
    tt_latency_ms_mean: float
    speedup: float
    compression_ratio: float


def _timed(module: nn.Module, x: torch.Tensor, trials: int, warmup: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            module(x)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 48, 72])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32], dest="batch_sizes")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("bench_real_model_results.csv"))
    args = parser.parse_args()

    torch.manual_seed(0)
    up_proj = nn.Linear(EMBED_DIM, MLP_HIDDEN, bias=False)
    down_proj = nn.Linear(MLP_HIDDEN, EMBED_DIM, bias=False)

    results: list[RealModelBenchResult] = []

    for rank in args.ranks:
        cfg_up = TTMatrixConfig(out_factors=MLP_FACTORS, in_factors=EMBED_FACTORS, ranks=(rank,))
        cfg_down = TTMatrixConfig(out_factors=EMBED_FACTORS, in_factors=MLP_FACTORS, ranks=(rank,))
        tt_up = TTLinear.from_dense(up_proj, cfg_up)
        tt_down = TTLinear.from_dense(down_proj, cfg_down)
        report_up = tt_up.compression_report()
        report_down = tt_down.compression_report()

        for batch in args.batch_sizes:
            x_up = torch.randn(batch, EMBED_DIM)
            x_down = torch.randn(batch, MLP_HIDDEN)

            dm, _ = _timed(up_proj, x_up, args.trials, args.warmup)
            tm, _ = _timed(tt_up, x_up, args.trials, args.warmup)
            results.append(RealModelBenchResult("up_proj", rank, batch, dm, tm, dm / tm, report_up["compression_ratio"]))

            dm2, _ = _timed(down_proj, x_down, args.trials, args.warmup)
            tm2, _ = _timed(tt_down, x_down, args.trials, args.warmup)
            results.append(RealModelBenchResult("down_proj", rank, batch, dm2, tm2, dm2 / tm2, report_down["compression_ratio"]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"{'direction':>10} {'rank':>6} {'batch':>6} {'dense_ms':>10} {'tt_ms':>10} {'speedup':>9} {'compression':>12}")
    for r in results:
        print(f"{r.direction:>10} {r.rank:>6} {r.batch_size:>6} {r.dense_latency_ms_mean:>10.4f} "
              f"{r.tt_latency_ms_mean:>10.4f} {r.speedup:>8.3f}x {r.compression_ratio:>11.4f}")

    print(f"\nResults written to {args.out}")
    print("\nNote: random weights at real architecture dimensions — valid for latency, "
          "not for accuracy/compressibility. See bench_real_checkpoint.py for that question.")


if __name__ == "__main__":
    main()
