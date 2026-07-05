"""
benchmarks/bench_tt_linear.py — latency, throughput, and memory benchmarks
comparing dense nn.Linear against TTLinear across ranks and batch sizes.

This is a *measurement* tool, not a correctness test (see tests/test_core.py
for correctness). It answers a different, honest question: does TT format
actually run faster, and how much memory does it actually save, on the
hardware you run it on — not on hardware assumptions baked into the code.

Usage:
    python benchmarks/bench_tt_linear.py
    python benchmarks/bench_tt_linear.py --device cuda --dtype float16
    python benchmarks/bench_tt_linear.py --sizes 512 1024 --ranks 16 32 64 128 --out results.csv

Design notes (read before trusting the numbers):
  - GPU timing requires torch.cuda.synchronize() around the timed region,
    otherwise you're timing kernel *launch*, not kernel *completion* — a
    classic and completely silent source of wrong benchmark numbers.
  - Warmup iterations are discarded (JIT/cuDNN autotuning, cache effects).
  - Each configuration reports mean and std over --trials repetitions, not
    a single sample — a single wall-clock measurement is not a benchmark.
  - Every configuration is wrapped so a single OOM or shape error skips
    that configuration (logged) rather than crashing the entire sweep —
    a multi-hour sweep should not die on configuration #40 of 200.
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tt_linear import TTLinear, TTMatrixConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bench_tt_linear")


@dataclass
class BenchResult:
    size: int
    rank: int
    batch_size: int
    device: str
    dtype: str
    dense_latency_ms_mean: float
    dense_latency_ms_std: float
    tt_latency_ms_mean: float
    tt_latency_ms_std: float
    speedup: float  # dense_mean / tt_mean; <1 means TT is SLOWER
    dense_params: int
    tt_params: int
    compression_ratio: float
    status: str  # "ok" or an error description


def _factor_pair(n: int) -> tuple[int, int]:
    """Pick a roughly-square (a, b) with a*b == n, for building a 2-factor
    TTMatrixConfig from a plain square layer size. Raises if n has no such
    factorization — the caller should pick benchmark sizes that do."""
    for a in range(int(n**0.5), 0, -1):
        if n % a == 0:
            return a, n // a
    raise ValueError(f"{n} could not be factorized — pick a size with a clean factor pair.")


def _timed_forward(module: nn.Module, x: torch.Tensor, device: str, trials: int, warmup: int) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) over `trials` repetitions, after `warmup`
    discarded iterations. Synchronizes CUDA explicitly — required for
    correct GPU timing, silently wrong without it."""
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        if device == "cuda":
            torch.cuda.synchronize()

        samples_ms = []
        for _ in range(trials):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            module(x)
            if device == "cuda":
                torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000)

    return statistics.mean(samples_ms), statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0


def run_one(size: int, rank: int, batch_size: int, device: str, dtype: torch.dtype, trials: int, warmup: int) -> BenchResult:
    dtype_name = str(dtype).replace("torch.", "")
    try:
        a, b = _factor_pair(size)
        config = TTMatrixConfig(out_factors=(a, b), in_factors=(a, b), ranks=(rank,))

        dense = nn.Linear(size, size, dtype=dtype).to(device).eval()
        tt = TTLinear.from_dense(dense, config).to(device).eval()
        x = torch.randn(batch_size, size, dtype=dtype, device=device)

        dense_mean, dense_std = _timed_forward(dense, x, device, trials, warmup)
        tt_mean, tt_std = _timed_forward(tt, x, device, trials, warmup)
        report = tt.compression_report()

        return BenchResult(
            size=size, rank=rank, batch_size=batch_size, device=device, dtype=dtype_name,
            dense_latency_ms_mean=dense_mean, dense_latency_ms_std=dense_std,
            tt_latency_ms_mean=tt_mean, tt_latency_ms_std=tt_std,
            speedup=dense_mean / tt_mean if tt_mean > 0 else float("nan"),
            dense_params=report["dense_params"], tt_params=report["tt_params"],
            compression_ratio=report["compression_ratio"], status="ok",
        )
    except torch.cuda.OutOfMemoryError as e:  # noqa: F821 - only defined when CUDA built
        logger.warning("OOM at size=%d rank=%d batch=%d — skipping", size, rank, batch_size)
        if device == "cuda":
            torch.cuda.empty_cache()
        return BenchResult(size, rank, batch_size, device, dtype_name, 0, 0, 0, 0, float("nan"), 0, 0, 0, f"OOM: {e}")
    except Exception as e:  # noqa: BLE001 - deliberately broad: this is a sweep, not a single run
        logger.warning("Failed at size=%d rank=%d batch=%d: %s — skipping", size, rank, batch_size, e)
        return BenchResult(size, rank, batch_size, device, dtype_name, 0, 0, 0, 0, float("nan"), 0, 0, 0, f"error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                         help="Square layer sizes (in_features == out_features) to benchmark.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64],
                         help="Bond dimensions to sweep per size.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32], dest="batch_sizes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trials", type=int, default=50, help="Timed repetitions per configuration.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations per configuration.")
    parser.add_argument("--out", type=Path, default=Path("bench_results.csv"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available. Falling back to cpu.")
        args.device = "cpu"

    dtype = getattr(torch, args.dtype)
    if args.device == "cpu" and dtype in (torch.float16,):
        logger.warning("float16 on CPU is typically unsupported or very slow for matmul — "
                        "results at this dtype/device combination may not be meaningful.")

    results: list[BenchResult] = []
    total = len(args.sizes) * len(args.ranks) * len(args.batch_sizes)
    done = 0
    for size in args.sizes:
        for rank in args.ranks:
            if rank > size:
                logger.info("Skipping rank=%d > size=%d (not a meaningful config).", rank, size)
                continue
            for batch_size in args.batch_sizes:
                done += 1
                logger.info("[%d/%d] size=%d rank=%d batch=%d device=%s dtype=%s",
                            done, total, size, rank, batch_size, args.device, args.dtype)
                results.append(run_one(size, rank, batch_size, args.device, dtype, args.trials, args.warmup))

    ok_results = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\n{'size':>6} {'rank':>6} {'batch':>6} {'dense_ms':>12} {'tt_ms':>12} {'speedup':>9} {'compression':>12}")
    for r in ok_results:
        print(f"{r.size:>6} {r.rank:>6} {r.batch_size:>6} "
              f"{r.dense_latency_ms_mean:>9.3f}±{r.dense_latency_ms_std:<3.2f} "
              f"{r.tt_latency_ms_mean:>9.3f}±{r.tt_latency_ms_std:<3.2f} "
              f"{r.speedup:>8.2f}x {r.compression_ratio:>11.3f}")

    print(f"\n{len(ok_results)}/{len(results)} configurations succeeded. Results written to {args.out}")
    if failed:
        print(f"{len(failed)} configuration(s) failed or were skipped — see log above and 'status' column in {args.out}.")


if __name__ == "__main__":
    main()
