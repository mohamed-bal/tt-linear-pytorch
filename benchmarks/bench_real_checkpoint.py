"""
benchmarks/bench_real_checkpoint.py — profile REAL trained layers from an
actually-loaded model checkpoint, using REAL captured activations.

Everything in bench_real_model.py uses correct architecture *dimensions*
but random weights — valid for the latency question, not the accuracy
question (compressibility depends on a layer's *actual* trained entropy
structure, which random weights don't have; see the main article/README).
This script is for the question that actually requires a real checkpoint:
"how much does this specific trained layer degrade at rank X?"

Requires: pip install transformers accelerate
Not a dependency of the tt_linear package itself — this is a standalone
example script, not something the library requires at import time.

Usage:
    python benchmarks/bench_real_checkpoint.py --model google/gemma-3-1b-it
    python benchmarks/bench_real_checkpoint.py --model <local-path> --layer-path model.layers.5.mlp.up_proj

Security notes (read before pointing this at a model you don't fully trust):
  - Loads with `trust_remote_code=False` explicitly — never flip this to True
    for a checkpoint you haven't personally audited; it permits arbitrary
    Python execution from the model repo.
  - Prefers safetensors (`use_safetensors=True`) over pickle-based .bin
    checkpoints wherever the repo provides both — safetensors cannot
    execute arbitrary code on load, standard pickle can.
  - This script only reads weights and runs inference; it never writes
    back to the source checkpoint.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tt_linear import TTMatrixConfig  # noqa: E402
from tt_linear.profiling import profile_layer_compressibility  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bench_real_checkpoint")


def _resolve_layer(model: nn.Module, layer_path: str) -> nn.Linear:
    """Walk a dotted attribute path (e.g. 'model.layers.5.mlp.up_proj') to
    the target nn.Linear. Raises a clear error rather than a bare
    AttributeError if any segment of the path doesn't exist — the exact
    path varies by model family, and a wrong guess should fail loudly."""
    obj = model
    for part in layer_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(
                f"'{part}' not found while resolving '{layer_path}'. "
                f"Available attributes at this point: {[n for n, _ in obj.named_children()][:20]}"
            )
        obj = getattr(obj, part)
    if not isinstance(obj, nn.Linear):
        raise ValueError(f"'{layer_path}' resolved to {type(obj)}, not nn.Linear.")
    return obj


def _capture_real_activation(model: nn.Module, target_layer: nn.Linear, input_ids: torch.Tensor) -> torch.Tensor:
    """Runs one real forward pass and captures the actual input activation
    to `target_layer` via a hook — this is the real, input-conditioned
    activation distribution, not synthetic noise, which is the entire
    point of profile_layer_compressibility over a weight-norm-only check."""
    captured = {}

    def hook(_module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        captured["activation"] = inputs[0].detach()

    handle = target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError(
            "Hook never fired — the target layer was not reached during this forward pass. "
            "Check that --layer-path points to a layer actually used for this input."
        )
    activation = captured["activation"]
    # Flatten any leading (batch, seq) dims into one batch dimension —
    # profile_layer_compressibility expects (batch, in_features).
    return activation.reshape(-1, activation.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model id or local path, e.g. google/gemma-3-1b-it")
    parser.add_argument("--layer-path", default="model.layers.5.mlp.up_proj",
                         help="Dotted attribute path to the target nn.Linear.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 48, 72])
    parser.add_argument("--sample-text", default="The quick brown fox jumps over the lazy dog.",
                         help="Text used to generate a real forward pass and capture real activations.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("This script requires `transformers` (and likely `accelerate`). "
                     "Install with: pip install transformers accelerate")
        sys.exit(1)

    logger.info("Loading tokenizer and model for %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,   # never load arbitrary remote code for an unaudited checkpoint
        use_safetensors=True,      # prefer safetensors over pickle-based .bin where available
        torch_dtype=torch.float32,
    ).to(args.device).eval()

    target_layer = _resolve_layer(model, args.layer_path)
    logger.info("Target layer resolved: %s -> Linear(%d, %d)",
                args.layer_path, target_layer.in_features, target_layer.out_features)

    input_ids = tokenizer(args.sample_text, return_tensors="pt").input_ids.to(args.device)
    real_activations = _capture_real_activation(model, target_layer, input_ids)
    logger.info("Captured %d real activation vectors of dimension %d.",
                real_activations.shape[0], real_activations.shape[1])

    print(f"\n{'rank':>6} {'compression_ratio':>18} {'functional_rel_error':>22}")
    for rank in args.ranks:
        try:
            in_f, out_f = target_layer.in_features, target_layer.out_features

            def factor_pair(n: int) -> tuple[int, int]:
                for a in range(int(n**0.5), 0, -1):
                    if n % a == 0:
                        return a, n // a
                raise ValueError(f"{n} has no clean factor pair.")

            in_factors = factor_pair(in_f)
            out_factors = factor_pair(out_f)
            config = TTMatrixConfig(out_factors=out_factors, in_factors=in_factors, ranks=(rank,))
            report = profile_layer_compressibility(target_layer, config, real_activations)
            print(f"{rank:>6} {report['compression_ratio']:>18.4f} {report['functional_rel_error']:>22.6f}")
        except ValueError as e:
            logger.warning("Skipping rank=%d: %s", rank, e)

    print("\nThis is the REAL functional error on THIS checkpoint's THIS layer against THIS sample text — "
          "not a generic figure. Re-run with representative production inputs, and sweep more layers, "
          "before picking a production rank. See the main article's decision framework for the full procedure.")


if __name__ == "__main__":
    main()
