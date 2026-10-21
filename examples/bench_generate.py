"""Measure greedy generation speed with and without the KV cache on a trained checkpoint.

Usage: uv run python examples/bench_generate.py [--ckpt out/tiny/ckpt.pt] [--new 120 500]
Reports the median of 5 runs (CPU, torch's default thread count) and checks that both
decoding paths produce identical tokens.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from tinygpt import train as training
from tinygpt.tokenizer import Tokenizer


def time_generate(model, ids: torch.Tensor, n_new: int, use_cache: bool, reps: int = 5) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.generate(ids, n_new, temperature=0.0, use_cache=use_cache)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", default="out/tiny/ckpt.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--new", type=int, nargs="+", default=[120, 500])
    args = parser.parse_args()

    ckpt = training.load_checkpoint(args.ckpt)
    model = training.model_from_checkpoint(ckpt).eval()
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    ids = torch.tensor([tok.encode(args.prompt)])
    print(
        f"{model.num_params():,} params | block_size {model.config.block_size} | "
        f"prompt {ids.shape[1]} tokens | torch threads {torch.get_num_threads()}"
    )
    print(f"{'new tokens':>10} | {'kv cache':>12} | {'no cache':>12} | speedup")
    for n_new in args.new:
        cached = time_generate(model, ids, n_new, use_cache=True)
        plain = time_generate(model, ids, n_new, use_cache=False)
        a = model.generate(ids, n_new, temperature=0.0, use_cache=True)
        b = model.generate(ids, n_new, temperature=0.0, use_cache=False)
        assert torch.equal(a, b), "cache and no-cache greedy outputs differ"
        print(
            f"{n_new:>10} | {n_new / cached:>8.0f} t/s | {n_new / plain:>8.0f} t/s | "
            f"{plain / cached:.2f}x"
        )


if __name__ == "__main__":
    main()
