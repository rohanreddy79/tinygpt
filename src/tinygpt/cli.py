"""The ``tinygpt`` command line: prepare -> train -> sample -> plot."""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from tinygpt import train as training
from tinygpt.plot import plot_loss_curve
from tinygpt.tokenizer import Tokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

app = typer.Typer(
    help="A GPT trained from scratch, readable end to end.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", err=True, fg=typer.colors.RED)
    return typer.Exit(code=1)


def _download(url: str, dest: Path, timeout: float = 30.0) -> None:
    typer.echo(f"downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
            dest.write_bytes(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise _fail(f"download failed ({e}); pass --input with a local text file") from e


@app.command()
def prepare(
    input_path: Annotated[
        Path | None, typer.Option("--input", help="UTF-8 text corpus (default: Tiny Shakespeare)")
    ] = None,
    data_dir: Annotated[Path, typer.Option(help="Where to write tokenizer and token files")] = Path(
        "data"
    ),
    vocab_size: Annotated[int, typer.Option(min=256, max=65535)] = 512,
    val_frac: Annotated[float, typer.Option(min=0.01, max=0.5, help="Tail fraction for val")] = 0.1,
    url: Annotated[
        str, typer.Option(help="Download URL when --input is omitted")
    ] = TINY_SHAKESPEARE_URL,
) -> None:
    """Fetch a corpus, train the BPE tokenizer and write train.bin / val.bin (uint16)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if input_path is None:
        input_path = data_dir / "input.txt"
        if not input_path.exists():
            _download(url, input_path)
    if not input_path.exists():
        raise _fail(f"input file not found: {input_path}")
    text = input_path.read_text(encoding="utf-8")
    typer.echo(f"corpus: {input_path} ({len(text):,} chars, {len(text.encode()):,} bytes)")

    t0 = time.perf_counter()
    tok = Tokenizer.train(text, vocab_size)
    tok.save(data_dir / "tokenizer.json")
    typer.echo(f"tokenizer: {tok.vocab_size} tokens trained in {time.perf_counter() - t0:.1f}s")

    split = int(len(text) * (1 - val_frac))
    t0 = time.perf_counter()
    train_ids = np.array(tok.encode(text[:split]), dtype=np.uint16)
    val_ids = np.array(tok.encode(text[split:]), dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    meta = {
        "vocab_size": tok.vocab_size,
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "bytes_per_token": round(len(text.encode()) / (train_ids.size + val_ids.size), 3),
        "source": str(input_path),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    typer.echo(
        f"encoded in {time.perf_counter() - t0:.1f}s: train {meta['train_tokens']:,} tokens, "
        f"val {meta['val_tokens']:,} tokens, {meta['bytes_per_token']} bytes/token"
    )


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="YAML with model: and train: sections")] = Path(
        "configs/tiny.yaml"
    ),
    device: Annotated[str | None, typer.Option(help="auto | cpu | mps (overrides config)")] = None,
    max_steps: Annotated[int | None, typer.Option(help="Override train.max_steps")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Override train.out_dir")] = None,
    resume: Annotated[bool, typer.Option(help="Continue from out_dir/ckpt.pt")] = False,
) -> None:
    """Train a model and write ckpt.pt + log.csv into the output directory."""
    try:
        model_cfg, train_cfg = training.load_config(config)
    except training.ConfigError as e:
        raise _fail(str(e)) from e
    if device is not None:
        train_cfg.device = device
    if max_steps is not None:
        train_cfg.max_steps = max_steps
    if out_dir is not None:
        train_cfg.out_dir = str(out_dir)

    tok_src = Path(train_cfg.data_dir) / "tokenizer.json"
    if not tok_src.exists():
        raise _fail(f"{tok_src} not found; run `tinygpt prepare` first")
    Path(train_cfg.out_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(tok_src, Path(train_cfg.out_dir) / "tokenizer.json")

    try:
        result = training.train(model_cfg, train_cfg, resume=resume, log=typer.echo)
    except (training.ConfigError, FileNotFoundError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo("")
    typer.echo(f"params        {result.num_params:,}")
    typer.echo(f"steps         {result.steps}")
    typer.echo(f"wall time     {result.wall_time_s:.1f}s")
    typer.echo(f"tokens/sec    {result.tokens_per_sec:,.0f} (whole run incl. eval)")
    typer.echo(f"train loss    {result.train_loss:.4f}")
    typer.echo(f"val loss      {result.val_loss:.4f} (best {result.best_val_loss:.4f})")
    typer.echo(f"checkpoint    {result.ckpt_path}")


@app.command()
def sample(
    ckpt: Annotated[Path, typer.Option(help="Checkpoint from `tinygpt train`")] = Path(
        "out/tiny/ckpt.pt"
    ),
    prompt: Annotated[str, typer.Option(help="Text to continue")] = "\n",
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 200,
    temperature: Annotated[float, typer.Option(min=0.0, help="0 = greedy")] = 0.8,
    top_k: Annotated[int | None, typer.Option(min=1)] = 50,
    top_p: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option(help="auto | cpu | mps")] = "cpu",
    tokenizer: Annotated[
        Path | None, typer.Option(help="tokenizer.json (default: next to the checkpoint)")
    ] = None,
) -> None:
    """Generate text from a checkpoint using the KV cache."""
    try:
        dev = training.resolve_device(device)
        payload = training.load_checkpoint(ckpt, dev)
        model = training.model_from_checkpoint(payload, dev).eval()
        tok = Tokenizer.load(tokenizer or ckpt.parent / "tokenizer.json")
    except (training.ConfigError, FileNotFoundError, ValueError) as e:
        raise _fail(str(e)) from e
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    typer.echo(tok.decode(out[0].tolist()))


@app.command()
def plot(
    log: Annotated[Path, typer.Option(help="log.csv from `tinygpt train`")] = Path(
        "out/tiny/log.csv"
    ),
    out: Annotated[Path, typer.Option(help="PNG to write")] = Path("assets/loss_curve.png"),
    title: Annotated[str, typer.Option()] = "tinygpt: train / val loss",
) -> None:
    """Plot train and val loss from the CSV log."""
    try:
        path = plot_loss_curve(log, out, title=title)
    except (FileNotFoundError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
