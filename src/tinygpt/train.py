"""Training loop: random-batch loader, AdamW param groups, warmup+cosine LR, checkpoints.

Everything the loop needs is in ``TrainConfig`` (read from the ``train:`` section of a YAML
file) plus a ``GPTConfig`` (the ``model:`` section). The loop writes ``log.csv`` and
``ckpt.pt`` into ``out_dir`` and can resume from the checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tinygpt.model import GPT, GPTConfig

CSV_COLUMNS = ["step", "train_loss", "val_loss", "lr", "tokens_per_sec"]


class ConfigError(ValueError):
    """Raised when a YAML config is missing or inconsistent."""


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping settings for one run."""

    data_dir: str = "data"
    out_dir: str = "out/tiny"
    device: str = "auto"
    seed: int = 1337
    batch_size: int = 32
    max_steps: int = 2000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 50
    notes: str = ""


@dataclass
class TrainResult:
    """Summary of a finished run; every number is measured, nothing is estimated."""

    steps: int
    num_params: int
    wall_time_s: float
    tokens_per_sec: float
    train_loss: float
    val_loss: float
    best_val_loss: float
    ckpt_path: str
    log_path: str
    history: list[dict[str, float]] = field(default_factory=list)


def _build(cls: type, section: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"unknown keys in '{name}' section: {sorted(unknown)}")
    return cls(**section)


def load_config(path: str | Path) -> tuple[GPTConfig, TrainConfig]:
    """Parse a YAML file with ``model:`` and ``train:`` sections."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "model" not in raw or "train" not in raw:
        raise ConfigError(f"{path} must contain 'model' and 'train' sections")
    return _build(GPTConfig, raw["model"], "model"), _build(TrainConfig, raw["train"], "train")


def resolve_device(device: str) -> torch.device:
    """Map ``auto|cpu|mps`` to a torch device, falling back to CPU when MPS is unavailable."""
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("device 'mps' requested but MPS is not available on this machine")
    if device not in {"cpu", "mps", "cuda"}:
        raise ConfigError(f"unsupported device {device!r}; use auto, cpu or mps")
    return torch.device(device)


# ---------------------------------------------------------------------------------- data
def load_tokens(path: str | Path) -> np.ndarray:
    """Memory-map a ``uint16`` token file written by ``tinygpt prepare``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `tinygpt prepare` first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; ``y`` is ``x`` shifted left by one token."""
    if len(data) <= block_size + 1:
        raise ValueError(f"dataset of {len(data)} tokens is too short for block_size {block_size}")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts]
    )
    return x, y


# ------------------------------------------------------------------------ optimisation
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to ``lr`` over ``warmup_steps``, then cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / (cfg.warmup_steps + 1)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def build_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices (dim >= 2) only; biases and LayerNorm gains are
    not decayed, since shrinking them toward zero has no regularising interpretation."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Mean loss over ``eval_iters`` random batches for each split, in eval mode."""
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(data, model.config.block_size, cfg.batch_size, generator)
            _, loss, _ = model(x.to(device), y.to(device))
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------- checkpoints
def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    train_cfg: TrainConfig,
    best_val_loss: float = math.inf,
    rng: dict[str, torch.Tensor] | None = None,
) -> None:
    """Write model/optimizer state plus both configs so a run can be resumed or sampled."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": asdict(model.config),
        "train_config": asdict(train_cfg),
        "best_val_loss": best_val_loss,
        "rng": rng or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` (weights-only safe load)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def model_from_checkpoint(ckpt: dict[str, Any], device: torch.device | str = "cpu") -> GPT:
    """Rebuild the model whose architecture and weights are stored in ``ckpt``."""
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


# ---------------------------------------------------------------------------------- loop
def _append_csv(path: Path, row: dict[str, float | int]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def train(
    model_cfg: GPTConfig,
    cfg: TrainConfig,
    resume: bool = False,
    log: Any = print,
) -> TrainResult:
    """Run the training loop described by the two configs and return measured results."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    data_dir, out_dir = Path(cfg.data_dir), Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out_dir / "ckpt.pt", out_dir / "log.csv"

    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["vocab_size"] != model_cfg.vocab_size:
            raise ConfigError(
                f"model.vocab_size={model_cfg.vocab_size} but {meta_path} says "
                f"{meta['vocab_size']}; re-run prepare or fix the config"
            )
    splits = {name: load_tokens(data_dir / f"{name}.bin") for name in ("train", "val")}

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    data_gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)
    step, best_val = 0, math.inf
    last_eval_step = -1
    if resume:
        ckpt = load_checkpoint(ckpt_path, device)
        if ckpt["config"] != asdict(model_cfg):
            raise ConfigError(
                f"checkpoint architecture {ckpt['config']} != config {asdict(model_cfg)}"
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val_loss"]
        last_eval_step = step  # the checkpoint was written right after an eval
        if "data" in ckpt["rng"]:
            data_gen.set_state(ckpt["rng"]["data"])
        log(f"resumed from {ckpt_path} at step {step}")
    else:
        if log_path.exists():
            log_path.unlink()

    n_params = model.num_params()
    start_step = step
    log(f"model: {n_params / 1e6:.3f}M params | device: {device} | {cfg.max_steps} steps")

    tokens_per_step = cfg.batch_size * model_cfg.block_size
    history: list[dict[str, float]] = []
    t_start = time.perf_counter()
    train_time, tokens_seen, last_loss = 0.0, 0, math.nan
    last_train, last_val = math.nan, math.nan
    model.train()

    def evaluate(step: int) -> None:
        nonlocal best_val, train_time, tokens_seen, last_train, last_val, last_eval_step
        last_eval_step = step
        losses = estimate_loss(model, splits, cfg, device, eval_gen)
        tps = tokens_seen / train_time if train_time > 0 else 0.0
        row = {
            "step": step,
            "train_loss": round(losses["train"], 4),
            "val_loss": round(losses["val"], 4),
            "lr": lr_at(step, cfg),
            "tokens_per_sec": round(tps, 1),
        }
        _append_csv(log_path, row)
        history.append(dict(row))
        last_train, last_val = losses["train"], losses["val"]
        best_val = min(best_val, losses["val"])
        log(
            f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
            f"| lr {row['lr']:.2e} | {tps:,.0f} tok/s"
        )
        train_time, tokens_seen = 0.0, 0
        save_checkpoint(
            ckpt_path, model, optimizer, step, cfg, best_val, {"data": data_gen.get_state()}
        )

    while step < cfg.max_steps:
        if step % cfg.eval_interval == 0 and step != last_eval_step:
            evaluate(step)
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        t0 = time.perf_counter()
        x, y = get_batch(splits["train"], model_cfg.block_size, cfg.batch_size, data_gen)
        _, loss, _ = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = loss.item()  # sync point; also keeps MPS from queueing unboundedly
        train_time += time.perf_counter() - t0
        tokens_seen += tokens_per_step
        step += 1
        if step % cfg.log_interval == 0:
            log(f"step {step:5d} | loss {last_loss:.4f} | lr {lr:.2e}")

    if step != last_eval_step:
        evaluate(step)
    wall = time.perf_counter() - t_start
    overall_tps = ((step - start_step) * tokens_per_step) / wall
    return TrainResult(
        steps=step,
        num_params=n_params,
        wall_time_s=wall,
        tokens_per_sec=overall_tps,
        train_loss=last_train,
        val_loss=last_val,
        best_val_loss=best_val,
        ckpt_path=str(ckpt_path),
        log_path=str(log_path),
        history=history,
    )
