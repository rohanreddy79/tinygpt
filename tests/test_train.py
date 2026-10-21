"""Trainer: schedule, optimizer groups, batches, overfitting, checkpoints, resume, config."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tinygpt import train as training
from tinygpt.model import GPT, GPTConfig
from tinygpt.train import ConfigError, TrainConfig


def write_repeated_dataset(data_dir: Path, period: int = 16, repeats: int = 400) -> np.ndarray:
    pattern = np.arange(1, period + 1, dtype=np.uint16)
    tokens = np.tile(pattern, repeats)
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens.tofile(data_dir / "train.bin")
    tokens[: period * 40].tofile(data_dir / "val.bin")
    return tokens


def test_cosine_schedule_boundaries() -> None:
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=1100)
    assert math.isclose(training.lr_at(0, cfg), 1e-3 / 101)
    assert training.lr_at(50, cfg) < training.lr_at(99, cfg) < cfg.lr
    assert math.isclose(training.lr_at(100, cfg), cfg.lr)
    assert math.isclose(training.lr_at(600, cfg), (cfg.lr + cfg.min_lr) / 2)
    assert math.isclose(training.lr_at(1100, cfg), cfg.min_lr)
    assert math.isclose(training.lr_at(5000, cfg), cfg.min_lr)
    lrs = [training.lr_at(s, cfg) for s in range(100, 1101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))


def test_optimizer_param_groups(tiny_config: GPTConfig) -> None:
    model = GPT(GPTConfig(**{**tiny_config.__dict__, "bias": True}))
    opt = training.build_optimizer(model, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    total = sum(p.numel() for p in decay["params"]) + sum(p.numel() for p in no_decay["params"])
    assert total == model.num_params()


def test_get_batch_shapes_and_shift() -> None:
    data = np.arange(1000, dtype=np.uint16)
    gen = torch.Generator().manual_seed(0)
    x, y = training.get_batch(data, block_size=8, batch_size=4, generator=gen)
    assert x.shape == y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y[:, 0], x[:, 0] + 1)
    with pytest.raises(ValueError):
        training.get_batch(data[:5], block_size=8, batch_size=1, generator=gen)


def test_two_layer_model_overfits_repeated_sequence(tmp_path: Path) -> None:
    write_repeated_dataset(tmp_path / "data")
    model_cfg = GPTConfig(vocab_size=32, block_size=32, n_layer=2, n_head=2, n_embd=32)
    cfg = TrainConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        batch_size=8,
        max_steps=200,
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=10,
        eval_interval=100,
        eval_iters=2,
        log_interval=1000,
    )
    result = training.train(model_cfg, cfg, log=lambda _: None)
    assert result.steps == 200
    assert result.history[0]["train_loss"] > 2.0  # near ln(32) at init
    assert result.train_loss < 0.1 and result.val_loss < 0.1
    assert Path(result.ckpt_path).exists() and Path(result.log_path).exists()
    rows = Path(result.log_path).read_text().splitlines()
    assert rows[0] == "step,train_loss,val_loss,lr,tokens_per_sec"
    assert len(rows) == 1 + 3  # steps 0, 100, 200
    assert result.tokens_per_sec > 0


def test_checkpoint_roundtrip_identical_logits(tiny_config: GPTConfig, tmp_path: Path) -> None:
    model = GPT(tiny_config).eval()
    opt = training.build_optimizer(model, TrainConfig())
    path = tmp_path / "ckpt.pt"
    training.save_checkpoint(path, model, opt, step=7, train_cfg=TrainConfig(), best_val_loss=1.5)
    ckpt = training.load_checkpoint(path)
    assert ckpt["step"] == 7 and ckpt["best_val_loss"] == 1.5
    assert ckpt["config"] == tiny_config.__dict__
    restored = training.model_from_checkpoint(ckpt).eval()
    x = torch.randint(0, tiny_config.vocab_size, (2, 10))
    assert torch.equal(model(x)[0], restored(x)[0])
    with pytest.raises(FileNotFoundError):
        training.load_checkpoint(tmp_path / "missing.pt")


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    write_repeated_dataset(tmp_path / "data")
    model_cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16)
    cfg = TrainConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        batch_size=4,
        max_steps=10,
        eval_interval=5,
        eval_iters=1,
        log_interval=1000,
    )
    first = training.train(model_cfg, cfg, log=lambda _: None)
    assert training.load_checkpoint(first.ckpt_path)["step"] == 10
    cfg.max_steps = 20
    second = training.train(model_cfg, cfg, resume=True, log=lambda _: None)
    assert second.steps == 20
    assert training.load_checkpoint(second.ckpt_path)["step"] == 20
    assert [r["step"] for r in second.history] == [15, 20]
    assert second.tokens_per_sec > 0
    rows = Path(second.log_path).read_text().splitlines()
    assert [int(r.split(",")[0]) for r in rows[1:]] == [0, 5, 10, 15, 20]
    with pytest.raises(ConfigError):
        training.train(GPTConfig(**{**model_cfg.__dict__, "n_embd": 32}), cfg, resume=True)


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"model": {"n_layer": 1}, "train": {"max_steps": 3}}))
    model_cfg, train_cfg = training.load_config(path)
    assert model_cfg.n_layer == 1 and train_cfg.max_steps == 3
    path.write_text(yaml.safe_dump({"model": {"n_layers": 1}, "train": {}}))
    with pytest.raises(ConfigError, match="unknown keys"):
        training.load_config(path)
    path.write_text("just a string")
    with pytest.raises(ConfigError):
        training.load_config(path)
    with pytest.raises(ConfigError):
        training.load_config(tmp_path / "nope.yaml")


def test_vocab_mismatch_with_meta_is_rejected(tmp_path: Path) -> None:
    write_repeated_dataset(tmp_path / "data")
    (tmp_path / "data" / "meta.json").write_text('{"vocab_size": 999}')
    cfg = TrainConfig(data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "out"), max_steps=1)
    with pytest.raises(ConfigError, match="vocab_size"):
        training.train(GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=1, n_embd=8), cfg)


def test_resolve_device() -> None:
    assert training.resolve_device("cpu").type == "cpu"
    assert training.resolve_device("auto").type in {"cpu", "mps"}
    with pytest.raises(ConfigError):
        training.resolve_device("tpu")
