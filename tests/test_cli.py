"""End-to-end CLI run on a synthetic corpus: prepare -> train -> sample -> plot (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tinygpt.cli import app
from tinygpt.tokenizer import Tokenizer

runner = CliRunner()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, corpus: str) -> Path:
    root = tmp_path_factory.mktemp("ws")
    (root / "input.txt").write_text(corpus, encoding="utf-8")
    config = {
        "model": {
            "vocab_size": 300,
            "block_size": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
        },
        "train": {
            "data_dir": str(root / "data"),
            "out_dir": str(root / "out"),
            "device": "cpu",
            "batch_size": 4,
            "max_steps": 6,
            "eval_interval": 3,
            "eval_iters": 1,
            "log_interval": 2,
        },
    }
    (root / "tiny.yaml").write_text(yaml.safe_dump(config))
    return root


def test_prepare(workspace: Path) -> None:
    result = runner.invoke(
        app,
        [
            "prepare",
            "--input",
            str(workspace / "input.txt"),
            "--data-dir",
            str(workspace / "data"),
            "--vocab-size",
            "300",
        ],
    )
    assert result.exit_code == 0, result.output
    data = workspace / "data"
    meta = json.loads((data / "meta.json").read_text())
    assert meta["vocab_size"] == 300
    train_ids = np.fromfile(data / "train.bin", dtype=np.uint16)
    assert train_ids.size == meta["train_tokens"] and train_ids.max() < 300
    tok = Tokenizer.load(data / "tokenizer.json")
    text = (workspace / "input.txt").read_text()
    assert tok.decode(train_ids.tolist()) == text[: int(len(text) * 0.9)]


def test_train(workspace: Path) -> None:
    result = runner.invoke(app, ["train", "--config", str(workspace / "tiny.yaml")])
    assert result.exit_code == 0, result.output
    out = workspace / "out"
    assert (out / "ckpt.pt").exists() and (out / "tokenizer.json").exists()
    rows = (out / "log.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in rows[1:]] == [0, 3, 6]
    assert "val loss" in result.output and "params" in result.output


def test_sample(workspace: Path) -> None:
    args = ["sample", "--ckpt", str(workspace / "out" / "ckpt.pt"), "--prompt", "the king"]
    result = runner.invoke(app, [*args, "--max-new-tokens", "12", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("the king") and len(result.output) > len("the king\n")
    again = runner.invoke(app, [*args, "--max-new-tokens", "12", "--seed", "1"])
    assert again.output == result.output  # seeded sampling is deterministic
    greedy = runner.invoke(app, [*args, "--max-new-tokens", "8", "--temperature", "0"])
    assert greedy.exit_code == 0, greedy.output


def test_plot(workspace: Path) -> None:
    out = workspace / "assets" / "loss.png"
    log = workspace / "out" / "log.csv"
    result = runner.invoke(app, ["plot", "--log", str(log), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_error_paths(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["prepare", "--input", str(tmp_path / "nope.txt")])
    assert missing.exit_code == 1 and "not found" in missing.output
    bad_cfg = runner.invoke(app, ["train", "--config", str(tmp_path / "nope.yaml")])
    assert bad_cfg.exit_code == 1 and "config file not found" in bad_cfg.output
    no_ckpt = runner.invoke(app, ["sample", "--ckpt", str(tmp_path / "nope.pt")])
    assert no_ckpt.exit_code == 1 and "checkpoint not found" in no_ckpt.output
    no_log = runner.invoke(app, ["plot", "--log", str(tmp_path / "nope.csv")])
    assert no_log.exit_code == 1 and "log not found" in no_log.output
