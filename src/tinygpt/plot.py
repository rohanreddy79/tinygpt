"""Render the ``log.csv`` written by the trainer as a loss-curve PNG (headless matplotlib)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen before pyplot import)


def read_log(path: str | Path) -> list[dict[str, float]]:
    """Read the trainer CSV into a list of float-valued rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"log not found: {path}")
    with path.open(newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def plot_loss_curve(log_path: str | Path, out_path: str | Path, title: str = "tinygpt") -> Path:
    """Plot train/val loss against step and save a PNG; returns the output path."""
    rows = read_log(log_path)
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(steps, [r["train_loss"] for r in rows], label="train", color="#4C72B0")
    ax.plot(steps, [r["val_loss"] for r in rows], label="val", color="#DD8452")
    final = rows[-1]
    ax.annotate(
        f"val {final['val_loss']:.3f}",
        xy=(final["step"], final["val_loss"]),
        xytext=(-60, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats / token)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
