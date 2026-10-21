"""Shared fixtures: a deterministic synthetic corpus and tiny model configs."""

from __future__ import annotations

import random

import pytest
import torch

from tinygpt.model import GPTConfig
from tinygpt.tokenizer import Tokenizer

WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "king",
    "and", "queen", "speak", "softly", "of", "love", "war", "honour", "death", "night", "day",
    "sun", "moon", "stars", "sea", "thou", "thee", "thy", "hath", "doth", "shall", "will",
    "not", "never", "ever", "here", "there", "where", "when", "what", "why", "who", "which",
]  # fmt: skip


def make_corpus(n_words: int = 4000, seed: int = 0) -> str:
    """Pseudo-English with punctuation and newlines; deterministic for a given seed."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(n_words):
        w = rng.choice(WORDS)
        if rng.random() < 0.08:
            w = w.capitalize()
        out.append(w)
        if rng.random() < 0.12:
            out[-1] += rng.choice([",", ".", "!", "?", ";", "'s"])
        if i % 11 == 10:
            out[-1] += "\n"
    return " ".join(out).replace("\n ", "\n")


@pytest.fixture(scope="session")
def corpus() -> str:
    return make_corpus()


@pytest.fixture(scope="session")
def tokenizer(corpus: str) -> Tokenizer:
    return Tokenizer.train(corpus, vocab_size=320)


@pytest.fixture
def tiny_config() -> GPTConfig:
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)
