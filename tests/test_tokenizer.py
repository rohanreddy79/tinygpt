"""Byte-level BPE: exact round-trips, merge count, determinism, persistence, errors."""

from __future__ import annotations

import json
import random

import pytest

from tinygpt.tokenizer import N_BYTES, Tokenizer, TokenizerError, pretokenize

SAMPLES = [
    "",
    "a",
    "hello world",
    "Hello, World! It's 2026.",
    "tab\there\r\nnewline  double  space   ",
    "snake_case_name and __dunder__",
    "emoji 🚀🔥 and flags 🇯🇵 and skin tones 👍🏽",
    "日本語のテキストと中文和한국어",
    "mixed: café naïve résumé – em—dash “quotes” …",
    "Ωmega αβγ δ ∑ ∫ √",
    "\x00\x01 control chars \x7f",
    "zero-width​joiner‍ and nbsp here",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip_exact(tokenizer: Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenize_partitions_text(text: str) -> None:
    assert "".join(pretokenize(text)) == text


def test_roundtrip_random_unicode(tokenizer: Tokenizer) -> None:
    rng = random.Random(1234)
    for _ in range(50):
        chars = [chr(rng.randrange(0x20, 0x2FFFF)) for _ in range(rng.randrange(0, 40))]
        text = "".join(c for c in chars if not (0xD800 <= ord(c) <= 0xDFFF))
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_untrained_tokenizer_is_bytes() -> None:
    tok = Tokenizer()
    assert tok.vocab_size == N_BYTES
    assert tok.encode("hi!") == list(b"hi!")
    assert tok.decode([104, 105]) == "hi"


def test_merge_count_equals_vocab_minus_256(corpus: str) -> None:
    for vocab_size in (256, 300, 400):
        tok = Tokenizer.train(corpus, vocab_size)
        assert len(tok.merges) == vocab_size - N_BYTES
        assert tok.vocab_size == vocab_size
        assert sorted(tok.vocab) == list(range(vocab_size))


def test_merges_are_ranked_and_compose(tokenizer: Tokenizer) -> None:
    for (a, b), new_id in tokenizer.merges.items():
        assert a < new_id and b < new_id
        assert tokenizer.vocab[new_id] == tokenizer.vocab[a] + tokenizer.vocab[b]
    assert sorted(tokenizer.merges.values()) == list(range(N_BYTES, tokenizer.vocab_size))


def test_compression_and_common_word(tokenizer: Tokenizer, corpus: str) -> None:
    ids = tokenizer.encode(corpus)
    assert len(ids) < len(corpus.encode("utf-8")) / 2
    assert len(tokenizer.encode(" the")) == 1  # most frequent chunk becomes one token


def test_training_is_deterministic(corpus: str) -> None:
    a = Tokenizer.train(corpus, 300)
    b = Tokenizer.train(corpus, 300)
    assert a.merges == b.merges


def test_save_load_roundtrip(tokenizer: Tokenizer, tmp_path, corpus: str) -> None:
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab == tokenizer.vocab
    assert loaded.encode(corpus[:2000]) == tokenizer.encode(corpus[:2000])
    payload = json.loads(path.read_text())
    assert payload["vocab_size"] == tokenizer.vocab_size
    assert len(payload["merges"]) == tokenizer.vocab_size - N_BYTES


def test_decode_invalid_utf8_does_not_raise(tokenizer: Tokenizer) -> None:
    assert tokenizer.decode([0xFF, 0xFE]) == "��"


def test_errors() -> None:
    with pytest.raises(TokenizerError):
        Tokenizer.train("abc", 255)
    with pytest.raises(TokenizerError):
        Tokenizer().decode([999])
    with pytest.raises(TokenizerError):
        Tokenizer({(1, 2): 300})  # id 300 is not the next free id
    with pytest.raises(TokenizerError):
        Tokenizer.train("ab", 300)  # too little text for 44 merges


def test_load_rejects_malformed(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"format": "other"}')
    with pytest.raises(TokenizerError):
        Tokenizer.load(path)
    path.write_text('{"format": "tinygpt-bpe-v1", "vocab_size": 257, "merges": [["a", 1]]}')
    with pytest.raises(TokenizerError):
        Tokenizer.load(path)
