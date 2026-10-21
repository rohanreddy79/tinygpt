"""Byte-level BPE tokenizer trained from scratch.

The base vocabulary is the 256 byte values, so any UTF-8 string round-trips exactly and no
``<unk>`` token is ever needed. Text is first split into "chunks" by a GPT-2-style regex
(words with their leading space, numbers, punctuation runs, whitespace) so merges never cross
chunk boundaries. Training counts byte pairs over the *unique* chunks weighted by their
frequency and updates the pair counts incrementally after every merge, which keeps a
1 MB corpus to a few seconds rather than minutes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Letters are ``[^\W\d_]`` (unicode-aware stand-in for ``\p{L}``); numbers are ``\d``;
# "other" is any non-space char that is neither, which also covers ``_`` and emoji.
# ``\s+(?!\S)`` keeps trailing whitespace separate so a word keeps its single leading space.
PRETOKENIZE_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_PRETOKENIZE_RE = re.compile(PRETOKENIZE_PATTERN)

N_BYTES = 256
Pair = tuple[int, int]


class TokenizerError(ValueError):
    """Raised for invalid tokenizer arguments or malformed tokenizer files."""


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into chunks whose concatenation is exactly ``text``."""
    return _PRETOKENIZE_RE.findall(text)


def _merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _adjacent_pairs(ids: list[int]) -> list[Pair]:
    return list(zip(ids, ids[1:], strict=False))


class Tokenizer:
    """Byte-level BPE with a regex pre-tokenizer.

    ``merges`` maps a byte-pair ``(a, b)`` to the id it merges into; ids are assigned in
    training order starting at 256, so ``merges[pair] - 256`` is the merge rank.
    """

    def __init__(self, merges: dict[Pair, int] | None = None) -> None:
        self.merges: dict[Pair, int] = dict(merges or {})
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            if a not in self.vocab or b not in self.vocab or new_id != len(self.vocab):
                raise TokenizerError(
                    f"merge {(a, b)} -> {new_id} is out of order or references unknown ids"
                )
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        """Number of token ids, i.e. 256 bytes plus one id per merge."""
        return len(self.vocab)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> Tokenizer:
        """Learn ``vocab_size - 256`` merges from ``text`` by greedy most-frequent-pair merging.

        Ties are broken by the smaller pair so training is deterministic regardless of
        dictionary iteration order.
        """
        if vocab_size < N_BYTES:
            raise TokenizerError(f"vocab_size must be >= {N_BYTES}, got {vocab_size}")
        n_merges = vocab_size - N_BYTES

        chunk_freq = Counter(pretokenize(text))
        chunks: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_freq]
        freqs: list[int] = list(chunk_freq.values())

        # pair -> total weighted count, and pair -> indices of the chunks containing it.
        pair_counts: Counter[Pair] = Counter()
        where: dict[Pair, set[int]] = {}
        for idx, (ids, f) in enumerate(zip(chunks, freqs, strict=True)):
            for p in _adjacent_pairs(ids):
                pair_counts[p] += f
                where.setdefault(p, set()).add(idx)

        merges: dict[Pair, int] = {}
        for rank in range(n_merges):
            if not pair_counts:
                raise TokenizerError(
                    f"corpus exhausted after {rank} merges; lower vocab_size or use more text"
                )
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = N_BYTES + rank
            merges[best] = new_id
            if verbose and (rank % 100 == 0 or rank == n_merges - 1):
                print(f"merge {rank + 1}/{n_merges}: {best} -> {new_id} ({pair_counts[best]}x)")

            # Incremental update: only chunks containing ``best`` change their pair counts.
            for idx in list(where.pop(best)):
                ids, f = chunks[idx], freqs[idx]
                old_pairs = _adjacent_pairs(ids)
                new_ids = _merge_pair(ids, best, new_id)
                new_pairs = _adjacent_pairs(new_ids)
                chunks[idx] = new_ids
                for p in old_pairs:
                    pair_counts[p] -= f
                for p in new_pairs:
                    pair_counts[p] += f
                for p in set(old_pairs) - set(new_pairs):
                    if p != best:
                        where[p].discard(idx)
                for p in set(new_pairs):
                    where.setdefault(p, set()).add(idx)
            pair_counts.pop(best, None)
            for p in [p for p, c in pair_counts.items() if c <= 0]:
                del pair_counts[p]
                where.pop(p, None)
        return cls(merges)

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = _adjacent_pairs(ids)
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _merge_pair(ids, best, self.merges[best])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids. Identical chunks are encoded once per call."""
        cache: dict[str, list[int]] = {}
        out: list[int] = []
        for chunk in pretokenize(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = cache[chunk] = self._encode_chunk(chunk)
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        """Decode ids to text. Invalid UTF-8 (possible for model samples) becomes U+FFFD."""
        try:
            data = b"".join(self.vocab[i] for i in ids)
        except KeyError as e:
            raise TokenizerError(f"unknown token id {e.args[0]}") from e
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        """Write the merges (the only learned state) as JSON."""
        payload = {
            "format": "tinygpt-bpe-v1",
            "pattern": PRETOKENIZE_PATTERN,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair, _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
        }
        Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        """Load a tokenizer written by :meth:`save`."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("format") != "tinygpt-bpe-v1":
                raise TokenizerError(f"unrecognised tokenizer format in {path}")
            merges = {
                (int(a), int(b)): N_BYTES + rank for rank, (a, b) in enumerate(payload["merges"])
            }
        except (KeyError, ValueError, TypeError) as e:
            raise TokenizerError(f"malformed tokenizer file {path}: {e}") from e
        tok = cls(merges)
        if tok.vocab_size != payload["vocab_size"]:
            raise TokenizerError(f"{path}: vocab_size {payload['vocab_size']} != {tok.vocab_size}")
        return tok
