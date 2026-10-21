"""Decoder-only transformer (GPT-2 style) with explicit attention and KV-cache generation.

Shapes use B = batch, T = tokens in this call, P = tokens already in the cache,
H = heads, D = head dimension, E = n_embd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# One (k, v) pair per layer, each of shape (B, H, P, D).
LayerKV = tuple[torch.Tensor, torch.Tensor]
PastKV = list[LayerKV]


@dataclass
class GPTConfig:
    """Hyperparameters that define the architecture (saved inside every checkpoint)."""

    vocab_size: int = 512
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    # Use torch's fused scaled_dot_product_attention kernel instead of the explicit path.
    # Both compute the same thing; the explicit path exists to be read.
    sdpa: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if self.vocab_size <= 0 or self.block_size <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size, block_size and n_layer must be positive")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional key/value cache.

    attention(Q, K, V) = softmax(Q K^T / sqrt(D) + mask) V, where the mask is -inf above the
    diagonal so position t sees only positions <= t.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.sdpa = config.sdpa
        # q, k, v projections fused into one matmul: (E) -> (3E).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, past_kv: LayerKV | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, LayerKV | None]:
        B, T, E = x.shape
        q, k, v = self.c_attn(x).split(E, dim=2)
        # (B, T, E) -> (B, H, T, D) so each head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)  # (B, H, P+T, D)
            v = torch.cat([past_v, v], dim=2)
        present: LayerKV | None = (k, v) if use_cache else None

        if self.sdpa and past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # Query i sits at absolute position P+i and may attend to keys 0..P+i.
            allowed = torch.ones(T, past_len + T, dtype=torch.bool, device=x.device).tril(past_len)
            if self.sdpa:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
                )
            else:
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, P+T)
                att = att.masked_fill(~allowed, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, E)  # concat heads
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    """Position-wise feed-forward network: E -> 4E -> GELU -> E."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, past_kv: LayerKV | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, LayerKV | None]:
        a, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    """Token + learned position embeddings, N pre-LN blocks, final LayerNorm, tied LM head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output projection reuses the input embedding matrix.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 scales residual-branch output projections by 1/sqrt(2 * n_layer) so the
        # residual stream's variance does not grow with depth at init.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Parameter count. Tied weights count once; ``non_embedding`` also drops ``wpe``."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: PastKV | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PastKV | None]:
        """Return ``(logits, loss, new_kv)``.

        ``logits`` is (B, T, vocab). ``loss`` is mean cross-entropy when ``targets`` is given
        (positions with target -1 are ignored). ``new_kv`` is the per-layer cache covering
        all P+T positions when ``use_cache`` is set, else None.
        """
        B, T = idx.shape
        past_len = past_kv[0][0].size(2) if past_kv else 0
        if past_len + T > self.config.block_size:
            raise ValueError(
                f"sequence of {past_len + T} tokens exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        new_kv: PastKV = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, past_kv=past_kv[i] if past_kv else None, use_cache=use_cache)
            if use_cache and present is not None:
                new_kv.append(present)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss, (new_kv if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (B, T0) by ``max_new_tokens`` tokens.

        ``temperature == 0`` is greedy decoding. With the cache, each step feeds only the
        newest token; the cache is rebuilt from the last ``block_size`` tokens whenever the
        context would overflow, because learned absolute positions cannot slide.
        """
        block = self.config.block_size
        past: PastKV | None = None
        logits: torch.Tensor | None = None
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _, _ = self(idx[:, -block:])
            elif past is None or past[0][0].size(2) + 1 > block:
                logits, _, past = self(idx[:, -block:], use_cache=True)
            else:
                logits, _, past = self(idx[:, -1:], past_kv=past, use_cache=True)
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Set disallowed logits to -inf. ``top_k`` keeps the k largest; ``top_p`` (nucleus) keeps
    the smallest prefix of the descending-sorted distribution whose mass reaches ``p``."""
    logits = logits.clone()
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop a token if the mass *before* it already reaches p; the top token always stays.
        drop = (cum - torch.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def sample_next(
    logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Pick one next-token id per row from (B, vocab) logits; returns (B, 1)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1)
