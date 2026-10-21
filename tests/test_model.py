"""Model invariants: causality, KV cache, param count, init, SDPA parity, sampling filters."""

from __future__ import annotations

import pytest
import torch

from tinygpt.model import GPT, GPTConfig, filter_logits, sample_next


def expected_params(c: GPTConfig) -> int:
    """Closed-form parameter count for the tied-weight GPT."""
    E, L = c.n_embd, c.n_layer
    ln = 2 * E if c.bias else E
    attn = 4 * E * E + (4 * E if c.bias else 0)  # c_attn (3E x E) + c_proj (E x E)
    mlp = 8 * E * E + (5 * E if c.bias else 0)  # c_fc (4E x E) + c_proj (E x 4E)
    return c.vocab_size * E + c.block_size * E + L * (2 * ln + attn + mlp) + ln


@pytest.mark.parametrize("bias", [False, True])
def test_num_params_matches_formula(bias: bool) -> None:
    cfg = GPTConfig(vocab_size=100, block_size=24, n_layer=3, n_head=4, n_embd=48, bias=bias)
    model = GPT(cfg)
    assert model.num_params() == expected_params(cfg)
    assert model.num_params(non_embedding=True) == expected_params(cfg) - 24 * 48


def test_weight_tying(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()


def test_init_statistics() -> None:
    cfg = GPTConfig(vocab_size=2000, block_size=64, n_layer=4, n_head=4, n_embd=256)
    model = GPT(cfg)
    assert abs(model.wte.weight.std().item() - 0.02) < 0.002
    proj = model.blocks[0].attn.c_proj.weight
    assert abs(proj.std().item() - 0.02 / (2 * 4) ** 0.5) < 0.002


@pytest.mark.parametrize("sdpa", [True, False])
def test_causality(tiny_config: GPTConfig, sdpa: bool) -> None:
    cfg = GPTConfig(**{**tiny_config.__dict__, "sdpa": sdpa})
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    base, _, _ = model(x)
    for t in (0, 5, cfg.block_size - 1):
        y = x.clone()
        y[:, t] = (y[:, t] + 1) % cfg.vocab_size
        out, _, _ = model(y)
        assert torch.allclose(out[:, :t], base[:, :t], atol=1e-6)
        assert not torch.allclose(out[:, t], base[:, t])


def test_sdpa_matches_explicit_attention(tiny_config: GPTConfig) -> None:
    a = GPT(GPTConfig(**{**tiny_config.__dict__, "sdpa": True})).eval()
    b = GPT(GPTConfig(**{**tiny_config.__dict__, "sdpa": False})).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, tiny_config.vocab_size, (3, 10))
    la, _, _ = a(x)
    lb, _, _ = b(x)
    assert torch.allclose(la, lb, atol=1e-5)
    # Cached path uses the explicit mask for both implementations.
    la_c, _, kv_a = a(x[:, :6], use_cache=True)
    lb_c, _, kv_b = b(x[:, :6], use_cache=True)
    la2, _, _ = a(x[:, 6:], past_kv=kv_a)
    lb2, _, _ = b(x[:, 6:], past_kv=kv_b)
    assert torch.allclose(la2, lb2, atol=1e-5)


def test_kv_cache_logits_match_full_forward(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config).eval()
    x = torch.randint(0, tiny_config.vocab_size, (2, 12))
    full, _, _ = model(x)
    logits, _, kv = model(x[:, :4], use_cache=True)
    assert len(kv) == tiny_config.n_layer and kv[0][0].shape == (2, 2, 4, 16)
    steps = [logits[:, -1]]
    for t in range(4, 12):
        logits, _, kv = model(x[:, t : t + 1], past_kv=kv, use_cache=True)
        steps.append(logits[:, -1])
    assert kv[0][0].shape[2] == 12
    incremental = torch.stack(steps, dim=1)  # predictions after tokens 3..11
    assert torch.allclose(incremental, full[:, 3:], atol=1e-5)


@pytest.mark.parametrize("new_tokens", [4, 30])  # 30 overflows block_size=16 -> cache rebuild
def test_kv_cache_greedy_equals_no_cache(tiny_config: GPTConfig, new_tokens: int) -> None:
    model = GPT(tiny_config).eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (2, 5))
    with_cache = model.generate(prompt, new_tokens, temperature=0.0, use_cache=True)
    without = model.generate(prompt, new_tokens, temperature=0.0, use_cache=False)
    assert with_cache.shape == (2, 5 + new_tokens)
    assert torch.equal(with_cache, without)
    assert torch.equal(with_cache[:, :5], prompt)


def test_forward_rejects_overlong_sequence(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, tiny_config.block_size + 1, dtype=torch.long))


def test_loss_is_cross_entropy_and_ignores_minus_one(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config).eval()
    x = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, loss, _ = model(x, targets=x)
    manual = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
    assert torch.allclose(loss, manual)
    masked = x.clone()
    masked[:, :4] = -1
    _, loss_masked, _ = model(x, targets=masked)
    manual_masked = torch.nn.functional.cross_entropy(
        logits[:, 4:].reshape(-1, logits.size(-1)), x[:, 4:].reshape(-1)
    )
    assert torch.allclose(loss_masked, manual_masked)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GPTConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(n_layer=0)


def test_top_k_filter_and_sampling() -> None:
    logits = torch.randn(3, 50)
    filtered = filter_logits(logits, top_k=5)
    allowed = torch.topk(logits, 5, dim=-1).indices
    assert (torch.isfinite(filtered).sum(dim=-1) == 5).all()
    for row in range(3):
        assert set(torch.nonzero(torch.isfinite(filtered[row])).flatten().tolist()) == set(
            allowed[row].tolist()
        )
        draws = torch.cat([sample_next(logits[row : row + 1], 1.0, 5, None) for _ in range(300)])
        assert set(draws.flatten().tolist()) <= set(allowed[row].tolist())


def test_top_p_filter_keeps_minimal_prefix() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]]))
    kept = torch.isfinite(filter_logits(logits, top_p=0.75))[0]
    assert kept.tolist() == [True, True, False, False, False]
    kept = torch.isfinite(filter_logits(logits, top_p=0.81))[0]
    assert kept.tolist() == [True, True, True, False, False]
    assert torch.isfinite(filter_logits(logits, top_p=0.01))[0].tolist()[0] is True
    draws = torch.cat([sample_next(logits, 1.0, None, 0.75) for _ in range(300)])
    assert set(draws.flatten().tolist()) <= {0, 1}
    assert torch.equal(filter_logits(logits, top_p=1.0), logits)


def test_greedy_and_temperature_validation() -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3]])
    assert sample_next(logits, 0.0, None, None).item() == 1
    with pytest.raises(ValueError):
        sample_next(logits, -1.0, None, None)
