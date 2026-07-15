from __future__ import annotations

import torch

from fan.attention import FuzzyTemporalAttention, FuzzyTemporalAttentionEncoder


def test_fuzzy_attention_shape_and_rows_sum():
    attn = FuzzyTemporalAttention(d_model=16, n_heads=4)
    x = torch.randn(3, 5, 16)
    y, weights, aux = attn(x)
    assert y.shape == x.shape
    assert weights.shape == (3, 4, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4, 5), atol=1e-5)
    assert aux["mu_q"].shape == (3, 4, 5, 3, 4)


def test_padding_keys_zero_and_queries_zero_output():
    attn = FuzzyTemporalAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -2:] = True
    y, weights, _ = attn(x, key_padding_mask=mask)
    assert torch.all(weights[..., -2:] == 0)
    assert torch.allclose(y[:, -2:], torch.zeros_like(y[:, -2:]), atol=1e-6)


def test_widths_positive_and_membership_gradients_nonzero():
    attn = FuzzyTemporalAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)
    y, _, _ = attn(x)
    y.square().mean().backward()
    assert torch.all(attn.membership.widths > 0)
    assert attn.membership.centers.grad is not None
    assert attn.membership.centers.grad.abs().sum().item() > 0


def test_no_nan_under_extreme_inputs():
    attn = FuzzyTemporalAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16) * 100
    y, weights, _ = attn(x)
    assert torch.isfinite(y).all()
    assert torch.isfinite(weights).all()


def test_fuzzy_encoder_outputs_sequence():
    enc = FuzzyTemporalAttentionEncoder(input_dim=7, sequence_length=8, d_model=16, n_heads=4, n_layers=2, d_ffn=32)
    x = torch.randn(4, 8, 7)
    h, weights = enc(x, return_attention=True)
    assert h.shape == (4, 8, 16)
    assert len(weights) == 2
