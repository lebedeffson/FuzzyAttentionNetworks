from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fan.concept.stability import (
    contribution_consistency_loss,
    make_masked_view,
    minimum_width_penalty,
    ordered_centers_penalty,
)
from fan.sctc.adaptive import (
    AdaptiveSparseTranscoder,
    TopKAnnealingSchedule,
    effective_rank,
    layer_specific_capacity,
)


def test_adaptive_sctc_normalizes_and_reconstructs_raw_shape():
    x = torch.randn(8, 6, 12) * 3.0 + 2.0
    model = AdaptiveSparseTranscoder.from_train_activation(x, n_features=24, top_k=8)
    out = model(x)
    assert out["reconstructed"].shape == x.shape
    assert out["reconstructed_normalized"].shape == x.shape
    assert model.reconstruction_bias.shape == (12,)
    assert torch.all(model.train_std >= 1e-5)


def test_topk_annealing_schedule_reaches_target():
    sched = TopKAnnealingSchedule(target_top_k=8, no_topk_epochs=3, anneal_until_epoch=10, initial_top_k=32)
    assert sched.top_k_for_epoch(1, 64) == 32
    assert sched.top_k_for_epoch(10, 64) == 8
    assert sched.top_k_for_epoch(99, 64) == 8


def test_dead_feature_resampling_uses_residual_directions():
    x = torch.randn(16, 4, 10)
    model = AdaptiveSparseTranscoder.from_train_activation(x, n_features=12, top_k=2)
    with torch.no_grad():
        model.encoder.bias[:] = -100.0
    count = model.resample_dead_features(x, frequency_threshold=1e-5, max_resamples=3)
    assert count == 3
    assert torch.allclose(model.decoder.weight.norm(dim=0), torch.ones(12), atol=1e-5)


def test_effective_rank_and_layer_capacity_are_bounded():
    x = torch.randn(40, 5, 16)
    rank = effective_rank(x)
    cap = layer_specific_capacity(x, multiplier=4.0, minimum=16, maximum=96)
    assert rank > 0
    assert 16 <= cap <= 96


def test_contribution_consistency_loss_zero_for_identical_vectors():
    contrib = torch.tensor([[1.0, -2.0, 0.5], [0.1, 0.2, -0.3]])
    loss = contribution_consistency_loss(contrib, contrib.clone())
    assert loss.item() < 1e-6


def test_masked_view_changes_some_entries_with_full_mask():
    x = torch.ones(4, 3, 2)
    y = make_masked_view(x, mask_probability=1.0, noise_std=0.0)
    assert torch.count_nonzero(y).item() == 0


def test_basis_penalties_detect_invalid_order_and_width():
    centers = torch.tensor([[0.2, 0.1, 0.8]])
    widths = torch.tensor([[0.1, 0.0001, 0.2]])
    assert ordered_centers_penalty(centers).item() > 0
    assert minimum_width_penalty(widths, min_width=0.001).item() > 0
