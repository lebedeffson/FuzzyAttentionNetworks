from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fan.concept import TemporalConceptAggregator, TemporalConceptFANModel
from scripts.medical.v2.diagnose_concept_leakage import diagnose
from scripts.medical.v2.run_v2_1_program import sequence_targets


def test_temporal_concept_weights_sum_to_one():
    agg = TemporalConceptAggregator(n_concepts=5, sequence_length=36)
    q, w = agg(torch.rand(7, 36, 5))
    assert q.shape == (7, 5)
    assert w.shape == (7, 36, 5)
    assert torch.allclose(w.sum(dim=1), torch.ones(7, 5), atol=1e-6)


def test_temporal_oracle_fan_shapes():
    model = TemporalConceptFANModel(27, 36, 16, 5, oracle=True)
    out = model(torch.randn(4, 36, 27), torch.rand(4, 36, 5))
    assert out.concept_trajectories.shape == (4, 36, 5)
    assert out.temporal_concept_weights.shape == (4, 36, 5)
    assert out.concept_summaries.shape == (4, 5)
    assert torch.allclose(out.concept_weights.sum(dim=1), torch.ones(4), atol=1e-6)


def test_strict_projector_is_frozen_during_task_training():
    model = TemporalConceptFANModel(27, 36, 16, 5, oracle=False)
    model.freeze_concept_path()
    assert not any(p.requires_grad for p in model.projector.parameters())
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.decision_head.parameters())


def test_task_gradient_does_not_reach_projector_when_frozen():
    model = TemporalConceptFANModel(27, 36, 16, 5, oracle=False)
    model.freeze_concept_path()
    out = model(torch.randn(3, 36, 27))
    loss = out.logit.mean()
    loss.backward()
    assert all(p.grad is None for p in model.projector.parameters())


def test_concept_leakage_fixture():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(80, 5))
    y = (true[:, 0] > 0).astype(int)
    pred = true + rng.normal(scale=0.01, size=true.shape)
    result = diagnose(pred[:40], true[:40], y[:40], pred[40:], true[40:], y[40:], seed=0)
    assert result["predicted_concepts_auprc"] >= result["prevalence"]
    assert "residual_auprc" in result


def test_project_memory_present():
    root = Path(__file__).resolve().parents[3]
    assert (root / "AGENTS.md").exists()
    assert (root / "docs/medical/PROJECT_STATE.md").exists()

