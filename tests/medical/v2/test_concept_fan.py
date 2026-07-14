from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fan.concept import ConceptFANConfig, ConceptFANLossConfig, OracleConceptFAN, PredictedConceptFAN, concept_fan_loss
from fan.concept.interventions import insert_contributions, remove_contributions


def test_concept_fan_weights_sum_to_one():
    model = PredictedConceptFAN(ConceptFANConfig(input_dim=27, latent_dim=16, n_concepts=5))
    out = model(torch.randn(4, 36, 27))
    assert set(out.as_dict()) == {
        "logit",
        "probability",
        "latent",
        "concepts",
        "memberships",
        "concept_weights",
        "concept_contributions",
    }
    assert torch.allclose(out.concept_weights.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out.concept_contributions, out.concept_weights * out.memberships)


def test_concept_fan_loss_components_are_independent():
    model = PredictedConceptFAN(ConceptFANConfig(input_dim=27, latent_dim=16, n_concepts=5))
    x = torch.randn(8, 36, 27)
    y = torch.randint(0, 2, (8,)).float()
    c = torch.rand(8, 5)
    parts = concept_fan_loss(model(x), y, c, ConceptFANLossConfig())
    assert set(parts) == {"task_loss", "concept_loss", "alignment_loss", "sparsity_loss", "total_loss"}
    assert parts["total_loss"].requires_grad
    values = {round(float(v.detach()), 6) for k, v in parts.items() if k != "total_loss"}
    assert len(values) > 1


def test_oracle_fan_requires_concepts_and_decision_uses_contributions():
    model = OracleConceptFAN(ConceptFANConfig(input_dim=27, latent_dim=16, n_concepts=4))
    x = torch.randn(3, 36, 27)
    c = torch.rand(3, 4)
    out = model(x, c)
    idx = torch.tensor([[0], [1], [2]])
    removed = remove_contributions(out.concept_contributions, idx)
    inserted = insert_contributions(out.concept_contributions, idx)
    assert removed.shape == out.concept_contributions.shape
    assert inserted.shape == out.concept_contributions.shape
    assert not torch.allclose(model.decision_from_contributions(removed), model.decision_from_contributions(inserted))
