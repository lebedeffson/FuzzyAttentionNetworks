import torch

from src.concept_fan import (
    ConceptMembership,
    ConceptFANBlock,
    SafetyCriticalConceptFAN,
    structural_alignment_loss,
)


def test_membership_families_are_bounded():
    concepts = torch.rand(5, 4)

    for family in ("gaussian", "bell", "sigmoid", "mixed"):
        membership = ConceptMembership(num_concepts=4, family=family)
        values = membership(concepts)
        assert values.shape == (5, 4)
        assert torch.all((values >= 0) & (values <= 1))


def test_concept_fan_block_returns_weighted_membership_evidence():
    block = ConceptFANBlock(num_concepts=4, hidden_dim=8, membership_family="mixed")
    concepts = torch.rand(3, 4)

    evidence, alpha, membership = block(concepts)

    assert evidence.shape == (3, 4)
    assert alpha.shape == (3, 4)
    assert membership.shape == (3, 4)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(evidence, alpha * membership)
    assert torch.all((membership >= 0) & (membership <= 1))


def test_safety_critical_concept_fan_forward_contract():
    model = SafetyCriticalConceptFAN(in_channels=5, num_concepts=4, hidden_dim=8, latent_dim=6)
    x = torch.rand(2, 5, 12)

    out = model(x)

    assert set(out) == {"logit", "z", "concepts", "alpha", "membership"}
    assert out["logit"].shape == (2,)
    assert out["z"].shape == (2, 6)
    assert out["concepts"].shape == (2, 4)
    assert out["alpha"].shape == (2, 4)
    assert out["membership"].shape == (2, 4)


def test_structural_alignment_loss_is_finite():
    z = torch.rand(4, 6)
    concepts = torch.rand(4, 4)

    loss = structural_alignment_loss(z, concepts)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
