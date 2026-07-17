from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fan.concept import MultiSetAdditiveTemporalConceptFANModel, TemporalConceptFANModel
from scripts.medical.v3.run_research_program import FINAL_STATUSES, set_all_seeds, validate_claims, validate_delivery


def test_temporal_fan_output_exposes_latent_sequence():
    model = TemporalConceptFANModel(input_dim=27, sequence_length=36, latent_dim=32, n_concepts=5)
    out = model(torch.randn(2, 36, 27))
    assert out.latent_sequence.shape == (2, 36, 32)
    assert out.as_dict()["latent_sequence"].shape == (2, 36, 32)


def test_multiset_additive_fan_shapes_and_signed_contributions():
    model = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=5,
        n_memberships=3,
        oracle=True,
        positive_decision_weights=True,
    )
    x = torch.randn(4, 36, 27)
    c = torch.rand(4, 36, 5)
    out = model(x, c)
    assert out.memberships.shape == (4, 5, 3)
    assert out.membership_local_weights.shape == (4, 5, 3)
    assert out.fuzzy_values.shape == (4, 5)
    assert out.concept_evidence.shape == (4, 5)
    assert out.signed_decision_contributions.shape == (4, 5)
    assert torch.allclose(out.membership_local_weights.sum(dim=-1), torch.ones(4, 5), atol=1e-6)
    assert torch.allclose(out.concept_weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out.logit, model.decision_head.bias + out.signed_decision_contributions.sum(dim=-1))
    assert torch.all(model.decision_head.weight > 0)


def test_multiset_additive_fan_concept_mask_zero_subset_is_bias_only():
    model = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=5,
        n_memberships=3,
        oracle=True,
    )
    x = torch.randn(3, 36, 27)
    c = torch.rand(3, 36, 5)
    full = model(x, c)
    empty_mask = torch.zeros(3, 5, dtype=torch.bool)
    masked = model(x, c, concept_mask=empty_mask)
    assert torch.allclose(masked.concept_evidence, torch.zeros_like(masked.concept_evidence), atol=1e-7)
    assert torch.allclose(masked.logit, model.decision_head.bias.expand_as(masked.logit), atol=1e-6)
    keep = torch.tensor([[True, False, True, False, False]]).expand(3, -1)
    partial = model.forward_from_summaries(
        full.latent_sequence,
        full.concept_trajectories,
        full.concept_summaries,
        full.temporal_concept_weights,
        concept_mask=keep,
    )
    assert torch.allclose(partial.concept_weights[:, [1, 3, 4]], torch.zeros(3, 3), atol=1e-7)
    assert torch.allclose(partial.concept_weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_multiset_alpha_modes_have_expected_weight_behavior():
    x = torch.randn(2, 36, 27)
    c = torch.rand(2, 36, 5)
    no_alpha = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=5,
        n_memberships=3,
        oracle=True,
        alpha_mode="no_alpha",
    )
    out_no_alpha = no_alpha(x, c)
    assert torch.allclose(out_no_alpha.concept_weights, torch.ones_like(out_no_alpha.concept_weights), atol=1e-6)

    uniform = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=5,
        n_memberships=3,
        oracle=True,
        alpha_mode="uniform_alpha",
    )
    out_uniform = uniform(x, c)
    assert torch.allclose(out_uniform.concept_weights, torch.full_like(out_uniform.concept_weights, 0.2), atol=1e-6)

    residual = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=5,
        n_memberships=3,
        oracle=True,
        alpha_mode="residual_floor_alpha",
        gamma_init=0.5,
        gamma_max=0.9,
    )
    out_residual = residual(x, c)
    assert 0.0 < float(residual.aggregator.gamma.detach()) < 0.9
    assert torch.all(out_residual.concept_weights > 0)
    assert torch.allclose(out_residual.concept_weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_no_alpha_uniform_alpha_equivalence_under_weight_rescaling():
    k = 5
    x = torch.randn(3, 36, 27)
    c = torch.rand(3, 36, k)
    no_alpha = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=k,
        n_memberships=3,
        oracle=True,
        alpha_mode="no_alpha",
    )
    uniform = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=27,
        sequence_length=36,
        latent_dim=32,
        n_concepts=k,
        n_memberships=3,
        oracle=True,
        alpha_mode="uniform_alpha",
    )
    uniform.load_state_dict(no_alpha.state_dict(), strict=False)
    with torch.no_grad():
        uniform.decision_head.raw_weight.copy_(no_alpha.decision_head.raw_weight * k)
    out_no_alpha = no_alpha(x, c)
    out_uniform = uniform(x, c)
    assert torch.allclose(out_no_alpha.logit, out_uniform.logit, atol=1e-6)


def test_set_all_seeds_controls_random_numpy_and_torch():
    set_all_seeds(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_all_seeds(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_v3_entrypoint_files_exist():
    base = Path("scripts/medical/v3")
    expected = [
        "run_research_program.py",
        "run_fan_iteration.py",
        "run_iteration.py",
        "analyze_iteration.py",
        "select_next_iteration.py",
        "train_fan.py",
        "evaluate_fan.py",
        "run_leakage_audit.py",
        "run_faithfulness.py",
        "diagnose_fan_faithfulness.py",
        "run_oracle_alpha_ablation.py",
        "run_predicted_fan_strict.py",
        "train_planted_sctc.py",
        "run_representation_audit.py",
        "train_standard_sctc.py",
        "train_fan_sctc.py",
        "run_interventions.py",
        "aggregate_results.py",
        "build_paper.py",
        "validate_paper_claims.py",
        "validate_delivery.py",
    ]
    assert all((base / name).exists() for name in expected)


def test_claim_validator_fixture(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "paper").mkdir()
    pd.DataFrame({"model": ["m", "m"], "AUPRC": [0.2, 0.4]}).to_csv(tmp_path / "results" / "fan_results.csv", index=False)
    claims = [
        {
            "claim_id": "x",
            "value": 0.3,
            "source_file": "results/fan_results.csv",
            "filters": {"model": "m"},
            "column": "AUPRC",
            "aggregation": "mean",
            "split": "validation",
        }
    ]
    (tmp_path / "paper" / "claims.json").write_text(json.dumps(claims))
    assert validate_claims(tmp_path)["passed"]


def test_delivery_validator_fixture(tmp_path: Path):
    required = [
        "results/program_status.json",
        "results/aggregate_metrics.csv",
        "results/fan_gate.json",
        "results/predicted_fan_results.csv",
        "results/planted_results.csv",
        "results/representation_audit.parquet",
        "results/standard_sctc_results.csv",
        "results/fan_sctc_results.csv",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/supplement.tex",
        "paper/supplement.pdf",
        "paper/claims.json",
        "paper/claims_validation.json",
        "manifests/test_unlock_manifest.json",
        "manifests/test_consumed.lock",
    ]
    for rel in required:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            pd.DataFrame({"x": [1]}).to_parquet(path)
        else:
            path.write_text("ok")
    (tmp_path / "results" / "program_status.json").write_text(json.dumps({"final_status": sorted(FINAL_STATUSES)[0]}))
    assert validate_delivery(tmp_path)["passed"]
