from __future__ import annotations

import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fan.concept.stability import (
    contribution_consistency_loss,
    make_masked_view,
    minimum_width_penalty,
    ordered_centers_penalty,
)
from fan.sctc.adaptive import (
    AdaptiveSCTCTrainConfig,
    AdaptiveSparseTranscoder,
    TopKAnnealingSchedule,
    effective_rank,
    layer_specific_capacity,
    train_adaptive_sctc,
)
from fan.sctc.joint_causal import (
    CompactCausalTranscoder,
    JointCausalSCTC,
    WhiteningTransform,
    compact_capacity,
    compact_top_k,
    decoder_incoherence,
)
from fan.sctc.concept_aligned import (
    ConceptAlignedInterventionalSCTC,
    ConceptAlignedInterventionalTrainConfig,
    correlation_matrix,
    sinkhorn,
    train_concept_aligned_interventional_sctc,
)
from scripts.medical.v3_1.run_planted_adaptive_sctc import (
    final_activation_stats,
    full_gate_status,
    intervention_validate_matches,
    node_matching,
    node_recovery_metrics,
)
from scripts.medical.v3_1.audit_concept_aligned_negative_result import build_validity_gate
from med_circuitbench.planted.model import PlantedCircuitModel


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


def test_smoke_gate_cannot_report_scientific_pass():
    result = pd.DataFrame(
        [
            {
                "delta_AUPRC": 0.0,
                "final_dead_feature_fraction": 0.0,
            }
        ]
    )
    gate = full_gate_status(
        result,
        full=False,
        run_interventions=False,
        run_negative_controls=False,
        smoke_reason="single seed, single layer, one epoch, warm-up top-k",
    )
    assert gate["status"] == "SMOKE_PASS"
    assert gate["scientific_gate_evaluated"] is False


def test_node_recovery_counts_unique_nodes_not_duplicate_features():
    matches = pd.DataFrame(
        [
            {"layer": 0, "feature_id": 0, "node": "I", "match_type": "PRIMARY_HUNGARIAN_MATCH", "accepted": True},
            {"layer": 0, "feature_id": 1, "node": "I", "match_type": "REDUNDANT_CORRELATED_FEATURE", "accepted": True},
            {"layer": 0, "feature_id": 2, "node": "I", "match_type": "REDUNDANT_CORRELATED_FEATURE", "accepted": False},
        ]
    )
    metrics = node_recovery_metrics(matches, layer=0)
    assert metrics["accepted_feature_matches"] == 2
    assert metrics["unique_recovered_nodes"] == 1
    assert metrics["redundant_matches"] == 2
    assert metrics["node_recall"] == 1.0


def test_final_activation_stats_uses_final_target_topk():
    x = torch.randn(10, 4, 8)
    model = AdaptiveSparseTranscoder.from_train_activation(x, n_features=16, top_k=4)
    model.set_active_top_k(16)
    stats = final_activation_stats(model, x.numpy(), target_top_k=4, dead_threshold=1e-5)
    assert stats["final_active_top_k"] == 4
    assert stats["final_L0_per_token"] <= 4.1
    assert "effective_active_feature_count" in stats


def test_intervention_validation_emits_true_and_negative_control_rows():
    planted = PlantedCircuitModel(seed=1, d_model=16)
    states = np.random.default_rng(1).random((8, 4, 5)).astype("float32")
    with torch.no_grad():
        out = planted(torch.from_numpy(states))
    activation = out.layers[0].numpy()
    nodes = out.nodes.numpy()
    transcoder = AdaptiveSparseTranscoder.from_train_activation(torch.from_numpy(activation).float(), n_features=4, top_k=2)
    matches = node_matching(planted, transcoder, activation, nodes, layer=0, seed=1)
    updated, evidence = intervention_validate_matches(
        matches,
        planted,
        transcoder,
        activation,
        nodes,
        layer=0,
        seed=1,
        run_interventions=True,
        run_negative_controls=True,
        n_nulls=2,
    )
    assert len(updated) == len(matches)
    assert {"true_feature_intervention", "matched_random_ablation", "wrong_layer_node_label", "permuted_node_label"}.issubset(
        set(evidence["evidence_type"])
    )


def test_adaptive_training_respects_min_epochs_before_early_stopping():
    x = torch.randn(12, 3, 6)

    def behavior_forward(repl: torch.Tensor) -> torch.Tensor:
        return repl.mean(dim=(1, 2))

    _, log = train_adaptive_sctc(
        x,
        torch.zeros(12),
        behavior_forward,
        AdaptiveSCTCTrainConfig(n_features=8, target_top_k=4, epochs=5, min_epochs=3, patience=1),
        torch.device("cpu"),
    )
    assert len(log) >= 3


def test_joint_causal_whitening_roundtrip_shape_and_rank():
    x = torch.randn(20, 5, 12)
    whitening = WhiteningTransform.fit(x, variance=0.995)
    xw = whitening.whiten(x)
    xr = whitening.unwhiten(xw)
    assert xw.shape[-1] <= x.shape[-1]
    assert xr.shape == x.shape
    assert torch.mean((xr - x) ** 2).item() < 1.0


def test_compact_capacity_and_topk_are_registered():
    x = torch.randn(20, 4, 10)
    cap = compact_capacity(x, multiplier=2.0, minimum=4, maximum=32)
    assert cap in {4, 6, 8, 12, 16, 24, 32}
    assert compact_top_k(4) == 2
    assert compact_top_k(12) == 4
    assert compact_top_k(24) == 6
    assert compact_top_k(32) == 8


def test_joint_causal_forward_outputs_all_layers():
    xs = [torch.randn(6, 3, 8) for _ in range(4)]
    transcoders = [CompactCausalTranscoder.from_train_activation(x, n_features=4, top_k=2) for x in xs]
    model = JointCausalSCTC(transcoders)
    out = model(xs)
    assert len(out["z"]) == 4
    assert len(model.transitions) == 3
    assert out["reconstructed"][0].shape == xs[0].shape
    assert decoder_incoherence(transcoders[0]).item() >= 0.0


def test_sinkhorn_returns_finite_soft_assignment():
    scores = torch.randn(5, 2)
    assignment = sinkhorn(scores)
    assert torch.isfinite(assignment).all()
    assert assignment.shape == scores.shape
    assert torch.all(assignment >= 0)
    assert torch.allclose(assignment.sum(dim=0), torch.ones(2), atol=1e-4)


def test_correlation_matrix_shape():
    features = torch.randn(4, 3, 6)
    targets = torch.randn(4, 3, 2)
    corr = correlation_matrix(features, targets)
    assert corr.shape == (6, 2)
    assert torch.isfinite(corr).all()


def test_concept_aligned_model_emits_layer_readouts():
    xs = [torch.randn(6, 3, 8) for _ in range(4)]
    transcoders = [CompactCausalTranscoder.from_train_activation(x, n_features=4, top_k=2) for x in xs]
    model = ConceptAlignedInterventionalSCTC(transcoders)
    out = model(xs)
    preds = model.concept_predictions(out["z"])
    assert [p.shape[-1] for p in preds] == [1, 1, 1, 2]


def test_concept_aligned_training_uses_downstream_callback():
    xs = [torch.randn(10, 4, 6) for _ in range(4)]
    concepts = torch.randn(10, 4, 5)
    calls = {"count": 0}

    def behavior_forward(_layer: int, repl: torch.Tensor) -> torch.Tensor:
        return repl.mean(dim=(1, 2))

    def downstream_forward(layer: int, repl: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return xs[layer + 1][: repl.shape[0]].to(repl.device) + 0.0 * repl.mean()

    model, log = train_concept_aligned_interventional_sctc(
        xs,
        concepts,
        behavior_forward,
        downstream_forward,
        torch.zeros(10),
        ConceptAlignedInterventionalTrainConfig(epochs=1, batch_size=5, lambda_interventional=0.03),
        torch.device("cpu"),
        control_seed=123,
    )
    assert isinstance(model, ConceptAlignedInterventionalSCTC)
    assert calls["count"] > 0
    assert {"concept_loss", "matching_loss", "interventional_loss"}.issubset(log.columns)


def test_negative_result_validity_gate_blocks_when_oracle_fails(tmp_path):
    gradients = pd.DataFrame(
        [
            {
                "alignment_reaches_encoder": True,
                "alignment_reaches_decoder": False,
            }
        ]
    )
    continuous = pd.DataFrame(
        [
            {"stage": "correct_concepts", "concept_R2": 0.9},
            {"stage": "permuted_concepts", "concept_R2": 0.1},
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "metric": "concept_R2",
                "ci_lower": 0.1,
            }
        ]
    )
    gate = build_validity_gate(
        {"status": "FAIL"},
        gradients,
        continuous,
        bootstrap,
        {"status": "PASS", "prediction_rows": 30000},
        tmp_path,
    )
    assert gate["status"] == "EVALUATION_PROTOCOL_INVALID"
    assert gate["scientific_gate"]["oracle_evaluator"] == "FAIL"
