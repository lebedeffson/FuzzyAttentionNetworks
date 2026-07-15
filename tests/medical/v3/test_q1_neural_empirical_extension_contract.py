from __future__ import annotations

import torch

from scripts.medical.v3_1.run_q1_neural_empirical_extension import (
    ARMS,
    make_model,
    model_config,
    model_output_dict,
    run_grid,
    state_dict_sha256,
)


def _cfg():
    return {
        "model": {"input_dim": 27},
        "dataset": {"observed_window": 36},
    }


def test_q1_neural_grid_requires_30_fits_per_arm():
    grid = run_grid(30)
    assert len(grid) == 30
    assert len({row["initialization_seed"] for row in grid}) == 10
    assert len({row["data_order_seed"] for row in grid}) == 3
    assert len(ARMS) == 5
    assert len(grid) * len(ARMS) == 150


def test_q1_conceptfan_noalpha_forward_uses_raw_sequence_input():
    cfg = model_config(_cfg())
    model = make_model("ConceptFAN-NoAlpha", cfg)
    out = model_output_dict(model(torch.randn(4, 36, 27)))
    assert out["probability"].shape == (4,)
    assert out["concept_trajectories"].shape == (4, 36, 5)
    assert out["local_contributions"].shape == (4, 5)
    assert model.aggregator.alpha_mode == "no_alpha"
    assert model.oracle is False


def test_q1_all_neural_arms_accept_same_raw_input_contract():
    cfg = model_config(_cfg())
    x = torch.randn(3, 36, 27)
    for arm in ARMS:
        out = model_output_dict(make_model(arm, cfg)(x))
        assert out["probability"].shape == (3,)
        assert out["latent_sequence"].shape[:2] == (3, 36)
        if arm != "PlainTransformer":
            assert out["concept_trajectories"].shape == (3, 36, 5)
            assert out["local_contributions"].shape == (3, 5)


def test_q1_parameter_sha_depends_on_model_parameters():
    cfg = model_config(_cfg())
    model = make_model("NoFuzzy", cfg)
    before = state_dict_sha256(model.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    after = state_dict_sha256(model.state_dict())
    assert before != after
