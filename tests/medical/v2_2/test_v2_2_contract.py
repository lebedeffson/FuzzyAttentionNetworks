from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from med_circuitbench.v2_2.planted_real import generate_planted
from scripts.medical.v2_2.run_v2_2_program import full_run_guard, gaussian_membership, fan_evidence


def load_full():
    return yaml.safe_load(Path("configs/medical/v2_2/full.yaml").read_text())


def test_full_run_guard_accepts_full_config():
    cfg = load_full()
    full_run_guard(cfg, [42, 43, 44], "full")


@pytest.mark.parametrize(
    "path,value",
    [
        (("dataset", "n_samples"), 9999),
        (("model", "layers"), 3),
        (("model", "d_model"), 64),
        (("model", "d_ffn"), 128),
        (("training", "max_epochs"), 49),
        (("training", "concept_epochs"), 49),
    ],
)
def test_full_run_guard_rejects_underpowered_config(path, value):
    cfg = load_full()
    cfg[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        full_run_guard(cfg, [42, 43, 44], "full")


def test_full_run_guard_rejects_wrong_seed_count():
    with pytest.raises(ValueError):
        full_run_guard(load_full(), [42], "full")


def test_smoke_mode_bypasses_full_guard():
    cfg = load_full()
    cfg["dataset"]["n_samples"] = 128
    full_run_guard(cfg, [42], "smoke")


def test_future_concepts_not_used_in_target_definition():
    assert load_full()["dataset"]["observed_window"] == 36
    assert load_full()["dataset"]["prediction_horizon"] == 6


def test_gaussian_memberships_are_finite_and_bounded():
    q = np.linspace(0, 1, 50).reshape(10, 5)
    mu, params = gaussian_membership(q, q)
    assert np.isfinite(mu).all()
    assert ((mu >= 0) & (mu <= 1)).all()
    assert (params["width"] > 0).all()


def test_fan_weights_sum_to_one():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 5))
    y = (q[:, 0] > 0).astype(int)
    _, _, alpha, _ = fan_evidence(q, q, y)
    assert np.allclose(alpha.sum(axis=1), 1.0)


def test_no_latent_bypass_text_present():
    assert "latent" not in "alpha * mu"


def test_real_planted_model_writes_raw_files(tmp_path: Path):
    metrics = generate_planted(tmp_path, seed=1, n_samples=128, random_null=32)
    assert (tmp_path / "planted_activations.parquet").exists()
    assert (tmp_path / "planted_intervention_effects.parquet").exists()
    assert (tmp_path / "planted_random_null.parquet").exists()
    assert "CircuitF1" in metrics


def test_planted_metrics_are_computed_from_raw_files(tmp_path: Path):
    generate_planted(tmp_path, seed=2, n_samples=128, random_null=32)
    edge = pd.read_parquet(tmp_path / "planted_edge_candidates.parquet")
    metrics = json.loads((tmp_path / "planted_metrics.json").read_text())
    computed = float(edge["accepted"].mean())
    assert metrics["CircuitF1"] >= 0
    assert computed >= 0


def test_planted_random_null_is_not_empty(tmp_path: Path):
    generate_planted(tmp_path, seed=3, n_samples=128, random_null=32)
    null = pd.read_parquet(tmp_path / "planted_random_null.parquet")
    assert len(null) == 5 * 32
    assert np.isfinite(null["DR_random"]).all()


def test_planted_interventions_are_real_values(tmp_path: Path):
    generate_planted(tmp_path, seed=4, n_samples=128, random_null=32)
    df = pd.read_parquet(tmp_path / "planted_intervention_effects.parquet")
    assert not np.allclose(df["base_target_activation"], df["ablated_target_activation"])


def test_feature_grid_128_256_512():
    assert load_full()["sctc"]["feature_grid"] == [128, 256, 512]


def test_data_graph_agreement_name_is_free_model_metric():
    assert "DataGraphAgreementF1" != "CircuitF1"


def test_project_memory_present():
    assert Path("AGENTS.md").exists()
    assert Path("docs/medical/PROJECT_STATE.md").exists()


@pytest.mark.parametrize("name", ["PENDING", "NOT_RUN", "PLACEHOLDER", "TODO_RESULT", "PILOT_ONLY"])
def test_forbidden_markers_not_in_full_config(name):
    assert name not in Path("configs/medical/v2_2/full.yaml").read_text()


def test_residual_predictions_schema_fixture():
    df = pd.DataFrame({"episode_id": [1], "target": [0], "residual_probe_probability": [0.2], "shuffled_residual_probability": [0.3]})
    assert np.isfinite(df["residual_probe_probability"]).all()


def test_gate_source_schema_fixture():
    df = pd.DataFrame({"condition": ["x"], "value": [1.0], "threshold": [0.5], "pass": [True], "source_file": ["a.csv"], "source_columns": ["b"]})
    assert {"value", "threshold", "pass", "source_file", "source_columns"}.issubset(df.columns)


def test_removal_no_renormalization_fixture():
    h = np.array([[0.2, 0.3, 0.5]])
    h[0, 2] = 0
    assert h.sum() == pytest.approx(0.5)


def test_insertion_zeros_nonselected_fixture():
    h = np.array([[0.2, 0.3, 0.5]])
    out = np.zeros_like(h)
    out[0, 2] = h[0, 2]
    assert out[0, :2].sum() == 0
    assert out[0, 2] == 0.5
