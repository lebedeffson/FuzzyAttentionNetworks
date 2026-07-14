from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from scripts.medical.v3.run_research_program import FINAL_STATUSES


ROOT = Path("artifacts/medical/v3_real")
ZIP = Path("artifacts/medical/Med_CircuitBench_V3_REAL_RESEARCH_FINAL.zip")


def require(path: str) -> Path:
    p = ROOT / path
    assert p.exists(), path
    return p


def test_program_status_is_terminal_real_status():
    status = json.loads(require("results/program_status.json").read_text())
    assert status["final_status"] in FINAL_STATUSES
    assert status["test_opened"] is True


def test_fan_gate_file_is_validation_only():
    gate = json.loads(require("results/fan_gate.json").read_text())
    assert gate["status"] in {"FAN_VALIDATED", "FAN_VALIDATED_NEGATIVE"}
    assert gate["test_opened"] is False


def test_test_unlock_and_lock_exist():
    assert require("manifests/test_unlock_manifest.json").exists()
    assert require("manifests/test_consumed.lock").exists()


def test_predicted_fan_has_three_seeds_if_fan_outputs_exist():
    df = pd.read_csv(require("results/predicted_vs_oracle_noalpha.csv"))
    assert set(df["seed"]) == {42, 43, 44}


def test_real_representation_audit_capture_points():
    df = pd.read_parquet(require("results/representation_audit.parquet"))
    assert {"residual_pre", "attention_output", "residual_mid", "mlp_output", "residual_post"} <= set(df["capture_point"])
    assert "activation_sample_sha256" in df.columns


def test_standard_sctc_has_real_fidelity_columns():
    df = pd.read_csv(require("results/standard_sctc_results.csv"))
    for col in ["original_AUPRC", "reconstructed_AUPRC", "delta_AUPRC", "probability_MAE", "L0_per_token", "dead_feature_fraction"]:
        assert col in df.columns
    assert "CircuitF1" not in df.columns
    assert "DataGraphAgreementF1" in df.columns


def test_planted_uses_circuit_f1_and_grid():
    assert "CircuitF1" in pd.read_csv(require("results/planted_results.csv")).columns
    assert set(pd.read_csv(require("results/planted_feature_grid.csv"))["n_features"]) == {128, 256, 512}
    assert require("results/planted_interventions.parquet").exists()


def test_fan_sctc_is_real_or_gate_skipped():
    df = pd.read_csv(require("results/fan_sctc_results.csv"))
    if "status" in df.columns and set(df["status"]) == {"SKIPPED_BY_GATE"}:
        assert "reason" in df.columns
    else:
        assert {"n_features", "delta_AUPRC", "probability_MAE", "L0_per_token"} <= set(df.columns)


def test_explicit_vs_discovered_is_real_or_gate_skipped():
    df = pd.read_csv(require("results/explicit_vs_discovered.csv"))
    if "status" in df.columns and set(df["status"]) == {"SKIPPED_BY_GATE"}:
        assert "reason" in df.columns
    else:
        assert {"concept", "best_feature", "activation_Pearson", "contribution_correlation"} <= set(df.columns)


def test_validation_vs_test_contains_only_test_rows():
    assert set(pd.read_csv(require("results/validation_vs_test.csv"))["split"]) == {"test"}


def test_claims_and_delivery_validation_passed():
    assert json.loads(require("paper/claims_validation.json").read_text())["passed"]
    assert json.loads(require("results/delivery_validation.json").read_text())["passed"]


def test_zip_under_500_mib_and_contains_memory():
    assert ZIP.exists()
    assert ZIP.stat().st_size < 524288000
    with zipfile.ZipFile(ZIP) as zf:
        names = set(zf.namelist())
    assert "Med_CircuitBench_V3_REAL_RESEARCH_FINAL/AGENTS.md" in names
    assert "Med_CircuitBench_V3_REAL_RESEARCH_FINAL/PROJECT_MEMORY/PROJECT_STATE.md" in names
    assert "Med_CircuitBench_V3_REAL_RESEARCH_FINAL/paper/main.pdf" in names


def test_no_forbidden_delivery_tokens():
    for rel in ["results/delivery_validation.json", "results/program_status.json", "paper/claims_validation.json"]:
        text = require(rel).read_text()
        for token in ["PENDING", "PLACEHOLDER", "TODO_RESULT", "PILOT_ONLY", "NOT_RUN"]:
            assert token not in text
