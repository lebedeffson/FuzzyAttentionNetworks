from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from scripts.medical.v3.finalize_real_research import FINAL_STATUSES_REAL
from scripts.medical.v3.validate_final_zip import validate as validate_final_zip
from scripts.medical.v3.validate_result_provenance import validate as validate_provenance


ROOT = Path("artifacts/medical/v3_real_final")


def require(rel: str) -> Path:
    path = ROOT / rel
    assert path.exists(), rel
    return path


def test_finalize_runner_exists():
    assert Path("scripts/medical/v3/finalize_real_research.py").exists()


def test_final_status_is_not_v3_go_if_available():
    if (ROOT / "results/program_status.json").exists():
        status = json.loads(require("results/program_status.json").read_text())
        assert status["final_status"] in FINAL_STATUSES_REAL
        assert status["final_status"] != "V3_GO"


def test_frozen_model_manifest_has_required_models_if_available():
    if (ROOT / "manifests/frozen_models.json").exists():
        rows = json.loads(require("manifests/frozen_models.json").read_text())["checkpoints"]
        models = {row["model"] for row in rows}
        assert {"Standard Transformer", "Predicted FAN-NoAlpha", "Oracle FAN-NoAlpha", "Planted model", "Selected Standard SCTC", "Selected planted SCTC"} <= models
        assert all(row["status"] == "OK" for row in rows)


def test_fan_metrics_recomputed_file_if_available():
    if (ROOT / "results/fan_validation_metrics.csv").exists():
        df = pd.read_csv(require("results/fan_validation_metrics.csv"))
        assert {"AUPRC_from_raw_forward", "direct_macro_R2", "macro_Pearson"} <= set(df.columns)


def test_circuit_f1_uses_node_and_edge_counts_if_available():
    if (ROOT / "results/planted_metrics.csv").exists():
        df = pd.read_csv(require("results/planted_metrics.csv"))
        assert {"Node F1", "Edge F1", "CircuitF1"} <= set(df.columns)
        assert not (df["CircuitF1"] == df["Node F1"]).all()


def test_dead_fraction_from_activity_if_available():
    if (ROOT / "results/planted_sctc_activity.csv").exists():
        df = pd.read_csv(require("results/planted_sctc_activity.csv"))
        assert "dead_feature_fraction" in df.columns


def test_standard_fidelity_from_prediction_pairs_if_available():
    if (ROOT / "results/standard_prediction_pairs.parquet").exists():
        df = pd.read_parquet(require("results/standard_prediction_pairs.parquet"))
        assert {"original_probability", "reconstructed_probability", "target"} <= set(df.columns)


def test_standard_edge_requires_intervention_if_available():
    if (ROOT / "results/standard_candidate_edges.parquet").exists():
        df = pd.read_parquet(require("results/standard_candidate_edges.parquet"))
        assert {"ablation_effect", "matched_random_distribution", "accepted"} <= set(df.columns)


def test_graph_f1_uses_tp_fp_fn_if_available():
    if (ROOT / "results/standard_graph_agreement.csv").exists():
        df = pd.read_csv(require("results/standard_graph_agreement.csv"))
        assert {"TP", "FP", "FN", "DataGraphAgreementF1"} <= set(df.columns)


def test_graph_f1_is_not_correlation_product_source():
    text = Path("scripts/medical/v3/finalize_real_research.py").read_text()
    assert "source_correlation * target_correlation" not in text


def test_representation_marks_unsupported_patching_if_available():
    if (ROOT / "results/representation_audit.parquet").exists():
        df = pd.read_parquet(require("results/representation_audit.parquet"))
        unsupported = df[df["capture_point"] != "mlp_output"]
        assert "UNSUPPORTED_CAPTURE_POINT" in set(unsupported["patching_status"])


def test_partial_test_marked_partial_if_available():
    if (ROOT / "results/partial_test_metrics.csv").exists():
        assert set(pd.read_csv(require("results/partial_test_metrics.csv"))["status"]) == {"PARTIAL_TEST_CONSUMED"}


def test_original_test_not_reopened_if_available():
    if (ROOT / "manifests/original_test_consumed.json").exists():
        assert json.loads(require("manifests/original_test_consumed.json").read_text())["test_reopened"] is False


def test_replication_manifest_if_available():
    if (ROOT / "manifests/replication_dataset_manifest.json").exists():
        manifest = json.loads(require("manifests/replication_dataset_manifest.json").read_text())
        assert manifest["seed"] == 20260715
        assert manifest["episode_count"] == 10000


def test_replication_does_not_select_models_source():
    text = Path("scripts/medical/v3/finalize_real_research.py").read_text()
    assert "models_changed_after_dataset_generation" in text


def test_every_claim_has_provenance_if_available():
    if (ROOT / "paper/claims.json").exists():
        claims = json.loads(require("paper/claims.json").read_text())
        assert claims
        assert all("provenance_id" in claim for claim in claims)


def test_provenance_validator_if_available():
    if (ROOT / "manifests/result_provenance.jsonl").exists():
        assert validate_provenance(ROOT)["passed"]


def test_source_tests_configs_in_delivery_if_available():
    delivery = ROOT / "delivery"
    if delivery.exists() and (ROOT / "manifests/final_zip_validation.json").exists():
        assert (delivery / "SOURCE/src/fan").exists()
        assert (delivery / "SOURCE/scripts/medical/v3").exists()
        assert (delivery / "TESTS/tests/medical/v3").exists()
        assert (delivery / "CONFIGS/configs/medical/v3").exists()


def test_final_zip_hash_sidecar_if_available():
    zips = sorted(Path("artifacts/medical").glob("Med_CircuitBench_V3_REAL_FINAL_*.zip"))
    if zips:
        sidecar = zips[-1].with_suffix(zips[-1].suffix + ".sha256")
        assert sidecar.exists()
        assert validate_final_zip(zips[-1])["passed"]


def test_no_duplicate_upper_lower_result_dirs_if_available():
    if (ROOT / "delivery").exists():
        names = [p.name for p in (ROOT / "delivery").iterdir() if p.is_dir()]
        assert len({name.lower() for name in names}) == len(names)


def test_main_and_supplement_have_expected_page_counts_if_available():
    if (ROOT / "paper/main.pdf").exists() and (ROOT / "paper/supplement.pdf").exists():
        assert require("paper/main.pdf").stat().st_size > 5000
        assert require("paper/supplement.pdf").stat().st_size > 5000
