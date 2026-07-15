from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from scripts.medical.v3_1.run_q1_oracle_concept_surrogate_ablation import (
    REQUIRED_Q1_ARTIFACTS,
    best_temperature,
    run_grid,
    sigmoid,
    summarize_requirements,
    validate_q1_outputs,
)


def test_q1_run_grid_crosses_initialization_and_data_order_seeds():
    grid = run_grid(30)
    assert len(grid) == 30
    assert len({row["initialization_seed"] for row in grid}) == 10
    assert len({row["data_order_seed"] for row in grid}) == 3
    assert len({(row["initialization_seed"], row["data_order_seed"]) for row in grid}) == 30


def test_q1_requirements_matrix_covers_reviewer_items(tmp_path):
    for rel in set(REQUIRED_Q1_ARTIFACTS.values()):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact\n", encoding="utf-8")
    matrix = summarize_requirements(tmp_path)
    assert matrix["present"].all()
    requirements = set(matrix["requirement"])
    for expected in [
        "30 independent runs",
        "PCBM baseline",
        "CEM baseline",
        "NoFuzzy ablation",
        "temperature scaling",
        "generator/distribution shift",
        "article-ready Q1 figures",
    ]:
        assert expected in requirements


def test_temperature_scaling_search_can_reduce_nll():
    y = np.asarray([0, 0, 0, 1, 1, 1])
    logits = np.asarray([-6.0, -4.0, 2.0, -2.0, 4.0, 6.0])
    overconfident = logits * 4.0
    temp = best_temperature(overconfident, y)
    raw = log_loss(y, sigmoid(overconfident), labels=[0, 1])
    scaled = log_loss(y, sigmoid(overconfident / temp), labels=[0, 1])
    assert temp > 1.0
    assert scaled <= raw


def test_strict_validator_fails_when_required_content_is_missing(tmp_path):
    (tmp_path / "TABLES").mkdir()
    (tmp_path / "FIGURES").mkdir()
    metric_cols = "run_id,initialization_seed,data_order_seed,model_arm,calibrated,n_memberships,AUROC,AUPRC,Brier,ECE,NLL,temperature\n"
    (tmp_path / "TABLES" / "q1_run_metrics.csv").write_text(
        metric_cols + "1,1001,2001,ConceptFAN,True,3,0.5,0.5,0.2,0.1,0.7,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_calibration.csv").write_text(
        "run_id,initialization_seed,data_order_seed,model_arm,calibrated,temperature,Brier,ECE,NLL\n"
        "1,1001,2001,ConceptFAN,True,1.0,0.2,0.1,0.7\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_m_sensitivity.csv").write_text(
        metric_cols + "1,1001,2001,ConceptFAN,True,3,0.5,0.5,0.2,0.1,0.7,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_stability_pairwise.csv").write_text(
        "run_a,run_b,spearman,kendall,jaccard_top2\n1,2,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_seed_variance_decomposition.csv").write_text(
        "quantity,total_variance,initialization_seed_variance,data_order_seed_variance,residual_variance\nAUPRC,0,0,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_controls.csv").write_text(
        "run_id,initialization_seed,data_order_seed,control,AUROC,AUPRC,Brier,ECE,NLL\n"
        "1,1001,2001,random_concepts,0.5,0.5,0.2,0.1,0.7\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_leakage_audit.csv").write_text(
        "concept_error_noise,AUROC,AUPRC,Brier,ECE,NLL\n0.01,0.5,0.5,0.2,0.1,0.7\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_robustness.csv").write_text(
        "run_id,initialization_seed,data_order_seed,model_arm,scenario,level,AUROC,AUPRC,Brier,ECE,NLL\n"
        "1,1001,2001,ConceptFAN,noise,0.0,0.5,0.5,0.2,0.1,0.7\n",
        encoding="utf-8",
    )
    (tmp_path / "TABLES" / "q1_model_summary.csv").write_text(
        "model_arm,calibrated,runs,AUPRC_mean,AUPRC_std,AUROC_mean,Brier_mean,ECE_mean,NLL_mean,temperature_mean\n"
        "ConceptFAN,True,1,0.5,0,0.5,0.2,0.1,0.7,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "FIGURES" / "q1_model_auprc.png").write_bytes(b"x")
    report = validate_q1_outputs(tmp_path)
    assert report["status"] == "Q1_STRICT_VALIDATION_FAIL"
    assert report["failed_checks"]
