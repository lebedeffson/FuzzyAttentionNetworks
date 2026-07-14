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

from fan.concept import TemporalConceptFANModel
from scripts.medical.v3.run_research_program import FINAL_STATUSES, set_all_seeds, validate_claims, validate_delivery


def test_temporal_fan_output_exposes_latent_sequence():
    model = TemporalConceptFANModel(input_dim=27, sequence_length=36, latent_dim=32, n_concepts=5)
    out = model(torch.randn(2, 36, 27))
    assert out.latent_sequence.shape == (2, 36, 32)
    assert out.as_dict()["latent_sequence"].shape == (2, 36, 32)


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
        "run_iteration.py",
        "analyze_iteration.py",
        "select_next_iteration.py",
        "train_fan.py",
        "evaluate_fan.py",
        "run_leakage_audit.py",
        "run_faithfulness.py",
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
        "results/concept_sufficiency.csv",
        "results/fan_family_comparison.csv",
        "results/fan_results.csv",
        "results/concept_leakage_metrics.csv",
        "results/leakage_bootstrap.parquet",
        "results/faithfulness_results.csv",
        "results/planted_results.csv",
        "results/planted_interventions.parquet",
        "results/sctc_grid_results.csv",
        "results/representation_audit.parquet",
        "results/standard_sctc_results.csv",
        "results/fan_sctc_results.csv",
        "results/explicit_vs_discovered.csv",
        "results/aggregate_metrics.csv",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/supplement.tex",
        "paper/supplement.pdf",
        "paper/references.bib",
        "paper/claims.json",
        "paper/claims_validation.json",
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
