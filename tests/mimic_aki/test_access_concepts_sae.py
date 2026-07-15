from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import zipfile
import gzip
import io

from fan.sae import TopKSAE, dictionary_health
from mimic_aki.access import verify_mimic_access
from mimic_aki.concepts import compute_window_concepts
from mimic_aki.demo_baseline import build_demo_window_feature_table, run_demo_logistic_baseline
from mimic_aki.io import MimicSource
from mimic_aki.preprocessing import TrainNormalizer


def test_verify_access_blocks_without_root(monkeypatch):
    monkeypatch.delenv("MIMIC_IV_ROOT", raising=False)
    monkeypatch.delenv("MIMIC_IV_DEMO_ZIP", raising=False)
    monkeypatch.chdir("/tmp")
    status = verify_mimic_access()
    assert status.status == "BLOCKED_DATA_ACCESS"


def test_demo_zip_source_reads_nested_gzip(tmp_path):
    zip_path = tmp_path / "mimic-iv-clinical-database-demo-test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        payload = io.BytesIO()
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz:
            gz.write(b"subject_id,anchor_age\n1,65\n")
        zf.writestr("mimic-iv-clinical-database-demo-test/hosp/patients.csv.gz", payload.getvalue())
    source = MimicSource.open(zip_path)
    df = source.read_csv("hosp/patients.csv.gz")
    assert source.kind == "demo_zip"
    assert df.iloc[0]["anchor_age"] == 65


def test_demo_feature_table_uses_observed_creatinine_only(tmp_path):
    zip_path = tmp_path / "mimic-iv-clinical-database-demo-test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        payload = io.BytesIO()
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz:
            gz.write(b"subject_id,stay_id\n1,10\n")
        zf.writestr("mimic-iv-clinical-database-demo-test/icu/icustays.csv.gz", payload.getvalue())
    source = MimicSource.open(zip_path)
    creat = pd.DataFrame(
        {
            "stay_id": [10, 10, 10],
            "subject_id": [1, 1, 1],
            "hadm_id": [5, 5, 5],
            "charttime": ["2026-01-01 06:00", "2026-01-01 23:00", "2026-01-02 02:00"],
            "creatinine": [1.0, 1.5, 9.9],
        }
    )
    windows = pd.DataFrame({"stay_id": [10], "window_end": [pd.Timestamp("2026-01-02 00:00")], "label": [1]})
    features = build_demo_window_feature_table(source, creat, windows, observation_hours=24)
    assert len(features) == 1
    assert features.iloc[0]["creatinine_last"] == 1.5
    assert features.iloc[0]["creatinine_max"] == 1.5


def test_demo_logistic_baseline_writes_prediction_rows():
    table = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "subject_id": [1, 2, 3, 4, 5, 6],
            "window_end": pd.date_range("2026-01-01", periods=6, freq="h"),
            "label": [0, 1, 0, 1, 0, 1],
            "creatinine_first": [1, 2, 1, 2, 1, 2],
            "creatinine_last": [1, 2.5, 1, 2.5, 1, 2.5],
            "creatinine_mean": [1, 2.2, 1, 2.2, 1, 2.2],
            "creatinine_max": [1, 2.5, 1, 2.5, 1, 2.5],
            "creatinine_min": [1, 2, 1, 2, 1, 2],
            "creatinine_delta": [0, 0.5, 0, 0.5, 0, 0.5],
            "creatinine_slope_per_event": [0, 0.5, 0, 0.5, 0, 0.5],
            "creatinine_count": [1, 2, 1, 2, 1, 2],
            "concept_renal_function_trajectory": [0, 1, 0, 1, 0, 1],
            "concept_oliguria_burden": [0, 0, 0, 0, 0, 0],
            "concept_hemodynamic_instability": [0, 0, 0, 0, 0, 0],
            "concept_volume_imbalance": [0, 0, 0, 0, 0, 0],
            "concept_systemic_stress": [0, 0, 0, 0, 0, 0],
            "concept_mask_renal_function_trajectory": [1, 1, 1, 1, 1, 1],
            "concept_mask_oliguria_burden": [0, 0, 0, 0, 0, 0],
            "concept_mask_hemodynamic_instability": [1, 1, 1, 1, 1, 1],
            "concept_mask_volume_imbalance": [1, 1, 1, 1, 1, 1],
            "concept_mask_systemic_stress": [1, 1, 1, 1, 1, 1],
        }
    )
    pred, metrics, summary = run_demo_logistic_baseline(table, seed=1)
    assert len(pred) == len(table)
    assert set(metrics["split"]) == {"train", "validation", "test", "all"}
    assert summary["status"] == "DEMO_BASELINE_COMPLETE"


def test_concept_mask_for_missing_urine():
    window = pd.DataFrame({"creatinine": [1.0, 1.2], "map": [70, 60]})
    concepts, mask = compute_window_concepts(window, {"baseline_creatinine": 1.0})
    assert concepts.shape == (5,)
    assert mask[1] == 0


def test_train_normalizer_uses_masks():
    x = np.array([[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32)
    mask = np.isfinite(x)
    norm = TrainNormalizer.fit(x, mask)
    z = norm.transform(x, mask)
    assert z[0, 0, 1] == 0.0


def test_topk_sae_health_metrics():
    sae = TopKSAE(input_dim=8, n_features=16, top_k=4)
    x = torch.randn(32, 8)
    out = sae(x)
    health = dictionary_health(out["z"], x, out["reconstructed"], sae.decoder.weight)
    assert health["L0"] <= 4.0
    assert 0.0 <= health["dead_feature_fraction"] <= 1.0
