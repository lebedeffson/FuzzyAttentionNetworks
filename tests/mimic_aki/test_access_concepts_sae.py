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
