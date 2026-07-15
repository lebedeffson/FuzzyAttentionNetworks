from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fan.sae import TopKSAE, dictionary_health
from mimic_aki.access import verify_mimic_access
from mimic_aki.concepts import compute_window_concepts
from mimic_aki.preprocessing import TrainNormalizer


def test_verify_access_blocks_without_root(monkeypatch):
    monkeypatch.delenv("MIMIC_IV_ROOT", raising=False)
    status = verify_mimic_access()
    assert status.status == "BLOCKED_DATA_ACCESS"


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
