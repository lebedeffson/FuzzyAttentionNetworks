from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from conceptfan_realdata.data import PreparedData


@pytest.fixture
def config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "physionet2012" / "data.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["model"] = {**value["model"], "latent_dim": 16, "layers": 1, "heads": 4, "ffn": 32, "dropout": 0.0}
    value["training"] = {
        **value["training"],
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 1,
        "batch_size": 4,
    }
    value["stability_regularization"] = {
        **value["stability_regularization"],
        "warmup_epochs": 0,
        "ramp_epochs": 1,
    }
    return value


@pytest.fixture
def prepared() -> PreparedData:
    rng = np.random.default_rng(20260717)
    patients, hours, variables, concepts = 20, 48, 3, 5
    split = np.asarray(["train"] * 10 + ["validation"] * 4 + ["calibration"] * 2 + ["test"] * 4)
    y = np.asarray([0, 1] * 10, dtype=np.int8)
    v = rng.normal(size=(patients, hours, variables)).astype(np.float32)
    m = (rng.random((patients, hours, variables)) > 0.25).astype(np.float32)
    d = rng.random((patients, hours, variables)).astype(np.float32)
    concept_values = rng.random((patients, hours, concepts)).astype(np.float32)
    concept_mask = (rng.random((patients, hours, concepts)) > 0.1).astype(np.float32)
    return PreparedData(
        record_ids=np.arange(100000, 100000 + patients),
        split=split,
        y=y,
        v=v,
        m=m,
        d=d,
        static=rng.normal(size=(patients, 8)).astype(np.float32),
        concepts=concept_values,
        concept_mask=concept_mask,
        variables=["HR", "MAP", "Creatinine"],
        concept_names=[f"concept_{index}" for index in range(concepts)],
        metadata={
            "status": "TEST",
            "input_dims": {"V": 11, "V+M": 14, "V+D": 14, "V+M+D": 17},
            "split_sha256": "split-test",
            "preprocessing_sha256": "preprocess-test",
        },
    )
