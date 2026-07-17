from __future__ import annotations

import json

import torch

from conceptfan_realdata.models import MODEL_ARMS
from conceptfan_realdata.training import train_run


def test_validation_selection_never_materializes_calibration_or_test(prepared, config, tmp_path) -> None:
    run_dir = tmp_path / "selection"
    manifest = train_run(
        "ConceptFAN-StabilityReg",
        prepared,
        config,
        run_dir,
        11,
        1001,
        "V+M+D",
        torch.device("cpu"),
        4,
        {"lambda_stab": 0.01, "lambda_logit": 0.0},
        max_epochs_override=1,
        validation_only=True,
    )
    assert manifest["status"] == "VALIDATION_SELECTION_RUN_COMPLETE"
    assert manifest["evaluation_scope"] == "train_validation_only"
    assert not (run_dir / "logits_calibration.parquet").exists()
    assert not (run_dir / "logits_test.parquet").exists()
    assert not (run_dir / "metrics_test.json").exists()


def test_two_epoch_integration_all_models_and_resume(prepared, config, tmp_path) -> None:
    hashes = []
    for position, arm in enumerate(MODEL_ARMS):
        run_dir = tmp_path / arm
        selected = {"lambda_stab": 0.01, "lambda_logit": 0.0} if arm == "ConceptFAN-StabilityReg" else None
        manifest = train_run(
            arm,
            prepared,
            config,
            run_dir,
            101 + position,
            1001,
            "V+M+D",
            torch.device("cpu"),
            4,
            selected,
            max_epochs_override=2,
        )
        assert manifest["status"] == "RUN_COMPLETE"
        assert manifest["test_used_for_selection"] is False
        assert manifest["max_decomposition_error"] < 1e-5
        hashes.append(manifest["parameter_sha256"])
        resumed = train_run(
            arm,
            prepared,
            config,
            run_dir,
            101 + position,
            1001,
            "V+M+D",
            torch.device("cpu"),
            4,
            selected,
            max_epochs_override=2,
        )
        assert resumed == json.loads((run_dir / "run_manifest.json").read_text())
    assert len(set(hashes)) == len(hashes)
