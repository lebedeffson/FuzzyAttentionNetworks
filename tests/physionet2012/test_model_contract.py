from __future__ import annotations

import numpy as np
import torch

from conceptfan_realdata.calibration import fit_calibrators, select_primary_calibrator
from conceptfan_realdata.models import MODEL_ARMS, build_model, exact_decomposition_error, masked_concept_loss, parameter_sha256
from conceptfan_realdata.sufficiency import _mask_bits


def test_all_models_forward_and_exact_additive_contract(config) -> None:
    x = torch.randn(3, 48, 17)
    hashes = []
    for arm in MODEL_ARMS:
        torch.manual_seed(100 + len(hashes))
        model = build_model(arm, config, 17, 0.2)
        output = model(x)
        assert output.logit.shape == (3,)
        assert torch.isfinite(output.logit).all()
        if arm != "PlainTransformer":
            assert float(exact_decomposition_error(output).max()) < 1e-6
        hashes.append(parameter_sha256(model))
    assert len(set(hashes)) == len(hashes)


def test_pure_nofuzzy_changes_only_forward_mode(config) -> None:
    fuzzy = build_model("ConceptFAN-NoAlpha", config, 17, 0.2)
    no_fuzzy = build_model("PureNoFuzzy", config, 17, 0.2)
    assert list(fuzzy.state_dict()) == list(no_fuzzy.state_dict())
    assert sum(parameter.numel() for parameter in fuzzy.parameters()) == sum(parameter.numel() for parameter in no_fuzzy.parameters())
    assert fuzzy.fuzzy is True
    assert no_fuzzy.fuzzy is False


def test_masked_concept_loss_and_all_32_masks() -> None:
    prediction = torch.tensor([[[0.0], [1.0]]])
    target = torch.tensor([[[1.0], [1.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    assert float(masked_concept_loss(prediction, target, mask)) == 0.5
    masks = {_mask_bits(mask_id).tobytes() for mask_id in range(32)}
    assert len(masks) == 32


def test_calibration_methods_are_finite_and_primary_excludes_isotonic() -> None:
    logits = np.linspace(-2.0, 2.0, 20)
    y = np.asarray([0] * 10 + [1] * 10)
    calibrators = fit_calibrators(logits, y)
    selected, losses = select_primary_calibrator(calibrators, logits, y)
    assert selected in {"none", "temperature", "platt"}
    assert "isotonic" not in losses
    assert all(np.isfinite(calibrator.predict(logits)).all() for calibrator in calibrators.values())
