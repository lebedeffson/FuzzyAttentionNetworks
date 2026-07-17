from __future__ import annotations

import numpy as np
import torch

from conceptfan_realdata.posthoc_attribution import gradient_shap, integrated_gradients


def test_attribution_methods_are_deterministic_with_fixed_schedule() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6, 1), torch.nn.Flatten(0))
    inputs = torch.randn(3, 2, 3)
    baseline = torch.zeros(2, 3)
    first, _ = integrated_gradients(model, inputs, baseline, 16, 3)
    second, _ = integrated_gradients(model, inputs, baseline, 16, 3)
    assert torch.equal(first, second)
    background = torch.stack([torch.zeros(2, 3), torch.ones(2, 3)])
    indices = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int16)
    alphas = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]], dtype=np.float32)
    first_shap = gradient_shap(model, inputs, background, indices, alphas, 0.0)
    second_shap = gradient_shap(model, inputs, background, indices, alphas, 0.0)
    assert torch.equal(first_shap, second_shap)

