from __future__ import annotations

import torch

from conceptfan_realdata.posthoc_attribution import integrated_gradients


def test_integrated_gradients_completeness_for_linear_logit() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 1), torch.nn.Flatten(0))
    torch.manual_seed(7)
    model[1].reset_parameters()
    inputs = torch.randn(4, 3, 4)
    baseline = torch.zeros(3, 4)
    attributions, errors = integrated_gradients(model, inputs, baseline, n_steps=16, internal_batch_size=2)
    assert attributions.shape == inputs.shape
    assert float(errors.max()) < 1e-6

