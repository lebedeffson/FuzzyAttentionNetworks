import json

import torch

from src.med_circuitbench.sctc.transcoder import sctc_loss


def test_sctc_loss_fixture_value():
    expected = json.load(open("tests/medical/fixtures/reference_pipeline_case.json"))
    a = torch.tensor([1.0, 0.0])
    a_hat = torch.tensor([0.8, 0.1])
    z = torch.tensor([0.5, 0.0])
    logit = torch.tensor([0.70])
    logit_hat = torch.tensor([0.65])
    temporal = torch.tensor(0.013646933333333333)
    parts = sctc_loss(a_hat, a, z, logit_hat, logit, lambda_1=0.1, lambda_b=0.2, lambda_t=0.3, temporal_override=temporal)
    assert abs(parts.total.item() - expected["sctc_loss_total"]) < 1e-7
