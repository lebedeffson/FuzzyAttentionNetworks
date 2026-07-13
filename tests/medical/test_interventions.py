import torch

from src.med_circuitbench.sctc.interventions import amplify_feature, remove_feature


def test_remove_and_amplify_feature():
    a = torch.ones(2, 3)
    z = torch.tensor([0.5, 1.0])
    direction = torch.tensor([2.0, 0.0, 1.0])
    removed = remove_feature(a, z, direction)
    assert torch.allclose(removed[0], torch.tensor([0.0, 1.0, 0.5]))
    assert torch.allclose(removed[1], torch.tensor([-1.0, 1.0, 0.0]))
    amplified = amplify_feature(a, 0.25, direction)
    assert torch.allclose(amplified[0], torch.tensor([1.5, 1.0, 1.25]))
