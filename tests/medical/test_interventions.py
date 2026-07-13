import torch

import numpy as np

from src.med_circuitbench.sctc.interventions import amplify_feature, downstream_response, remove_feature


def test_remove_and_amplify_feature():
    a = torch.ones(2, 3)
    z = torch.tensor([0.5, 1.0])
    direction = torch.tensor([2.0, 0.0, 1.0])
    removed = remove_feature(a, z, direction)
    assert torch.allclose(removed[0], torch.tensor([0.0, 1.0, 0.5]))
    assert torch.allclose(removed[1], torch.tensor([-1.0, 1.0, 0.0]))
    amplified = amplify_feature(a, 0.25, direction)
    assert torch.allclose(amplified[0], torch.tensor([1.5, 1.0, 1.25]))


def test_intervention_dr_linear_fixture():
    q_base = np.array([2.0, 4.0])
    q_ablate = np.array([0.0, 2.0])
    q_push = np.array([4.0, 6.0])
    dr = downstream_response(q_base, q_ablate, q_push, sigma=1.0)
    assert np.isclose(dr["DR_ablate"], 2.0)
    assert np.isclose(dr["DR_push"], 2.0)
    assert np.isclose(dr["DR"], 2.0)
    assert dr["inconsistent"] is False
