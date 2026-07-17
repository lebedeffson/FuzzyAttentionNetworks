from __future__ import annotations

import torch
import yaml

from fan.concept.temporal import MultiSetMembershipLayer
from scripts.medical.v3_1.run_q1_neural_final import config_contract


def test_q1_neural_final_uses_canonical_v3_config():
    cfg = yaml.safe_load(open("configs/medical/v3/full.yaml", encoding="utf-8"))
    checks = config_contract(cfg)
    assert all(checks.values()), checks


def test_multiset_membership_supports_required_m_sensitivity_values():
    x = torch.rand(64, 5)
    for n_memberships in [2, 3, 4, 5]:
        layer = MultiSetMembershipLayer(5, n_memberships, "gaussian")
        layer.initialize_from_quantiles(x)
        y = layer(x[:8])
        assert y.shape == (8, 5, n_memberships)
        assert layer.widths.shape == (5, n_memberships)
        assert torch.isfinite(y).all()
