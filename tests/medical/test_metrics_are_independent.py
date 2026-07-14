import numpy as np

from src.med_circuitbench.metrics.circuit_metrics import cie, completeness, intervention_predictability, off_target_effect


def test_metrics_are_not_copied_from_one_field():
    c = cie(np.array([0.8, 0.2, 0.1]), np.array([0.5, 0.25, 0.2]))["CIE_abs"]
    ip = intervention_predictability(np.array([0.1, 0.4, 0.9]), np.array([0.3, -0.05, -0.1]))["IP_pearson"]
    comp = completeness(0.2, 0.5)
    ote = off_target_effect(np.array([0.0, 1.0]), np.array([0.3, 0.8]), np.array([1.0, 2.0]))
    assert len({round(c, 6), round(ip, 6), round(comp, 6)}) > 1
    assert ote != 0.0
    assert np.isnan(completeness(0.1, 0.0))

