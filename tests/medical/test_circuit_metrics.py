import numpy as np

from src.med_circuitbench.metrics.circuit_metrics import cie, completeness, intervention_predictability, off_target_effect


def test_circuit_metrics_fixture():
    c = cie(np.array([0.8, 0.2]), np.array([0.6, 0.1]))
    assert np.isclose(c["CIE_abs"], 0.15)
    ip = intervention_predictability(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert np.isclose(ip["IP_pearson"], 1.0)
    assert np.isclose(completeness(0.2, 0.4), 0.5)
    ote = off_target_effect(np.array([1.0, 2.0]), np.array([1.1, 1.8]), np.array([1.0, 2.0]))
    assert np.isclose(ote, 0.1)
