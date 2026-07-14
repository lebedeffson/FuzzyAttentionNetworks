import numpy as np

from src.med_circuitbench.metrics.circuit_metrics import cie, completeness, error_coverage_at3, intervention_predictability, off_target_effect


def test_chain_metrics_fixture_values():
    base = np.array([0.9, 0.7, 0.2, 0.1])
    ablated = np.array([0.6, 0.5, 0.25, 0.2])
    strength = np.array([0.8, 0.6, 0.2, 0.1])
    c = cie(base, ablated)
    ip = intervention_predictability(strength, base - ablated)
    comp = completeness(0.1875, 0.25)
    ote = off_target_effect(np.array([[1.0, 2.0]]), np.array([[1.2, 1.5]]), np.array([[1.0, 2.0]]))
    ec = error_coverage_at3(base, ablated, np.array([0, 0, 1, 1]))
    assert np.isclose(c["CIE_abs"], 0.1625)
    assert ip["IP_pearson"] > 0.8
    assert np.isclose(comp, 0.75)
    assert ote > 0
    assert np.isclose(ec["ErrorCoverageAt3"], 0.75)
