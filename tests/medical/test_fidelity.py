import numpy as np

from src.med_circuitbench.metrics.fidelity import probability_fidelity_metrics


def test_fidelity_metrics_are_measured():
    base = np.array([0.1, 0.8, 0.7, 0.2])
    repl = np.array([0.2, 0.7, 0.6, 0.3])
    y = np.array([0, 1, 1, 0])
    metrics = probability_fidelity_metrics(base, repl, y)
    assert metrics["probability_error"] == np.mean(np.abs(repl - base))
    assert set(metrics) == {"delta_auroc", "delta_auprc", "probability_error"}
