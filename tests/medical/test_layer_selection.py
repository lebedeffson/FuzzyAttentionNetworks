import numpy as np

from src.med_circuitbench.sctc.layer_screening import compute_cls


def test_layer_selection_max_three():
    auc = np.array([[0.7], [0.8], [0.9], [0.85]])
    sparsity = np.array([0.1, 0.2, 0.3, 0.4])
    sensitivity = np.array([0.1, 0.3, 0.4, 0.2])
    robustness = np.array([0.4, 0.3, 0.2, 0.1])
    result = compute_cls(auc, sparsity, sensitivity, robustness, max_layers=3)
    assert len(result.selected_report) <= 3
