import numpy as np

from src.med_circuitbench.metrics.circuit_f1 import layer_aware_circuit_f1


def test_layer_aware_circuit_f1_with_duplicate_feature_ids():
    states = np.eye(5, dtype=float)[np.arange(20) % 5]
    node_activations = {
        (0, 0): states[:, 0],
        (1, 0): states[:, 1],
        (2, 0): states[:, 2],
        (3, 0): states[:, 3],
    }
    recovered = [
        {"source_layer": 0, "source_feature": 0, "target_layer": 1, "target_feature": 0, "DR": 1.0},
        {"source_layer": 1, "source_feature": 0, "target_layer": 2, "target_feature": 0, "DR": 1.0},
        {"source_layer": 0, "source_feature": 0, "target_layer": 3, "target_feature": 0, "DR": 1.0},
    ]
    true_graph = {"I": {"R": 0.8}, "R": {"V": 0.7}, "V": {"O": 0.65}, "O": {}, "S": {}}
    result, edges, assignment = layer_aware_circuit_f1(node_activations, states, recovered, true_graph)
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 1
    assert np.isclose(result.circuit_f1, 2 / 3)
    assert set(assignment["layer"]) == {0, 1, 2, 3}
    assert "TP" in set(edges["status"])
