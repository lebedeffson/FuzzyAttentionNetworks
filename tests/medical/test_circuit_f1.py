import json

import numpy as np

from src.med_circuitbench.metrics import circuit_f1


def test_circuit_f1_fixture_counts():
    expected = json.load(open("tests/medical/fixtures/reference_pipeline_case.json"))["circuit_f1"]
    rng = np.random.default_rng(42)
    states = rng.normal(size=(128, 3))
    features = np.column_stack([states[:, 0], states[:, 1], states[:, 2], rng.normal(size=128)])
    true_graph = {"I": {"R": 0.8}, "R": {"S": 0.9}, "S": {"O": 0.9}, "O": {}}
    edges = [
        {"source_feature": 0, "target_feature": 1, "DR": 0.5},
        {"source_feature": 1, "target_feature": 2, "DR": 0.5},
        {"source_feature": 0, "target_feature": 3, "DR": 0.5},
    ]
    result = circuit_f1(features, states, edges, true_graph, state_names=("I", "R", "S"), assignment_threshold=0.5)
    assert result.tp == expected["TP"]
    assert result.fp == expected["FP"]
    assert result.fn == expected["FN"]
    assert abs(result.circuit_f1 - expected["CircuitF1"]) < 1e-6
