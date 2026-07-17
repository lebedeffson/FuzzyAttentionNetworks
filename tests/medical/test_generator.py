import json

import numpy as np

from src.med_circuitbench.benchmark.generator import BenchmarkConfig, generate_episode, true_graph


def test_true_graph_matches_fixture():
    expected = json.load(open("tests/medical/fixtures/true_graph_expected.json"))
    assert true_graph() == expected


def test_generator_is_reproducible_and_shapes_are_valid():
    cfg = BenchmarkConfig(n_samples=2)
    a = generate_episode(0, cfg, np.random.default_rng(42))
    b = generate_episode(0, cfg, np.random.default_rng(42))
    for key in ("states", "treatments", "observations", "model_input"):
        assert np.array_equal(a[key], b[key])
    assert a["states"].shape == (42, 5)
    assert a["treatments"].shape == (42, 3)
    assert a["observations"].shape == (42, 8)
    assert a["model_input"].shape == (36, 27)
    assert a["target"] in {0, 1}
    assert np.isfinite(a["model_input"]).all()
