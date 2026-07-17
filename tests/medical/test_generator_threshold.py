import numpy as np

from src.med_circuitbench.benchmark.generator import BenchmarkConfig, generate_episode, infection_input_at


def test_infection_input_at_v4():
    assert np.array_equal(infection_input_at(3.0), np.array([3.0, 0.0, 0.0, 0.0, 0.0]))


def test_fixed_threshold_positive_rate_fixture_seeds():
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        cfg = BenchmarkConfig(seed=seed, n_samples=1000, target_threshold=0.0715, allow_target_fallback=False)
        targets = [generate_episode(i, cfg, rng)["target"] for i in range(cfg.n_samples)]
        positive_rate = float(np.mean(targets))
        assert 0.20 <= positive_rate <= 0.30
        assert cfg.allow_target_fallback is False
