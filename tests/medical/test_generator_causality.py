import numpy as np

from src.med_circuitbench.benchmark.generator import BenchmarkConfig, generate_episode, initial_state


def test_generator_direct_injection_only_changes_i_at_t0():
    no_impulse = initial_state(0.0, np.zeros(5))
    with_impulse = initial_state(3.0, np.zeros(5))
    delta = with_impulse - no_impulse
    assert delta[0] > 0
    assert np.allclose(delta[1:], 0.0, atol=1e-12)


def test_no_future_leakage_in_model_input():
    cfg = BenchmarkConfig(seed=42, n_samples=1, allow_target_fallback=False)
    episode = generate_episode(0, cfg, np.random.default_rng(42))
    model_input_before = np.asarray(episode["model_input"]).copy()
    states = np.asarray(episode["states"]).copy()
    states[36:, 4] = 1.0 - states[36:, 4]
    target_after = int(states[36:, 4].max() >= cfg.target_threshold)
    assert np.array_equal(model_input_before, np.asarray(episode["model_input"]))
    assert target_after in {0, 1}


def test_full_config_does_not_apply_target_fallback():
    cfg = BenchmarkConfig(seed=42, n_samples=8, allow_target_fallback=False)
    assert cfg.allow_target_fallback is False
