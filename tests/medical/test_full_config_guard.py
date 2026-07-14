from scripts.medical.run_benchmark_pipeline import _guard_full_config


def test_full_config_guard_rejects_smoke_like_config():
    cfg = {
        "dataset": {"n_samples": 128},
        "model": {"layers": 2},
        "training": {"maximum_epochs": 3},
        "sctc": {"n_features": 16},
        "interventions": {"random_directions": 5},
    }
    failures = _guard_full_config(cfg)
    assert "n_samples < 10000" in failures
    assert "random_directions < 1000" in failures

