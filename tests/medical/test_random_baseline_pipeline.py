from pathlib import Path


def test_random_baseline_uses_ridge_decoder_and_pipeline_stage():
    train = Path("scripts/medical/train_random_directions.py").read_text()
    runner = Path("scripts/medical/run_benchmark_pipeline.py").read_text()
    assert "ridge_decoder" in train
    assert '"train_random_directions"' in runner
    assert '"build_edges_random_directions"' in runner

