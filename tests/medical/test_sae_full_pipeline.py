from pathlib import Path


def test_sae_pipeline_entrypoints_exist():
    train = Path("scripts/medical/train_sae.py").read_text()
    runner = Path("scripts/medical/run_benchmark_pipeline.py").read_text()
    assert 'input_kind": "a_ffn"' in train
    assert '"train_sae"' in runner
    assert '"build_edges_sae"' in runner

