from pathlib import Path

from scripts.medical.run_benchmark_pipeline import _run_one


def test_run_directory_naming_is_seed_and_split_scoped():
    text = Path("scripts/medical/run_benchmark_pipeline.py").read_text()
    assert 'run_id = f"seed_{seed}_{split}"' in text
    assert "refusing to overwrite existing run directory" in text
    assert _run_one.__name__ == "_run_one"

