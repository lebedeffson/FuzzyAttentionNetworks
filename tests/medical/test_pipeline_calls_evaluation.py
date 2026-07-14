from pathlib import Path


def test_pipeline_calls_benchmark_evaluation_stage():
    text = Path("scripts/medical/run_benchmark_pipeline.py").read_text()
    assert '"evaluate_benchmark"' in text
    assert "evaluation_summary.json" in text

