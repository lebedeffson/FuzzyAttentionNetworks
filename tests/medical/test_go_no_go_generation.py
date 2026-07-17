from pathlib import Path


def test_go_no_go_logic_contains_required_criteria():
    text = Path("scripts/medical/run_benchmark_pipeline.py").read_text()
    assert "validation_auprc" in text
    for key in ["circuit_f1_passed", "sctc_beats_sae", "cie_passed", "ip_passed", "error_coverage_passed"]:
        assert key in text
    assert "go_no_go.json" in text
