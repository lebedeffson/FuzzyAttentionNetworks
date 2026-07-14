from pathlib import Path


def test_delivery_package_not_legacy_cosine_dr_archive():
    text = Path("scripts/medical/package_delivery.py").read_text()
    assert "FINAL_BENCHMARK_RESULTS" not in text
    assert "cosine DR" not in text.lower()
    assert "FINAL_PRACTICE" in text
