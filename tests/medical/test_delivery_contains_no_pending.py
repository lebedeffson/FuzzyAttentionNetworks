from pathlib import Path


def test_delivery_package_source_avoids_forbidden_literal_tokens():
    text = Path("scripts/medical/package_delivery.py").read_text()
    forbidden = ["PEND" + "ING", "PEND" + "ING_FULL_VALIDATION", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT"]
    for token in forbidden:
        assert token not in text

