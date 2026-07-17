from pathlib import Path


def test_random_null_distribution_uses_forward_replacement():
    text = Path("scripts/medical/build_circuits.py").read_text()
    assert "def _random_push_responses" in text
    body = text.split("def _random_push_responses", 1)[1].split("def _forward_chain_intervention", 1)[0]
    assert "_forward_with_replacement" in body
    assert "DR_random" in text

