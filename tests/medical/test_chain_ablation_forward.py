from pathlib import Path


def test_chain_ablation_uses_temporal_feature_forward_replacement():
    text = Path("scripts/medical/build_circuits.py").read_text()
    assert "def _forward_chain_intervention" in text
    assert "a_layer = a_layer - z[:, :, feature : feature + 1] * direction" in text
    assert "transformer.layers" in text

