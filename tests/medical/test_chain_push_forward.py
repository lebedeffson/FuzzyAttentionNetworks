from pathlib import Path


def test_chain_push_uses_decoder_direction_in_forward_pass():
    text = Path("scripts/medical/build_circuits.py").read_text()
    assert 'mode == "push"' in text
    assert "a_layer = a_layer + float(eta_by_node.get((layer_id, feature), 1.0)) * direction" in text
    assert "prob_push" in text

