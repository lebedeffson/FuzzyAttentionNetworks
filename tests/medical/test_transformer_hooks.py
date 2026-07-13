import torch

from src.med_circuitbench.models.hooks import max_prediction_delta_with_hooks
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig


def test_transformer_hooks_do_not_change_predictions():
    torch.manual_seed(42)
    model = ClinicalTransformer(TransformerConfig(input_dim=27, d_model=32, d_ffn=64, layers=2, heads=4))
    model.eval()
    x = torch.randn(4, 36, 27)
    assert max_prediction_delta_with_hooks(model, x) < 1e-6


def test_transformer_returns_ffn_activations():
    model = ClinicalTransformer(TransformerConfig(input_dim=24, d_model=32, d_ffn=64, layers=2, heads=4))
    out = model(torch.randn(3, 36, 24), return_activations=True)
    assert out["h_ffn"][0].shape == (3, 36, 32)
    assert out["a_ffn"][1].shape == (3, 36, 32)
