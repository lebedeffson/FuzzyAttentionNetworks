import torch

from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder, sctc_loss


def test_behavior_loss_nonzero_and_gradients():
    torch.manual_seed(42)
    transformer = ClinicalTransformer(TransformerConfig(input_dim=5, d_model=8, d_ffn=16, layers=1, heads=2))
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)
    sctc = SparseClinicalTranscoder(d_model=8, n_features=12)
    x = torch.randn(4, 36, 5)
    base = transformer(x, return_activations=True)
    h = base["h_ffn"][0]
    a = base["a_ffn"][0]
    out = sctc(h)
    replaced = transformer(x, replacements={0: out["a_hat"]})
    parts = sctc_loss(out["a_hat"], a, out["z"], replaced["logit"], base["logit"].detach(), x=x[:, :, :5])
    assert parts.behavior.item() > 0
    parts.total.backward()
    sctc_grad = sum(p.grad.abs().sum().item() for p in sctc.parameters() if p.grad is not None)
    transformer_grad = sum((0.0 if p.grad is None else p.grad.abs().sum().item()) for p in transformer.parameters())
    assert sctc_grad > 0
    assert transformer_grad == 0
