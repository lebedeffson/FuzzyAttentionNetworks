import torch

from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder, sctc_loss


def test_sctc_uses_h_to_predict_a():
    torch.manual_seed(42)
    model = SparseClinicalTranscoder(d_model=4, n_features=8)
    h = torch.randn(2, 3, 4)
    a = h + 2.0
    out = model(h)
    parts = sctc_loss(out["a_hat"], a, out["z"], torch.zeros(2, 3), torch.zeros(2, 3))
    assert out["a_hat"].shape == a.shape
    assert parts.reconstruction.item() > 0
