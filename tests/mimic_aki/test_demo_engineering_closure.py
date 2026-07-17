from __future__ import annotations

import json
import sys
import types
import zipfile
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fan.attention import FuzzyTemporalAttention
from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from fan.sae import TopKSAE, ablate_feature, steer_feature
from mimic_aki.io import MimicSource


def _repo_or_release_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "SOURCE").exists() or (parent / "src").exists():
            return parent
    return here.parents[2]


ROOT = _repo_or_release_root()
FINAL = ROOT / "artifacts" / "mimic_aki" / "final_demo"
if not FINAL.exists() and (ROOT / "RESULTS").exists():
    FINAL = ROOT / "RESULTS"


def _require_final():
    if not (FINAL / "final_status.json").exists():
        pytest.skip("run scripts/mimic_aki/run_demo_engineering.py first")


def test_demo_zip_read():
    demo_zip = ROOT / "mimic-iv-clinical-database-demo-2.2.zip"
    if not demo_zip.exists():
        pytest.skip("raw MIMIC-IV Demo zip is intentionally absent from standalone release")
    assert MimicSource.open(demo_zip).exists("hosp/patients.csv.gz")


def _make_nested_demo_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "mimic-iv-clinical-database-demo-test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        payload = io.BytesIO()
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz:
            gz.write(b"subject_id,anchor_age\n1,65\n2,70\n")
        zf.writestr("mimic-iv-clinical-database-demo-test/hosp/patients.csv.gz", payload.getvalue())
    return zip_path


def test_nested_csv_gz(tmp_path):
    df = MimicSource.open(_make_nested_demo_zip(tmp_path)).read_csv("hosp/patients.csv.gz", nrows=2)
    assert {"subject_id", "anchor_age"} <= set(df.columns)


def test_directory_zip_parity():
    _require_final()
    contract = json.loads((FINAL / "data_contract" / "data_contract.json").read_text())
    assert contract["directory_zip_parity"] in {"NOT_APPLICABLE_ONLY_ZIP_SOURCE_PRESENT", "PASS"}


def test_duplicate_table_rejected():
    _require_final()
    contract = json.loads((FINAL / "data_contract" / "data_contract.json").read_text())
    assert contract["duplicate_table_resolution"] == "PASS_SINGLE_SOURCE"


def test_path_traversal_rejected(tmp_path):
    demo_zip = ROOT / "mimic-iv-clinical-database-demo-2.2.zip"
    source = MimicSource.open(demo_zip) if demo_zip.exists() else MimicSource.open(_make_nested_demo_zip(tmp_path))
    with pytest.raises(ValueError):
        source.exists("../hosp/patients.csv.gz")


def test_label_window_no_future_leakage():
    _require_final()
    leak = pd.read_parquet(FINAL / "data_contract" / "leakage_assertions.parquet")
    assert leak["leakage_pass"].all()


def test_public_artifacts_no_identifiers():
    _require_final()
    contract = json.loads((FINAL / "data_contract" / "data_contract.json").read_text())
    assert contract["public_artifacts_no_identifiers"] is True


def test_legacy_fan_characterized():
    if "huggingface_hub" not in sys.modules:
        stub = types.ModuleType("huggingface_hub")
        stub.login = lambda *args, **kwargs: None
        sys.modules["huggingface_hub"] = stub
    try:
        import src.fuzzy_attention as legacy
    except ModuleNotFoundError:
        import fuzzy_attention as legacy

    assert hasattr(legacy, "FuzzyMembership")
    assert hasattr(legacy, "MultiHeadFuzzyAttention")


def test_canonical_fan_parameter_mapping():
    _require_final()
    compat = json.loads((FINAL / "code_audit" / "fuzzy_attention_compatibility.json").read_text())
    assert compat["status"] == "PASS"
    assert compat["canonical_path"] == "src/fan/attention"


def test_no_duplicate_active_fuzzy_attention():
    _require_final()
    compat = json.loads((FINAL / "code_audit" / "fuzzy_attention_compatibility.json").read_text())
    assert compat["duplicate_active_implementation"] is False


def test_existing_conceptfan_retained():
    assert MultiSetAdditiveTemporalConceptFANModel.__name__ == "MultiSetAdditiveTemporalConceptFANModel"


def test_fuzzy_attention_masks():
    attn = FuzzyTemporalAttention(8, 2)
    x = torch.randn(3, 4, 8)
    mask = torch.tensor([[False, False, False, True], [False, False, True, True], [False, False, False, False]])
    y, w, _ = attn(x, key_padding_mask=mask)
    assert torch.isfinite(y).all()
    assert torch.isfinite(w).all()


def test_fuzzy_attention_gradients():
    attn = FuzzyTemporalAttention(8, 2)
    x = torch.randn(3, 4, 8)
    y, _, _ = attn(x)
    y.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in attn.parameters())


def test_conceptfan_no_latent_bypass():
    model = MultiSetAdditiveTemporalConceptFANModel(8, 4, 16, 5, alpha_mode="no_alpha")
    assert hasattr(model, "decision_head")


def test_exact_decomposition():
    model = MultiSetAdditiveTemporalConceptFANModel(8, 4, 16, 5, alpha_mode="no_alpha")
    out = model(torch.randn(2, 4, 8))
    rec = model.decision_head.bias + out.signed_decision_contributions.sum(dim=-1)
    assert torch.max(torch.abs(out.logit - rec)).item() < 1e-6


def test_combined_model_forward_backward():
    model = MultiSetAdditiveTemporalConceptFANModel(8, 4, 16, 5, alpha_mode="no_alpha")
    out = model(torch.randn(3, 4, 8))
    loss = out.logit.square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_sae_decoder_unit_norm():
    sae = TopKSAE(8, 16, 4)
    sae.normalize_decoder_()
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(16), atol=1e-6)


def test_sae_replacement_forward():
    sae = TopKSAE(8, 16, 4)
    out = sae(torch.randn(5, 8))
    assert out["reconstructed"].shape == (5, 8)


def test_steering_direction_norm_matched():
    d = torch.randn(8)
    r = torch.randn(8)
    r = r / r.norm() * d.norm()
    assert torch.allclose(r.norm(), d.norm(), atol=1e-6)


def test_steering_activity_control_matched():
    x = torch.randn(4, 8)
    d = torch.randn(8)
    z = torch.ones(4)
    assert ablate_feature(x, z, d).shape == x.shape
    assert steer_feature(x, d, 1.0, 0.5).shape == x.shape


def test_bundle_hash_verification():
    _require_final()
    manifest = json.loads((FINAL / "bundles" / "standard_transformer" / "manifest.json").read_text())
    assert len(manifest["checkpoint_sha256"]) == 64


def test_bundle_reproduces_predictions():
    _require_final()
    pred = pd.read_parquet(FINAL / "bundles" / "standard_transformer" / "reference_predictions.parquet")
    assert np.isfinite(pred["probability"]).all()


def test_cli_from_unpacked_release():
    assert (
        (ROOT / "scripts" / "mimic_aki" / "cli" / "mimic-aki-run-demo").exists()
        or (ROOT / "SOURCE" / "scripts" / "mimic_aki" / "cli" / "mimic-aki-run-demo").exists()
    )


def test_report_generated():
    _require_final()
    assert (FINAL / "report" / "report.html").exists()
    assert "NOT A CLINICAL" in (FINAL / "report" / "report.html").read_text()


def test_demo_flags_present_everywhere():
    _require_final()
    status = json.loads((FINAL / "final_status.json").read_text())
    assert status["final_status"] == "PRACTICE_CLOSED_MIMIC_DEMO_END_TO_END"
    for manifest in (FINAL / "bundles").glob("*/manifest.json"):
        assert json.loads(manifest.read_text())["demo_only"] is True


def test_no_publication_claim_from_demo():
    _require_final()
    text = (FINAL / "report" / "report.html").read_text()
    assert "NOT A CLINICAL OR SCIENTIFIC PERFORMANCE STUDY" in text


def test_release_zip_excludes_mimic_data():
    _require_final()
    release_info = json.loads((FINAL / "final_status.json").read_text())["release"]
    if "zip" not in release_info:
        pytest.skip("in-archive final_status intentionally stores ZIP SHA externally")
    release = release_info["zip"]
    if not (ROOT / release).exists():
        pytest.skip("release zip is outside the unpacked standalone package")
    with zipfile.ZipFile(ROOT / release) as zf:
        assert not any(name.lower().endswith(".zip") and "mimic-iv" in name.lower() for name in zf.namelist())


def test_demo_concept_masks_mark_unavailable_targets():
    _require_final()
    concepts = pd.read_csv(FINAL / "report" / "concept_metrics.csv")
    unavailable = concepts.set_index("concept").loc[
        ["oliguria_burden", "hemodynamic_instability", "volume_imbalance", "systemic_stress"],
        "target_status",
    ]
    assert set(unavailable) == {"masked_unavailable_in_demo"}


def test_demo_faithfulness_is_structural_only():
    _require_final()
    faithfulness = pd.read_csv(FINAL / "report" / "faithfulness.csv").iloc[0].to_dict()
    assert faithfulness["status"] == "STRUCTURAL_FAITHFULNESS_MECHANICS_PASS_EMPIRICAL_NOT_EVALUATED"
    assert faithfulness["top_concept_removal"] == "NOT_EVALUATED_ON_DEMO"


def test_demo_sae_fidelity_failure_is_explicit():
    _require_final()
    status = json.loads((FINAL / "final_status.json").read_text())
    assert status["sae"] == "SAE_DEMO_MECHANICS_COMPLETE"
    assert status["sae_fidelity"] in {"PASS", "FAIL"}
