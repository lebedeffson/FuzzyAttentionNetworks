from __future__ import annotations

from pathlib import Path

from scripts.medical.v3_1.verify_canonical_3_seed_parity import SEEDS, validate_config


def test_canonical_parity_config_matches_registered_v3_full_config():
    report = validate_config(Path("configs/medical/v3/full.yaml"))
    assert report["passed"]
    assert report["checks"]["latent_dim_128"]
    assert report["checks"]["layers_4"]
    assert report["checks"]["heads_4"]
    assert report["checks"]["ffn_512"]
    assert report["checks"]["max_epochs_50"]
    assert report["checks"]["seeds_42_43_44"]
    assert SEEDS == [42, 43, 44]
