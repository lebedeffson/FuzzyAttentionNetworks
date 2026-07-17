from __future__ import annotations

import numpy as np

from conceptfan_realdata.data import PreparedData
from conceptfan_realdata.posthoc_attribution import build_channel_map, map_matrices, validate_channel_map


def test_channel_map_is_derived_without_outcomes() -> None:
    data = PreparedData(
        record_ids=np.array([1, 2]), split=np.array(["train", "test"]), y=np.array([0, 1]),
        v=np.zeros((2, 3, 3), dtype=np.float32), m=np.zeros((2, 3, 3), dtype=np.float32),
        d=np.zeros((2, 3, 3), dtype=np.float32), static=np.zeros((2, 2), dtype=np.float32),
        concepts=np.zeros((2, 3, 5), dtype=np.float32), concept_mask=np.ones((2, 3, 5), dtype=np.float32),
        variables=["MAP", "PaO2", "ALP"], concept_names=["a", "b", "c", "d", "e"],
        metadata={"input_dims": {"V+M+D": 11}},
    )
    config = {
        "concepts": {"definitions": {
            "hemodynamic_instability": {"low": ["MAP"]},
            "respiratory_dysfunction": {"low": ["PFratio"]},
            "renal_dysfunction": {"low": ["Urine6h"]},
            "neurological_dysfunction": {"low": ["GCS"]},
            "metabolic_inflammatory_dysregulation": {"two_sided": ["pH"]},
        }}
    }
    mapping = build_channel_map(data, config)
    validate_channel_map(mapping, 11)
    primary, sensitivity = map_matrices(mapping)
    assert mapping.loc[mapping.channel_name == "V_MAP", "proxy_concept"].item() == "hemodynamic_instability"
    assert mapping.loc[mapping.channel_name == "M_MAP", "proxy_concept"].item() == "observation_process"
    assert primary[:, mapping.channel_name == "M_MAP"].sum() == 0
    assert sensitivity[:, mapping.channel_name == "M_MAP"].sum() == 1

