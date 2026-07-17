from __future__ import annotations

import numpy as np
import pandas as pd

from conceptfan_realdata.data import _forward_fill, compute_delta_time, create_split, parse_hour


def test_parse_time_and_official_right_boundary() -> None:
    assert parse_hour("00:00") == (0, 0)
    assert parse_hour("47:59") == (47, 59)
    assert parse_hour("48:00") == (47, 59)


def test_forward_fill_never_uses_future() -> None:
    values = np.asarray([[np.nan], [2.0], [np.nan], [4.0]], dtype=np.float32)
    result = _forward_fill(values)
    assert np.isnan(result[0, 0])
    assert result[:, 0].tolist()[1:] == [2.0, 2.0, 4.0]


def test_delta_time_resets_only_on_observation() -> None:
    mask = np.asarray([[0], [1], [0], [0], [1]], dtype=np.float32)
    delta = compute_delta_time(mask, cap=4)
    assert delta[:, 0].tolist() == [1.0, 0.0, 0.25, 0.5, 0.0]


def test_deterministic_disjoint_split(tmp_path) -> None:
    patient = pd.DataFrame(
        {
            "record_id": np.arange(4000),
            "icu_type": np.tile([1, 2, 3, 4], 1000),
            "In-hospital_death": np.tile([0] * 7 + [1], 500),
        }
    )
    first = create_split(patient, tmp_path / "first")
    second = create_split(patient, tmp_path / "second")
    assert first.equals(second)
    assert first["RecordID"].nunique() == 4000
    assert first.groupby("RecordID")["split"].nunique().max() == 1
