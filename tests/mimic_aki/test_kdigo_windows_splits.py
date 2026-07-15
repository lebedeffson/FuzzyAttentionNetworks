from __future__ import annotations

import pandas as pd

from mimic_aki.kdigo import creatinine_kdigo_events
from mimic_aki.splits import split_subjects
from mimic_aki.windows import incident_aki_windows


def test_creatinine_03_in_48h_positive():
    creat = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": ["2026-01-01 00:00", "2026-01-02 00:00"],
            "creatinine": [1.0, 1.31],
        }
    )
    events = creatinine_kdigo_events(creat)
    assert len(events) == 1
    assert events.iloc[0]["criterion"] == "creatinine"


def test_creatinine_15x_in_7d_positive():
    creat = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": ["2026-01-01 00:00", "2026-01-04 00:00"],
            "creatinine": [1.0, 1.5],
        }
    )
    events = creatinine_kdigo_events(creat)
    assert len(events) == 1


def test_patient_split_isolation():
    split = split_subjects([1, 2, 3, 4, 5, 6], seed=1)
    assert split.groupby("subject_id")["split"].nunique().max() == 1
    assert set(split["split"]).issubset({"train", "validation", "test"})


def test_label_horizon_boundary_and_preexisting_excluded():
    times = pd.date_range("2026-01-01", periods=49, freq="h")
    events = pd.DataFrame({"stay_id": 1, "charttime": times})
    aki = pd.DataFrame({"stay_id": [1], "aki_time": [pd.Timestamp("2026-01-02 12:00")]})
    windows = incident_aki_windows(events, aki, observation_hours=24, horizon_hours=12, stride_hours=6)
    assert windows[windows["window_end"].eq(pd.Timestamp("2026-01-02 00:00"))]["label"].iloc[0] == 1
    assert not (windows["window_end"] >= pd.Timestamp("2026-01-02 12:00")).any()
