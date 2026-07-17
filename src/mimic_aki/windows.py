from __future__ import annotations

import pandas as pd


def incident_aki_windows(events: pd.DataFrame, aki_events: pd.DataFrame, observation_hours: int = 24, horizon_hours: int = 12, stride_hours: int = 6) -> pd.DataFrame:
    rows = []
    events = events.copy()
    events["charttime"] = pd.to_datetime(events["charttime"])
    aki_events = aki_events.copy()
    if len(aki_events):
        aki_events["aki_time"] = pd.to_datetime(aki_events["aki_time"])
    for stay_id, group in events.groupby("stay_id"):
        start, end = group["charttime"].min(), group["charttime"].max()
        aki = aki_events[aki_events["stay_id"].eq(stay_id)] if len(aki_events) else pd.DataFrame()
        t = start + pd.Timedelta(hours=observation_hours)
        while t + pd.Timedelta(hours=horizon_hours) <= end:
            preexisting = len(aki[aki["aki_time"] <= t]) > 0 if len(aki) else False
            future = len(aki[(aki["aki_time"] > t) & (aki["aki_time"] <= t + pd.Timedelta(hours=horizon_hours))]) > 0 if len(aki) else False
            if not preexisting:
                rows.append({"stay_id": stay_id, "window_end": t, "label": int(future)})
            t += pd.Timedelta(hours=stride_hours)
    return pd.DataFrame(rows)
