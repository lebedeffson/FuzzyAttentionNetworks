from __future__ import annotations

import pandas as pd


def creatinine_kdigo_events(creatinine: pd.DataFrame, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Creatinine-only KDIGO AKI events.

    Required columns: stay_id, charttime, creatinine.
    Optional baseline columns: stay_id, baseline_creatinine.
    """
    df = creatinine.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df = df.sort_values(["stay_id", "charttime"])
    if baseline is not None:
        df = df.merge(baseline, on="stay_id", how="left")
    else:
        df["baseline_creatinine"] = df.groupby("stay_id")["creatinine"].transform("min")
    events = []
    for stay_id, group in df.groupby("stay_id"):
        g = group.reset_index(drop=True)
        for i, row in g.iterrows():
            now = row["charttime"]
            cur = float(row["creatinine"])
            last48 = g[(g["charttime"] >= now - pd.Timedelta(hours=48)) & (g["charttime"] <= now)]["creatinine"]
            base7 = g[(g["charttime"] >= now - pd.Timedelta(days=7)) & (g["charttime"] <= now)]["creatinine"].min()
            baseline_value = float(row.get("baseline_creatinine", base7))
            stage1 = (cur - float(last48.min()) >= 0.3) or (baseline_value > 0 and cur >= 1.5 * baseline_value)
            if stage1:
                events.append({"stay_id": stay_id, "aki_time": now, "aki_stage": 1, "criterion": "creatinine"})
                break
    return pd.DataFrame(events, columns=["stay_id", "aki_time", "aki_stage", "criterion"])
