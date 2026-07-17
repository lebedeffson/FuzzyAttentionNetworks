from __future__ import annotations

import pandas as pd

from .io import MimicSource


def demo_cohort_summary(source: MimicSource) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = source.read_csv("hosp/patients.csv.gz")
    stays = source.read_csv("icu/icustays.csv.gz")
    admissions = source.read_csv("hosp/admissions.csv.gz")
    merged = stays.merge(patients[["subject_id", "anchor_age", "gender", "anchor_year_group"]], on="subject_id", how="left")
    merged = merged.merge(admissions[["subject_id", "hadm_id", "admission_type"]], on=["subject_id", "hadm_id"], how="left")
    merged["adult"] = merged["anchor_age"] >= 18
    merged["icu_hours"] = merged["los"] * 24.0
    merged["eligible_age_los"] = merged["adult"] & (merged["icu_hours"] >= 36.0)
    flow = pd.DataFrame(
        [
            {"step": "patients", "count": int(patients["subject_id"].nunique())},
            {"step": "icu_stays", "count": int(stays["stay_id"].nunique())},
            {"step": "adult_icu_stays", "count": int(merged["adult"].sum())},
            {"step": "adult_icu_stays_los_ge_36h", "count": int(merged["eligible_age_los"].sum())},
        ]
    )
    return merged, flow
