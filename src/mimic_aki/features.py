from __future__ import annotations

import pandas as pd

from .io import MimicSource


CREATININE_ITEMIDS = {50912, 52546, 52024}


def demo_creatinine_events(source: MimicSource) -> pd.DataFrame:
    labs = source.read_csv("hosp/labevents.csv.gz", usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    creat = labs[labs["itemid"].isin(CREATININE_ITEMIDS)].copy()
    creat = creat.rename(columns={"valuenum": "creatinine"})
    creat = creat.dropna(subset=["creatinine", "charttime"])
    stays = source.read_csv("icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id"])
    creat = creat.merge(stays, on=["subject_id", "hadm_id"], how="inner")
    return creat[["stay_id", "subject_id", "hadm_id", "charttime", "creatinine"]]
