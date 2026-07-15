from __future__ import annotations

import numpy as np
import pandas as pd


CONCEPT_NAMES = [
    "renal_function_trajectory",
    "oliguria_burden",
    "hemodynamic_instability",
    "volume_imbalance",
    "systemic_stress",
]


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_window_concepts(window: pd.DataFrame, thresholds: dict) -> tuple[np.ndarray, np.ndarray]:
    c = np.zeros(5, dtype=np.float32)
    mask = np.ones(5, dtype=np.float32)
    creat = window.get("creatinine", pd.Series(dtype=float)).dropna()
    baseline = float(thresholds.get("baseline_creatinine", 1.0))
    if len(creat):
        ratio = float(creat.iloc[-1]) / max(baseline, 1e-6)
        slope = float(creat.iloc[-1] - creat.iloc[0]) / max(1, len(creat) - 1)
        c[0] = clip01(0.5 * (ratio - 1.0) + 2.0 * max(0.0, slope))
    else:
        mask[0] = 0.0
    urine = window.get("urine_output_rate", pd.Series(dtype=float)).dropna()
    if len(urine):
        c[1] = clip01(float((urine < thresholds.get("oliguria_ml_kg_h", 0.5)).mean()))
    else:
        mask[1] = 0.0
    mapv = window.get("map", pd.Series(dtype=float)).dropna()
    lactate = window.get("lactate", pd.Series(dtype=float)).dropna()
    c[2] = clip01((float((mapv < 65).mean()) if len(mapv) else 0.0) + 0.2 * max(0.0, float(lactate.max()) - 2.0 if len(lactate) else 0.0))
    fluid_in = float(window.get("fluid_input", pd.Series([0.0])).fillna(0).sum())
    urine_out = float(window.get("urine_output", pd.Series([0.0])).fillna(0).sum())
    c[3] = clip01((fluid_in - urine_out) / max(1.0, float(thresholds.get("fluid_balance_scale", 5000.0))))
    hr = window.get("heart_rate", pd.Series(dtype=float)).dropna()
    rr = window.get("respiratory_rate", pd.Series(dtype=float)).dropna()
    temp = window.get("temperature", pd.Series(dtype=float)).dropna()
    c[4] = clip01(0.01 * max(0.0, float(hr.mean()) - 100 if len(hr) else 0.0) + 0.03 * max(0.0, float(rr.mean()) - 20 if len(rr) else 0.0) + 0.2 * max(0.0, abs(float(temp.mean()) - 37.0) if len(temp) else 0.0))
    return c, mask
