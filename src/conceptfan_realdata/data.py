from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


STATIC_PARAMETERS = {"Age", "Gender", "Height", "ICUType"}
FORBIDDEN_INPUTS = {"RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"}
KNOWN_PARAMETERS = {
    "RecordID", "Age", "Gender", "Height", "ICUType", "Weight", "Albumin", "ALP", "ALT", "AST",
    "Bilirubin", "BUN", "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3",
    "HCT", "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
    "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_hour(value: str) -> tuple[int, int]:
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid PhysioNet time {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if hour == 48 and minute == 0:
        # The official Set A contains an exact right-boundary timestamp. Keep it
        # in the final hour of the 48-bin representation and audit it separately.
        return 47, 59
    if not 0 <= hour <= 47 or not 0 <= minute <= 59:
        raise ValueError(f"PhysioNet time outside 00:00-47:59: {value!r}")
    return hour, minute


def _read_record(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    expected = ["Time", "Parameter", "Value"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected patient record schema: {list(frame.columns)}")
    frame["Parameter"] = frame["Parameter"].astype(str).str.strip()
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame["boundary_48_00"] = frame["Time"].astype(str).str.strip().eq("48:00")
    parsed = frame["Time"].map(parse_hour)
    frame["hour"] = parsed.map(lambda pair: pair[0]).astype(np.int16)
    frame["minute"] = parsed.map(lambda pair: pair[1]).astype(np.int16)
    return frame


def ensure_outcomes(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ConceptFAN-audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except Exception as exc:
        command = f"curl -fL {url} -o {path}"
        raise RuntimeError(f"Outcomes file is required and automatic download failed: {exc}. Run: {command}") from exc
    if not payload:
        raise RuntimeError(f"Downloaded empty outcomes file from {url}")
    path.write_bytes(payload)
    return path


def load_outcomes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"RecordID", "In-hospital_death"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Outcomes schema lacks {sorted(required - set(frame.columns))}")
    frame["RecordID"] = pd.to_numeric(frame["RecordID"], errors="raise").astype(np.int64)
    frame["In-hospital_death"] = pd.to_numeric(frame["In-hospital_death"], errors="raise").astype(np.int8)
    if frame["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID values in Outcomes-a.txt")
    if set(frame["In-hospital_death"].unique()) - {0, 1}:
        raise ValueError("In-hospital_death must contain only 0/1")
    return frame


def audit_raw(set_zip: Path, outcomes_path: Path, output_dir: Path, expected_patients: int = 4000) -> dict:
    if not set_zip.exists():
        raise FileNotFoundError(set_zip)
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = load_outcomes(outcomes_path)
    patient_rows: list[dict] = []
    variable_rows: list[dict] = []
    record_ids: set[int] = set()
    unknown_parameters: set[str] = set()
    boundary_48_00_rows = 0
    with zipfile.ZipFile(set_zip) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".txt"))
        if len(members) != expected_patients:
            raise ValueError(f"Expected {expected_patients} patient files, found {len(members)}")
        for name in members:
            payload = archive.read(name)
            frame = _read_record(payload)
            boundary_48_00_rows += int(frame["boundary_48_00"].sum())
            filename_id = int(Path(name).stem)
            record_values = frame.loc[frame["Parameter"].eq("RecordID"), "Value"].dropna().astype(np.int64).unique()
            if len(record_values) != 1:
                raise ValueError(f"{name}: expected exactly one RecordID value")
            record_id = int(record_values[0])
            if record_id != filename_id:
                raise ValueError(f"{name}: filename/content RecordID mismatch ({filename_id} != {record_id})")
            if record_id in record_ids:
                raise ValueError(f"Duplicate RecordID in set-a.zip: {record_id}")
            record_ids.add(record_id)
            icu = frame.loc[frame["Parameter"].eq("ICUType"), "Value"].dropna()
            patient_rows.append(
                {
                    "record_id": record_id,
                    "member": name,
                    "rows": int(len(frame)),
                    "max_hour": int(frame["hour"].max()),
                    "icu_type": int(icu.iloc[0]) if len(icu) and float(icu.iloc[0]) != -1 else -1,
                    "record_sha256": sha256_bytes(payload),
                }
            )
            counts = frame.groupby("Parameter", sort=False).agg(
                rows=("Value", "size"),
                sentinel_minus_one=("Value", lambda values: int(np.sum(values == -1))),
                non_missing=("Value", lambda values: int(np.sum(values.notna() & (values != -1)))),
            )
            for parameter, row in counts.iterrows():
                variable_rows.append(
                    {
                        "record_id": record_id,
                        "parameter": parameter,
                        "rows": int(row["rows"]),
                        "sentinel_minus_one": int(row["sentinel_minus_one"]),
                        "non_missing": int(row["non_missing"]),
                    }
                )
                if parameter not in KNOWN_PARAMETERS:
                    unknown_parameters.add(parameter)
    outcome_ids = set(outcomes["RecordID"].astype(int))
    if len(outcomes) != expected_patients:
        raise ValueError(f"Expected {expected_patients} outcomes, found {len(outcomes)}")
    if record_ids != outcome_ids:
        missing_labels = sorted(record_ids - outcome_ids)
        missing_records = sorted(outcome_ids - record_ids)
        raise ValueError(f"RecordID mismatch: missing_labels={missing_labels[:10]}, missing_records={missing_records[:10]}")
    patient_manifest = pd.DataFrame(patient_rows).merge(
        outcomes[["RecordID", "In-hospital_death"]], left_on="record_id", right_on="RecordID", validate="one_to_one"
    ).drop(columns="RecordID")
    patient_manifest.to_parquet(output_dir / "patient_manifest.parquet", index=False, compression="zstd")
    variables = pd.DataFrame(variable_rows)
    variable_summary = variables.groupby("parameter", as_index=False).agg(
        patient_count=("record_id", "nunique"),
        rows=("rows", "sum"),
        non_missing=("non_missing", "sum"),
        sentinel_minus_one=("sentinel_minus_one", "sum"),
    )
    variable_summary["patient_coverage"] = variable_summary["patient_count"] / expected_patients
    variable_summary.to_csv(output_dir / "variable_summary.csv", index=False)
    missing = variable_summary[["parameter", "rows", "non_missing", "sentinel_minus_one", "patient_coverage"]].copy()
    missing["missing_or_sentinel_rate"] = 1.0 - missing["non_missing"] / missing["rows"].clip(lower=1)
    missing.to_csv(output_dir / "missingness_summary.csv", index=False)
    hashes = {
        "set_a_zip": sha256_file(set_zip),
        "outcomes": sha256_file(outcomes_path),
    }
    (output_dir / "sha256.txt").write_text(
        f"{hashes['set_a_zip']}  {set_zip.name}\n{hashes['outcomes']}  {outcomes_path.name}\n", encoding="utf-8"
    )
    report = {
        "status": "PHYSIONET2012_RAW_AUDIT_PASS",
        "created_utc": utc_now(),
        "patients": expected_patients,
        "unique_record_ids": len(record_ids),
        "outcome_rows": len(outcomes),
        "outcome_prevalence": float(outcomes["In-hospital_death"].mean()),
        "unknown_parameters": sorted(unknown_parameters),
        "boundary_48_00_rows_mapped_to_hour_47": boundary_48_00_rows,
        "hashes": hashes,
        "checks": {
            "zip_integrity": True,
            "exact_patient_count": True,
            "record_id_alignment": True,
            "unique_record_ids": True,
            "complete_binary_labels": True,
            "time_range_normalized_to_48_hour_bins": True,
            "minus_one_recorded_as_missing": True,
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def create_split(patient_manifest: pd.DataFrame, output_dir: Path, seed: int = 20260717) -> pd.DataFrame:
    frame = patient_manifest.rename(columns={"record_id": "RecordID", "icu_type": "ICUType"}).copy()
    frame["stratum"] = frame["In-hospital_death"].astype(str) + "_" + frame["ICUType"].astype(str)
    ids = frame.index.to_numpy()
    train_val_cal, test = train_test_split(ids, test_size=600, random_state=seed, stratify=frame.loc[ids, "stratum"])
    train_val, calibration = train_test_split(
        train_val_cal,
        test_size=400,
        random_state=seed + 1,
        stratify=frame.loc[train_val_cal, "stratum"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=600,
        random_state=seed + 2,
        stratify=frame.loc[train_val, "stratum"],
    )
    split_by_index = {int(index): split for split, values in {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test": test,
    }.items() for index in values}
    frame["split"] = [split_by_index[int(index)] for index in frame.index]
    expected = {"train": 2400, "validation": 600, "calibration": 400, "test": 600}
    counts = frame["split"].value_counts().to_dict()
    if counts != expected:
        raise AssertionError(f"Split size mismatch: {counts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in expected:
        ids_for_split = sorted(frame.loc[frame["split"].eq(split), "RecordID"].astype(int))
        (output_dir / f"{split}_ids.txt").write_text("\n".join(map(str, ids_for_split)) + "\n", encoding="utf-8")
        if split == "validation":
            (output_dir / "val_ids.txt").write_text("\n".join(map(str, ids_for_split)) + "\n", encoding="utf-8")
    summary = frame.groupby("split", as_index=False).agg(
        patients=("RecordID", "nunique"),
        deaths=("In-hospital_death", "sum"),
        mortality_rate=("In-hospital_death", "mean"),
    )
    for icu_type in sorted(frame["ICUType"].unique()):
        shares = frame.assign(flag=frame["ICUType"].eq(icu_type).astype(float)).groupby("split")["flag"].mean()
        summary[f"icu_type_{icu_type}_share"] = summary["split"].map(shares)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    split_payload = {
        "status": "PHYSIONET2012_SPLIT_FROZEN",
        "seed": seed,
        "created_utc": utc_now(),
        "counts": expected,
        "stratification": ["In-hospital_death", "ICUType"],
        "record_ids": {
            split: sorted(frame.loc[frame["split"].eq(split), "RecordID"].astype(int).tolist()) for split in expected
        },
    }
    canonical = json.dumps(split_payload, indent=2, sort_keys=True)
    split_path = output_dir / f"split_seed_{seed}.json"
    split_path.write_text(canonical, encoding="utf-8")
    (output_dir / f"split_seed_{seed}.sha256").write_text(f"{sha256_file(split_path)}  {split_path.name}\n", encoding="utf-8")
    return frame[["RecordID", "ICUType", "In-hospital_death", "split"]].sort_values("RecordID").reset_index(drop=True)


def _aggregate_patient(frame: pd.DataFrame, variables: list[str], hours: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = np.full((hours, len(variables)), np.nan, dtype=np.float32)
    mask = np.zeros((hours, len(variables)), dtype=np.float32)
    static: dict[str, float] = {}
    for parameter in STATIC_PARAMETERS:
        rows = frame.loc[frame["Parameter"].eq(parameter) & frame["Value"].notna() & frame["Value"].ne(-1), "Value"]
        static[parameter] = float(rows.iloc[0]) if len(rows) else float("nan")
    weight_rows = frame.loc[frame["Parameter"].eq("Weight") & frame["Value"].notna() & frame["Value"].ne(-1)].sort_values(["hour", "minute"])
    static["Weight_at_admission"] = float(weight_rows.iloc[0]["Value"]) if len(weight_rows) else float("nan")
    index = {name: position for position, name in enumerate(variables)}
    temporal = frame.loc[~frame["Parameter"].isin(STATIC_PARAMETERS | {"RecordID"})].copy()
    temporal.loc[temporal["Value"].eq(-1), "Value"] = np.nan
    temporal = temporal.loc[temporal["Value"].notna() & temporal["Parameter"].isin(index)]
    grouped = temporal.groupby(["hour", "Parameter"], sort=False)["Value"]
    for (hour, parameter), series in grouped:
        aggregate = float(series.max()) if parameter == "MechVent" else float(series.median())
        values[int(hour), index[parameter]] = aggregate
        mask[int(hour), index[parameter]] = 1.0
    return values, mask, static


def _forward_fill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = np.full(values.shape[1], np.nan, dtype=np.float32)
    for hour in range(values.shape[0]):
        observed = np.isfinite(out[hour])
        last[observed] = out[hour, observed]
        missing = ~observed & np.isfinite(last)
        out[hour, missing] = last[missing]
    return out


def compute_delta_time(mask: np.ndarray, cap: int = 48) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    last_seen = np.full(mask.shape[1], -1, dtype=np.int16)
    for hour in range(mask.shape[0]):
        observed = mask[hour] > 0
        last_seen[observed] = hour
        elapsed = np.where(last_seen >= 0, hour - last_seen, cap)
        out[hour] = np.minimum(elapsed, cap) / float(cap)
    return out


def _robust_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(values, axis=(0, 1)),
        "iqr": np.nanquantile(values, 0.75, axis=(0, 1)) - np.nanquantile(values, 0.25, axis=(0, 1)),
        "q005": np.nanquantile(values, 0.005, axis=(0, 1)),
        "q995": np.nanquantile(values, 0.995, axis=(0, 1)),
    }


def _score_component(values: np.ndarray, available: np.ndarray, direction: str, quantiles: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(values, dtype=np.float32)
    if direction in {"low", "two_sided"}:
        denominator = max(quantiles["q25"] - quantiles["q05"], 1e-6)
        low = np.clip((quantiles["q25"] - values) / denominator, 0.0, 1.0)
        score = np.maximum(score, low.astype(np.float32))
    if direction in {"high", "two_sided"}:
        denominator = max(quantiles["q95"] - quantiles["q75"], 1e-6)
        high = np.clip((values - quantiles["q75"]) / denominator, 0.0, 1.0)
        score = np.maximum(score, high.astype(np.float32))
    score[~available] = 0.0
    return score


def _concept_components(raw_values: np.ndarray, raw_mask: np.ndarray, variables: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index = {name: position for position, name in enumerate(variables)}
    components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in variables:
        position = index[name]
        values = np.stack([_forward_fill(patient)[:, position] for patient in raw_values], axis=0)
        available = np.cumsum(raw_mask[:, :, position], axis=1) > 0
        components[name] = (values, available)
    if "PaO2" in components and "FiO2" in components:
        pao2, pao2_available = components["PaO2"]
        fio2, fio2_available = components["FiO2"]
        fio2_fraction = np.where(fio2 > 1.5, fio2 / 100.0, fio2)
        available = pao2_available & fio2_available & np.isfinite(fio2_fraction) & (fio2_fraction > 1e-6)
        ratio = np.divide(pao2, np.maximum(fio2_fraction, 1e-6), out=np.zeros_like(pao2), where=available)
        components["PFratio"] = (ratio, available)
    if "Urine" in components:
        urine = raw_values[:, :, index["Urine"]]
        observed = raw_mask[:, :, index["Urine"]] > 0
        rolling = np.zeros_like(urine, dtype=np.float32)
        available = np.zeros_like(observed)
        for hour in range(urine.shape[1]):
            start = max(0, hour - 5)
            window = urine[:, start : hour + 1]
            window_observed = observed[:, start : hour + 1]
            rolling[:, hour] = np.nansum(np.where(window_observed, window, np.nan), axis=1)
            available[:, hour] = window_observed.any(axis=1)
        components["Urine6h"] = (rolling, available)
    return components


@dataclass
class PreparedData:
    record_ids: np.ndarray
    split: np.ndarray
    y: np.ndarray
    v: np.ndarray
    m: np.ndarray
    d: np.ndarray
    static: np.ndarray
    concepts: np.ndarray
    concept_mask: np.ndarray
    variables: list[str]
    concept_names: list[str]
    metadata: dict

    def indices(self, split_name: str) -> np.ndarray:
        return np.flatnonzero(self.split == split_name)

    def inputs(self, channels: str = "V+M+D") -> np.ndarray:
        pieces = [self.v]
        if "M" in channels:
            pieces.append(self.m)
        if "D" in channels:
            pieces.append(self.d)
        static = np.repeat(self.static[:, None, :], self.v.shape[1], axis=1)
        pieces.append(static)
        return np.concatenate(pieces, axis=-1).astype(np.float32)


def save_prepared(data: PreparedData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        record_ids=data.record_ids,
        split=data.split,
        y=data.y,
        v=data.v,
        m=data.m,
        d=data.d,
        static=data.static,
        concepts=data.concepts,
        concept_mask=data.concept_mask,
        variables=np.asarray(data.variables),
        concept_names=np.asarray(data.concept_names),
        metadata=np.asarray(json.dumps(data.metadata, sort_keys=True)),
    )


def load_prepared(path: Path) -> PreparedData:
    with np.load(path, allow_pickle=False) as payload:
        return PreparedData(
            record_ids=payload["record_ids"],
            split=payload["split"].astype(str),
            y=payload["y"],
            v=payload["v"],
            m=payload["m"],
            d=payload["d"],
            static=payload["static"],
            concepts=payload["concepts"],
            concept_mask=payload["concept_mask"],
            variables=payload["variables"].astype(str).tolist(),
            concept_names=payload["concept_names"].astype(str).tolist(),
            metadata=json.loads(str(payload["metadata"].item())),
        )


def prepare_dataset(set_zip: Path, outcomes_path: Path, split_frame: pd.DataFrame, config: dict, output_dir: Path) -> PreparedData:
    hours = int(config["data"]["hours"])
    outcomes = load_outcomes(outcomes_path).set_index("RecordID")
    split_frame = split_frame.set_index("RecordID")
    frames: dict[int, pd.DataFrame] = {}
    all_variables: set[str] = set()
    with zipfile.ZipFile(set_zip) as archive:
        for name in sorted(member for member in archive.namelist() if member.lower().endswith(".txt")):
            record_id = int(Path(name).stem)
            frame = _read_record(archive.read(name))
            frames[record_id] = frame
            all_variables.update(frame.loc[~frame["Parameter"].isin(STATIC_PARAMETERS | FORBIDDEN_INPUTS), "Parameter"].unique())
    candidate_variables = sorted(all_variables - FORBIDDEN_INPUTS)
    record_ids = np.asarray(sorted(frames), dtype=np.int64)
    raw_values = np.full((len(record_ids), hours, len(candidate_variables)), np.nan, dtype=np.float32)
    raw_mask = np.zeros_like(raw_values, dtype=np.float32)
    static_rows: list[dict[str, float]] = []
    for position, record_id in enumerate(record_ids):
        values, mask, static = _aggregate_patient(frames[int(record_id)], candidate_variables, hours)
        raw_values[position] = values
        raw_mask[position] = mask
        static_rows.append(static)
    splits = np.asarray([split_frame.loc[int(record_id), "split"] for record_id in record_ids])
    train_index = np.flatnonzero(splits == "train")
    coverage = (raw_mask[train_index].sum(axis=1) > 0).mean(axis=0)
    keep = coverage >= float(config["data"]["minimum_train_patient_coverage"])
    variables = [name for name, accepted in zip(candidate_variables, keep) if accepted]
    raw_values = raw_values[:, :, keep]
    raw_mask = raw_mask[:, :, keep]
    coverage_table = pd.DataFrame({"parameter": candidate_variables, "train_patient_coverage": coverage, "included": keep})
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_table.to_csv(output_dir / "feature_coverage.csv", index=False)
    train_stats = _robust_stats(raw_values[train_index])
    for name, values in train_stats.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite train-only preprocessing statistic: {name}")
    filled = np.empty_like(raw_values)
    for patient in range(len(record_ids)):
        patient_values = _forward_fill(raw_values[patient])
        missing = ~np.isfinite(patient_values)
        patient_values[missing] = np.broadcast_to(train_stats["median"], patient_values.shape)[missing]
        filled[patient] = patient_values
    clipped = np.clip(filled, train_stats["q005"], train_stats["q995"])
    scaled = (clipped - train_stats["median"]) / np.maximum(train_stats["iqr"], 1e-6)
    delta = np.stack([compute_delta_time(mask, hours) for mask in raw_mask], axis=0)
    static_frame = pd.DataFrame(static_rows, index=record_ids)
    continuous_names = ["Age", "Height", "Weight_at_admission"]
    static_continuous = static_frame[continuous_names].to_numpy(dtype=np.float32).copy()
    train_static = static_continuous[train_index]
    static_median = np.nanmedian(train_static, axis=0)
    static_iqr = np.nanquantile(train_static, 0.75, axis=0) - np.nanquantile(train_static, 0.25, axis=0)
    missing_static = ~np.isfinite(static_continuous)
    static_continuous[missing_static] = np.broadcast_to(static_median, static_continuous.shape)[missing_static]
    static_continuous = (static_continuous - static_median) / np.maximum(static_iqr, 1e-6)
    gender = static_frame["Gender"].fillna(-1).to_numpy(dtype=np.float32)[:, None]
    icu_raw = static_frame["ICUType"].fillna(-1).to_numpy(dtype=np.int16)
    icu_one_hot = np.stack([(icu_raw == value).astype(np.float32) for value in [1, 2, 3, 4]], axis=1)
    static = np.concatenate([static_continuous, gender, icu_one_hot], axis=1).astype(np.float32)
    components = _concept_components(raw_values, raw_mask, variables)
    definitions = config["concepts"]["definitions"]
    concept_names = list(config["concepts"]["names"])
    concept_quantiles: dict[str, dict[str, float]] = {}
    concept_values = np.zeros((len(record_ids), hours, len(concept_names)), dtype=np.float32)
    concept_mask = np.zeros_like(concept_values, dtype=np.float32)
    for concept_position, concept_name in enumerate(concept_names):
        component_scores: list[np.ndarray] = []
        component_available: list[np.ndarray] = []
        for direction, names in definitions[concept_name].items():
            for component_name in names:
                if component_name not in components:
                    continue
                values, available = components[component_name]
                train_values = values[train_index][available[train_index] & np.isfinite(values[train_index])]
                if len(train_values) == 0:
                    continue
                quantiles = {
                    "q05": float(np.quantile(train_values, 0.05)),
                    "q25": float(np.quantile(train_values, 0.25)),
                    "q75": float(np.quantile(train_values, 0.75)),
                    "q95": float(np.quantile(train_values, 0.95)),
                }
                concept_quantiles[component_name] = quantiles
                if direction == "binary":
                    score = np.where(available, np.clip(values, 0.0, 1.0), 0.0).astype(np.float32)
                else:
                    score = _score_component(values, available, direction, quantiles)
                component_scores.append(score)
                component_available.append(available)
        if not component_scores:
            raise ValueError(f"No available components for concept {concept_name}")
        scores = np.stack(component_scores, axis=-1)
        available = np.stack(component_available, axis=-1)
        scores = np.where(available, scores, -np.inf)
        sorted_scores = np.sort(scores, axis=-1)
        available_count = available.sum(axis=-1)
        top1 = sorted_scores[..., -1]
        top2 = sorted_scores[..., -2] if scores.shape[-1] > 1 else top1
        aggregate = np.where(available_count >= 2, 0.5 * (top1 + top2), top1)
        aggregate = np.where(available_count > 0, aggregate, 0.0)
        concept_values[..., concept_position] = np.clip(aggregate, 0.0, 1.0)
        concept_mask[..., concept_position] = (available_count > 0).astype(np.float32)
    y = np.asarray([outcomes.loc[int(record_id), "In-hospital_death"] for record_id in record_ids], dtype=np.int8)
    preprocessing_stats = {
        "variables": variables,
        "train_only": True,
        "median": dict(zip(variables, map(float, train_stats["median"]))),
        "iqr": dict(zip(variables, map(float, train_stats["iqr"]))),
        "q005": dict(zip(variables, map(float, train_stats["q005"]))),
        "q995": dict(zip(variables, map(float, train_stats["q995"]))),
        "static_continuous": continuous_names,
        "static_median": dict(zip(continuous_names, map(float, static_median))),
        "static_iqr": dict(zip(continuous_names, map(float, static_iqr))),
    }
    stats_path = output_dir / "preprocessing_stats.json"
    stats_path.write_text(json.dumps(preprocessing_stats, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "feature_schema.json").write_text(
        json.dumps({"temporal_variables": variables, "channels": ["V", "M", "D"], "static_dim": int(static.shape[1])}, indent=2),
        encoding="utf-8",
    )
    concept_dir = output_dir.parent / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    (concept_dir / "concept_definition.yaml").write_text(yaml.safe_dump(config["concepts"], sort_keys=False), encoding="utf-8")
    (concept_dir / "concept_quantiles.json").write_text(json.dumps(concept_quantiles, indent=2, sort_keys=True), encoding="utf-8")
    long_rows = {
        "RecordID": np.repeat(record_ids, hours * len(concept_names)),
        "hour": np.tile(np.repeat(np.arange(hours), len(concept_names)), len(record_ids)),
        "concept": np.tile(concept_names, len(record_ids) * hours),
        "target": concept_values.reshape(-1),
        "observed": concept_mask.reshape(-1).astype(np.int8),
        "split": np.repeat(splits, hours * len(concept_names)),
    }
    pd.DataFrame(long_rows).to_parquet(concept_dir / "concept_trajectories.parquet", index=False, compression="zstd")
    observability = pd.DataFrame(long_rows).groupby(["split", "concept", "hour"], as_index=False).agg(
        observability=("observed", "mean"), mean_target=("target", "mean")
    )
    observability.to_csv(concept_dir / "concept_observability.csv", index=False)
    metadata = {
        "status": "PHYSIONET2012_PREPARED_PASS",
        "created_utc": utc_now(),
        "patients": len(record_ids),
        "hours": hours,
        "temporal_variables": len(variables),
        "input_dims": {
            "V": len(variables) + static.shape[1],
            "V+M": 2 * len(variables) + static.shape[1],
            "V+D": 2 * len(variables) + static.shape[1],
            "V+M+D": 3 * len(variables) + static.shape[1],
        },
        "split_counts": pd.Series(splits).value_counts().to_dict(),
        "split_sha256": sha256_file(output_dir.parent / "splits" / f"split_seed_{int(config['program']['split_seed'])}.json"),
        "preprocessing_sha256": sha256_file(stats_path),
        "forbidden_outcome_inputs": sorted(FORBIDDEN_INPUTS),
        "future_fill_used": False,
        "normalization_scope": "train_only",
    }
    data = PreparedData(
        record_ids=record_ids,
        split=splits,
        y=y,
        v=scaled.astype(np.float32),
        m=raw_mask.astype(np.float32),
        d=delta.astype(np.float32),
        static=static,
        concepts=concept_values,
        concept_mask=concept_mask,
        variables=variables,
        concept_names=concept_names,
        metadata=metadata,
    )
    prepared_path = output_dir / "prepared_physionet2012.npz"
    save_prepared(data, prepared_path)
    metadata["prepared_sha256"] = sha256_file(prepared_path)
    data.metadata = metadata
    (output_dir / "prepare_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data
