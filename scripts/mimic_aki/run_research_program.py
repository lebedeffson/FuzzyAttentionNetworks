#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimic_aki.access import verify_mimic_access  # noqa: E402
from mimic_aki.cohort import demo_cohort_summary  # noqa: E402
from mimic_aki.features import demo_creatinine_events  # noqa: E402
from mimic_aki.io import MimicSource  # noqa: E402
from mimic_aki.kdigo import creatinine_kdigo_events  # noqa: E402
from mimic_aki.windows import incident_aki_windows  # noqa: E402


STAGES = [
    "01_verify_access",
    "02_build_cohort",
    "03_build_aki_labels",
    "04_build_windows",
    "05_data_audit",
    "06_train_baselines",
    "07_train_standard_conceptfan",
    "08_train_fuzzy_encoder_models",
    "09_predictive_evaluation",
    "10_concept_evaluation",
    "11_fan_faithfulness",
    "12_oracle_evaluator_regression",
    "13_representation_audit",
    "14_sae_training",
    "15_sae_grounding",
    "16_steering",
    "17_frozen_test",
    "18_article_tables_figures",
    "19_validation",
    "20_delivery_package",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    access = verify_mimic_access()
    rows = [{"stage": STAGES[0], "status": access.status, "reason": "MIMIC_IV_ROOT missing or incomplete" if access.status != "OK" else "OK"}]
    if access.status != "OK":
        if access.status == "OK_DEMO":
            source = MimicSource.open(access.root)
            cohort, flow = demo_cohort_summary(source)
            creat = demo_creatinine_events(source)
            labels = creatinine_kdigo_events(creat)
            event_times = creat.rename(columns={"charttime": "charttime"})[["stay_id", "charttime"]]
            windows = incident_aki_windows(event_times, labels)
            flow.to_csv(output / "cohort_flow.csv", index=False)
            labels.to_parquet(output / "aki_labels_demo.parquet", index=False)
            windows.to_parquet(output / "aki_windows_demo.parquet", index=False)
            cohort.drop(columns=["subject_id"], errors="ignore").head(100).to_parquet(output / "cohort_demo_deidentified_preview.parquet", index=False)
            rows = [
                {"stage": STAGES[0], "status": "OK_DEMO", "reason": "MIMIC-IV demo zip detected"},
                {"stage": STAGES[1], "status": "COMPLETED_DEMO", "reason": f"{len(cohort)} ICU stays parsed"},
                {"stage": STAGES[2], "status": "COMPLETED_DEMO", "reason": f"{len(labels)} creatinine KDIGO events"},
                {"stage": STAGES[3], "status": "COMPLETED_DEMO", "reason": f"{len(windows)} incident windows"},
                {"stage": STAGES[4], "status": "COMPLETED_DEMO", "reason": "cohort_flow and demo manifests written"},
            ]
            for stage in STAGES[5:]:
                rows.append({"stage": stage, "status": "SKIPPED_BY_GATE", "reason": "DEMO_DEBUG_ONLY_FULL_TRAINING_REQUIRES_FULL_MIMIC"})
            final = {
                "program": "MIMIC_AKI_FAN_SAE",
                "status": "DEMO_PIPELINE_COMPLETE",
                "dataset_kind": access.dataset_kind,
                "root": access.root,
                "config": cfg,
                "stage_results": rows,
                "counts": {
                    "cohort_rows": int(len(cohort)),
                    "creatinine_rows": int(len(creat)),
                    "aki_label_rows": int(len(labels)),
                    "window_rows": int(len(windows)),
                },
            }
            (output / "program_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
            print(json.dumps(final, indent=2))
            return 0
        for stage in STAGES[1:]:
            rows.append({"stage": stage, "status": "SKIPPED_BY_GATE", "reason": "BLOCKED_DATA_ACCESS"})
        final = {
            "program": "MIMIC_AKI_FAN_SAE",
            "status": "BLOCKED_DATA_ACCESS",
            "config": cfg,
            "missing_files": access.missing_files,
            "stage_results": rows,
        }
        (output / "program_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final, indent=2))
        return 2
    final = {"program": "MIMIC_AKI_FAN_SAE", "status": "DATA_ACCESS_OK_IMPLEMENTATION_READY", "stage_results": rows}
    (output / "program_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
