#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile


FORBIDDEN = ["PEND" + "ING", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT", "PILOT" + "_ONLY", "HARDCODED" + "_RESULT"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        text = ""
        for name in names:
            if name.endswith((".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".log")):
                text += zf.read(name).decode("utf-8", errors="ignore")
        required = [
            "AGENTS.md",
            "PROJECT_MEMORY/PROJECT_STATE.md",
            "CONFIGS/full.yaml",
            "RESULTS/concept_sufficiency.csv",
            "RESULTS/concept_leakage_metrics.csv",
            "RESULTS/concept_residual_predictions.parquet",
            "RESULTS/membership_diagnostics.csv",
            "RESULTS/fan_weight_diagnostics.csv",
            "RESULTS/fan_aggregation_diagnostics.csv",
            "RESULTS/faithfulness_results.csv",
            "RESULTS/shortcut_audit.csv",
            "RESULTS/planted_metrics.csv",
            "RESULTS/planted_intervention_effects.parquet",
            "RESULTS/representation_audit.parquet",
            "RESULTS/sctc_feature_grid.csv",
            "RESULTS/standard_sctc_results.csv",
            "RESULTS/fan_sctc_results.csv",
            "RESULTS/explicit_vs_discovered.csv",
            "RESULTS/aggregate_metrics.csv",
            "RESULTS/program_status.json",
            "TESTS/pytest_exit_code.txt",
            "TESTS/compileall_exit_code.txt",
        ]
        checks = {
            "zip_opens": True,
            "sha256": sha256(args.zip),
            "no_forbidden_strings": not any(token in text for token in FORBIDDEN),
            "three_seeds": all(any(f"RUNS/seed_{seed}/" in n for n in names) for seed in [42, 43, 44]),
            "required_files": all(any(n.endswith(req) for n in names) for req in required),
            "raw_planted_files": all(any(req in n for n in names) for req in ["planted_feature_matching.parquet", "planted_edge_candidates.parquet", "planted_random_null.parquet"]),
            "checkpoints": any("best_checkpoint.pt" in n for n in names),
            "logs": any("status.json" in n for n in names),
            "data_graph_name": "DataGraphAgreementF1" in text,
            "circuitf1_present": "CircuitF1" in text,
            "skipped_has_reason": ("SKIPPED_BY_GATE" not in text) or ("reason" in text),
        }
    print(json.dumps(checks, indent=2))
    return 0 if all(v for k, v in checks.items() if k != "sha256") else 1


if __name__ == "__main__":
    raise SystemExit(main())

