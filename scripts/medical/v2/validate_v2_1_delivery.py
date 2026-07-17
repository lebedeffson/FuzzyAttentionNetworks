#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile


FORBIDDEN = ["PEND" + "ING", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT", "PILOT" + "_ONLY"]


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
            "RESULTS/concept_sufficiency.csv",
            "RESULTS/static_vs_temporal_fan.csv",
            "RESULTS/concept_leakage_metrics.csv",
            "RESULTS/membership_diagnostics.csv",
            "RESULTS/fan_weight_diagnostics.csv",
            "RESULTS/standard_sctc_results.csv",
            "RESULTS/fan_sctc_results.csv",
            "RESULTS/program_status.json",
            "TESTS/pytest_exit_code.txt",
            "TESTS/compileall_exit_code.txt",
        ]
        checks = {
            "zip_opens": True,
            "sha256": sha256(args.zip),
            "no_forbidden_strings": not any(token in text for token in FORBIDDEN),
            "three_seeds_present": all(any(f"RUNS/seed_{seed}/" in n for n in names) for seed in [42, 43, 44]),
            "required_files_present": all(any(n.endswith(req) for n in names) for req in required),
            "project_memory_present": any(n.endswith("AGENTS.md") for n in names)
            and any(n.endswith("PROJECT_MEMORY/PROJECT_STATE.md") for n in names),
            "skipped_by_gate_has_reason": ("SKIPPED_BY_GATE" not in text) or ("reason" in text),
            "circuitf1_planted_only": "DataGraphAgreementF1" in text,
        }
    print(json.dumps(checks, indent=2))
    return 0 if all(v for k, v in checks.items() if k != "sha256") else 1


if __name__ == "__main__":
    raise SystemExit(main())
