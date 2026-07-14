#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import PurePosixPath


FORBIDDEN = [
    "PEND" + "ING",
    "NOT" + "_RUN",
    "PLACE" + "HOLDER",
    "PILOT" + "_ONLY",
    "TODO" + "_RESULT",
]


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
    checks = {}
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        checks["zip_opens"] = True
        text_files = [n for n in names if n.endswith((".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".log"))]
        joined = ""
        for name in text_files:
            try:
                joined += zf.read(name).decode("utf-8", errors="ignore")
            except KeyError:
                pass
        checks["no_forbidden_strings"] = not any(token in joined for token in FORBIDDEN)
        checks["three_seeds_present"] = all(any(f"RUNS/seed_{seed}/" in n or f"runs/seed_{seed}/" in n for n in names) for seed in [42, 43, 44])
        required = [
            "RESULTS/fan_results.csv",
            "RESULTS/concept_metrics.csv",
            "RESULTS/faithfulness_results.csv",
            "RESULTS/shortcut_audit.csv",
            "RESULTS/planted_results.csv",
            "RESULTS/program_status.json",
            "README_FIRST.md",
            "DELIVERY_REPORT.md",
        ]
        checks["required_results_present"] = all(any(n.endswith(req) for n in names) for req in required)
        checks["oracle_fan_present"] = "oracle_fan" in joined
        checks["predicted_fan_present"] = "predicted_fan" in joined
        checks["fan4_present"] = "fan_4" in joined
        checks["shortcut_present"] = "shortcut_audit" in joined
        checks["planted_present"] = "CircuitF1" in joined and "planted" in joined
        checks["skipped_has_reason"] = ("SKIPPED_BY_GATE" not in joined) or ("reason" in joined)
    checks["sha256"] = sha256(args.zip)
    ok = all(v for k, v in checks.items() if k != "sha256")
    print(json.dumps(checks, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
