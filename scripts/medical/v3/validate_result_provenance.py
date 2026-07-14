#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {
    "fan_validation_metrics",
    "planted_metrics",
    "planted_sctc_activity",
    "standard_sctc_fidelity",
    "standard_graph_agreement",
    "representation_audit",
    "partial_test_metrics",
    "replication_metrics",
}


def validate(output: Path) -> dict:
    path = output / "manifests" / "result_provenance.jsonl"
    findings = []
    if not path.exists():
        return {"passed": False, "findings": [{"reason": "missing_result_provenance"}]}
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {entry["metric_id"] for entry in entries}
    missing = sorted(REQUIRED - ids)
    for metric_id in missing:
        findings.append({"metric_id": metric_id, "reason": "missing_required_provenance"})
    for entry in entries:
        raw = output / entry["raw_file"].replace("RESULTS/", "results/")
        if not raw.exists():
            findings.append({"metric_id": entry["metric_id"], "reason": "missing_raw_file", "raw_file": entry["raw_file"]})
        for key in ["checkpoint_sha256", "dataset_sha256", "split_sha256", "aggregation_script", "code_commit"]:
            if not entry.get(key):
                findings.append({"metric_id": entry["metric_id"], "reason": f"missing_{key}"})
    return {"passed": not findings, "entry_count": len(entries), "findings": findings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = validate(Path(args.output))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
