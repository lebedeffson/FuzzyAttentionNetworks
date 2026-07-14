#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


FORBIDDEN_TOKENS = ["PEND" + "ING", "PEND" + "ING_FULL_VALIDATION", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT"]
REQUIRED_AGGREGATE = [
    "aggregate_metrics.csv",
    "benchmark_comparison.csv",
    "circuit_f1_by_seed.csv",
    "circuit_metrics_by_seed.csv",
    "bootstrap_intervals.csv",
    "feature_stability.csv",
    "edge_stability.csv",
    "go_no_go.json",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="ignore")


def validate(zip_path: Path) -> dict:
    failures: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        root = names[0].split("/", 1)[0] if names else ""
        name_set = set(names)
        if not root:
            failures.append("empty zip")
        for suffix in REQUIRED_AGGREGATE:
            if not any(name.endswith("/" + suffix) or name == suffix for name in names):
                failures.append(f"missing aggregate file: {suffix}")
        for seed in (42, 43, 44):
            marker = f"RUNS/seed_{seed}_validation/"
            if not any(marker in name for name in names):
                failures.append(f"missing validation run directory for seed {seed}")
            config_candidates = [name for name in names if name.endswith(f"RUNS/seed_{seed}_validation/config_resolved.yaml")]
            if not config_candidates:
                failures.append(f"missing resolved config for seed {seed}")
            else:
                text = _read_text(zf, config_candidates[0])
                if "n_samples: 10000" not in text or "maximum_epochs: 50" not in text or "random_directions: 1000" not in text:
                    failures.append(f"seed {seed} resolved config is not full benchmark config")
        if not any("CircuitF1" in _read_text(zf, name) for name in names if name.endswith("circuit_f1_by_seed.csv") or name.endswith("circuit_metrics_by_seed.csv")):
            failures.append("CircuitF1 metric absent")
        if not any("sae" in _read_text(zf, name).lower() for name in names if name.endswith(".csv") or name.endswith(".json")):
            failures.append("SAE results absent")
        if not any("forward_replacement" in _read_text(zf, name) or "sequential_forward_chain_ablation" in _read_text(zf, name) for name in names if name.endswith(".json") or name.endswith(".py")):
            failures.append("forward intervention marker absent")
        if not any(name.endswith("TESTS/pytest_stdout.log") for name in names):
            failures.append("pytest stdout log absent")
        if not any(name.endswith("TESTS/end_to_end_smoke_stdout.log") for name in names):
            failures.append("smoke stdout log absent")
        if not any(name.endswith("checksums.sha256") for name in names):
            failures.append("checksums file absent")
        for name in names:
            if not name.lower().endswith((".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".py", ".log")):
                continue
            text = _read_text(zf, name)
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    failures.append(f"forbidden token {token} in {name}")
                    break
        checksum_members = [name for name in names if name.endswith("checksums.sha256")]
        if checksum_members:
            checksum_text = _read_text(zf, checksum_members[0])
            for line in checksum_text.splitlines():
                if not line.strip():
                    continue
                expected, rel = line.split(maxsplit=1)
                member = f"{root}/{rel}" if not rel.startswith(root + "/") else rel
                if member not in name_set:
                    failures.append(f"checksum target missing: {rel}")
                    continue
                actual = _sha256_bytes(zf.read(member))
                if actual != expected:
                    failures.append(f"checksum mismatch: {rel}")
                    break
        go_files = [name for name in names if name.endswith("go_no_go.json")]
        if not go_files:
            failures.append("go_no_go.json absent")
        else:
            payload = json.loads(_read_text(zf, go_files[0]))
            if payload.get("overall_status") not in {"GO", "NO_GO"}:
                failures.append("invalid GO/NO-GO status")
        run_manifest_texts = [_read_text(zf, name) for name in names if name.endswith("_validation/manifest.json")]
        if len(set(run_manifest_texts)) < min(3, len(run_manifest_texts)):
            failures.append("validation run manifests are duplicated")
    return {"status": "PASS" if not failures else "FAIL", "zip": str(zip_path), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.zip)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
