#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


FORBIDDEN_PATTERNS = {
    "hardcoded_data_graph_agreement": r"DataGraphAgreementF1['\"]?\s*:\s*0\.38",
    "hardcoded_fan_sctc_agreement": r"explicit_concept_intervention_agreement['\"]?\s*:\s*0\.66",
    "formula_circuit_f1": r"0\.82\s*\+\s*0\.03",
    "hardcoded_sign_agreement": r"sign_agreement['\"]?\s*:\s*0\.94",
    "hardcoded_negative_fpr": r"negative_control_fpr['\"]?\s*:\s*0\.02",
    "validation_plus_random_test": r"AUPRC.*validation.*\+\s*delta|test_minus_validation_AUPRC",
    "mse_to_delta_auprc": r"delta_AUPRC['\"]?\s*:\s*.*mse\s*/\s*10",
    "mse_to_probability_mae": r"probability_MAE['\"]?\s*:\s*.*mse\s*/\s*5",
    "synthetic_representation_strength": r"base_strength\s*=|patching_effect[^\n]*r2",
    "catalog_arithmetic_sequence": r"activation_Pearson[^\n]*\+[^\n]*idx|ablation_effect[^\n]*idx|p_value[^\n]*edge_id",
    "synthetic_fan_metric_arrays": r"np\.r_\[\s*np\.ones\(\s*100\s*\)|linspace\(\s*0\.55\s*,\s*0\.95|linspace\(\s*0\.05\s*,\s*0\.45",
    "manual_fan_stability_triplet": r"0\.79[,\s\S]{0,120}0\.78[,\s\S]{0,120}0\.80",
    "csv_copied_to_parquet": r"copy2\([^\n]+\.csv[^\n]+\.parquet",
    "validation_copied_to_replication": r"\b(fan|std|planted)\s*=\s*pd\.read_csv\([^\n]+predicted_vs_oracle_noalpha\.csv[\s\S]{0,500}replication_metrics\.csv|replication_model_predictions\.parquet[\s\S]{0,120}\breplication_metrics\.csv",
    "placeholder_figure_series": r"ax\.plot\(\s*\[0,\s*1,\s*2\]\s*,\s*\[0\.2,\s*0\.5,\s*0\.3\]",
    "matplotlib_paper_pdf": r"PdfPages\([^\n]+main\.pdf|PdfPages\([^\n]+supplement\.pdf",
    "forced_edge_rejection": r"accepted\s*=\s*False[\s\S]{0,200}NO_VALIDATED_EDGES",
}


def _expand(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts))
        elif path.exists():
            out.append(path)
    return out


def validate(paths: list[Path]) -> dict:
    findings = []
    files = _expand(paths)
    for path in files:
        if not path.exists() or path.is_dir() or path.suffix == ".pyc":
            continue
        if path.name == "validate_no_synthetic_results.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                findings.append({"file": str(path), "pattern": name})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    except Exception:
        commit = None
    return {
        "passed": not findings,
        "files_scanned": [str(path) for path in files],
        "rules_checked": sorted(FORBIDDEN_PATTERNS),
        "findings": findings,
        "code_commit": commit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=["scripts/medical/v3/run_research_program.py"])
    parser.add_argument("--source-root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.source_root:
        base = Path(args.source_root)
        paths = [base / "scripts" / "medical" / "v3", base / "src" / "fan", base / "src" / "med_circuitbench"]
    else:
        paths = [Path(p) for p in args.paths]
    result = validate(paths)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
