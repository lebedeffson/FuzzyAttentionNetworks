#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_filtered(src: Path, dst: Path) -> None:
    def ignore(_dir, names):
        banned = {"__pycache__", ".pytest_cache"}
        return [n for n in names if n in banned or n.endswith(".zip")]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fan-stable-output", default="artifacts/medical/v3_1/fan_noalpha_stable")
    parser.add_argument("--release-root", default="artifacts/medical/v3_1_practical_release")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    args = parser.parse_args(argv)
    release = Path(args.release_root)
    if release.exists():
        shutil.rmtree(release)
    for sub in ["SOURCE", "CONFIGS", "TESTS", "RESULTS", "BUNDLES", "MANIFESTS", "LIMITATIONS"]:
        (release / sub).mkdir(parents=True, exist_ok=True)
    copytree_filtered(ROOT / "src" / "fan", release / "SOURCE" / "src" / "fan")
    copytree_filtered(ROOT / "src" / "med_circuitbench", release / "SOURCE" / "src" / "med_circuitbench")
    copytree_filtered(ROOT / "scripts" / "medical" / "v3_1", release / "SOURCE" / "scripts" / "medical" / "v3_1")
    copytree_filtered(ROOT / "scripts" / "cli", release / "SOURCE" / "scripts" / "cli")
    copytree_filtered(ROOT / "configs" / "medical" / "v3_1", release / "CONFIGS" / "medical" / "v3_1")
    copytree_filtered(ROOT / "tests" / "medical" / "v3", release / "TESTS" / "medical" / "v3")
    fan_out = ROOT / args.fan_stable_output
    if fan_out.exists():
        for name in [
            "fan_stable_gate.json",
            "fan_stable_metrics.csv",
            "fan_cross_seed_stability.csv",
            "resolved_config.yaml",
        ]:
            if (fan_out / name).exists():
                shutil.copy2(fan_out / name, release / "RESULTS" / name)
        if (fan_out / "bundles").exists():
            copytree_filtered(fan_out / "bundles", release / "BUNDLES")
    for src, dst in [
        (ROOT / "AGENTS.md", release / "MANIFESTS" / "AGENTS.md"),
        (ROOT / "docs" / "medical" / "PROJECT_STATE.md", release / "MANIFESTS" / "PROJECT_STATE.md"),
        (ROOT / "docs" / "medical" / "KNOWN_LIMITATIONS.md", release / "LIMITATIONS" / "KNOWN_LIMITATIONS.md"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    manifest = {
        "status": "Med-CircuitBench V3.1 Research Complete",
        "code_commit": commit,
        "git_status_clean": status.strip() == "",
        "sctc_line_status": [
            "SPARSE_FIDELITY_VALIDATED",
            "MECHANISTIC_RECOVERY_NOT_SEMANTICALLY_SPECIFIC",
            "DECODER_COUPLING_BREAKS_FIDELITY",
        ],
        "papers_included": False,
    }
    (release / "MANIFESTS" / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (release / "README_FIRST.md").write_text(
        "\n".join(
            [
                "# Med-CircuitBench V3.1 Practical Release",
                "",
                "This package contains practical code, configs, tests, results, and frozen bundles.",
                "Papers are intentionally not included in this practical release.",
                "",
                "SCTC development is closed as a mixed/negative research line:",
                "- sparse behavioral fidelity is validated;",
                "- mechanistic recovery is not semantically specific;",
                "- decoder-coupled semantic supervision breaks fidelity and remains non-specific.",
            ]
        ),
        encoding="utf-8",
    )
    zip_dir = Path(args.zip_output_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    short = commit[:8]
    zip_path = zip_dir / f"Med_CircuitBench_V3_1_RESEARCH_COMPLETE_{short}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(release.parent))
    sha = sha256_file(zip_path)
    (zip_path.with_suffix(zip_path.suffix + ".sha256")).write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(zip_path), "sha256": sha, "size_bytes": zip_path.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
