#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.report
    selected = [
        report / "README.md",
        report / "MANIFESTS" / "posthoc_attribution_manifest.json",
        report / "MANIFESTS" / "posthoc_attribution_validation.json",
        report / "MANIFESTS" / "plain_transformer_checkpoint_sha256.csv",
        report / "CONFIG" / "posthoc_attribution_stability.yaml",
        report / "CONFIG" / "channel_to_proxy_concept_map.csv",
        report / "TABLES" / "posthoc_article_table.csv",
        report / "TABLES" / "posthoc_attribution_summary.csv",
        report / "TABLES" / "posthoc_method_comparison.csv",
        report / "TABLES" / "posthoc_completeness.csv",
        report / "TABLES" / "posthoc_baseline_sensitivity.csv",
        report / "TABLES" / "posthoc_mapping_controls.csv",
        report / "TABLES" / "posthoc_randomization_control.csv",
        *sorted((report / "FIGURES").glob("*.svg")),
    ]
    missing = [path for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conceptfan-posthoc-") as temporary:
        root = Path(temporary) / "ConceptFAN_PhysioNet_POSTHOC_FINAL_PACKAGE"
        root.mkdir()
        shutil.copy2(args.docx, root / args.docx.name)
        shutil.copy2(args.pdf, root / args.pdf.name)
        for source in selected:
            relative = source.relative_to(report)
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        sums = "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
        (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                archive.write(path, Path(root.name) / path.relative_to(root))
    archive_sha = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{archive_sha}  {args.output.name}\n", encoding="utf-8"
    )
    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        if any("DATA/" in name or name.endswith(".pt") for name in names):
            raise RuntimeError("Compact package unexpectedly contains raw attribution data or checkpoints")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt archive member: {bad}")
    print(f"{archive_sha}  {args.output}")


if __name__ == "__main__":
    main()
