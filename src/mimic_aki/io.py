from __future__ import annotations

import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MimicSource:
    path: Path
    kind: str
    zip_prefix: str = ""

    def _safe_rel(self, rel: str) -> str:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe MIMIC relative path: {rel}")
        return rel

    @classmethod
    def open(cls, path: str | Path) -> "MimicSource":
        p = Path(path)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
            prefixes = sorted({name.split("/")[0] for name in names if "/" in name})
            prefix = prefixes[0] if prefixes else ""
            kind = "demo_zip" if "demo" in p.name.lower() else "zip"
            return cls(path=p, kind=kind, zip_prefix=prefix)
        return cls(path=p, kind="directory")

    def exists(self, rel: str) -> bool:
        rel = self._safe_rel(rel)
        if self.kind in {"zip", "demo_zip"}:
            target = f"{self.zip_prefix}/{rel}" if self.zip_prefix else rel
            with zipfile.ZipFile(self.path) as zf:
                return target in zf.namelist()
        return (self.path / rel).exists()

    def read_csv(self, rel: str, **kwargs) -> pd.DataFrame:
        rel = self._safe_rel(rel)
        if self.kind in {"zip", "demo_zip"}:
            target = f"{self.zip_prefix}/{rel}" if self.zip_prefix else rel
            with zipfile.ZipFile(self.path) as zf:
                with zf.open(target) as raw:
                    if rel.endswith(".gz"):
                        with gzip.GzipFile(fileobj=raw) as fh:
                            return pd.read_csv(fh, **kwargs)
                    return pd.read_csv(raw, **kwargs)
        return pd.read_csv(self.path / rel, **kwargs)
