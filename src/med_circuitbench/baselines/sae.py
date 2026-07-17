from __future__ import annotations

from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


class SparseAutoEncoder(SparseClinicalTranscoder):
    """SAE baseline with the same sparse bottleneck contract as SCTC."""
