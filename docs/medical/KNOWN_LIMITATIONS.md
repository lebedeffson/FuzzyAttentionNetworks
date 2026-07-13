# Known Limitations

- Full V4 validation/test/PhysioNet runs are not claimed complete unless produced by `run_benchmark_pipeline.py`/`run_physionet_pipeline.py`.
- Med-CircuitBench uses the V4 fixed threshold `0.0715`; older `0.65` results are invalid for publication.
- Circuit construction still needs full downstream forward intervention propagation for final scientific claims.
- Raw PhysioNet data is intentionally excluded from Git and delivery packages.
