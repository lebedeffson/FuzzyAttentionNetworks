# Reproduce from raw files

Run from the repository root at commit `e60d9a3da40fc05b66874f89ad61015d6e07d33c` plus the PhysioNet last-run implementation on this branch:

```bash
bash scripts/run_physionet2012_last_run.sh \
  --set-a-zip data/physionet2012/raw/set-a.zip \
  --outcomes data/physionet2012/raw/Outcomes-a.txt \
  --artifacts-dir artifacts/physionet2012_last_run \
  --device cuda \
  --resume
```

If `Outcomes-a.txt` is absent, the pipeline downloads it from the configured official PhysioNet endpoint before the raw-data audit. Raw patient files are never added to the report archive.
