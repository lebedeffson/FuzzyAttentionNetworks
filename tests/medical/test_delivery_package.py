import zipfile

from scripts.medical.package_delivery import build_delivery


def test_delivery_package_schema(tmp_path):
    zip_path, sha_path = build_delivery(tmp_path)
    assert zip_path.exists()
    assert sha_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    required_suffixes = {
        "README_FIRST.txt",
        "DELIVERY_REPORT.md",
        "GIT_INFO.txt",
        "SOURCE/src/med_circuitbench/__init__.py",
        "SOURCE/scripts/medical/package_delivery.py",
        "SOURCE/configs/medical/smoke.yaml",
        "SOURCE/tests/medical/test_delivery_package.py",
        "CONFIGS/medical/smoke.yaml",
        "TESTS/pytest_stdout.log",
        "TESTS/compileall_exit_code.txt",
        "RESULTS/go_no_go_validation.json",
        "MANIFESTS/delivery_manifest.json",
        "MANIFESTS/checksums.sha256",
        "LIMITATIONS/known_issues.md",
    }
    for suffix in required_suffixes:
        assert any(name.endswith(suffix) for name in names), suffix
    assert not any("physionet2019/raw" in name.lower() for name in names)
    assert not any("__pycache__" in name for name in names)
