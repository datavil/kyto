"""The MinIO runner skips visibly without a container runtime (task 2.8)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote_store_tests.py"


def _run_without_runtime(
    tmp_path: Path, require: bool
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PATH": str(tmp_path)}
    environment.pop("KYTO_REQUIRE_MINIO", None)
    if require:
        environment["KYTO_REQUIRE_MINIO"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_skips_with_a_notice_without_a_runtime(tmp_path: Path) -> None:
    result = _run_without_runtime(tmp_path, require=False)
    assert result.returncode == 0, result.stderr
    assert "SKIPPED: remote store tests on MinIO" in result.stdout


def test_fails_without_a_runtime_when_required(tmp_path: Path) -> None:
    result = _run_without_runtime(tmp_path, require=True)
    assert result.returncode == 1
    assert "SKIPPED" in result.stdout
