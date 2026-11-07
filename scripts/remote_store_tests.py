"""Run the tile reader test suite against S3 on a local MinIO container (task 2.8).

Does: start MinIO on a free local port, from a native `minio` binary on PATH (in a
temporary data directory) or else from a pinned image in a container runtime (docker
or podman, as in CI); create a test bucket; run `cargo test -p kyto-store --test
tile_reader --test remote_storage` with the `KYTO_TEST_S3_*` variables set (the
suites then run every test on the local, HTTP and S3 backends); and stop MinIO and
remove its data. With neither a `minio` binary nor a working container runtime it
prints a visible SKIPPED notice and exits 0, so the local gate stays green on
machines without one; set `KYTO_REQUIRE_MINIO=1` to fail instead.

Does not: install MinIO or a container runtime, or test GCS (no emulator; see the
plan).

Usage (from the repo root)::

    uv run python scripts/remote_store_tests.py
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import hmac
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pinned: the community MinIO images are no longer published, so this tag stays put.
IMAGE = "docker.io/minio/minio:RELEASE.2025-04-22T22-12-26Z"
USER = "kyto-test"
PASSWORD = "kyto-test-secret"
BUCKET = "kyto-test"
REGION = "us-east-1"
READY_TIMEOUT_SECONDS = 90.0


def find_runtime() -> str | None:
    """Return the first container runtime whose daemon answers, or None.

    Returns
    -------
    str | None
        The path of `docker` or `podman`, or None if neither runs.
    """
    for name in ("docker", "podman"):
        path = shutil.which(name)
        if path is None:
            continue
        probe = subprocess.run(
            [path, "info"], capture_output=True, text=True, check=False
        )
        if probe.returncode == 0:
            return path
    return None


def free_port() -> int:
    """Return a TCP port that is free on 127.0.0.1 right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def wait_ready(endpoint: str) -> None:
    """Wait until MinIO answers its readiness probe.

    Raises
    ------
    TimeoutError
        If it does not within `READY_TIMEOUT_SECONDS`.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/minio/health/ready", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"MinIO at {endpoint} not ready after {READY_TIMEOUT_SECONDS}s")


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def create_bucket(endpoint: str, bucket: str) -> None:
    """Create `bucket` with an S3 `CreateBucket` request signed with SigV4.

    Raises
    ------
    urllib.error.HTTPError
        If the server refuses the request.
    """
    host = endpoint.removeprefix("http://")
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    day = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical = "\n".join(
        [
            "PUT",
            f"/{bucket}",
            "",
            f"host:{host}",
            f"x-amz-content-sha256:{payload_hash}",
            f"x-amz-date:{amz_date}",
            "",
            "host;x-amz-content-sha256;x-amz-date",
            payload_hash,
        ]
    )
    scope = f"{day}/{REGION}/s3/aws4_request"
    to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical.encode()).hexdigest(),
        ]
    )
    key = _sign(("AWS4" + PASSWORD).encode(), day)
    for part in (REGION, "s3", "aws4_request"):
        key = _sign(key, part)
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        f"{endpoint}/{bucket}",
        method="PUT",
        headers={
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_hash,
            "Authorization": (
                f"AWS4-HMAC-SHA256 Credential={USER}/{scope}, "
                "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
                f"Signature={signature}"
            ),
        },
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


@contextlib.contextmanager
def native_minio(binary: str, port: int) -> Generator[str, None, None]:
    """Run `binary` as a MinIO server on `port` with a temporary data directory.

    Yields
    ------
    str
        A description of the server, for the log.
    """
    with tempfile.TemporaryDirectory(prefix="kyto-minio-") as data:
        process = subprocess.Popen(
            [binary, "server", data, "--address", f"127.0.0.1:{port}", "--quiet"],
            env={
                **os.environ,
                "MINIO_ROOT_USER": USER,
                "MINIO_ROOT_PASSWORD": PASSWORD,
            },
            stdout=subprocess.DEVNULL,
            stderr=None,  # inherited: MinIO errors show in the log
            text=True,
        )
        try:
            yield f"native {binary}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextlib.contextmanager
def container_minio(runtime: str, port: int) -> Generator[str, None, None]:
    """Run the pinned MinIO image in `runtime` on `port`; remove it afterwards.

    Yields
    ------
    str
        A description of the server, for the log.

    Raises
    ------
    RuntimeError
        If the container does not start.
    """
    started = subprocess.run(
        [
            runtime,
            "run",
            "--detach",
            "--rm",
            "--publish",
            f"127.0.0.1:{port}:9000",
            "--env",
            f"MINIO_ROOT_USER={USER}",
            "--env",
            f"MINIO_ROOT_PASSWORD={PASSWORD}",
            IMAGE,
            "server",
            "/data",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        raise RuntimeError(f"could not start MinIO:\n{started.stderr}")
    container = started.stdout.strip()
    try:
        yield f"{IMAGE} in {runtime}"
    finally:
        subprocess.run(
            [runtime, "rm", "--force", container], capture_output=True, check=False
        )


def main() -> int:
    """Run the suites on MinIO; return the exit status."""
    port = free_port()
    binary = shutil.which("minio")
    runtime = None if binary is not None else find_runtime()
    if binary is not None:
        server = native_minio(binary, port)
    elif runtime is not None:
        server = container_minio(runtime, port)
    else:
        notice = (
            "SKIPPED: remote store tests on MinIO (task 2.8) need a `minio` binary "
            "on PATH or a running docker or podman; none found. The tile reader "
            "suite still ran on the local and HTTP backends in `cargo test`."
        )
        print(f"\n{'!' * 78}\n{notice}\n{'!' * 78}\n", flush=True)
        return 1 if os.environ.get("KYTO_REQUIRE_MINIO") == "1" else 0
    endpoint = f"http://127.0.0.1:{port}"
    with server as description:
        wait_ready(endpoint)
        create_bucket(endpoint, BUCKET)
        print(f"MinIO ({description}) at {endpoint}, bucket {BUCKET}", flush=True)
        environment = {
            **os.environ,
            "KYTO_TEST_S3_ENDPOINT": endpoint,
            "KYTO_TEST_S3_BUCKET": BUCKET,
            "KYTO_TEST_S3_ACCESS_KEY_ID": USER,
            "KYTO_TEST_S3_SECRET_ACCESS_KEY": PASSWORD,
            "KYTO_TEST_REQUIRE_S3": "1",
        }
        command = [
            "cargo",
            "test",
            "-p",
            "kyto-store",
            "--locked",
            "--test",
            "tile_reader",
            "--test",
            "remote_storage",
        ]
        return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    sys.exit(main())
