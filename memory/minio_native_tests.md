# MinIO for the remote store tests: native binary on this Mac (task 2.8)

**Works:** `scripts/remote_store_tests.py` prefers a native `minio` on PATH (temporary
data dir, `--address 127.0.0.1:<free port>`), else docker/podman with a pinned image
(CI). It creates the bucket with a hand-signed SigV4 `PUT` (no `mc`, no boto3) and runs
`tile_reader` and `remote_storage` with `KYTO_TEST_S3_*` set. Without either it prints
SKIPPED and exits 0; CI sets `KYTO_REQUIRE_MINIO=1` so it cannot skip there.

**Watch out:** this Mac has no container runtime (Docker Desktop removed, its
`/usr/local/bin/docker` symlink dangles). MinIO comes from Homebrew
(RELEASE.2025-10-15T17-29-55Z), a formula marked deprecated (upstream archived) that
Homebrew disables on **2027-02-17**. After that the gate silently SKIPs S3 locally:
install MinIO another way or a container runtime. Community MinIO images are no
longer published, so the pinned tag must stay an existing one.

**Also:** object_store retries a dead endpoint for minutes; a wrong
`KYTO_TEST_S3_ENDPOINT` looks like a hang, not a failure. `Path::from(key)` encodes a
`%` again in the backend; use `Path::parse` for zarr keys.
