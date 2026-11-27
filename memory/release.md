# Releases: wheels, trusted publishing, the read policy (task 10.6)

**Works:** `.github/workflows/release.yml` calls `ci.yml` (`workflow_call`) and publishes
the wheels CI built (artifacts `wheel-*` are shared within the caller's run) plus an
sdist, by PyPI trusted publishing (environments `pypi` for `v*` tags, `testpypi` for runs
by hand). `scripts/release_version.py` checks tag = `v<Cargo version>` and a dated
`CHANGELOG.md` section first. `scripts/wheel_smoke.py` runs a small pipeline with an
installed wheel (`python -I`, so the source tree's `python/` is not imported).

**Broke / watch for:**
- Until 10.6 the wheels declared no dependencies, though the extension imports
  `polars`, `numpy` and `scipy.sparse` at run time: `pip install kyto` gave a package
  whose first terminal failed. `tests/test_release.py` ties `[project] dependencies`
  to the `import("...")` calls in `crates/kyto-py`.
- No `concurrency` on the release workflow: a called workflow's `github.workflow` is
  the caller's name, so the same group in both would cancel CI.
- Install a built wheel by its path (`uv pip install dist/*.whl`), not
  `--find-links dist kyto` with the index on: the index's `kyto` could win.
- NumPy and SciPy have no Windows ARM64 wheels for Python 3.10: the smoke test uses
  3.13 there.
- The store read policy (docs/release.md) is tested by rewriting a real store's root
  `format_version`; refusal (`KY-E115`) happens at `ky.scan`, before any read.

**Unverified until the first dry run:** the manylinux, macOS x86_64 and Windows wheel
builds with the bundled HDF5 ([hdf5_static_build.md](hdf5_static_build.md)).
