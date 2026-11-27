# Releases and compatibility

How kyto is versioned, which stores a kyto version reads, and the checklist a release
follows (task 10.6). Changes per version are in the
[changelog](https://github.com/datavil/kyto/blob/main/CHANGELOG.md).

## Installing

```bash
pip install kyto
```

PyPI carries one wheel per platform, built against CPython's stable ABI (abi3), so the
same wheel serves CPython 3.10 and every later version:

| Platform | Architectures |
| --- | --- |
| macOS | arm64 (first-class), x86_64 |
| Linux (manylinux, glibc) | x86_64, aarch64 |
| Windows | x86_64, ARM64 |

Apple Silicon Macs are the first-class platform (decided by the user, Nov 27, 2026):
the macOS arm64 wheel is the one built and checked by hand before a release (the
checklist below). The other wheels are built and smoke-tested by CI only.

The wheels bundle a static HDF5 and need no system libraries. They contain the CPU
executor only: the `wgpu` and `cuda` GPU features are opt-in Cargo features, so a GPU
build comes from source (`MATURIN_PEP517_ARGS="--features wgpu" pip install kyto
--no-binary kyto`, which needs Rust and CMake). Elsewhere, pip builds the sdist the same
way. kyto depends on NumPy, Polars and SciPy; `kyto[plot]` adds anndata and cellestial
for plotting.

## Versions

kyto follows [semantic versioning 2.0](https://semver.org/spec/v2.0.0.html). The
version has one source, `[workspace.package] version` in the workspace `Cargo.toml`:
maturin writes it into the wheels' metadata and the extension reports it as
`kyto.__version__`. A version is always `MAJOR.MINOR.PATCH`, without a pre-release or
build suffix, because it is also the PyPI version. The release workflow refuses any
other form, and a tag that is not `v<version>`.

The public API that a version number covers:

- the Python API: every name in `kyto.__all__`, `kyto.qc`, `kyto.dtypes`,
  `kyto.errors`, their signatures and what they return;
- the error codes (`KY-Exxx`) and the exception class each one raises;
- the plan JSON (`to_json`, `from_json`, `plan_json_schema`);
- the store format versions kyto writes and reads (below).

The Rust crates are not published and are not part of it.

| Change | Before 1.0 | From 1.0 on |
| --- | --- | --- |
| Removes or changes public API, an error code's meaning, or which store versions are read | MINOR (0.1 to 0.2) | MAJOR |
| Adds public API, a new error code, or a store version that is read | MINOR or PATCH | MINOR |
| Fixes a bug without changing the API | PATCH | PATCH |
| Changes results within the parity tolerances (`tests/parity/tolerances.toml`) | PATCH, stated in the changelog | PATCH, stated in the changelog |

Results outside the tolerances, or a new deviation from scanpy, count as a change of
meaning: they are listed in `deviations.md` and in the changelog.

## Reading older stores

A store records its format as `format_version` (`"MAJOR.MINOR"`, `format.md` §3). The
store format is versioned on its own, apart from the package: most kyto releases do not
change it. The rules for each bump are in `format.md` §11. This is the read policy
for kyto releases:

- **Writes.** A kyto release writes exactly one `format_version`, the newest it
  implements.
- **Reads, before 1.0.** A kyto 0.y release reads exactly the format versions it lists
  (in the changelog and in the table below). It refuses every other version, older or
  newer, with `KY-E115` (`FormatVersionError`), which names the store's version and the
  versions this kyto reads. Nothing is read on a guess. A kyto release that stops
  reading a format version is a MINOR bump and says so in the changelog.
- **Reads, from 1.0 on.** A kyto 1.y release reads every store of format 1.x: its own
  1.m and older 1.x in full, newer 1.x (x > m) by ignoring the members and JSON keys it
  does not know (`format.md` §11). It refuses format 0.x and 2.x with `KY-E115`. A
  format major bump comes only with a kyto major bump.
- **Never in place.** Reading never upgrades or modifies a store. To carry a store to
  a newer format, rerun the plan that made it with the newer kyto (the store's
  provenance holds the plan), or export it as `.h5ad` with the kyto that reads it
  (`sink_h5ad`) and import that file with the newer one (`kyto.convert`).

| kyto | Writes | Reads |
| --- | --- | --- |
| 0.1.x | 0.1 | 0.1 |

The table's last row is checked against `SUPPORTED_FORMAT_VERSIONS` in
`crates/kyto-store/src/store_version.rs` by `tests/test_release.py`, which also opens a
store written as an older and as a newer version and checks the refusal.

## Release checklist

The workflow is `.github/workflows/release.yml`. It runs in four steps:

1. It checks the release with `scripts/release_version.py`: the commit is on `main`,
   the tag matches the version, the changelog section is dated, and the licence texts
   exist.
2. It calls the CI workflow. CI runs lint with every AGENTS.md command, then the tests
   and the wheels on all six platforms. Each wheel is installed in a clean environment
   on Python 3.10 and run through a small pipeline by `scripts/wheel_smoke.py`.
3. It builds the sdist.
4. It publishes CI's wheels and the sdist with
   [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/). Publishing
   uses GitHub's OIDC token, so no API token is stored anywhere.

### Once: set up publishing

1. On [PyPI](https://pypi.org/manage/account/publishing/), add a pending trusted
   publisher for the project `kyto`: owner `datavil`, repository `kyto`, workflow
   `release.yml`, environment `pypi`.
2. On [TestPyPI](https://test.pypi.org/manage/account/publishing/), the same with
   environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi`.
   Give `pypi` a required reviewer and limit it to `v*` tags, so a publish waits for
   a person.
4. Add the licence texts `LICENSE-MIT` and `LICENSE-APACHE` at the repo root and
   `license-files = ["LICENSE-*"]` under `[project]` in `pyproject.toml`: both
   licences require their text to ship with the wheels, and the release check
   refuses to publish without them.

### Each release

1. Pick the version by the table above. Set it in `Cargo.toml`
   (`[workspace.package] version`), then run `uv lock` and `cargo build` so
   `uv.lock` and `Cargo.lock` follow.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, check
   that it names the store format versions written and read, and update the table
   above if they changed. `uv run --no-project python scripts/release_version.py`
   prints the version, or what is missing.
3. Run the local gate (`uv run --no-project python scripts/agents_commands.py
   --check-clean`), commit as `release X.Y.Z`, and push `main`.
4. Dry run: `gh workflow run release.yml --ref main` (publishes to TestPyPI). When it
   passes, install from TestPyPI in a fresh environment and run the smoke test:

    ```bash
    uv venv --python 3.10 /path/to/scratch/env
    uv pip install --python /path/to/scratch/env --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match kyto==X.Y.Z
    /path/to/scratch/env/bin/python -I scripts/wheel_smoke.py
    ```

    TestPyPI keeps the first upload of a version: a repeated dry run of the same
    version builds and tests everything but uploads nothing new.
5. Tag and push: `git tag -a vX.Y.Z -m "kyto X.Y.Z"`, then
   `git push origin vX.Y.Z`. The workflow runs CI again on the tag and, after the
   `pypi` environment's reviewer approves, publishes.
6. On an Apple Silicon Mac, `pip install kyto==X.Y.Z` in a fresh environment and run
   `scripts/wheel_smoke.py` with it, once on its synthetic data and once on pbmc3k
   (`--matrix-dir .../filtered_gene_bc_matrices/hg19 --expect-cells 2638`). Then
   check that `pip install kyto==X.Y.Z` imports on Linux and Windows.
7. Open a new `## [Unreleased]` section in `CHANGELOG.md`.

A failed publish leaves PyPI unchanged; fix, bump PATCH, and release again. A version
on PyPI is never replaced: a broken release is yanked on PyPI and followed by a PATCH.
