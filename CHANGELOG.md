# Changelog

All notable changes to kyto are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and kyto uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) as stated in
[docs/release.md](docs/release.md): the version covers the Python API, the error codes
and the store `format_version` kyto reads. Every release section names the store
format versions it reads and writes.

A release turns `## [Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD` (the release
workflow refuses a tag without that section, `scripts/release_version.py`).

## [Unreleased]

The first release, 0.1.0. Store format: writes `format_version` "0.1", reads "0.1".
Wheels: abi3 for CPython 3.10 and later on Linux (x86_64, aarch64), macOS (x86_64,
arm64) and Windows (x86_64, ARM64); the GPU features are not in the wheels.

### Added

- Storage: the tiled sparse store format v0 (`docs/format.md`), a streaming writer,
  importers for `.h5ad`, 10x `.h5` and Matrix Market (`kyto.convert`), `.h5ad`
  export (`sink_h5ad`), and `kyto.concat` for samples. The Rust reader also reads
  stores over S3, GCS and HTTP; `kyto.scan` does not take URLs yet.
- A lazy plan: `kyto.scan` returns an `Atlas` whose chained methods build a typed,
  validated plan; `explain()`, `estimate()`, `schema()`, `peek()` and `features()`
  inspect it without reading the matrix; plans round-trip as JSON (`to_json`,
  `from_json`, `validate_plan`, `plan_json_schema`).
- A streaming CPU executor with a memory limit, deterministic reductions at any
  thread count, progress, and Ctrl-C.
- Analysis: QC metrics per cell and per gene, `normalize_total`, `log1p`,
  `highly_variable_genes`, `scale`, PCA (on a gene subset too), exact and
  approximate neighbours, connectivities, Leiden and Louvain, UMAP,
  `rank_genes_groups`, Scrublet, and `with_embedding` for scvi-tools and Harmony
  outputs. Results match scanpy on the reference fixtures within
  `tests/parity/tolerances.toml`; the differences are in `deviations.md`.
- Data hand-off: Polars frames, NumPy arrays, SciPy matrices, `to_anndata`, and
  DLPack (`to_torch`, `__dlpack__`), each docstring stating its copies.
- Coded errors (`KY-Exxx`, `docs/errors.md`) with a fix and close-match suggestions;
  provenance in every store; `llms.txt` for agents; typed stubs.
- GPU placement rules and kernels behind the opt-in `wgpu` and `cuda` features
  (build from source to use them).
