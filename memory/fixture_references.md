# Fixtures and scanpy references (task 1.5)

**Works:** fixtures are built in memory with numpy only and checked by a content
digest of canonical array values (`tests/fixtures/hashing.py`: little-endian widened
dtypes, one NaN bit pattern, length-prefixed UTF-8), never file bytes. The synthetic
generator uses only uniform/integer draws and + - * / (no `poisson`/`gamma`, whose
libm differs per OS), so its digest holds on all six CI runners.

**Breaks:** scanpy's `highly_variable_genes(flavor="cell_ranger")` raises "Bin edges
must be unique" on unfiltered pbmc3k (~19k all-zero genes). The pbmc3k references for
HVG and later steps therefore run on genes seen in >= 3 cells (scanpy tutorial); the
synthetic ones keep empty genes. Parity tests must apply the same gene filter
(`min_cells_per_gene` in `manifest.json`).

**Platforms:** scanpy/anndata/scikit-misc are dev deps only where wheels exist (not
Windows ARM64, not macOS x86_64); tests needing them skip there. Fixture digests and
reference integrity run everywhere. Upgrading scanpy fails
`test_reference_versions_match_installed_packages` until references are regenerated:
`uv run --directory tests python -m fixtures.make_reference`.

**Pyright:** numpy >= 2.3 stubs only type-check at Python >= 3.12, so
`[tool.pyright] pythonVersion` is 3.13 (user decision, 1.5); ruff keeps the 3.10 floor.
