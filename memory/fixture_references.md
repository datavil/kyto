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

**Breaks (CI, 1.5):** scanpy's own output is not reproducible across runners. Rerunning
the synthetic references on Linux x86_64/aarch64, macOS arm64 and Windows x86_64 gave
PCA differences up to 1e-3 (arpack, nearly flat spectrum after the ~3 cell-type
components), which then flip kNN neighbours, one HVG `highly_variable_nbatches` value
and `rank_genes_groups` order at ties. qc, normalize/log1p and scale reproduce. So the
stored `.npz` files are the reference (never regenerate per platform), and parity for
6.3/6.5-6.10 must compare robustly: PCA on leading components or subspace angles, kNN
as set overlap, ranks allowing ties.
