---
name: verify
description: How to run kyto for real (Python package on pbmc3k) to verify a change at its surface, not through the test suites.
---

# Verifying a kyto change

Tasks land straight on `main` and are pushed, so `git diff @{u}..` is empty: the scope
is the latest task commit (`git show --stat HEAD`).

## Handle

- Surface: the Python package (`import kyto`), run with `uv run --no-sync python script.py`
  from the repo root. The extension is current if no `crates/**/*.rs` is newer than
  `python/kyto/_kyto.abi3.so`; otherwise `uv sync --reinstall-package kyto`.
- Work in a scratch directory, never in the repo.
- Data: `tar -xzf .cache/fixtures/pbmc3k_filtered_gene_bc_matrices.tar.gz`, then
  `kyto.convert("<dir>/filtered_gene_bc_matrices/hg19", "pbmc3k.kyto")` (2700 cells, one
  4096-cell block). For several blocks, concatenate copies with anndata, write an
  `.h5ad` and `kyto.convert` it (12 copies: 8 blocks, converts in about 1 s).
- Until the public terminals exist (7.1 to 7.3), plans are built as JSON:
  `kyto._kyto.scan(store).to_json()`, append nodes (`kyto.plan_json_schema()` gives every
  step and expression shape; QC metrics need `"layer": "counts"`), `kyto.from_json(text)`,
  then `._execute(threads=..., progress=...)`. A sink is a last node
  `{"method": "sink_zarr" | "sink_h5ad", "path": ...}`.
- Reading results back: `Table.to_polars` and friends are `KY-E601` until 7.3. Read
  `schema.json`, `cells.parquet` and `genes.parquet` directly, and turn a store into an
  `.h5ad` with `kyto._kyto.sink_h5ad(store, target, x_layer=None, overwrite=False)` to
  read layers with anndata. Compare with scanpy run on the same input.

## Flows worth driving

- The reference pipeline: QC columns, cell filter (genes detected, MT percent), gene
  filter, `normalize_total`, `log1p`; pbmc3k keeps 2643 cells and 32 725 genes.
- Thread counts 1/2/8 give the same store bytes (hash the tree without
  `provenance.json`; `.h5ad` bytes differ per run, compare content).
- Cancel: raise from `progress`, and send a real SIGINT to a child process at several
  moments (wait for a line it prints before `_execute`, sleep, `send_signal`).

## Gotchas

- anndata 0.13 lists `X` under the layer key `None`; `sorted(adata.layers)` fails.
- A plan without a default layer writes an `.h5ad` without `X`.
