# Interop hooks: scvi-tools and Harmony against scanpy (task 6.11)

**Works:**
- `with_embedding` is a cross-cell plan node holding its values (`GivenValues`, behind
  an `Arc`); the plan JSON writes them as numbers (`float32` widened to the exact
  `float64`, read back with `as f32`: exact, no double rounding). Finite values only,
  since the plan JSON writes only finite floats.
- The executor needs no new pass: rows are keyed to store positions `0..n_cells`, or
  to the cells reaching the node (`cells_pass`, one pass) after a cell filter or
  `take_cells`; then it is an `EmbeddingSource` like `pca`'s (statistics keyed by node).
- scvi-tools on kyto's `to_anndata()` (counts `uint32`) equals scvi-tools on scanpy's
  AnnData (`float32`) bit for bit at seed 0 on the CPU: same cells, genes, counts.

**Breaks:**
- scanpy 1.12.5's `sc.external.pp.harmony_integrate` stores `Z_corr.T`; harmonypy 2.x
  already returns cells x components, so it raises a shape error. Call
  `harmonypy.run_harmony` on `obsm["X_pca"]` directly (orientation-agnostic).
- Harmony amplifies kyto's trailing-PC differences from scanpy (leading 10 PCs agree to
  5e-5, PCs 20-30 to 1.6e-3): Harmony outputs differ by a median 1.1e-3 per cell, worst
  7.5e-2, kNN overlap 0.991. scanpy's PCA with another seed moves Harmony only 6e-4, and
  iteration counts are equal, so it is the kind of PCA difference, not Harmony noise.
  Compare by median row error and kNN overlap (`interop.*`), not element-wise.
- A `Protocol` in `crates/kyto-py/stubs/expressions.pyi` (`_SupportsDLPack`) has no
  runtime object: `tests/test_stubs.py` lists `Protocol` as a type-only base.
- The `interop` dependency group (PyTorch) is not a default group: the gate's `uv sync`
  removes it; `uv run --group interop pytest tests/test_interop.py` runs the parity
  tests. Notebook imports of scvi/harmonypy carry `pyright: ignore[reportMissingImports]`.
