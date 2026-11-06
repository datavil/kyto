# kyto — Implementation Plan

Nov 6, 2026 · @Zafer Kosar

## 0. How to use this plan

kyto is built in 10 workstreams of numbered tasks. An agent takes one unblocked task, finishes it until its **Done when** check passes, then ticks it.

**Task protocol**

- IDs such as `3.4` never change. New tasks take the next free number; dropped tasks stay, struck through.
- Start a task only when every task under its **Depends** is ticked.
- One task = one commit on `main` (no task branches or PRs; decided by the user, Nov 6, 2026). Commit messages start with the ID: `[3.4] feature resolver`.
- A task is done when its check passes in CI, not before.
- Reviews cite a rule by its number (`R5`). If a task is unclear, add an open question under it instead of guessing.

**Design rules** (cite as R1–R12)

- **R1 Lazy by default.** Only `collect`, `sink_*`, `pull`, `select` and `to_*` read the matrix.
- **R2 Immutable, no views.** Every method returns a new plan; every `to_*` returns a copy the caller owns.
- **R3 One return type per function.** No `Union` returns; one gene still comes back as a `pl.DataFrame`.
- **R4 Named axes.** `cells` and `genes`; no `__getitem__`, no `.X`, no `obs`/`var`.
- **R5 Typed expressions.** `CellExpr` and `GeneExpr` never mix; mixing them fails when the plan is built.
- **R6 Names fixed at plan build.** A short name is resolved once and stored in full (`log1p:CD3E`).
- **R7 No hidden state.** Layer, seed, device and memory limit are arguments; there are no global settings.
- **R8 No implicit dense arrays.** Dense output only through `to_dense(max_bytes=...)`.
- **R9 Zeros are not nulls.** An absent sparse value is 0, missing metadata is null, a removed gene raises an error.
- **R10 Coded errors.** Every error has a stable `KY-Exxx` code and a suggested fix.
- **R11 Deterministic.** The same input, plan and seed give identical output on any number of threads.
- **R12 Documented surface.** Every public item has a NumPy docstring, a `.pyi` stub and a runnable example.

**Code style**

- Rust: `rustfmt`, `clippy -D warnings`. `unsafe` only in `kyto-exec` kernels and `kyto-gpu`, each block with a `// SAFETY:` note.
- Python: `ruff format`, `ruff check` with import sorting, NumPy docstrings, full names (`cell_index`, not `cell_idx`), Polars frames named `frame`.

**Repo map**

| Path | Holds |
| --- | --- |
| `crates/kyto-error` | `KytoError`, generated from `errors.toml` (8.1) |
| `crates/kyto-store` | Zarr layout, tile I/O, importers |
| `crates/kyto-core` | Schema, expressions, name resolver, handles |
| `crates/kyto-plan` | Logical plan, optimizer, `explain`, `estimate` |
| `crates/kyto-exec` | Streaming CPU executor, SIMD kernels |
| `crates/kyto-algo` | QC, HVG, PCA, kNN, clustering, DE |
| `crates/kyto-gpu` | CubeCL kernels, cudarc bindings |
| `crates/kyto-py` | PyO3 bindings |
| `python/kyto` | Python package and `.pyi` stubs |

## 1. Workspace and tooling

Goal: a repo where `import kyto` works, CI is green on Linux, macOS and Windows, and an agent can learn the rules from one file.

- [x] **1.1 Cargo workspace.** Create the crates in the repo map as a virtual workspace with shared lint settings.
  - Depends: none
  - Done when: `cargo build --workspace` and `cargo test --workspace` pass.
- [x] **1.2 Python package.** `kyto-py` as a PyO3 extension (`abi3-py310`), maturin as build backend, mixed Rust/Python layout, dev environment managed with uv.
  - Depends: 1.1
  - Done when: `uv run maturin develop` then `uv run python -c "import kyto; print(kyto.__version__)"` succeeds.
- [x] **1.3 Lint and format.** `rustfmt`, `clippy -D warnings`, `ruff format`, `ruff check` (import sorting on), `pyright --strict` on stubs, all in pre-commit.
  - Depends: 1.2
  - Done when: `pre-commit run --all-files` passes on a clean checkout.
- [x] **1.4 CI.** GitHub Actions matrix: Linux, macOS, Windows on x86\_64 and aarch64; wheels built with `maturin-action`; GPU job reserved for a self-hosted runner (9.7).
  - Depends: 1.3
  - Done when: a PR runs lint, tests and wheel builds, all green.
  - Note: kyto commits straight to `main`, so `.github/workflows/ci.yml` runs on push to `main` as well as on `pull_request`; a green push to `main` satisfies this check.
- [x] **1.5 Test fixtures.** A seeded synthetic generator (cells, genes, sparsity, batches) plus a cached public PBMC dataset; reference outputs from scanpy stored for parity tests.
  - Depends: 1.2
  - Done when: `uv run pytest tests/fixtures` builds every fixture and checks its hash.
- [x] **1.6 AGENTS.md.** Copies R1–R12, the task protocol, the repo map and the exact commands for build, test, lint and bench. `CLAUDE.md` links to it.
  - Depends: 1.4
  - Done when: CI fails if a command listed in AGENTS.md fails.

## 2. Storage (`kyto-store`)

Goal: a Zarr v3 store where reading a few genes costs about the same as reading a few cells.

- [x] **2.1 Format spec v0.** Write `docs/format.md`: `/cells` and `/genes` as Parquet, `/layers/<name>` as tiled sparse arrays, `/embeddings`, `/graphs`, `/schema.json`, `/provenance.json`, and a `format_version` field.
  - Depends: 1.1
  - Done when: the spec is merged and 2.2–2.5 cite its section numbers.
- [x] **2.2 Tiled sparse matrix.** Tiles of cell block × gene block, CSR inside each tile, Zarr sharding codec with zstd. Tile sizes are parameters; defaults come from 10.4.
  - Depends: 2.1
  - Done when: a `proptest` round trip (random shape, sparsity, dtype) is exact.
  - Spec: `docs/format.md` §4.1-§4.6 (tiling, arrays, CSR inside tile, chunks and codecs, limits, zeros), §2.1 (names).
  - Provisional writer defaults until 10.4: `C` = 4096, `G` = 1024, `K` = 65536, `S` = 1048576 entries, zstd level 3 (`kyto_store::Tiling::default`, `LayerWriteOptions::default`).
- [ ] **2.3 Metadata tables.** Cell and gene tables via Polars Parquet; IDs unique and checked on write; strings stored as categoricals.
  - Depends: 2.1
  - Done when: writing duplicate IDs raises `KY-E101`; round trip keeps dtypes.
  - Spec: `docs/format.md` §5 (tables, IDs, dtypes, Enum, nulls, names, Parquet settings), §8 (`schema.json` columns).
- [ ] **2.4 Tile reader.** Parallel reads that skip tiles outside the selected genes (projection) and cells (predicate); yields a stream of `CellBlock`s (5.1).
  - Depends: 2.2
  - Done when: a test counter shows that selecting 3 genes reads one gene block per cell block, nothing more.
  - Spec: `docs/format.md` §4.7 (reading a tile), §4.8 (tiles to CellBlock), §3 and §11 (version check on open).
- [ ] **2.5 Streaming writer.** Writes `CellBlock`s as they arrive; writes to a temporary path and renames on success, so a failed run leaves no half store.
  - Depends: 2.2, 2.3
  - Done when: killing a write midway leaves the target path absent.
  - Spec: `docs/format.md` §10 (atomic write, write order, determinism), §8, §9, §3 (root `zarr.json` last).
- [ ] **2.6 Importers.** `ky.convert()` from `.h5ad` (dense, CSR, CSC; streamed through the `hdf5` crate), 10x `.h5`, and `.mtx` + TSV. `on_duplicate_ids="error"` by default, `"suffix"` opt-in.
  - Depends: 2.5
  - Done when: every fixture converts and matches anndata's values exactly.
- [ ] **2.7 h5ad export.** `sink_h5ad()` for handing results to scanpy.
  - Depends: 2.6
  - Done when: `anndata.read_h5ad` on the export equals the source fixture.
- [ ] **2.8 Remote stores.** S3, GCS and HTTP through `object_store`, with the same reader as 2.4.
  - Depends: 2.4
  - Done when: the reader test suite passes against a local MinIO container.

## 3. Data model (`kyto-core`)

Goal: every name and type is checked when the plan is built, before any data is read.

- [ ] **3.1 Schema types.** `AtlasSchema`: cell and gene table schemas, layers (name, dtype), embeddings (name, width), graphs, modalities. Serializable with serde.
  - Depends: 2.1
  - Done when: schema read from a store equals the schema written; JSON round trip is exact.
- [ ] **3.2 Atlas.** Holds `Arc<Plan>`, exposed as a `#[pyclass(frozen)]`. Every method returns a new `Atlas` (R2).
  - Depends: 3.1
  - Done when: a test shows the original plan is unchanged after 10 chained calls.
- [ ] **3.3 Typed expressions.** `CellExpr` and `GeneExpr` with a dtype and a source tag: `Column`, `Expression { layer, gene_id }`, `Profile { layer, cell_id }`, `Embedding { name, component }`. Metadata-only parts lower to Polars `Expr`.
  - Depends: 3.1
  - Done when: `cell_column(...) & gene_column(...)` raises `KY-E202` at build; `.str` on a numeric expression fails in pyright.
- [ ] **3.4 Constructors.** `cell_column`, `gene_column`, `gene_expression`, `cell_profile`, `feature`.
  - Depends: 3.3
  - Done when: each has a stub, docstring and doctest (R12).
- [ ] **3.5 Feature resolver.** Maps short names (`CD3E`, `cluster`) to full names (`log1p:CD3E`, `meta:cluster`). No priority order: two matches raise `KY-E201` listing both. Short gene names need `default_layer`. Resolution is stored in the plan (R6).
  - Depends: 3.4
  - Done when: adding a column named `CD3E` after a plan is built does not change that plan's output.
- [ ] **3.6 Handles.** `Table`, `Matrix`, `Embedding`, `Graph`, `Modality`: read-only, lazy, each with `names()`, `schema()`, `estimate()` and its fixed `to_*` terminals.
  - Depends: 3.2
  - Done when: no handle exposes a method that mutates; checked by a stub test.
- [ ] **3.7 Missing-value rules.** Sparse absence densifies to 0; missing metadata is null; reading a gene removed by a filter raises `KY-E203` (R9).
  - Depends: 3.5
  - Done when: tests cover all three cases.
- [ ] **3.8 Derived-column lineage.** Stored columns record the layer and plan node they came from. Reading one after an upstream gene filter raises `KY-E301` unless `.recompute()` is called.
  - Depends: 3.5
  - Done when: `n_genes` read after `filter_genes` raises `KY-E301`.
- [ ] **3.9 Gene identity.** Genes indexed by ID. `by="symbol"` must match exactly one gene or raise `KY-E204` with the candidates.
  - Depends: 3.5
  - Done when: a fixture with duplicate symbols triggers `KY-E204`.

## 4. Lazy plan and optimizer (`kyto-plan`)

Goal: chained methods build a DAG that the optimizer rewrites into the fewest passes over the data.

- [ ] **4.1 Logical nodes.** `Scan`, `Filter`, `SelectGenes`, `TakeCells`, `MapElementwise`, `CrossCell` (scale, HVG, PCA), `WithColumns`, `Neighbors`, `Cluster`, `Sink`. Each declares which axis it changes and whether it is per-cell or cross-cell.
  - Depends: 3.3
  - Done when: every node has an output-schema function with unit tests.
- [ ] **4.2 Build-time validation.** Schema propagation through the DAG; name, type and axis errors raised here (R5, R6).
  - Depends: 4.1, 3.5
  - Done when: every `KY-E2xx` error fires from a plan that has read no data (I/O counter = 0).
- [ ] **4.3 Plan JSON.** Versioned serde format with stable node IDs; `Atlas.to_json()` and `ky.from_json()`.
  - Depends: 4.1
  - Done when: round trip is exact for every plan in the test suite.
- [ ] **4.4 Optimizer rules.** Predicate pushdown that stops at cross-cell nodes; projection pushdown; elementwise fusion (`normalize_total → log1p` in one pass); common sub-plan reuse.
  - Depends: 4.2
  - Done when: each rule has before/after plan snapshot tests, and optimized results equal unoptimized results on fixtures.
- [ ] **4.5 Physical plan.** Chooses device, block sizes and number of passes for each node.
  - Depends: 4.4
  - Done when: `explain()` shows device and pass count for every node.
- [ ] **4.6 Cost estimate.** Bytes read, peak memory, passes and device, without reading the matrix.
  - Depends: 4.5
  - Done when: on fixtures, estimated peak memory is within 25% of measured.
- [ ] **4.7 Explain.** `explain(format="text")` prints logical and optimized trees with resolved names; `format="json"` returns the same as a dict.
  - Depends: 4.5
  - Done when: snapshot tests cover text and JSON for 5 reference pipelines.

## 5. Streaming CPU executor (`kyto-exec`)

Goal: any plan runs within a memory limit, uses every core, and gives the same answer on 1 or 64 threads.

- [ ] **5.1 CellBlock.** A batch of CSR rows: `indptr: u64`, `indices: u32`, `data: f32 | u32`, 64-byte aligned, from a reusable buffer pool.
  - Depends: 2.4
  - Done when: allocation count stays flat across a 100-block stream (measured with a counting allocator).
- [ ] **5.2 Pipeline executor.** Pull-based stream through fused operators, rayon over blocks, bounded channels for backpressure.
  - Depends: 5.1, 4.5
  - Done when: CPU use scales with threads on a fixture pipeline (bench in 10.3).
- [ ] **5.3 SIMD kernels.** Row sums, row nnz, scaling, `log1p`, squares, via `pulp` with runtime dispatch (AVX2, AVX-512, NEON) and a scalar fallback.
  - Depends: 5.1
  - Done when: every kernel equals its scalar version on `proptest` inputs; criterion benches recorded.
- [ ] **5.4 Multi-pass operators.** Cross-cell nodes run a statistics pass, then an apply pass; the physical plan says how many.
  - Depends: 5.2
  - Done when: HVG on fixtures needs exactly the passes `explain()` reports.
- [ ] **5.5 Memory limit.** `collect(memory_limit="4GB")` shrinks blocks to fit; impossible plans raise `KY-E401` before running.
  - Depends: 5.2, 4.6
  - Done when: peak RSS stays under the limit on a dataset 4× larger than the limit.
- [ ] **5.6 Deterministic reductions.** Tree reduction in block order, never in thread-finish order (R11).
  - Depends: 5.2
  - Done when: outputs are bit-identical at 1, 4 and all threads.
- [ ] **5.7 GIL, progress, cancel.** Release the GIL during compute; optional progress callback; Ctrl-C cancels cleanly.
  - Depends: 5.2, 1.2
  - Done when: a Python test cancels a run and the process stays usable.

## 6. Algorithms (`kyto-algo`)

Goal: the standard scanpy workflow, streamed, with each step checked against scanpy on the 1.5 fixtures.

- [ ] **6.1 QC metrics.** Total counts, genes detected, percent of counts in a gene set (e.g. a mitochondrial pattern). Computed on read as expressions, so they never go stale.
  - Depends: 5.3, 3.8
  - Done when: values match `scanpy.pp.calculate_qc_metrics` to 1e-6 relative.
- [ ] **6.2 Normalize and log.** `normalize_total(target=, into=)` and `log1p(into=)`, fused into one pass by 4.4.
  - Depends: 5.3, 4.4
  - Done when: matches scanpy to 1e-6; `explain()` shows one fused pass.
- [ ] **6.3 Highly variable genes.** `seurat`, `cell_ranger` and `seurat_v3` flavours, optional `batch_key`, streaming moments (Welford). `seurat_v3` requires the `counts` layer.
  - Depends: 5.4
  - Done when: the selected gene sets equal scanpy's on fixtures.
- [ ] **6.4 Scale.** Z-score with optional clipping. Flagged as densifying; `explain()` suggests implicit centering in PCA instead.
  - Depends: 5.4
  - Done when: matches scanpy; `estimate()` reports the dense size.
- [ ] **6.5 PCA.** Streaming randomized SVD (Halko) with implicit centering on sparse data; faer for the small dense steps; `seed` required.
  - Depends: 5.4
  - Done when: explained variance matches scanpy to 1e-3 and subspace angles are small on fixtures.
- [ ] **6.6 Nearest neighbours.** Exact blocked search with SIMD distances, and HNSW (`hnsw_rs`) for approximate search. Output is a CSR graph.
  - Depends: 6.5
  - Done when: exact mode equals brute force; HNSW recall is reported in the test log.
- [ ] **6.7 Connectivities.** UMAP-style fuzzy graph weights, matching scanpy's `connectivities`.
  - Depends: 6.6
  - Done when: weights match scanpy to 1e-5 given the same neighbours.
- [ ] **6.8 Leiden and Louvain.** Rust implementation, `seed` required, resolution parameter.
  - Depends: 6.7
  - Done when: adjusted Rand index vs `leidenalg` is reported, and results are identical across runs with one seed.
- [ ] **6.9 UMAP layout.** SGD layout on the connectivities graph, `seed` required.
  - Depends: 6.7
  - Done when: trustworthiness score is within range of umap-learn on fixtures.
- [ ] **6.10 Differential expression.** Wilcoxon and t-test per group, streamed; returns one `pl.DataFrame` (group, gene, statistic, log fold change, p, adjusted p).
  - Depends: 5.4
  - Done when: matches `scanpy.tl.rank_genes_groups` to 1e-6.
- [ ] **6.11 Interop hooks.** Export embeddings and matrices to scvi-tools and Harmony through DLPack and NumPy; import their outputs with `with_embedding`.
  - Depends: 9.5, 7.3
  - Done when: a notebook example runs scvi on a kyto atlas and stores the latent space back.

## 7. Python API (`kyto-py`, `python/kyto`)

Goal: a small, chainable surface where every method has one return type and a typed stub.

- [ ] **7.1 Module layout.** `kyto` exports `scan`, `convert`, `from_json`, the expression constructors (3.4) and the exception classes (8.1); nothing else at top level.
  - Depends: 1.2
  - Done when: `kyto.__all__` matches a checked-in list.
- [ ] **7.2 Chain methods.** `filter`, `select_genes`, `take_cells`, `with_cells`, `with_genes`, `with_embedding`, `with_default_layer`, the algorithm methods from 6.x (each with `into=`), `on(device=)`, `collect()`, `sink_zarr()`.
  - Depends: 4.3, 3.2
  - Done when: the README pipeline runs end to end on a fixture.
- [ ] **7.3 Accessors and terminals.** `cells`, `genes`, `layer()`, `embedding()`, `graph()`, `modality()` return handles (3.6). Terminals: `to_polars`, `lazy`, `to_numpy`, `to_scipy`, `to_dense(max_bytes=)`, `to_torch(device=)`, `to_edges`.
  - Depends: 3.6, 5.2
  - Done when: each terminal's return type is asserted in tests (R3).
- [ ] **7.4 Feature access.** `pull` → `pl.Series`, `select` → `pl.DataFrame` with `cell_id`, `pull_genes`, `features()`, `cell_ids()`, `gene_ids()`, `count_cells()`, `count_genes()`, `metadata()`.
  - Depends: 3.5, 7.3
  - Done when: the AnnData migration table (10.5) has a passing test per row.
- [ ] **7.5 Data hand-off.** Polars via `pyo3-polars`, NumPy via the `numpy` crate, GPU tensors via DLPack. Copy behaviour stated in every terminal's docstring (R2).
  - Depends: 7.3
  - Done when: mutating a returned array never changes a later `pull` (test).
- [ ] **7.6 Stubs.** `.pyi` files generated with `pyo3-stub-gen`, NumPy docstrings, `@overload` on `filter` for `CellExpr` and `GeneExpr`.
  - Depends: 7.2, 7.3
  - Done when: `pyright --strict` passes on all doc examples; CI fails if stubs drift from bindings.
- [ ] **7.7 Repr.** One line, never triggers compute: `<Atlas plan: 7 nodes · cells: ? · genes: 2000 · layers: counts, log1p>`.
  - Depends: 4.6
  - Done when: a test asserts `repr()` performs zero I/O.

## 8. Agent-friendliness

Goal: an agent can inspect data, write a pipeline, check it and fix its own mistakes without reading the matrix or the source code.

- [x] **8.1 Error registry.** `errors.toml` lists code, message template, fix hint and docs link; a build script generates the Rust enum and the Python classes (`KytoError` → `ResolutionError`, `AxisError`, `MemoryLimitError`, `StaleColumnError`, ...).
  - Depends: 1.1
  - Done when: CI fails if any error is raised without a registered code (R10).
- [ ] **8.2 Suggestions.** Closest matches for unknown genes, columns and layers (Jaro-Winkler, top 3) in every `KY-E1xx`/`KY-E2xx` message.
  - Depends: 8.1, 3.5
  - Done when: `pull("CD3")` suggests `CD3E`, `CD3D`, `CD3G` on the PBMC fixture.
- [ ] **8.3 Inspection without reading.** `schema()`, `peek(n)`, `features()`, `estimate()`, `explain()` all read metadata only.
  - Depends: 4.6, 4.7
  - Done when: the I/O counter shows zero matrix-tile reads for each call.
- [ ] **8.4 Plans as JSON.** Publish a JSON Schema for plans; `ky.validate_plan(json)` returns a list of diagnostics (code, path, hint) instead of raising.
  - Depends: 4.3, 8.1
  - Done when: 20 deliberately broken plans each return the expected diagnostic.
- [ ] **8.5 Provenance.** Every store written by kyto holds the plan JSON, kyto and dependency versions, seeds, device and timings.
  - Depends: 2.5, 4.3
  - Done when: `ky.scan(path).provenance()` rebuilds a plan that reproduces the store.
- [ ] **8.6 Machine-readable docs.** `llms.txt`, one runnable example per method, doctests in CI.
  - Depends: 7.6
  - Done when: every public method appears in `llms.txt` and its example passes.
- [ ] **8.7 Reproducibility test.** Same input, plan and seed give a byte-identical store (R11).
  - Depends: 8.5, 5.6
  - Done when: CI writes a reference pipeline twice and compares checksums.

## 9. GPU backend (`kyto-gpu`)

Goal: the same plan runs on a GPU by changing `on(device=...)`, with results matching the CPU.

- [ ] **9.1 Device trait.** One `Device` interface with CPU, CUDA and wgpu/Metal implementations. `device="auto"` picks by fixed, documented rules shown in `explain()`.
  - Depends: 5.2
  - Done when: `explain()` names the device per node on a machine with and without a GPU.
- [ ] **9.2 CubeCL kernels.** The 5.3 kernels (row sums, nnz, scaling, `log1p`) written once in CubeCL for CUDA, ROCm and wgpu.
  - Depends: 9.1, 5.3
  - Done when: each kernel matches CPU output to 1e-6 relative.
- [ ] **9.3 CUDA libraries.** cuSPARSE SpMM and cuBLAS GEMM through `cudarc` for PCA; pinned buffers and double-buffered host-to-device copies so reads overlap compute.
  - Depends: 9.1, 6.5
  - Done when: GPU PCA matches CPU PCA within the 6.5 tolerances.
- [ ] **9.4 GPU neighbours.** Tiled brute-force distance via GEMM plus top-k selection.
  - Depends: 9.3, 6.6
  - Done when: results equal CPU exact mode.
- [ ] **9.5 DLPack export.** `to_torch(device=)` and `__dlpack__` on embeddings and dense matrices, copy-free on the same device.
  - Depends: 7.5, 9.1
  - Done when: a torch tensor from kyto shares the device pointer (test).
- [ ] **9.6 Fallback rules.** `device="cuda"` with a missing kernel raises `KY-E501`; `device="auto"` falls back to CPU and says so in `explain()`. Never silent.
  - Depends: 9.1, 8.1
  - Done when: both cases are tested.
- [ ] **9.7 GPU CI.** GPU tests on a self-hosted runner; skipped with a visible notice elsewhere.
  - Depends: 1.4, 9.2
  - Done when: the GPU job runs on every PR that touches `kyto-gpu` or `kyto-exec`.

## 10. Testing, benchmarks, docs, release

Goal: correctness and speed are measured on every PR, and 1.0 ships as wheels with a migration guide. 10.1–10.3 start with M1 and grow with each milestone.

- [ ] **10.1 Property tests.** `proptest` for storage and kernels; `hypothesis` for the Python API (random chains of valid methods must not crash).
  - Depends: 2.2, 7.2
  - Done when: both suites run in CI with fixed seeds.
- [ ] **10.2 Parity suite.** One test per algorithm against scanpy and anndata on the 1.5 fixtures, with tolerances kept in one table.
  - Depends: 1.5
  - Done when: every 6.x task links its parity test.
- [ ] **10.3 Benchmarks.** `criterion` for Rust and `pytest-benchmark` for Python, against scanpy and rapids-singlecell at increasing dataset sizes, tracking time and peak RSS.
  - Depends: 5.2
  - Done when: CI posts a benchmark table on PRs and flags regressions.
- [ ] **10.4 Tile size study.** Benchmark cell-block × gene-block sizes for cell scans, gene lookups and PCA; set the 2.2 defaults from the results.
  - Depends: 2.4, 10.3
  - Done when: the chosen defaults and the benchmark table are in `docs/format.md`.
- [ ] **10.5 Docs site.** mkdocs-material with mkdocstrings (NumPy style), a design-rules page (R1–R12), an error-code page from 8.1, and the AnnData → kyto migration table.
  - Depends: 7.6, 8.1
  - Done when: docs build in CI with zero warnings.
- [ ] **10.6 Release.** abi3 wheels via maturin, PyPI trusted publishing, semantic versioning, and a stated policy for reading older `format_version` stores.
  - Depends: 1.4, 10.5
  - Done when: `pip install kyto` works on Linux, macOS and Windows from a tagged release.

## 11. Milestones

A milestone closes when every task in it is ticked and the exit check in its box passes.

&#91;embedded content: milestone order · 6 milestones, exit check in each box\]

M4 can start 9.1 and 9.2 right after M2, but 9.3 and 9.4 wait for CPU PCA and neighbours (6.5, 6.6), shown by the dashed arrow.

| Milestone | Tasks |
| --- | --- |
| M0 Skeleton | 1.1–1.6, 8.1 |
| M1 Read and inspect | 2.1–2.6, 3.1–3.9, 4.1–4.7, 5.1–5.2, 7.1, 7.3–7.5, 7.7, 8.2–8.3 |
| M2 CPU pipeline | 5.3–5.7, 6.1–6.3, 7.2, 7.6, 10.1–10.3 |
| M3 Full analysis | 6.4–6.10, 8.4–8.7, 10.4 |
| M4 GPU backend | 9.1–9.7 |
| M5 1.0 release | 2.7–2.8, 6.11, 10.5–10.6 |
