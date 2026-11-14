# kyto — Implementation Plan

Nov 6, 2026 · @Zafer Kosar

## 0. How to use this plan

Ticked tasks live in [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md), grouped by workstream, with their notes. When you tick a task, move it there (under its workstream, in ID order) and add its ID to that workstream's "Done" line here. A `Depends` entry is satisfied when the task is in the Done file.

kyto is built in 10 workstreams of numbered tasks. An agent takes one unblocked task, finishes it until its **Done when** check passes, then ticks it.

**Task protocol**

- IDs such as `3.4` never change. New tasks take the next free number; dropped tasks stay, struck through.
- Start a task only when every task under its **Depends** is ticked.
- One task = one commit on `main` (no task branches or PRs; decided by the user, Nov 6, 2026). Commit messages start with the ID: `[3.4] feature resolver`.
- A task is done when its check and the local gate (AGENTS.md "Run the whole list") pass, not before. CI runs only on demand and on release tags, not on every push (decided by the user, Nov 7, 2026).
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

Done (1.1, 1.2, 1.3, 1.4, 1.5, 1.6): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#1-workspace-and-tooling).

## 2. Storage (`kyto-store`)

Goal: a Zarr v3 store where reading a few genes costs about the same as reading a few cells.

Done (2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#2-storage-kyto-store).

## 3. Data model (`kyto-core`)

Goal: every name and type is checked when the plan is built, before any data is read.

Done (3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#3-data-model-kyto-core).

## 4. Lazy plan and optimizer (`kyto-plan`)

Goal: chained methods build a DAG that the optimizer rewrites into the fewest passes over the data.

Done (4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#4-lazy-plan-and-optimizer-kyto-plan).

## 5. Streaming CPU executor (`kyto-exec`)

Goal: any plan runs within a memory limit, uses every core, and gives the same answer on 1 or 64 threads.

Done (5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#5-streaming-cpu-executor-kyto-exec).

## 6. Algorithms (`kyto-algo`)

Goal: the standard scanpy workflow, streamed, with each step checked against scanpy on the 1.5 fixtures.

Done (6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.12): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#6-algorithms-kyto-algo).

- [ ] **6.11 Interop hooks.** Export embeddings and matrices to scvi-tools and Harmony through DLPack and NumPy; import their outputs with `with_embedding`.
  - Depends: 9.5, 7.3
  - Done when: a notebook example runs scvi on a kyto atlas and stores the latent space back.
- [ ] **6.13 Per-gene QC metrics.** The rest of scanpy's `calculate_qc_metrics`: per-gene `n_cells_by_counts`, `total_counts`, `mean_counts` and `pct_dropout_by_counts` as gene expressions (a reduction over the plan's cells, 5.6), so `filter(<cells detected> >= 3)` is scanpy's `filter_genes(min_cells=3)` inside a plan; and `log1p` on numeric expressions for the `log1p_` QC columns (6.1 Open (2)). (proposed by the agent in 10.7, Nov 14, 2026: the clustering tutorial's gene filter had to be computed outside the plan)
  - Depends: 6.1, 5.6, 5.8
  - Done when: the per-gene columns and the `log1p_` cell columns equal `calculate_qc_metrics(log1p=True)` on the fixtures (dtypes as deviations.md "QC metric dtypes"), and `docs/tutorials/clustering.py` filters genes with a plan filter instead of its NumPy stand-in.
- [ ] **6.14 PCA on a gene subset.** `pca(..., genes=<GeneExpr>)`, scanpy's `mask_var`: the components are computed over the genes where the predicate holds (`gene_column("highly_variable")`) while the plan keeps every gene, so marker plots and `rank_genes_groups` after clustering see all genes in one plan. (proposed by the agent in 10.7, Nov 14, 2026: the clustering tutorial clusters on a branch filtered to the HVGs and ranks only those genes)
  - Depends: 6.5, 7.2
  - Done when: `pca(genes=gene_column("highly_variable"))` gives the same embedding as `filter(gene_column("highly_variable")).pca(...)` bit for bit, and the clustering tutorial runs as one plan with its Wilcoxon ranking over every kept gene compared with scanpy's.

## 7. Python API (`kyto-py`, `python/kyto`)

Goal: a small, chainable surface where every method has one return type and a typed stub.

Done (7.1, 7.2, 7.3, 7.4, 7.5, 7.8): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#7-python-api-kyto-py-pythonkyto).

- [ ] **7.6 Stubs.** `.pyi` files generated with `pyo3-stub-gen`, NumPy docstrings, `@overload` on `filter` for `CellExpr` and `GeneExpr`.
  - Depends: 7.2, 7.3
  - Done when: `pyright --strict` passes on all doc examples; CI fails if stubs drift from bindings.
- [ ] **7.7 Repr.** One line, never triggers compute: `<Atlas plan: 7 nodes · cells: ? · genes: 2000 · layers: counts, log1p>`.
  - Depends: 4.6
  - Done when: a test asserts `repr()` performs zero I/O.

## 8. Agent-friendliness

Goal: an agent can inspect data, write a pipeline, check it and fix its own mistakes without reading the matrix or the source code.

Done (8.1, 8.2, 8.4): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#8-agent-friendliness).

- [ ] **8.3 Inspection without reading.** `schema()`, `peek(n)`, `features()`, `estimate()`, `explain()` all read metadata only.
  - Depends: 4.6, 4.7
  - Done when: the I/O counter shows zero matrix-tile reads for each call.
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
  - Note (7.3): `Embedding.to_torch(device=)` exists and copies through NumPy (`torch.from_numpy(array).to(device)`); no `__dlpack__` and no `Matrix.to_torch` yet.
- [ ] **9.6 Fallback rules.** `device="cuda"` with a missing kernel raises `KY-E501`; `device="auto"` falls back to CPU and says so in `explain()`. Never silent.
  - Depends: 9.1, 8.1
  - Done when: both cases are tested.
- [ ] **9.7 GPU CI.** GPU tests on a self-hosted runner; skipped with a visible notice elsewhere.
  - Depends: 1.4, 9.2
  - Done when: the GPU job runs on every PR that touches `kyto-gpu` or `kyto-exec`.
- [ ] **9.8 Portable matrix kernels (Metal).** PCA's products (sparse × dense SpMM and dense GEMM) and 9.4's tiled distances plus top-k written in CubeCL, so PCA and neighbours run through wgpu on Metal (and ROCm) as well as CUDA. 9.3's cuSPARSE/cuBLAS stays the CUDA path; `device="auto"` picks it on NVIDIA and these kernels elsewhere, by the 9.1 rules shown in `explain()`. (added Nov 14, 2026, at the user's request: Apple GPUs otherwise run only the 9.2 kernels)
  - Depends: 9.2, 9.3, 9.4
  - Done when: on a Metal device, PCA matches CPU PCA within the 6.5 tolerances and neighbours equal CPU exact mode.

## 10. Testing, benchmarks, docs, release

Goal: correctness and speed are measured on every PR, and 1.0 ships as wheels with a migration guide. 10.1–10.3 start with M1 and grow with each milestone.

Done (10.7): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#10-testing-benchmarks-docs-release).

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
| M3 Full analysis | 2.9, 6.4–6.10, 6.12, 7.8, 8.4–8.7, 10.4, 10.7 |
| M4 GPU backend | 9.1–9.8 |
| M5 1.0 release | 2.7–2.8, 6.11, 10.5–10.6 |
