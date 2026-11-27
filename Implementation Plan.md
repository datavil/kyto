# kyto — Implementation Plan

Nov 6, 2026 · @Zafer Kosar

## 0. How to use this plan

Ticked tasks live in [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md), grouped by workstream, with their notes. When you tick a task, move it there (under its workstream, in ID order) and add its ID to that workstream's "Done" line here. A `Depends` entry is satisfied when the task is in the Done file.

kyto is built in 10 workstreams of numbered tasks. An agent takes one unblocked task, finishes it until its **Done when** check passes, then ticks it.

**Aim** (decided by the user, Nov 27, 2026): the first goal is a proof of concept that beats scanpy on an Apple Silicon Mac, in speed and in peak memory, preferably many times over. Other platforms (Linux, Windows, NVIDIA through CUDA, other GPUs through wgpu's Vulkan and DX12) are secondary: they keep building and passing where they are tested, but no Mac task waits for them.

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

Done (5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#5-streaming-cpu-executor-kyto-exec).

- [ ] **5.22 Preprocessing on 1 thread at scanpy's time.** In 10.8's workflow at 50k cells, kyto's QC + filters take 0.89 s against scanpy's 0.41 s (2.1x), and its HVG step takes 1.68 s against 0.53 s (3.2x; 1.29x at 4 threads). kyto decodes the store on every pass; scanpy holds the matrix after a 0.11 s read. Profile the passes on 1 thread (decode, QC, the HVG statistics pass) and cut the largest cost. (added Nov 20, 2026, from 10.8's measurement)
  - Depends: 5.18
  - Done when: `kyto_steps.py --threads 1,4` on the 50k data gives a `hvg` prefix (QC through HVG, cumulative) of at most 1.5 s on 1 thread (1.2x scanpy's warm read through HVG, 1.26 s) and an HVG step of at most 1.2x scanpy's at 4 threads; outputs bit-identical across thread counts (R11); `performance.md` updated.
- [ ] **5.23 PCA in the workflow on 1 thread.** In 10.8's workflow, the PCA step takes 2.95 s on 1 thread against scanpy's 1.92 s (1.54x), while PCA alone is at par (0.95x, 5.15). Find what the workflow step adds (the stream pass that holds the HVG-filtered input, the held copy) and cut it. (added Nov 20, 2026, from 10.8's measurement)
  - Depends: 5.18
  - Done when: `kyto_steps.py --threads 1` gives a PCA step (the `pca` prefix minus the `hvg` prefix) of at most 2.3 s (1.2x scanpy's warm 1.92 s); the embedding bit-identical to before; `performance.md` updated.
- [ ] **5.24 The workflow's write on 1 thread.** In 10.8's workflow, kyto's `sink_zarr` adds 0.70 s on 1 thread to the run, against scanpy's 0.13 s `.h5ad` write (5.4x); on 4 and 10 threads the difference is inside the run-to-run spread. That is not equal work: kyto compresses the `counts` and `log1p` layers and also writes its tables, embeddings and graphs. First time scanpy writing an AnnData with the same content (both layers, `obsm`, `obsp`, the same compression); then cut kyto's write. (added Nov 20, 2026, from 10.8's measurement)
  - Depends: 5.20
  - Done when: `benchmarks/scanpy_workflow` times scanpy's equal-content write, and kyto's `write` step (`kyto_steps.py --with-write`, or a direct timing of the sink if the difference stays noisy) is at most 1.2x that on 1 thread and no slower on 4; stores byte-identical across thread counts (except `provenance.json`); `performance.md` updated.

## 6. Algorithms (`kyto-algo`)

Goal: the standard scanpy workflow, streamed, with each step checked against scanpy on the 1.5 fixtures.

Done (6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 6.14, 6.17): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#6-algorithms-kyto-algo).

- [ ] **6.15 UMAP power kernel.** Replace libm `pow` in the UMAP SGD (`umap.rs` `optimize`) with kyto's own kernel (`exp(b · ln x)`), so the layout is faster and has the same bits on every platform (R11). (added Nov 14, 2026, at the user's request: `pow` is about 58% of the SGD's time on macOS, where kyto's UMAP equals scanpy's, 15 to 17 s against 15.6 s at 50k cells; performance.md measured 4× slower on a 2-core Linux box, not reproduced and not yet profiled there)
  - Depends: 6.9, 5.3
  - Done when: UMAP on the `benchmarks/scanpy_workflow` 50k data takes at most scanpy's time on Linux x86 and on macOS; trustworthiness stays inside the 6.9 range; the layout is bit-identical across thread counts.
  - Parity: [`umap_parity.rs`](crates/kyto-exec/tests/it/umap_parity.rs) must keep passing with the new kernel; [tolerances](tests/parity/tolerances.toml) `umap.*`.
  - Open: (1) profile the SGD on Linux x86 first, to confirm `pow` is the cause of the gap there.
- [ ] **6.16 Metacells by e-means (opt-in).** A `metacells(n_metacells=, n_components=, epochs=, seed=, into=)` method that groups cells into many small groups (about n/50 to n/500) with e-means, the mini-batch k-means of Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)", arXiv 2611.09501 (Nov 2026). The algorithm works in mini-batches. Each centroid is a running sum divided by a running count, both updated by EMA. The step size follows a cosine schedule, and clusters that end up nearly empty are re-seeded. Initialization takes k random cells (not k-means++), and the final centroids are the mean of the cells assigned to each in the last epoch. Input is a `pca` embedding; output is a `String` metacell label per cell plus a centroid embedding. **Opt-in only** (decided by the user, Nov 19, 2026): it runs only when the user calls `metacells`. No other method, default or `device="auto"` rule ever inserts it, and Leiden/Louvain stay the clustering. The method is written in Rust from the paper, with no Faiss dependency. Batches are shuffled with `seed` (R7), and assignments and sums use a fixed reduction order (R11). (added Nov 19, 2026, at the user's request: the paper reports beating Lloyd at 25 epochs in 1.4 to 2.6 epochs, at k from 256 to 16M on 96- to 1024-d embeddings; it was not tested on single-cell data)
  - Depends: 6.5, 7.2, 5.3
  - Done when: on pbmc3k and the `benchmarks/scanpy_workflow` 50k data (PCA, 50 components), e-means at 3 epochs or fewer reaches a training MSE at most that of Lloyd with 25 iterations (scikit-learn `KMeans`, same k and random init) at k = n/50 and n/200; labels and centroids are bit-identical at 1, 4 and all threads; a plan without `metacells` gives the same `explain()`, `estimate()` and output as before the task (test); `explain()` shows the node when it is called.
  - Parity: none: scanpy has no metacells; the task checks against its own reference (the paper's algorithm in NumPy on the fixtures), its tolerances in [tolerances](tests/parity/tolerances.toml).
  - Open: (1) the paper's step-size and batch-size laws (its Eq. 6 and 7) were fitted on DEEP (d = 96, k ≥ 256); refit or replace them for 50-component PCA and k ≈ n/50 before using their constants. (2) Which downstream steps take metacells as input (pseudobulk aggregation of counts, `neighbors`/`leiden` on centroids), and how the sinks store them, is undecided. (3) An IVF index trained by e-means for approximate `neighbors` on very large atlases (next to HNSW, 6.6; GPU, 9.4) is not part of this task; it would be a separate opt-in task if wanted.
- [ ] **6.18 Neighbours on 1 thread at scanpy's time.** In 10.8's workflow, neighbours on 1 thread take 29.0 s with exact search (blocked brute force, O(n²)) and 20.9 s with HNSW, against pynndescent's 9.8 s warm (3.0x, 2.1x). HNSW is also slower than exact search from 4 threads up (14 s, flat: insertion is sequential for R11). Make one search, with recall of at least 0.99, reach scanpy's time on 1 thread without losing the exact search's 4 and 10 thread times. (added Nov 20, 2026, from 10.8's measurement)
  - Depends: 6.17
  - Done when: `kyto_steps.py` on the 50k data gives a neighbours step of at most 11.8 s on 1 thread (1.2x scanpy's warm 9.8 s) with mean recall of at least 0.99 against exact search and Leiden ARI of 1.0 against scanpy; neighbours on 4 and 10 threads no slower than 10.8's exact times (8.2 s, 4.6 s); results identical across thread counts (R11); `performance.md` and deviations.md updated.
  - Parity: [`neighbors_parity.rs`](crates/kyto-exec/tests/it/neighbors_parity.rs) must keep passing; [tolerances](tests/parity/tolerances.toml) `neighbors.*`.

## 7. Python API (`kyto-py`, `python/kyto`)

Goal: a small, chainable surface where every method has one return type and a typed stub.

Done (7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#7-python-api-kyto-py-pythonkyto).

## 8. Agent-friendliness

Goal: an agent can inspect data, write a pipeline, check it and fix its own mistakes without reading the matrix or the source code.

Done (8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#8-agent-friendliness).

## 9. GPU backend (`kyto-gpu`)

Goal: the same plan runs on a GPU by changing `on(device=...)`, with results matching the CPU.

Done (9.1, 9.2, 9.5, 9.6, 9.9, 9.10): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#9-gpu-backend-kyto-gpu).

- [ ] **9.3 CUDA libraries.** cuSPARSE SpMM and cuBLAS GEMM through `cudarc` for PCA; pinned buffers and double-buffered host-to-device copies so reads overlap compute.
  - Depends: 9.1, 6.5, 5.12
  - Done when: GPU PCA matches CPU PCA within the 6.5 tolerances.
- [ ] **9.4 GPU neighbours.** Tiled brute-force distance via GEMM plus top-k selection.
  - Depends: 9.3, 6.6
  - Done when: results equal CPU exact mode.
- [ ] **9.7 GPU CI.** GPU tests on a self-hosted runner; skipped with a visible notice elsewhere.
  - Depends: 1.4, 9.2
  - Done when: the GPU job runs on every PR that touches `kyto-gpu` or `kyto-exec`.
- [ ] **9.8 Portable matrix kernels (Metal).** PCA's products (sparse × dense SpMM and dense GEMM) and 9.4's tiled distances plus top-k written in CubeCL, so PCA and neighbours run through wgpu on Metal (and ROCm) as well as CUDA. 9.3's cuSPARSE/cuBLAS stays the CUDA path; `device="auto"` picks it on NVIDIA and these kernels elsewhere, by the 9.1 rules shown in `explain()`. (added Nov 14, 2026, at the user's request: Apple GPUs otherwise run only the 9.2 kernels)
  - Depends: 9.2, 9.3, 9.4
  - Done when: on a Metal device, PCA matches CPU PCA within the 6.5 tolerances and neighbours equal CPU exact mode.
- [ ] **9.11 DLPack from device memory.** Since 9.5, `to_torch` and `__dlpack__` hand over host memory (`kDLCPU`): kyto's results are read back to the host, so `to_torch(device="cuda")` copies once, through PyTorch. Once a step keeps its output on the GPU (9.3, 12.2), export it as a `kDLCUDA` (or ROCm) DLPack tensor from `kyto-gpu`'s buffer, honouring the consumer's `stream`, so a CUDA tensor shares kyto's device pointer. Also decide whether `torch` joins the dev dependencies (today the torch tests of `tests/test_dlpack.py` skip without it; 9.5 ran them with `uv run --with torch`). (added Nov 22, 2026, by 9.5)
  - Depends: 9.3, 9.5
  - Done when: on a CUDA machine, a tensor from `to_torch(device="cuda")` of a GPU-computed embedding has the `data_ptr()` of kyto's device buffer (test, skipped visibly elsewhere).

## 10. Testing, benchmarks, docs, release

Goal: correctness and speed are measured on every PR, and 1.0 ships as wheels with a migration guide. 10.1–10.3 start with M1 and grow with each milestone.

Done (10.1, 10.2, 10.3, 10.5, 10.7, 10.8): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#10-testing-benchmarks-docs-release).

- [ ] **10.4 Tile size study.** Benchmark cell-block × gene-block sizes for cell scans, gene lookups and PCA; set the 2.2 defaults from the results.
  - Depends: 2.4, 10.3, 5.12, 5.13
  - Done when: the chosen defaults and the benchmark table are in `docs/format.md`.
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
| M3 Full analysis | 2.9, 5.12–5.15, 6.4–6.10, 6.12, 6.16–6.17, 7.8, 8.4–8.7, 10.7 |
| M4 GPU backend | 9.1–9.2, 9.5–9.6, 9.8–9.10 |
| M5 1.0 release | 2.7–2.8, 6.11, 10.5–10.6 |
| M6 Optimization | 12.1–12.3, and the speed tasks it absorbs: 5.22–5.24, 6.15, 6.18, 10.4 |
| Later: other platforms | 9.3, 9.4, 9.7, 9.11 (NVIDIA and GPU CI) |

M6 starts only when M3 to M5 are done (decided by the user, Nov 21, 2026: "premature optimization is a curse"). Since the Mac proof of concept is the aim (Nov 27, 2026), the NVIDIA-only and GPU CI tasks moved to "Later: other platforms" and do not hold M6 back, and neither does 10.6's publish to PyPI, which waits for the user. Until then tasks build features and correctness; a task that touches a hot path still records its time and peak memory in `performance.md` (AGENTS.md), but nobody chases speed. Speed tasks found earlier (5.22–5.24, 6.15, 6.18, 10.4) wait for M6 and feed 12.1.

## 12. Optimization (`M6`, after the feature milestones)

Goal: every performance-critical step is profiled, benchmarked against scanpy, and made faster by a written, approved strategy, with no loss of accuracy. The target machine is an Apple Silicon Mac, and the aim is to beat scanpy there in time and in peak memory, preferably many times over (section 0, Nov 27, 2026); other platforms are measured later.

Accuracy rule for all of 12: a change that keeps the algorithm gives bit-identical output (R11) at every thread count; a change that alters the algorithm (an approximation, a different solver, lower precision) stays inside the 10.2 tolerance table without loosening any entry, keeps the cluster labels (ARI against the unoptimized run reported), and is recorded in `deviations.md`. A speed-up that fails this is not taken.

- [ ] **12.1 Profile and benchmark every performance-critical step.** Steps: convert/read, QC + filters, normalize, log1p, HVG, PCA, neighbours (exact and approximate), Leiden, UMAP, rank_genes_groups, the sinks, and the GPU paths that exist by then. For each: a profile (flamegraph or `samply` on macOS, `perf` on Linux) showing where its time goes, and a benchmark against scanpy (golden rule 8: same data, same work) at 50k and about 1M cells and at 1, 4 and all threads, with time and peak RSS. (added Nov 21, 2026, at the user's request)
  - Depends: M3, M4, M5 done (10.6's publish excepted); 10.3 (the benchmark harness)
  - Done when: `performance.md` has one dated table of every step (time, peak memory, kyto and scanpy, load average stated), each step's top hotspots named from its profile, and the steps ranked by their share of the workflow's time and by their gap to scanpy.
- [ ] **12.2 A speed strategy for each step.** From 12.1, write one strategy per step that is slower than scanpy or large in the workflow's time: the cause (from the profile), the options (algorithmic, memory traffic, SIMD, threading, I/O, GPU; on Apple Silicon also Accelerate's BLAS and vDSP for the matrix products, the Metal GPU through wgpu (9.2, 9.8, 9.10), and unified memory to skip host copies between devices), the expected gain, the accuracy risk under the rule above, and the order to do them in. Fold the earlier findings (5.22–5.24, 6.15, 6.18, 10.4) into it. (added Nov 21, 2026)
  - Depends: 12.1
  - Done when: the strategies are in `performance.md` and the user has approved them and a stop condition for each (a target time or a time box, AGENTS.md); each approved strategy is a plan task under 12.3 with a measurable Done-when.
- [ ] **12.3 Implement the approved strategies.** One task per strategy (12.3.1, 12.3.2, ...), each measured before and after on the 12.1 benchmark, run alone on a quiet machine, with the accuracy rule checked. (added Nov 21, 2026)
  - Depends: 12.2
  - Done when: every 12.3.x task is done or closed by its stop condition, and the 12.1 table is re-measured with the new numbers.
