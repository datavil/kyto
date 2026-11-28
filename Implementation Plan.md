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

- [ ] **1.7 A faster local gate.** The gate (AGENTS.md "Run the whole list") takes 30 to 40 minutes, about 25 of them in `cargo test`, and sessions rerun it in full after every rebase and every small fix. Cut the waiting without weakening the rule that a task is done only when the gate passes: (a) drop the forced extension rebuild (`uv sync --reinstall-package kyto`) when the extension is already current, (b) run the Rust tests with `cargo-nextest`, (c) stop building the GPU features twice (clippy and test with `--all-features`), (d) write in AGENTS.md when a session may rerun only a subset: after a lint or stub fix, the failing command; after a rebase with no code conflict and no overlap with the task's files, build, the generated-file checks and the task's own tests; one full run always comes last before the commit. Tooling, not product speed, so M6's rule does not apply; runs after 12.1 so its timings stay clean. (added Nov 27, 2026, at the user's request)
  - Depends: 1.6
  - Done when: the gate passes on `main` with the same commands' coverage, its wall time before and after is in `performance.md` (same machine, load stated), and AGENTS.md states the subset rule.

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

- [ ] **5.22 Preprocessing on 1 thread at scanpy's time.** In 10.8's workflow at 50k cells, kyto's QC + filters take 0.89 s against scanpy's 0.41 s (2.1x), and its HVG step takes 1.68 s against 0.53 s (3.2x; 1.29x at 4 threads). kyto decodes the store on every pass; scanpy holds the matrix after a 0.11 s read. Profile the passes on 1 thread (decode, QC, the HVG statistics pass) and cut the largest cost. (added Nov 20, 2026, from 10.8's measurement) Carried out as 12.10 (Nov 28, 2026).
  - Depends: 5.18
  - Done when: `kyto_steps.py --threads 1,4` on the 50k data gives a `hvg` prefix (QC through HVG, cumulative) of at most 1.5 s on 1 thread (1.2x scanpy's warm read through HVG, 1.26 s) and an HVG step of at most 1.2x scanpy's at 4 threads; outputs bit-identical across thread counts (R11); `performance.md` updated.
- [ ] **5.23 PCA in the workflow on 1 thread.** In 10.8's workflow, the PCA step takes 2.95 s on 1 thread against scanpy's 1.92 s (1.54x), while PCA alone is at par (0.95x, 5.15). Find what the workflow step adds (the stream pass that holds the HVG-filtered input, the held copy) and cut it. (added Nov 20, 2026, from 10.8's measurement) Carried out as 12.10 (Nov 28, 2026).
  - Depends: 5.18
  - Done when: `kyto_steps.py --threads 1` gives a PCA step (the `pca` prefix minus the `hvg` prefix) of at most 2.3 s (1.2x scanpy's warm 1.92 s); the embedding bit-identical to before; `performance.md` updated.
- [ ] **5.24 The workflow's write on 1 thread.** In 10.8's workflow, kyto's `sink_zarr` adds 0.70 s on 1 thread to the run, against scanpy's 0.13 s `.h5ad` write (5.4x); on 4 and 10 threads the difference is inside the run-to-run spread. That is not equal work: kyto compresses the `counts` and `log1p` layers and also writes its tables, embeddings and graphs. First time scanpy writing an AnnData with the same content (both layers, `obsm`, `obsp`, the same compression); then cut kyto's write. (added Nov 20, 2026, from 10.8's measurement) Carried out as 12.8 (Nov 28, 2026).
  - Depends: 5.20
  - Done when: `benchmarks/scanpy_workflow` times scanpy's equal-content write, and kyto's `write` step (`kyto_steps.py --with-write`, or a direct timing of the sink if the difference stays noisy) is at most 1.2x that on 1 thread and no slower on 4; stores byte-identical across thread counts (except `provenance.json`); `performance.md` updated.

## 6. Algorithms (`kyto-algo`)

Goal: the standard scanpy workflow, streamed, with each step checked against scanpy on the 1.5 fixtures.

Done (6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 6.14, 6.16, 6.17): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#6-algorithms-kyto-algo).

- [ ] **6.15 UMAP power kernel.** Replace libm `pow` in the UMAP SGD (`umap.rs` `optimize`) with kyto's own kernel (`exp(b · ln x)`), so the layout is faster and has the same bits on every platform (R11). (added Nov 14, 2026, at the user's request: `pow` is about 58% of the SGD's time on macOS, where kyto's UMAP equals scanpy's, 15 to 17 s against 15.6 s at 50k cells; performance.md measured 4× slower on a 2-core Linux box, not reproduced and not yet profiled there) Carried out as 12.6 (Nov 28, 2026).
  - Depends: 6.9, 5.3
  - Done when: UMAP on the `benchmarks/scanpy_workflow` 50k data takes at most scanpy's time on Linux x86 and on macOS; trustworthiness stays inside the 6.9 range; the layout is bit-identical across thread counts.
  - Parity: [`umap_parity.rs`](crates/kyto-exec/tests/it/umap_parity.rs) must keep passing with the new kernel; [tolerances](tests/parity/tolerances.toml) `umap.*`.
  - Open: (1) profile the SGD on Linux x86 first, to confirm `pow` is the cause of the gap there.
- [ ] **6.18 Neighbours on 1 thread at scanpy's time.** In 10.8's workflow, neighbours on 1 thread take 29.0 s with exact search (blocked brute force, O(n²)) and 20.9 s with HNSW, against pynndescent's 9.8 s warm (3.0x, 2.1x). HNSW is also slower than exact search from 4 threads up (14 s, flat: insertion is sequential for R11). Make one search, with recall of at least 0.99, reach scanpy's time on 1 thread without losing the exact search's 4 and 10 thread times. (added Nov 20, 2026, from 10.8's measurement) Carried out as 12.4 and 12.5 (Nov 28, 2026).
  - Depends: 6.17
  - Done when: `kyto_steps.py` on the 50k data gives a neighbours step of at most 11.8 s on 1 thread (1.2x scanpy's warm 9.8 s) with mean recall of at least 0.99 against exact search and Leiden ARI of 1.0 against scanpy; neighbours on 4 and 10 threads no slower than 10.8's exact times (8.2 s, 4.6 s); results identical across thread counts (R11); `performance.md` and deviations.md updated.
  - Parity: [`neighbors_parity.rs`](crates/kyto-exec/tests/it/neighbors_parity.rs) must keep passing; [tolerances](tests/parity/tolerances.toml) `neighbors.*`.
- [ ] **6.19 Aggregate cells by a column (pseudobulk, metacell atlas).** An `aggregate(by=<cell column>, layer=, how="sum")` step that sums a layer's counts over the cells of each group of a cell column (a `metacells` label, a cluster, a sample) into a new plan whose cells are the groups: its cell table holds the group name and its cell count, its layer the summed counts, and, from `metacells`, its embedding the centroids. `neighbors`, `leiden`, `rank_genes_groups` and the sinks then run on it as on any atlas, which covers both downstream uses 6.16 left open (pseudobulk counts, and `neighbors`/`leiden` on metacell centroids). It changes the cell axis, so it needs its own estimate and lineage rules. Opt-in like 6.16. (added Nov 27, 2026, by 6.16 on the maestro's go-ahead for the user; milestone to be placed by the user)
  - Depends: 6.16, 7.2
  - Done when: on pbmc3k, `metacells(...).aggregate(by="metacells")` gives per-group sums equal to scanpy's `sc.get.aggregate(adata, by=..., func="sum")` bit for bit (integer counts), with the centroid embedding carried over; `neighbors` and `leiden` run on the result; `estimate()` and `explain()` show the new cell count bound.
  - Parity: scanpy's `get.aggregate`; tolerances in [tolerances](tests/parity/tolerances.toml) if any.

## 7. Python API (`kyto-py`, `python/kyto`)

Goal: a small, chainable surface where every method has one return type and a typed stub.

Done (7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#7-python-api-kyto-py-pythonkyto).

## 8. Agent-friendliness

Goal: an agent can inspect data, write a pipeline, check it and fix its own mistakes without reading the matrix or the source code.

Done (8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#8-agent-friendliness).

## 9. GPU backend (`kyto-gpu`)

Goal: the same plan runs on a GPU by changing `on(device=...)`, with results matching the CPU.

Done (9.1, 9.2, 9.5, 9.6, 9.8, 9.9, 9.10): see [Implementation Plan - Done.md](Implementation%20Plan%20-%20Done.md#9-gpu-backend-kyto-gpu).

- [ ] **9.3 CUDA libraries.** cuSPARSE SpMM and cuBLAS GEMM through `cudarc` for PCA; pinned buffers and double-buffered host-to-device copies so reads overlap compute.
  - Depends: 9.1, 6.5, 5.12
  - Done when: GPU PCA matches CPU PCA within the 6.5 tolerances.
  - Note (Nov 27, 2026, from 9.8, done first on the maestro's reorder): `pca` already runs on CUDA through 9.8's CubeCL kernels (`kyto-gpu/src/kernels_matrix.rs`; `"pca"` is in `CUDA_KERNELS`), unverified here (no NVIDIA GPU). What is left: cuSPARSE SpMM and cuBLAS GEMM as CUDA's preferred path, by branching on `self.device == Device::Cuda` inside `DeviceKernels::sparse_dense` (`X D`) and `DeviceKernels::sparse_transposed_dense` (`X^T D`, continuing `PairSums`) in `launch_matrix.rs`; same arguments and results, so kyto-exec's `pca/gpu.rs` and the placement are unchanged (keep R11: a fixed reduction order, and a leaf split into parts must give the bits of the whole block, which is why the sums are carried as pairs); the pinned buffers and double-buffered copies; and running `gpu_matrix::` on a CUDA machine.
- [ ] **9.4 GPU neighbours.** Tiled brute-force distance via GEMM plus top-k selection.
  - Depends: 9.3, 6.6
  - Done when: results equal CPU exact mode.
  - Note (Nov 27, 2026, from 9.8): exact `neighbors` already runs on any GPU through 9.8's CubeCL tiled distances (squared Euclidean, or cosine from a dense GEMM `A B^T`) and top-k (`DeviceKernels::nearest_keys`), with a host check that makes the graph equal the CPU's exact search (`kyto-exec/src/neighbors_gpu.rs`: candidates re-ranked with the CPU's distances, a row searched again on the CPU when the `f32` keys cannot prove its neighbours). Verified on Metal; unverified on CUDA. What is left: a cuBLAS GEMM for the distances on CUDA (behind `nearest_keys`, same results), and the CUDA run of `gpu_matrix::`.
- [ ] **9.7 GPU CI.** GPU tests on a self-hosted runner; skipped with a visible notice elsewhere.
  - Depends: 1.4, 9.2
  - Done when: the GPU job runs on every PR that touches `kyto-gpu` or `kyto-exec`.
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
  - Built (Nov 27, 2026), everything up to the publish: the version is 0.1.0, from one source (`[workspace.package] version`), semantic versioning and what it covers stated in `docs/release.md`; `pyproject.toml` declares the runtime dependencies (NumPy, Polars, SciPy, which the extension imports and which no wheel declared before) and PyPI metadata, abi3 wheels (`abi3-py310`) stripped. `.github/workflows/release.yml`: a `v*` tag publishes to PyPI by trusted publishing (OIDC, environment `pypi`), a run by hand is the TestPyPI dry run (environment `testpypi`, or `-f repository=none`); it checks that the commit is on `main`, the tag matches the version, the `CHANGELOG.md` section is dated and the licence texts exist (`scripts/release_version.py`), calls `ci.yml` (now `workflow_call` instead of its own tag trigger) and publishes CI's six wheels plus an sdist once lint, tests and wheels pass. CI's wheel job now installs each wheel on Python 3.10 (3.13 on Windows ARM64) and runs `scripts/wheel_smoke.py` (convert, QC, normalize, HVG, PCA, neighbours, Leiden). Store read policy in `docs/release.md` ("Reading older stores", linked from `format.md` §11): before 1.0 a release reads exactly the format versions it lists and refuses older and newer ones with `KY-E115`; from 1.0, 1.y reads every 1.x and refuses other majors; never in place. `tests/test_release.py` holds the version check, the refusal of older (`0.0`) and newer stores, the docs table against `SUPPORTED_FORMAT_VERSIONS`, and the workflow's publish rules. `CHANGELOG.md` has the 0.1.0 entries under `[Unreleased]`. macOS arm64 is the first-class wheel (the Mac proof of concept, Nov 27, 2026): checked by hand before each release; the Linux and Windows wheels are built and smoke-tested by CI only. Checked locally: the macOS arm64 abi3 wheel built with `maturin build --release`, installed in a fresh Python 3.10 venv, passes the smoke test and the pbmc3k pipeline (2,638 cells after QC).
  - Left for the user (the release checklist in `docs/release.md` has the commands):
    1. Create the PyPI and TestPyPI trusted publishers (project `kyto`, workflow `release.yml`, environments `pypi` and `testpypi`) and the two GitHub environments (a required reviewer on `pypi`).
    2. Add the licence files: `pyproject.toml` says `MIT OR Apache-2.0`, but the repo has no `LICENSE-MIT`/`LICENSE-APACHE` (they need the copyright holder's name), so the wheels would carry no licence text; the release check refuses until they exist (then set `license-files` in `pyproject.toml`).
    3. Date the changelog (`## [0.1.0] - YYYY-MM-DD`), run the TestPyPI dry run (`gh workflow run release.yml --ref main`) and install from TestPyPI. It is also the first run of the manylinux, Windows ARM64 and macOS x86_64 wheel builds with the bundled HDF5 (memory/hdf5_static_build.md).
    4. Push the first tag (`v0.1.0`) and approve the `pypi` environment.
    5. Check `pip install kyto` on Linux, macOS and Windows; then tick 10.6.

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
| M6 Optimization | 12.1, 12.4–12.10 (one task per slow step), and the speed tasks they absorb: 5.22–5.24, 6.15, 6.18, 10.4 |
| Later: other platforms | 9.3, 9.4, 9.7, 9.11 (NVIDIA and GPU CI) |

M6 starts only when M3 to M5 are done (decided by the user, Nov 21, 2026: "premature optimization is a curse"). Since the Mac proof of concept is the aim (Nov 27, 2026), the NVIDIA-only and GPU CI tasks moved to "Later: other platforms" and do not hold M6 back, and neither does 10.6's publish to PyPI, which waits for the user. Until then tasks build features and correctness; a task that touches a hot path still records its time and peak memory in `performance.md` (AGENTS.md), but nobody chases speed. Speed tasks found earlier (5.22–5.24, 6.15, 6.18, 10.4) wait for M6 and feed 12.1.

## 12. Optimization (`M6`, after the feature milestones)

Goal: every performance-critical step is profiled, benchmarked against scanpy, and made faster by a written, approved strategy, with no loss of accuracy. The target machine is an Apple Silicon Mac, and the aim is to beat scanpy there in time and in peak memory, preferably many times over (section 0, Nov 27, 2026); other platforms are measured later.

Accuracy rule for all of 12: a change that keeps the algorithm gives bit-identical output (R11) at every thread count; a change that alters the algorithm (an approximation, a different solver, lower precision) stays inside the 10.2 tolerance table without loosening any entry, keeps the cluster labels (ARI against the unoptimized run reported), and is recorded in `deviations.md`. A speed-up that fails this is not taken.

- [ ] **12.1 Profile and benchmark every performance-critical step.** Steps: convert/read, QC + filters, normalize, log1p, HVG, PCA, neighbours (exact and approximate), Leiden, UMAP, rank_genes_groups, the sinks, and the GPU paths that exist by then. For each: a profile (flamegraph or `samply` on macOS, `perf` on Linux) showing where its time goes, and a benchmark against scanpy (golden rule 8: same data, same work) at 50k and about 1M cells and at 1, 4 and all threads, with time and peak RSS. (added Nov 21, 2026, at the user's request)
  - Depends: M3, M4, M5 done (10.6's publish excepted); 10.3 (the benchmark harness)
  - Done when: `performance.md` has one dated table of every step (time, peak memory, kyto and scanpy, load average stated), each step's top hotspots named from its profile, and the steps ranked by their share of the workflow's time and by their gap to scanpy.
  - First pass (Nov 27, 2026, at `374d9ad`; `performance.md`, "Every step on the Mac, first pass"): every step timed at 50k cells against scanpy at 1, 4 and 10 threads with peak RSS, kyto alone at 1M (approximate at 1, 4, 10 threads; exact at 10), the Metal paths at 50k, and the ranked list that became 12.4–12.10. Left: scanpy at 1M (under `guard.py`, reporting an out-of-memory stop honestly), kyto on Metal at 1M, and the `samply` profiles with each step's hotspots. Each of 12.4–12.10 profiles its own step, so 12.1 closes once the 1M comparison is in.
- [ ] **12.2 A speed strategy for each step.** Split by step (user, Nov 28, 2026: one session per slow step, not one session for all of them): each of 12.4–12.10 writes and gets approval for its own strategy, so 12.2 is done when they are. From 12.1, write one strategy per step that is slower than scanpy or large in the workflow's time: the cause (from the profile), the options (algorithmic, memory traffic, SIMD, threading, I/O, GPU; on Apple Silicon also Accelerate's BLAS and vDSP for the matrix products, the Metal GPU through wgpu (9.2, 9.8, 9.10), and unified memory to skip host copies between devices), the expected gain, the accuracy risk under the rule above, and the order to do them in. Fold the earlier findings (5.22–5.24, 6.15, 6.18, 10.4) into it. (added Nov 21, 2026)
  - Depends: 12.1
  - Done when: the strategies are in `performance.md` and the user has approved them and a stop condition for each (a target time or a time box, AGENTS.md); each approved strategy is a plan task under 12.3 with a measurable Done-when.
- [ ] **12.3 Implement the approved strategies.** Carried out inside 12.4–12.10, one step each. One task per strategy (12.3.1, 12.3.2, ...), each measured before and after on the 12.1 benchmark, run alone on a quiet machine, with the accuracy rule checked. (added Nov 21, 2026)
  - Depends: 12.2
  - Done when: every 12.3.x task is done or closed by its stop condition, and the 12.1 table is re-measured with the new numbers.

Tasks 12.4–12.10 are the ranked list from 12.1's first pass (`performance.md`). Each one runs in its own session, alone on the machine, and does three things in order: profile its step with `samply` and name the hotspots; write the strategy in `performance.md` (cause, options, expected gain, accuracy risk under the rule above) and stop for the user's approval of it and of a stop condition; then implement, re-measure on the 12.1 benchmark (50k at 1, 4 and 10 threads, and 1M where the step runs there) and record before and after. Numbers below are kyto / scanpy at 50k cells, 10 threads, from Nov 27, 2026. (added Nov 28, 2026, at the user's request)

- [ ] **12.4 Exact neighbours.** 4.12 / 0.33 s (12x slower; 25.3 / 0.93 s on one thread, 27x), and 1,652 s of the 2,018 s exact run at 1M. The search is blocked brute force, O(n²). Carries 6.18's exact-search part.
  - Depends: 12.1 (first pass)
  - Done when: the approved stop condition is met or its time box runs out, the result is bit-identical across thread counts (R11), and the before/after table is in `performance.md`.
- [ ] **12.5 Approximate neighbours (HNSW).** 5.29 / 0.33 s (16x slower), slower than kyto's exact search at 50k and flat from 4 threads up (insertion is sequential for R11); 573 s of the 950 s approximate run at 1M. Recall of at least 0.99 against exact stays. Carries 6.18's HNSW part.
  - Depends: 12.1 (first pass)
  - Done when: as 12.4, plus recall stated.
- [ ] **12.6 UMAP.** 15.9 / 14.1 s (1.1x slower) but the largest share of the 50k run, and about 340 s at 1M. Carries 6.15 (libm `pow` was about 58% of the SGD).
  - Depends: 12.1 (first pass)
  - Done when: as 12.4.
- [ ] **12.7 The Metal paths.** At 50k, `on(device="auto")` is slower than kyto's CPU on every step except exact neighbours on one thread (normalize + log1p 3.4x, QC 1.8x, PCA 1.7x slower at 10 threads). Likely causes from 9.8: a host round trip per block (9.6 rule 5), a host CSC copy per PCA fold, untiled kernels. Options include keeping blocks on the device and unified memory.
  - Depends: 12.1 (first pass)
  - Done when: as 12.4, and `auto` never picks a GPU path that is slower than the CPU on the 12.1 benchmark (or the note says why).
- [ ] **12.8 The sinks.** `sink_h5ad` 0.64 / 0.14 s (4.5x slower at 10 threads), and `sink_zarr` 0.67 / 0.13 s on one thread (5.3x). Carries 5.24, including its rule to first time scanpy writing the same content.
  - Depends: 12.1 (first pass)
  - Done when: as 12.4.
- [ ] **12.9 `rank_genes_groups`, t-test.** 0.23 / 0.07 s (3.3x slower), the same at every thread count; Wilcoxon is already faster than scanpy.
  - Depends: 12.1 (first pass)
  - Done when: as 12.4, with the 10.2 tolerances unchanged.
- [ ] **12.10 Preprocessing on one thread.** On one thread QC + filters take 0.88 / 0.41 s (2.2x), HVG 1.67 / 0.53 s (3.1x) and PCA 2.91 / 1.91 s (1.5x); from 4 threads up all three are faster than scanpy. Carries 5.22 and 5.23.
  - Depends: 12.1 (first pass)
  - Done when: as 12.4, without losing the 4 and 10 thread times.
