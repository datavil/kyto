# Implementation Plan review (Fable 5.2)

Nov 11, 2026. Reviewed: `Implementation Plan.md` (all 77 tasks, 41 ticked), `AGENTS.md`,
`deviations.md`, `memory/README.md`, `docs/format.md`, and the repo state the plan
describes (crate layout, Cargo manifests, CI history, test files). Nothing but this
file was changed.

## Verdict

The engineering record is strong: Done-when checks are concrete, parity is measured
bit for bit, and bug-hunter findings are written down. The improvements are almost all
about the **plan as a document and a schedule**, not about the code already built:

1. The plan has become a changelog, and it is expensive to read.
2. About 100 open items and roughly 20 pending decisions are scattered with no owner.
3. The Python surface and the benchmarks are far behind the Rust core, so the two
   headline goals (usable from Python, faster than scanpy) are still unmeasured.
4. The remaining algorithm tasks (6.3 to 6.10) have gaps that will bite when they
   start: a crate dependency that points the wrong way, missing storage for embeddings
   and graphs, missing plan nodes, and soft Done-when checks.

Findings are ordered by how much they are worth fixing first.

## A. Fix now (state of the plan today)

**A1. Task 5.8 is ticked with placeholder text and is not committed.**
Lines 249 to 250 read `Done: DONE_TEXT` and `Open: OPEN_TEXT`. The 5.8 code
(`cell_steps.rs`, `gene_steps.rs`, `expression.rs`, `operator.rs`, four test files) is
in the working tree, untracked or modified, with no `[5.8]` commit. This breaks "one
task = one commit" and "tick only when the gate passes". Either finish the write-up and
commit, or untick it. `memory/executor_filters_columns.md` and one `deviations.md` row
already cite "plan 5.8 Open", which does not exist yet.

**A2. CI has not run since task 2.3, and that run failed.**
`gh run list` shows the last run on Nov 7 (`[2.3] metadata tables`, failure). Every
task from 2.4 on is verified on one Apple Silicon Mac only. The plan itself lists what
is waiting on "the next CI run": the 6-target static HDF5 build (Windows ARM64 above
all), the x86 SIMD paths (compiled, never executed), the pinned MinIO image, the arm
debug-link memory fix, Linux RSS for the memory limit (glibc arenas), and whether
anndata-dependent tests skip correctly. Suggestion: add a protocol line, "run
`gh workflow run ci.yml` when a workstream section or milestone closes, and record the
run ID in the plan". One run now would retire six or more open items or expose them
while they are still cheap.

**A3. The milestone section is broken.**
- Line 402 is a lost embed (`[embedded content: milestone order ...]`). The exit checks
  "in each box" that close a milestone are therefore not in the repo at all.
- 5.8 and 5.9 are in no milestone.
- M1 is not closed (7.1, 7.3 to 7.5, 7.7, 8.2, 8.3 are unticked), yet most of M2 is
  done. 2.7 and 2.8 sit in M5 but are ticked.
Suggestion: rewrite section 11 as a plain table with the exit check as a column, and
re-cut the milestones to match the order work is really done in (see C1).

**A4. Wording that the user's own decisions have made false.**
"On every PR" or "CI posts/fails" appears in the Done-when of 1.4, 7.6, 8.6, 8.7, 9.7,
10.1, 10.3, 10.5 and in the section 10 goal. There are no PRs and CI is on demand.
AGENTS.md covers "CI fails if" (must be a gate command), but "CI posts a benchmark
table on PRs" (10.3) and "runs on every PR that touches kyto-gpu" (9.7) cannot be met
as written. Rewrite each as a gate command or an on-purpose command with a stored
result file.

**A5. Stale open items.**
- 2.5 Open says `rust-version = "1.85"` no longer holds; `Cargo.toml` now says 1.95 and
  the `incompatible_msrv` allow is gone.
- 7.2 and 7.6 say `filter` (with an `@overload`); the code and 4.1 settled on
  `filter_cells`/`filter_genes`. 3.8 Open (5) still calls this undecided.
- 6.3 says "streaming moments (Welford)"; 5.4 and 5.6 built sums and sums of squares
  with a block-order tree, on purpose (R11).
- 6.2's title says `target=`; the builder and scanpy say `target_sum`.
- 8.7 says "byte-identical store"; `docs/format.md` §10.3 correctly exempts the `run`
  object of `provenance.json` (8.5 puts timings there). Cite §10.3 in 8.7.
- The plan's task protocol lacks AGENTS.md's "push after every commit" rule. Two copies
  of the protocol will keep drifting; keep it in AGENTS.md only and link to it.

## B. The plan as a document

**B1. 82% of the file is history.** Of 145 KB, `Done:` lines are 84 KB and `Open:`
lines 36 KB; the task definitions are 26 KB. One line (5.3) is 5.6 KB. Every session
starts by reading the plan (about 36k tokens), against AGENTS.md's own "read narrowly"
rule, and the unticked tasks an agent needs are buried under finished ones.

Suggestion:
- Move each `Done:` narrative to `docs/tasks/<id>.md` (or `history/`), verbatim. In the
  plan keep one line: what exists, where, and a link.
- Keep the plan to: definition, Depends, Done when, a one-line result, and links.
  Target size: under 30 KB.
- Because task IDs never change, this is a mechanical move with no information lost.

**B2. Open items need a register.** There are 101 numbered open items across 28 tasks.
Resolutions are appended inline (`**Resolved by 4.1:**`), so the reader must scan
paragraphs to learn what is still open. Suggestion: one file `open.md` (or a plan
section) with a row per item: ID (`4.6-O3`), text, owner (`user` / `task 7.2` /
`agent`), status. Tasks link to their rows. Closing a row is a one-line edit.

**B3. Pending decisions are invisible.** These wait for the user and are spread over
ten tasks and `deviations.md` (six rows marked "agent, to confirm"). Collected here so
they can be answered in one sitting:

| # | Decision | Where | Why it matters now |
| --- | --- | --- | --- |
| 1 | Pre-size the tile reader from tile nnz (use the pool) | 4.6 O2, 5.1 O1, 5.5 O5 | About halves peak memory on pbmc3k; cuts allocator overhead; open since 4.6 |
| 2 | Thread count in `PhysicalOptions`, fix the decode term | 4.6 O3, 5.2 O2, 5.5 O4 | Estimate is +176% to +2970% off with many gene blocks, so the memory limit shrinks blocks or raises `KY-E401` too early |
| 3 | Allocator overhead factor (50% assumed, 70 to 95% measured at small blocks) | 5.5 O1 | The 5.5 check passes with 6% to spare; 5.8 already pushed it "past its margin" per the memory note |
| 4 | Faster `f32` `log1p` (2x to 4x) vs correctly rounded | 5.3 O1 | A speed vs parity trade only the user can make |
| 5 | Fix `make_reference.py` (`normalized = adata.X` aliases the in-place `log1p`, line 211) and regenerate | 6.2 O1 | A known-wrong reference file is in the fixtures |
| 6 | Staleness rules: cell filter stales gene columns; layer rewrite stales lineage columns | 4.1 O1 | Semantics users will see |
| 7 | Dropping unused steps hides their run-time errors | 4.4 O6 | Semantics |
| 8 | QC dtypes `Float64`/`UInt32` vs scanpy `float32`/`int32` | 6.1 O3 | Parity table |
| 9 | `target_sum` range refusal; missing `normalize_total` parameters | 6.2 O2, O5 | API |
| 10 | Build-time cap on expanded expression size | 4.3 O2 | A plan can be built that `from_json` refuses |
| 11 | `convert` option to pick the gene ID column (scanpy output keeps symbols in the `var` index) | 3.9 Open | Without it `by="symbol"` fails on the most common `.h5ad` shape |
| 12 | Per-gene QC metrics (`n_cells_by_counts` and friends): which task | 6.1 O2 | See D4 |
| 13 | Default `blocks_in_flight` (2) | 5.2 O2 | See C3 |

## C. Schedule and scope

**C1. The Python API is the bottleneck, and 7.2 cannot be scheduled as written.**
The executor now runs filters, QC, normalize and log1p, but Python can only build
plans and call a private `_execute` that streams to nowhere. All of 7.x is unticked.
Open items defer 15 decisions to 7.2, 8 to 7.1 and 5 to 7.3. Problems:

- 7.2 lists "the algorithm methods from 6.x" and `collect()`/`sink_zarr()`, but
  depends only on 4.3 and 3.2. It really needs 5.8, 5.9 and every 6.x task, so it can
  only close after 6.10, while M2 schedules it before M3.
- 7.3 returns Polars, NumPy and SciPy objects, which need 7.5's hand-off; but 7.5
  depends on 7.3. The order is reversed.
- 7.4's Done-when needs the migration table of 10.5, which depends on 7.6 (M5 vs M1).
- 8.2's Done-when calls `pull("CD3")` (7.4) but depends on 8.1 and 3.5 only. 8.3 names
  `peek(n)` and `features()`, which no other task defines. 8.5 needs sinks (5.9) but
  depends on 2.5 and 4.3.

Suggestion: split 7.2 and make the rest vertical.
- **7.2a Core chain and terminals** (now, right after 5.9): `filter_cells`,
  `filter_genes`, `select_genes`, `take_cells`, `with_cells`, `with_genes`,
  `normalize_total`, `log1p`, `collect`, `sink_zarr`, `sink_h5ad`, with `threads=`,
  `progress=`, `memory_limit=`. Move 7.5 before 7.3. This also settles the 15 deferred
  naming questions while they are few.
- **Each 6.x task binds its own Python method** as part of its Done-when (stub,
  docstring, doctest, parity test from Python). Golden rule 2 asks agents to plot
  embeddings and clusterings; that is only practical from Python.
- 7.2 then shrinks to "the README pipeline runs", closed at the end of M3.

**C2. "Beat scanpy" has never been measured.** 10.3 depends only on 5.2 and is
unblocked, but there is no `benchmarks/` directory, no `pytest-benchmark` or
`hypothesis` dependency, and the only fixtures are synthetic and pbmc3k (2,700 cells:
one cell block). Three open items wait on benchmarks (5.5 O1, 5.6 O2, 4.6 O3), and
10.4 (tile sizes, which every estimate and pass count rests on) waits on 10.3.
Suggestion: split 10.3. **10.3a** now: a harness, one mid-size public dataset (for
example 68k PBMC) and one large synthetic, `read + QC + normalize + log1p` against
scanpy, time and peak RSS, result table checked in. **10.3b** later: rapids-singlecell
(needs the GPU runner, so it belongs with M4) and regression flags. Doing 10.3a before
6.5 means PCA is designed against numbers (see D2).

**C3. Default parallelism is capped at 2 blocks.** 5.2 measured 3.34x at 8 threads
with the default `blocks_in_flight = 2` versus 4.09x with 8, and pbmc3k is a single
4096-cell block, so small datasets get no block parallelism at all. Since most users
never pass options, the default decides the benchmark. Suggestion: with no memory
limit, default `blocks_in_flight` to the thread count; with one, let 5.5's fit choose.
Fold the choice of `C` for small data into 10.4.

**C4. Oversized tasks.** 6.6 (exact search plus HNSW), 6.8 (Leiden and Louvain, both
from scratch) and 6.10 (Wilcoxon and t-test) are each two tasks. 5.9 has become a
grab bag: sinks, embedding expression leaves, `.recompute()` columns, and writing
embeddings and graphs. Split them so each has one Done-when.

**C5. Soft Done-when checks.** These cannot fail: 6.6 "HNSW recall is reported", 6.8
"adjusted Rand index is reported", 6.9 "within range of umap-learn", 6.5 "subspace
angles are small". Give each a number (for example recall at 15 >= 0.95, ARI >= 0.9
on pbmc3k, trustworthiness within 0.02, largest principal angle < 1e-2 for the top 20
components). 6.6's "exact mode equals brute force" is checked against itself; compare
with scanpy's exact neighbours (`transformer="sklearn"`, or its brute-force path below
4,096 cells) instead.

**C6. The tolerance table does not exist.** AGENTS.md says "tolerances live in the
parity table, not in individual tests" (10.2), but 10.2 is unticked and tolerances are
inline today: `1e-6` in `qc_parity.rs` and `filter_qc_parity.rs`, `1e-5` in
`multi_pass.rs` and `filters_columns.rs`. 10.2 depends only on 1.5. Suggestion: create
the table now as **10.2a** (one TOML or Rust const table read by both test suites),
and leave "every 6.x task links its parity test" as the closing check.

## D. Technical gaps to settle before section 6 continues

**D1. `kyto-algo` depends on `kyto-exec`, so algorithms cannot live there.**
`crates/kyto-algo/src` is a 12-line `lib.rs`. QC moved into `kyto-exec` during 5.8,
HVG statistics were already there, and the memory note concludes "algorithms that run
inside the stream belong in kyto-exec". But the repo map (plan and AGENTS.md) still
says `kyto-algo` holds QC, HVG, PCA, kNN, clustering and DE, and section 6 is titled
`kyto-algo`. If the executor must run `neighbors`, `leiden` or the small dense steps
of PCA as stages, it needs to call them, which the current direction forbids. Decide
before 6.3:
- Option 1 (recommended): flip the edge. `kyto-algo` becomes pure, stream-free
  numerics (moments to ranks, loess, SVD steps, kNN, Leiden, UMAP, test statistics)
  depending on `kyto-core` and `kyto-error` only; `kyto-exec` depends on it and owns
  streaming. The SIMD kernels would move down with it, or into a small `kyto-kernels`.
- Option 2: retire `kyto-algo` and say so in the repo map.
AGENTS.md says to stop and tell the user when the foundation fights the feature; this
is that case, found by 5.8.

**D2. PCA at 16 passes will not beat scanpy.** 4.5 fixed streamed Halko at
`2 + 2 x 7 = 16` statistics passes, and each pass re-streams from the scan through
every stage below (decode, normalize, log1p, gene subset) because the executor has no
way to keep an intermediate. scanpy does this once in memory. Two additions to weigh
before 6.5, since both change `explain()` snapshots and `estimate()`:
- **A materialize operator.** When a cross-cell node's input (after filters and the
  HVG subset) fits the memory limit, hold it as CSR once and iterate in memory; else
  spill it to a temporary store. The physical plan picks, `explain()` shows it. This
  also helps `scale`, HVG after a median, and DE.
- **A covariance path for few genes.** After HVG the matrix has about 2,000 genes.
  The gene by gene Gram matrix (32 MB at 2,000 genes) can be accumulated in one pass
  with the existing block-order tree, then solved exactly with faer; a second pass
  projects the cells. That is 2 passes instead of 16, exact rather than randomized (so
  the 1e-3 explained-variance check against scanpy's ARPACK is easy, and `seed` is not
  needed on this path), and deterministic for free. Keep Halko for wide inputs.
  The cost is squared conditioning, fine for the top 50 components in `f64`; verify it.

**D3. Nothing stores or loads embeddings and graphs.** `store_writer.rs` says it
writes those groups empty and "6.x adds" them; there is no reader. The `.h5ad`
importer skips `obsm`/`obsp` and the exporter writes them empty. Yet 6.5, 6.6, 6.7 and
6.9 produce exactly these, `Embedding.to_numpy()` and `Graph.to_scipy()` (7.3) read
them, and 5.9's text mentions them only in passing. Add **2.9 Embedding and graph
I/O** (format.md §6 and §7: writer, reader, `obsm`/`obsp` import and export) before
6.5, and say in one design note how a node's in-memory result (HVG columns, PCA
embedding, neighbour graph) reaches later nodes and terminals; 5.4's `Statistics` is
the seed of that but no task owns it.

**D4. Missing plan nodes and expressions for the standard workflow.**
- No logical node for UMAP (6.9), differential expression (6.10) or `with_embedding`
  (7.2, 6.11). 4.1's list ends at `Cluster` and `Sink`. Adding a node currently
  touches the node file, `plan_json.rs`, `members.rs`, `physical_plan.rs`,
  `estimate.rs`, the executor's lowering and the snapshot corpora, which is far from
  AGENTS.md's "one new file plus a registration". Three more nodes are coming:
  consider one per-node trait that carries schema, members, execution, cost and JSON,
  so the files above iterate over it.
- scanpy's first step, `filter_genes(min_cells=3)`, needs a per-gene count over cells.
  6.1 lists the per-gene metrics as "not provided, user's call". The fixture note says
  pbmc3k needs that gene filter before `cell_ranger` HVG, so 6.3's parity check
  appears to depend on it. Give it a task number.
- 6.3 `seurat_v3` needs a loess fit (scanpy uses `skmisc`). 5.4 left a stand-in. The
  Done-when is "selected gene sets equal scanpy's", which a slightly different loess
  will miss at the cutoff rank. Name the loess port as its own sub-task with a
  numeric check on `reg_std` first.
- 6.4 says "`explain()` suggests implicit centering"; `explain` has no hint or warning
  channel (4.7). Add one, or drop the sentence.

**D5. R1 and R11 are about to be strained.**
- R1 lists the terminals that may read the matrix. `count_cells()`, `cell_ids()` and
  `gene_ids()` after a data-dependent filter (7.4), and the DE result frame (6.10),
  must read it and are not on the list. Extend R1 by decision, or route them through
  `collect`.
- R11 for the remaining algorithms is not free: `hnsw_rs` inserts in parallel in a
  thread-dependent order, UMAP's usual SGD is lock-free and racy, and Leiden's node
  order must be fixed. Put "bit-identical at 1, 4 and all threads" into the Done-when
  of 6.6, 6.8 and 6.9 now, so the design starts from it. `f64::exp_m1` (5.4 O1) is the
  same class of issue across platforms.

**D6. Orphans (work no task owns).**
- Remote `scan`: 2.8 made the tile reader remote, but `open_atlas_schema`, `read_ids`
  and `read_symbols` still take a `Path`. 2.8 handed this to "the data model tasks",
  which are all ticked. Atlases on S3 are a core promise of the storage goal.
- A README: 7.2's Done-when is "the README pipeline runs", and there is no README.
- Multi-sample input (concatenating stores or scanning many): absent from the plan,
  and usually the first thing an atlas needs. Even "not in 1.0" should be written down.
- A killed `sink_h5ad` leaves its temp file (2.7 Open); `convert` and `sink_h5ad`
  ignore Ctrl-C until they finish (5.7 O2).
- `estimate()` does not model table reads; the 5.8 note measured +54 MB for the
  pbmc3k `symbol` Enum column.

## E. Suggested order

1. Close or untick 5.8 (A1); run CI once (A2).
2. Slim the plan and build the open register (B1 to B3); repair section 11 and the
   stale wording (A3 to A5). One maintenance commit, no code.
3. Answer the decision table in B3, at least rows 1 to 3, 5 and 12.
4. Decide D1 (crate direction) and add 2.9 (D3) before any more of section 6.
5. 5.9 (sinks only), then 7.5, 7.2a, 7.3: first end-to-end Python pipeline.
6. 10.2a and 10.3a: tolerance table and the first scanpy benchmark.
7. 6.3 onward, each task with its Python method and numeric Done-when (C1, C5), with
   the PCA design of D2 settled before 6.5.

## Not checked

- I did not run the gate, the tests or any benchmark; the numbers quoted are the
  plan's own.
- I did not review the uncommitted 5.8 code, only its plan entry and memory note.
- D2's cost argument is an estimate from operation counts, not a measurement; it is a
  proposal to test in 10.3a, not a result.
- The `min_cells` dependency in D4 is inferred from the fixture note and 6.1's open
  item; I did not trace how `filter_qc_parity.rs` filters genes.

## Status on Nov 12, 2026

The plan fixes are applied in commit `61bb06e` on branch `claude/plan-review`
(worktree `.claude/worktrees/plan-review`), rebased on `origin/main`, not pushed.
Only `Implementation Plan.md`, `AGENTS.md` and the new `tasks/` directory changed.

| Finding | State |
| --- | --- |
| A1 5.8 placeholders | already fixed by the 5.8 commit |
| A2 CI | deferred by the user until the plan is finished; plan section 13 lists what waits |
| A3 milestones | re-cut with new exit checks (plan section 11; to confirm) |
| A4, A5 stale wording | fixed, except the 6.2 and 6.3 titles (entries in use by other sessions) |
| B1 plan size | history of 31 tasks moved word for word to `tasks/<id>.md`; 168 KB to 72 KB |
| B2 open register | Inbox lines under unticked tasks; open items stay numbered in `tasks/` |
| B3 decisions | plan section 12, 23 rows; rows 3 and 5 of the table above were answered meanwhile |
| C1 Python surface | 7.8 split from 7.2; 7.5 before 7.3; 6.x tasks bind their own method after 7.8 |
| C2 benchmarks | 10.7 split from 10.3 |
| C3 default parallelism | decision row 19 |
| C4 oversized tasks | 6.12 (HNSW) and 6.13 (Louvain) split off; 5.9 was split into 5.11 by its own session |
| C5 soft checks | R11 added to the checks; thresholds left as open questions with a suggestion |
| C6 tolerance table | 10.8 split from 10.2 |
| D1 crate direction, D2 PCA, D5 R1 | not decided: open questions in the plan and decision rows 14 to 16 |
| D3 embeddings and graphs | new task 2.10 (next to 5.11) |
| D4 missing nodes, `min_cells` | open questions under 6.9 to 6.11; new task 6.14 (needs the user's OK) |
| D6 orphans | 2.9 remote scan; README in 7.2; multi-sample input is decision row 22; table reads became 5.10 |

Entries 5.4, 5.9, 5.10, 5.11, 6.2 and 6.3 were left exactly as they are, because other
sessions were editing them; their history is still inline.
