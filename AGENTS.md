# AGENTS.md

Read this file fully before touching the code. It is the contract for every agent
(human or AI) working on **kyto**. The roadmap and task list live in
[Implementation Plan.md](Implementation%20Plan.md). Lessons learned live in
[memory/](memory/README.md).

## Vision

kyto is a lazy, streaming single-cell analysis library **written in Rust, used from
Python**.

- Chained methods build a plan; nothing reads the matrix until a terminal
  (`collect`, `sink_*`, `pull`, `select`, `to_*`) runs it.
- Rust does all the heavy work (storage, planning, execution, algorithms, GPU).
  Python is a thin, typed plan builder. Data crosses the boundary through Polars,
  NumPy or DLPack, with copy behaviour stated in every docstring.
- Results match scanpy and anndata on the reference fixtures, within the tolerances
  kept in one table (task 10.2).
- Performance target: beat scanpy and rapids-singlecell on equal work, in time and
  in peak memory. Any plan must run within a stated memory limit.
- The library is maintained by agents. Optimise for agent-friendliness (see below).

## Design rules (cite as R1-R12)

These are copied from the plan. Reviews and commit messages cite them by number.

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

If a change would break a rule, stop and ask the user. Rules change only by a
decision recorded in the plan.

## Task protocol

- An agent takes **one unblocked task**, finishes it until its **Done when** check
  passes, then ticks it in the plan.
- Start a task only when every task under its **Depends** is ticked.
- Task IDs such as `3.4` never change. New tasks take the next free number; dropped
  tasks stay, struck through.
- **One task = one commit on `main`.** No task branches or PRs. Commit messages
  start with the ID: `[3.4] feature resolver`.
- **Push after every commit.** As soon as a task commit (or a maintenance or decision
  commit) is on `main`, run `git push origin main`, so no finished work sits only on
  one machine (decided by the user, Nov 11, 2026). Never force-push. If the push is
  rejected, stop and tell the user instead of rebasing or merging on your own.
- A task is done when its **Done when** check and the local gate pass, not before. The
  local gate is "Run the whole list" below. CI (GitHub Actions) runs only on demand
  (`gh workflow run ci.yml`) and on `v*` release tags, not on every push (decided by
  the user, Nov 7, 2026). Where a **Done when** says "CI fails if ...", the check must
  be a command in `Run by CI`, so the local gate enforces it too.
- If a task is unclear, add an open question under it in the plan instead of guessing.

## Parity with scanpy (and when not to copy it)

The default is **parity with scanpy and anndata**: same results on the fixtures,
within the tolerance table, and the same meaning for the same parameters. But the
reference can be wrong or surprising (silent densification, order-dependent results,
in-place mutation, unseeded randomness). Do not copy those blindly. When the
reference output or behaviour looks wrong:

1. **Note it first.** Write down what looks wrong and how to reproduce it (in the commit
   message, and as a `memory/` entry if it is a lasting decision).
2. **Check a second implementation** (Seurat, rapids-singlecell, the original paper
   or reference package of the method) to see whether the behaviour is intended.
3. **Still unsure? Ask the user.** Do not decide a deviation on your own when it is
   not obvious.

**Record every deviation, no exceptions.** Any place where kyto gives different
results or semantics than scanpy or anndata gets a row in
[deviations.md](deviations.md), in the same commit that introduces it: big or small,
on purpose or a known gap left for later. Each row says what the reference does,
what kyto does, why, and when and by whom it was decided. Tolerances live in the
parity table, not in individual tests.

Parity covers results and semantics only. It does **not** cover API shape (the design
rules win: no `.X`, no in-place mutation) or code quality (type hints, docstrings and
error messages are held to the bar below, not the reference's).

## Golden rules

1. **Test everything before handing over. Not just smoke tests.** Run `cargo test`,
   `pytest`, the parity tests and the task's **Done when** check yourself.
2. **Verify results, not just that code runs.** Compare against scanpy on the
   fixtures. When a task produces something visual (an embedding, a clustering),
   plot it and look at it.
3. **Be surgical.** Do exactly the task. If you must touch anything outside it,
   **ask the user first. Do not fix it directly.** Note it as an open question or a
   new task instead.
4. **Simple beats clever.** Modular, re-purposable, no overengineering. 20/80 rule:
   do the 20 percent of work that delivers 80 percent of the value first.
5. **Ask the user** when unsure or stuck on a choice. Do not guess on design decisions.
6. **Bug-hunting sub agents.** After implementing a task, spin up sub agents whose
   duty is to assume the code is faulty, find the bug, and report back. Fix what they
   confirm (surgically). Re-verify. Use the `bug-hunter` agent
   (`.claude/agents/bug-hunter.md`, runs on Sonnet) for these.
7. **Record critical lessons** (an approach that works, or one that breaks things) as
   a file in `memory/` and add it to the index in [memory/README.md](memory/README.md).
8. **Never cheat benchmarks.** Same data, same work, same output on both sides, same
   precision (f32 vs f64), same thread count stated. Report time and peak memory
   honestly, including the cases where kyto loses.
9. **Format and test before every commit.** `cargo fmt --all`, `ruff format`,
   `ruff check --fix`. Tests must pass at each commit.
10. Mindset of an industry lead: elegant yet robust. Robustness over flexibility.
11. **Share progress.** At each milestone that produces something to look at (an
    `explain()` tree, a parity table, a benchmark table, a plot), show it to the user.

## Naming and style

- Polars-style naming: full words, no abbreviations (`cell_index`, not `cell_idx`;
  `frame`, not `df`). Well-known terms are fine (`n`, `x`, `pca`, `knn`, `hvg`, `umap`,
  `qc`, `nnz`, `csr`). Polars frames are named `frame`.
- Rust: `rustfmt`, `clippy -D warnings`. Errors via `thiserror`, no `unwrap()` on
  user-controlled input. Functions return one type (use `Result`/enums, not
  shape-shifting). `unsafe` only in `kyto-exec` kernels and `kyto-gpu`, each block
  with a `// SAFETY:` note.
- Python: `ruff format`, `ruff check` with import sorting, polars over pandas.
  Functions return a single type (R3).
- Python type hints are **comprehensive**: every public and private function,
  parameter and return value, using precise types (`Literal[...]`, `X | None`, typed
  containers, `@overload` where the input type decides the output type), not `Any`
  unless truly unavoidable. Checked with `pyright --strict`.
- Python docstrings are **NumPy style** on every public function and class: summary,
  `Parameters` (with `type, default=...`), `Returns`, `Raises`, and `Examples`
  (runnable, checked as doctests). Every docstring that returns data says whether it
  copies.
- No em dashes in code, docs, or comments.

## Errors are first-class

Errors are part of the API, Rust style: typed, informative and solution-oriented.

- Every error has a stable `KY-Exxx` code registered in `errors.toml` (R10). The Rust
  enum and the Python classes are generated from it; never raise an unregistered error.
  To add one: add an entry to `errors.toml` (its header states the rules), run
  `uv run python scripts/error_registry.py`, return the new `KytoError` variant from
  Rust (`kyto_error`), and convert it in `kyto-py` with `errors::to_py_err`. CI fails
  on drift and on any error built outside the registry (`--check`).
- Every error says **what** was wrong, **where** (which method, argument, column,
  gene, layer or plan node), **what was accepted**, and **how to fix it** (a
  suggestion, close matches for typos: `unknown gene 'CD3'; did you mean 'CD3E',
  'CD3D', 'CD3G'?`).
- Rust: one `thiserror` enum per crate boundary, variants with structured fields, no
  stringly-typed catch-alls. Never panic or `unwrap()` on user input.
- Python: specific exception classes under `KytoError`, each subclassing the matching
  builtin (`ValueError`, `TypeError`, `MemoryError`, ...) and each with a docstring.
- **Fail early.** Name, type and axis errors fire when the plan is built, before any
  data is read (R5, R6). Memory-limit errors fire before execution starts.
- Tests assert on the error code, the error type and the key parts of the message.

**Order of work:** implement and verify the feature first, but new code must not
leave a backlog: write hints, docstrings and coded errors as you go.

## Architecture (keep it this way)

| Path | Holds |
| --- | --- |
| `crates/kyto-error` | `KytoError`, generated from `errors.toml`; every crate returns it |
| `crates/kyto-store` | Zarr layout, tile I/O, importers |
| `crates/kyto-core` | Schema, expressions, name resolver, handles |
| `crates/kyto-plan` | Logical plan, optimizer, `explain`, `estimate` |
| `crates/kyto-exec` | Streaming CPU executor, SIMD kernels |
| `crates/kyto-algo` | QC, HVG, PCA, kNN, clustering, DE |
| `crates/kyto-gpu` | CubeCL kernels, cudarc bindings |
| `crates/kyto-py` | PyO3 bindings only. Thin: convert, call core, return. |
| `python/kyto` | Python package and `.pyi` stubs |
| `tests/` | pytest (API, parity, property tests) and fixtures |
| `benchmarks/` | criterion and pytest-benchmark, vs scanpy and rapids-singlecell (not yet: task 10.3) |
| `docs/` | `format.md` store format spec (2.1); docs site sources (not yet: task 10.5) |
| `errors.toml` | the error registry (codes, classes, templates, hints, docs links) |
| `scripts/` | repo tooling, such as `agents_commands.py` (runs the Commands below) |
| `memory/` | lessons learned (indexed) |
| `.github/workflows/` | CI (`ci.yml`) |
| `.claude/agents/` | sub agent definitions (`bug-hunter`) |
| `deviations.md` | every deviation from scanpy or anndata |

Pipeline:

```
Python chain of methods
  -> logical plan (typed, names resolved, validated, no data read)
  -> optimizer (pushdown, fusion, sub-plan reuse)
  -> physical plan (device, block sizes, passes)
  -> streaming executor over CellBlocks (CPU or GPU)
  -> terminal: Polars | NumPy | SciPy | DLPack | Zarr store | h5ad
```

**Extension points are small, well-named units.** Adding a plan node, kernel or
algorithm should be one new file plus a registration and tests. If adding a feature
requires editing many unrelated files, the foundation is wrong: stop and tell the user.

## Agent-friendliness

- Every module starts with a short doc comment: what it does, what it does not.
- Small files, one concept each. No hidden global state (R7).
- An agent must be able to inspect a dataset and a plan without reading the matrix:
  `schema()`, `peek()`, `features()`, `estimate()`, `explain()`.
- One command runs everything an agent needs before a commit (lint, type check,
  cargo test, pytest). Keep it fast and green.
- Snapshot and reference files live next to the tests that own them. Update them
  only on purpose, and say so in the commit message.

## Cost discipline

Tokens are a budget. Spend them on design and verification, not on re-reading or noise.

- **End the session at the end of a task.** After the task is committed to `main`, pushed, and
  the plan is ticked, stop. Tell the user the task is done and the next one should start in a
  fresh session.
- **Read narrowly.** Search before reading. Read the line range you need, not whole
  files. Do not re-read a file you just edited.
- **Test narrowly while iterating.** Run targeted tests (`cargo test -q -p <crate>
  <name>`, `pytest -q -k <name>`) while working. Run the full check once, before
  each commit.
- **Keep output short.** Pipe long command output through `tail` or `grep`. Never dump
  full matrices, plan JSON or benchmark logs into the context. Inspect slices.
- **Cheap sub agents for cheap work.** Searches and mechanical checks go to sub agents
  on a cheaper model. Do not spawn a sub agent for something a few tool calls settle.
- **No open-ended loops.** Before a performance or polish task, ask the user for a
  stop condition (a target number or a time box). If a fix fails twice the same way,
  stop and ask instead of trying a third variant.

## Workflow per task

1. Read the plan, pick an unblocked unticked task. Read relevant `memory/` entries.
2. Look at how the reference does it (scanpy, anndata, and for storage and planning,
   Polars and Zarr). Note the parameters and edge cases you must match.
3. Implement minimally. Write unit tests (Rust) and
   integration and parity tests (Python).
4. Run the task's **Done when** check. Compare with scanpy on the fixtures. If the
   reference looks wrong, follow "Parity with scanpy" above (note, check, ask).
5. Spin bug-hunting sub agents. Fix confirmed bugs surgically.
6. Benchmark if the task touches a hot path (time and peak memory).
7. Format, run all checks, tick the task in the plan, commit to `main` with the
   task ID, and push (`git push origin main`).

## Commands

This is the single list of commands. The local gate and CI both run the `Run by CI`
block below line by line (`scripts/agents_commands.py`; in CI, the lint job) and fail if any command fails or leaves the
git working tree changed. Change a command here and CI follows; never copy one into
`.github/workflows/ci.yml` by hand. Run from the repo root.

### Run by CI

```bash
uv sync                                   # dev env; builds and installs the extension into .venv
uv sync --reinstall-package kyto          # force an extension rebuild (never mix in `maturin develop`)
uv run cargo build --workspace --locked   # build all crates (uv run: PyO3 sees .venv python)
uv run cargo test --workspace --locked    # Rust tests (Linux: libpython on LD_LIBRARY_PATH, see below)
uv run python scripts/remote_store_tests.py   # tile reader suite on S3 in MinIO: native `minio` on PATH, else docker or podman (prints SKIPPED without any)
uv run cargo clippy --workspace --all-targets --locked -- -D warnings   # Rust lint
cargo fmt --all                           # Rust format
uv run ruff check --fix && uv run ruff format   # Python lint (import sorting on) and format
uv run pyright                            # Python type check (strict, set in pyproject.toml)
uv run python scripts/error_registry.py --check   # errors.toml: generated code in sync, no unregistered error (R10)
uv run pytest                             # Python tests (API, parity, property)
uv run pytest tests/fixtures              # build fixtures (pbmc3k downloads once to .cache/), check digests
uv run pre-commit install                 # run the hooks on every git commit (once)
uv run pre-commit run --all-files         # fmt, clippy, ruff, pyright, before every commit
```

### Run the whole list

The local gate before every task commit, and what CI does (`--list` only prints the
commands):

```bash
uv run --no-project python scripts/agents_commands.py --list
uv run --no-project python scripts/agents_commands.py
```

### On purpose only (not run by CI)

```bash
uv run --directory tests python -m fixtures.make_reference   # regenerate scanpy references
uv run cargo test --release -p kyto-exec --test thread_scaling -- --ignored --nocapture   # 5.2 thread scaling on pbmc3k (timings)
uv run cargo bench -p kyto-exec --bench kernels   # 5.3 SIMD kernel benches (criterion, timings)
uv run cargo test --release -p kyto-exec --test kernels -- --ignored --nocapture   # 5.3 log1p over every f32 (~2.5 min)
```

- **Bench:** only 5.3's kernel bench above (criterion, in `crates/kyto-exec/benches`). The
  suites against scanpy (`criterion`, `pytest-benchmark`) arrive with task 10.3, which adds
  them to `Run by CI` (or here, if too slow for every task).
- **Linux:** the Rust test binary links libpython, so `cargo test` needs it on the
  loader path once per shell:
  `export LD_LIBRARY_PATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"`.

## Memory index

See [memory/README.md](memory/README.md). Always read it before starting work.
