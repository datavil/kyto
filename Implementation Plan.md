# kyto — Implementation Plan

Nov 6, 2026 · @Zafer Kosar

## 0. How to use this plan

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
  - Note: kyto commits straight to `main`. Since Nov 7, 2026 (user decision) `.github/workflows/ci.yml` runs only on `workflow_dispatch` and `v*` tags, not on push or pull requests; the full matrix stays as is for those runs.
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
- [x] **2.3 Metadata tables.** Cell and gene tables via Polars Parquet; IDs unique and checked on write; strings stored as categoricals.
  - Depends: 2.1
  - Done when: writing duplicate IDs raises `KY-E101`; round trip keeps dtypes.
  - Spec: `docs/format.md` §5 (tables, IDs, dtypes, Enum, nulls, names, Parquet settings), §8 (`schema.json` columns).
- [x] **2.4 Tile reader.** Parallel reads that skip tiles outside the selected genes (projection) and cells (predicate); yields a stream of `CellBlock`s (5.1).
  - Depends: 2.2
  - Done when: a test counter shows that selecting 3 genes reads one gene block per cell block, nothing more.
  - Spec: `docs/format.md` §4.7 (reading a tile), §4.8 (tiles to CellBlock), §3 and §11 (version check on open).
  - Done: `kyto_store::LayerReader` (generic over Zarr storage; `tests/tile_reader.rs` counts reads with a wrapping store). Selections are plain canonical positions (`AxisSelection`: all, range, strictly increasing indices); unsorted, repeated or out-of-range positions raise `KY-E114` (the plan canonicalises, §4.8). The minimal `CellBlock` lives in kyto-store (exec depends on store); 5.1 adds pooling and alignment behind its slice accessors. Open checks `format_version` (`KY-E115`, `KY-E116`); `read_layer` (2.2) does not yet.
- [x] **2.5 Streaming writer.** Writes `CellBlock`s as they arrive; writes to a temporary path and renames on success, so a failed run leaves no half store.
  - Depends: 2.2, 2.3
  - Done when: killing a write midway leaves the target path absent.
  - Spec: `docs/format.md` §10 (atomic write, write order, determinism), §8, §9, §3 (root `zarr.json` last).
  - Done: `kyto_store::StoreWriter` (`create` with the gene table and `LayerSpec`s, `push` one `CellBlock` per layer, `finish` with the cell table and an optional `ImportSource`); public checked `CellBlock::from_f32`/`from_u32`. Re-blocks into cell blocks of `C` rows and writes shards as they fill; bytes equal `write_layer`'s for any batching and thread count. Target exists: `KY-E118` (overwrite replaces only a kyto store); bad blocks `KY-E117`; table length `KY-E119`. Stale temporaries are found by a free OS file lock and deleted by the next write (format.md §10.1 step 5). `tests/store_writer_kill.rs` SIGKILLs a child writer midway. Open: the declared `rust-version = "1.85"` no longer holds (`File::lock` is 1.89, polars needs 1.95); `atomic_dir.rs` allows `clippy::incompatible_msrv` until the workspace bumps it. Enum `ordered` flags are always `false` from this writer; 2.6 needs a way to pass them.
- [x] **2.6 Importers.** `ky.convert()` from `.h5ad` (dense, CSR, CSC; streamed through the `hdf5` crate), 10x `.h5`, and `.mtx` + TSV. `on_duplicate_ids="error"` by default, `"suffix"` opt-in.
  - Depends: 2.5
  - Done when: every fixture converts and matches anndata's values exactly.
  - Done: `kyto_store::convert` / `ky.convert(source, target, *, layer="counts", on_duplicate_ids, overwrite)`. `.h5ad` (anndata 0.8 to 0.13 encodings: `X` and `layers/*` dense, CSR, CSC; `obs`/`var` columns incl. categoricals with `ordered`, nullable arrays), 10x `.h5` v2/v3, 10x mtx directories (optionally gzipped). HDF5: `hdf5-metno` 0.15 with `static`, `zlib`, `lzf`, built from bundled source with CMake (`cmake` PyPI package in the uv dev group and `[build-system] requires`; user decision, Nov 7, 2026; memory/hdf5_static_build.md). Streaming: CSR/10x by cell block, dense in slabs of at most `memory_budget_bytes` (256 MiB), CSC and mtx by repeated scans collecting cell ranges within the budget (`1 + entries / budget` reads, no full load, R8). IDs checked (or suffixed as anndata's `make_index_unique`) before any matrix is read. New errors `KY-E120` (unsupported input), `KY-E121` (invalid input). Enum `ordered` passes through `StoreWriter::mark_ordered` (the 2.5 open item). `tests/import_parity.rs` converts both fixtures in 7 formats each and compares layers, tables and `schema.json` with anndata/scanpy exactly; deviations in deviations.md. Open: the 6-target build of the static HDF5 (Windows ARM64 above all, and manylinux containers having `cmake`) is verified only at the next on-demand or release CI run.
- [x] **2.7 h5ad export.** `sink_h5ad()` for handing results to scanpy.
  - Depends: 2.6
  - Done when: `anndata.read_h5ad` on the export equals the source fixture.
  - Done: `kyto_store::sink_h5ad(store, target, &SinkH5adOptions { x_layer, overwrite, blocks_in_flight })`, bound only as the private `kyto._kyto.sink_h5ad` (user decision, Nov 7, 2026: the public `sink_h5ad()` is a plan terminal added with 7.2/7.3, which wraps this function). Writes the anndata 0.13 h5py layout the 2.6 importer reads: `obs`/`var` dataframes (index `cell_id`/`gene_id`; `Enum` as `categorical` with `ordered` from `schema.json`; integers and booleans with nulls as `nullable-integer`/`nullable-boolean`; float nulls as `NaN`), `x_layer` as `X` and the other layers under `layers/`, all `csr_matrix` with the stored dtype (`int32` indices/indptr when they fit), empty `obsm`/`varm`/`obsp`/`varp`/`uns`. Streams each layer one cell block at a time through the tile reader (memory: the two tables and `blocks_in_flight` cell blocks, R8); written to `.kyto-part-<random>` next to the target and renamed on success. New `store_schema` reads `schema.json`. New errors `KY-E122` (invalid `schema.json`), `KY-E123` (target exists or lies inside the store; overwrite replaces files, never directories), `KY-E124` (names or strings h5ad cannot hold: empty, `/`, `.`, `_index`, NUL). `tests/test_sink_h5ad.py` round-trips both fixtures (anndata write, convert, sink_h5ad, `anndata.read_h5ad`) and compares with the source exactly; `crates/kyto-store/tests/export_h5ad.rs` round-trips every column dtype through the importer and checks encodings. Deviations in deviations.md.
  - Open: a killed export leaves its hidden `.kyto-part-*` file behind (no lock-file cleanup as for stores, `atomic_file.rs`). No compression option yet (anndata's default is none too).
- [x] **2.8 Remote stores.** S3, GCS and HTTP through `object_store`, with the same reader as 2.4.
  - Depends: 2.4
  - Done when: the reader test suite passes against a local MinIO container.
  - Done: `kyto_store::RemoteStorage::open(url, options)` (`s3://`, `gs://`, `http(s)://`) implements zarrs `ReadableStorageTraits` over `object_store` 0.13 (the version polars already uses), so the 2.4 `LayerReader` reads remote stores unchanged: projection and predicate tile skipping as before, every read a ranged GET (suffix ranges for shard indexes), up to 32 ranges of one call in flight, tiles in parallel across rayon threads, on a tokio runtime the storage owns (R7). Options are `object_store` config keys; S3 and GCS also read their standard env chain (`AWS_*`, `GOOGLE_*`), options win. New error `KY-E125`: bad URL or scheme, unknown option key, bucket options that would override the URL's bucket, a query, fragment or credentials in the URL, a rejected configuration; option values are never echoed. Not bound to Python yet (no reader is). Every test of `crates/kyto-store/tests/tile_reader.rs` (and `remote_storage.rs`) runs on the local, HTTP (in-process range server, always) and S3 backends; `scripts/remote_store_tests.py` (in the gate) starts MinIO, natively from a `minio` binary on PATH or from a pinned image in docker/podman, and runs them with S3; without either it prints SKIPPED (`KYTO_REQUIRE_MINIO=1`, set in the CI lint job, makes that a failure). Passed here against native MinIO RELEASE.2025-10-15T17-29-55Z (user decision, Nov 7, 2026: native binary instead of a container, no container runtime on this Mac).
  - Open: GCS is unit-tested for configuration only; no live test (no emulator in the gate). The Homebrew `minio` formula used on this Mac is deprecated (upstream archived) and will be disabled on 2027-02-17; after that, install MinIO otherwise or use docker/podman. The pinned container image tag (`RELEASE.2025-04-22T22-12-26Z`) is unverified until the next CI run. Remote reads of the tables and `schema.json` (still `Path`-based) come with the store-level open of the data model tasks.

## 3. Data model (`kyto-core`)

Goal: every name and type is checked when the plan is built, before any data is read.

- [x] **3.1 Schema types.** `AtlasSchema`: cell and gene table schemas, layers (name, dtype), embeddings (name, width), graphs, modalities. Serializable with serde.
  - Depends: 2.1
  - Done when: schema read from a store equals the schema written; JSON round trip is exact.
  - Done: `kyto_core::AtlasSchema` (`schema.rs`, `schema_column.rs`): `n_cells`, `n_genes`, `Tiling`, `cells`/`genes` `TableSchema` (`ColumnSchema { name, dtype: ColumnDtype, lineage }`, `Enum` with `n_categories` and `ordered`), `LayerSchema { name, dtype: LayerDtype, nnz }`, `EmbeddingSchema { name, axis, width, dtype }`, `GraphSchema { name, axis, dtype, nnz }`, `ModalitySchema`; serde in format.md §8 key order, unknown keys, wrong types and `null` optional keys rejected. It is the one model of `schema.json`: kyto-store now depends on kyto-core, `Tiling` (with its defaults), `ColumnDtype` and the column type (renamed `TableColumn` -> `ColumnSchema`) moved to kyto-core and are re-exported by kyto-store, the writer serializes an `AtlasSchema` (bytes unchanged), and the hand-written parser is replaced by `kyto_store::read_atlas_schema` (serde; the internal `StoreSchema` view of 2.7 is derived from it). Errors stay `KY-E110`/`KY-E122`/`KY-E104`/`KY-E105`, no new codes; `KY-E122` messages now give serde's reason with line and column instead of a key path. Tests: `crates/kyto-core/tests/schema_round_trip.rs` (proptest: any schema to JSON and back is equal and re-serializes to the same bytes; the §8 example byte for byte, `nnz` = `u64::MAX`), `crates/kyto-store/tests/atlas_schema.rs` (written store reads back equal, bytes equal), `import_parity.rs` (every converted fixture's `schema.json` re-serializes byte for byte). Not bound to Python (3.6/8.3 `schema()`).
  - Open: the reader is now strict: an unknown key, or a missing `embeddings`/`graphs` key, is `KY-E122` (2.x ignored them; every store kyto wrote has both). Consistent with format.md §11 while MAJOR is 0; from 1.0 readers must ignore unknown keys of newer minors, so revisit with 10.6. `ModalitySchema { name, n_features, features, layers }` is provisional: format v0 stores no modalities (the reader rejects a non-empty list, and an empty one is not written back); its shape is fixed by the format version that adds them.
- [x] **3.2 Atlas.** Holds `Arc<Plan>`, exposed as a `#[pyclass(frozen)]`. Every method returns a new `Atlas` (R2).
  - Depends: 3.1
  - Done when: a test shows the original plan is unchanged after 10 chained calls.
  - Done: `kyto_plan::Plan` (owned by kyto-plan per the repo map; kyto-core cannot hold it, kyto-plan depends on kyto-core) is an enum shaped like Polars' `DslPlan`: one variant per node, each node owning its input as `Arc<Plan>`, so extending a plan is one allocation and the old plan is shared, never changed. Nodes so far: `Scan { source, schema: Arc<AtlasSchema> }` (the root) and `WithDefaultLayer { input, layer }` (one file each; 4.1 adds variants the same way). Each node keeps its output schema as a shared `Arc` (`Plan::schema()` does not walk the chain); `nodes()` walks from a node to its scan; `default_layer()`, `root()`, `name()`, `Display` (one line per node). Equality, `Display` and `Drop` loop instead of recursing (a 100 000-node chain on a 256 KiB stack is tested). `with_default_layer` checks the layer against the schema and raises `KY-E109` (no new codes). Python: `kyto._kyto.Atlas` (`#[pyclass(frozen)]` holding `Arc<Plan>`, no constructor, no `__getitem__`/`X`/`obs`/`var`) with `with_default_layer(layer) -> Atlas`, `__str__`/`__repr__`; NumPy docstrings with doctests, stub in `_kyto.pyi`. Built by the private `kyto._kyto.scan(store)`, which checks the format version and reads only `schema.json` (`kyto_store::open_atlas_schema`, new). Tests: `crates/kyto-plan/tests/plan_chain.rs` (10 chained calls leave the original equal to a fresh scan, its text unchanged, each input shared by pointer; branches; KY-E109; long chains), `tests/test_atlas.py` (10 chained calls from Python, frozen, R4 attributes absent, KY-E109, missing store, doctests).
  - Open: `with_default_layer` is the one chain method so far, taken early from 7.2 because the Done-when needs a method to chain and it needs no expressions (3.3); confirm it belongs here or rename at 7.2. `scan` stays private (`kyto._kyto.scan`, local paths only) until 7.1 decides the public `kyto.scan` signature (remote URLs with `storage_options`, as 2.8). `Atlas` is not exported from `kyto` (7.1 fixes `__all__`); `__repr__`/`__str__` are a plain node list, not `explain()` (4.7).
- [x] **3.3 Typed expressions.** `CellExpr` and `GeneExpr` with a dtype and a source tag: `Column`, `Expression { layer, gene_id }`, `Profile { layer, cell_id }`, `Embedding { name, component }`. Metadata-only parts lower to Polars `Expr`.
  - Depends: 3.1
  - Done when: `cell_column(...) & gene_column(...)` raises `KY-E202` at build; `.str` on a numeric expression fails in pyright.
  - Done: `kyto_core::Expr<A>` (`expr.rs`) is generic over a sealed axis, with `CellExpr = Expr<CellAxis>` and `GeneExpr = Expr<GeneAxis>`, so Rust code cannot mix axes at all. An expression is an `Arc`-shared `ExprNode` tree (`Source`, `Literal`, `Binary` with `== != < <= > >= + - * / & |`, `Not`, `Str` with `starts_with`/`ends_with`) plus its `ExprDtype` kind (`expr_dtype.rs`: `Unknown`, `Boolean`, `Numeric`, `String`; `of_column` maps stored dtypes, `Enum` is `String`). `Source` (`expr_source.rs`) has the four tags; `layer` is `Option` (`None` = default layer). Columns start `Unknown`; `Expression`, `Profile` and `Embedding` are `Numeric`. Operand kinds are checked as each operation is built: a wrong kind is the new `KY-E205` (`ExpressionError`, a `TypeError`); `Unknown` is accepted and left to plan build. `Display` writes the Python code that builds the expression (`cell_column('n_genes') > 200`), used in messages and `repr`. `to_polars()` (`expr_polars.rs`) lowers metadata-only trees (`col`, `lit`, `/` as true division, `&`/`|` logical, `str.starts_with`/`ends_with` after a cast to String so `Enum` columns work), `None` when a matrix or embedding source is involved. Python (`crates/kyto-py/src/expr.rs`, one macro for both axes): frozen, unhashable `CellExpr`/`GeneExpr` with the operators and their reflected forms, `.dtype` (a `kyto.dtypes` class), `.str` namespaces, `__class_getitem__`; combining a `CellExpr` with a `GeneExpr` raises `KY-E202` (`AxisError`) as the two are combined, before any data is read (R5); other operand types are `KY-E205`; `bool(expr)` (also `and`/`or`/`not`, `a < b < c`) is the new `KY-E206`. Trees deeper than `MAX_EXPR_DEPTH` (1000) are the new `KY-E207` (a 300 000-term loop segfaulted before; bug-hunter); rendering and lowering grow the stack on demand with the `recursive` crate, as Polars does. `Display` escapes non-printable characters (`\x00`, `\u2028`), so it is always valid Python. Stubs (`_kyto.pyi`) are generic over the dtype kind (`python/kyto/dtypes.py`: `ExprDtype`, `Boolean`, `Numeric`, `String`, `Unknown`): `.str` is a property with a `self: CellExpr[String] | CellExpr[Unknown]` annotation, `&`/`|`/`~` take only `Boolean`, arithmetic only `Numeric`, ordering both sides `Numeric` or both `String` (overloads), and a cell/gene mix is a pyright error too. Polars dependency of kyto-core: `lazy`, `strings`, and `temporal` (polars-expr 0.55.2 does not compile `strings` without it). Tests: `crates/kyto-core/tests/expressions.rs` (dtypes, `KY-E205`, `KY-E202` text, rendering, lowering evaluated on a Polars frame), `tests/test_expressions.py` (`KY-E202` for every operator both ways, `KY-E205`/`KY-E206`, repr, immutability, runtime generics, doctests of the extension and stub docstrings), `tests/typing/expressions.py` (never run; checked by `uv run pyright` with `reportUnnecessaryTypeIgnoreComment=error`, so each expected error, such as `.str` on `CellExpr[Numeric]`, fails the gate if it stops erroring).
  - Open: `KY-E202` fires when the two expressions are combined, which is before plan build; there is no plan node taking expressions until 4.1 (`Filter`), and the check needs none. `cell_column` and `gene_column` (positional name only) were taken early from 3.4, which adds the rest and may extend their signatures; they are exported from `kyto` (7.1 lists the constructors at top level). `CellExpr`/`GeneExpr` live in `kyto._kyto` and the dtype markers in `kyto.dtypes`; 7.1 decides where they are exported. The dtype is a kind, not a storage dtype: 3.5 sets a column's kind from the schema and checks `Unknown` operands then (`KY-E205` at plan build). The constructor names in `Display` for the other sources (`gene_expression`, `cell_profile`, `cell_embedding`/`gene_embedding`) are placeholders until 3.4 names them. Static checks are looser than run time in two places: pyright cannot check the kinds of `==`/`!=` (it falls back to `object.__eq__`), and a `bool` passes where `float` is typed (`x + True` type-checks, raises `KY-E205`). Run-time choices to confirm: `x == None` is `KY-E205` (null tests come with 3.7); NumPy integer and bool scalars are `KY-E205` (NumPy floats subclass `float` and work). For 3.5: a column resolved to an integer must be rechecked under `~` (Polars `.not()` on integers is bitwise), and ordering on an `Enum` column compares category order in Polars, not text. Expressions are not picklable.
- [x] **3.4 Constructors.** `cell_column`, `gene_column`, `gene_expression`, `cell_profile`, `feature`.
  - Depends: 3.3
  - Done when: each has a stub, docstring and doctest (R12).
  - Note: minimal `cell_column(name)` and `gene_column(name)` were taken early by 3.3, whose Done-when needs them (`crates/kyto-py/src/expr.rs`).
  - Done: Python constructors in `crates/kyto-py/src/expr.rs`, exported from `kyto`, each with a NumPy docstring, a stub (`_kyto.pyi`) and doctests: `cell_column(name, /)` and `gene_column(name, /)` (unchanged, `Unknown`), `gene_expression(gene_id, /, *, layer=None) -> CellExpr[Numeric]` (`Source::Expression`), `cell_profile(cell_id, /, *, layer=None) -> GeneExpr[Numeric]` (`Source::Profile`), and `feature(name, /) -> CellExpr[Unknown]`. Embedding naming decided: `cell_embedding(name, component, /)` and `gene_embedding(name, component, /)` (`Source::Embedding`, `Numeric`), the names 3.3's `Display` already used; added because without them `Embedding` cannot be built from Python and its `repr` would name functions that do not exist. `layer=None` means the plan's default layer and is keyword-only, as `Display` writes it; names are positional-only, as in 3.3. `feature` only parses (R6 resolution is 3.5): `kyto_core::FeatureName` (`feature_name.rs`) is `Short { name }` (no `:`) or `Full { prefix, name }` (split at the first `:`, so gene IDs may hold `:`), with `Display` giving the parsed text back; `CellExpr::feature` stores it in a new, fifth source tag `Source::Feature { name: FeatureName }` (`Unknown`, not metadata, `to_polars()` is `None`, rendered `feature('log1p:CD3E')`), which 3.5 replaces with `Column`/`Expression`/`Embedding` at plan build. New errors: `KY-E208` (`ResolutionError`) for a name that cannot name any feature (empty, empty part around the first `:`, prefix outside `[A-Za-z0-9_.-]+`, format.md §2) and `KY-E209` (`ExpressionError`) for an embedding component that is not a non-negative `int` (`bool`, float, negative, above `2**64 - 1`). Doctests in the gate: `tests/test_docstrings.py` (run by `uv run pytest`, so no new `Run by CI` command) runs the runtime and stub examples of every function in `kyto.__all__` and fails if one has none. Tests: `crates/kyto-core/tests/feature_names.rs` (parse cases, `KY-E208` text, proptest `Display` round trip), `tests/test_constructors.py` (class, axis, dtype and `repr` of each; `repr` evaluates back to the same expression; keyword/positional-only; `KY-E202` across constructors; `KY-E208`, `KY-E209` fields).
  - Open: for 3.5, the meaning of a full name is not fixed by its syntax: `meta:<column>` and `<layer>:<gene_id>` are the forms the plan names, but a layer may be named `meta`, and a layer and an embedding may share a name (format.md §2), so `pca:0` could be a gene or an embedding component; 3.5 decides by kind against the schema (and raises `KY-E201` on a clash) or reserves a syntax. Embedding full names are not specified anywhere yet. `feature` is cell-axis only (the plan's examples are cell features); a gene-axis `feature` is not planned. A short gene name with a `:` cannot be reached through `feature` (use `gene_expression`). The stub types `feature` as `CellExpr[Unknown]` for every name. `cell_column`/`gene_column`/`gene_expression`/`cell_profile` accept any string, including empty; checks against the schema are 3.5's. NumPy integer components (`np.int64(3)`) are `KY-E209` at run time and rejected by pyright, consistent with 3.3's literal rule (NumPy ints are `KY-E205`); confirm with that choice. A name with a lone surrogate (`'a\ud800'`) raises PyO3's raw `UnicodeEncodeError`, not a coded error, in every constructor including 3.3's columns (bug-hunter, not fixed here).
- [x] **3.5 Feature resolver.** Maps short names (`CD3E`, `cluster`) to full names (`log1p:CD3E`, `meta:cluster`). No priority order: two matches raise `KY-E201` listing both. Short gene names need `default_layer`. Resolution is stored in the plan (R6).
  - Depends: 3.4
  - Done when: adding a column named `CD3E` after a plan is built does not change that plan's output.
  - Done: grammar in the new [docs/feature_names.md](docs/feature_names.md) (linked from format.md §2.1 and §5.6). Full names are `meta:<column>`, `<layer>:<gene_id>` and `<embedding>:<component>` (decimal, no sign or leading zeros), split at the first `:`; they are told apart **by kind against the schema**, with no reserved prefix and no forbidden layer name (format.md §2.1 already said so): every form that exists is a match, a short name matches a cell column and a gene of the default layer, and anything but exactly one match is an error. Decided by the user (Nov 7, 2026): the scan also reads the `gene_id` column of `genes.parquet` (new `kyto_store::read_ids`, only that column; `KY-E113` on a non-String or null ID or a row count other than `n_genes`), held as `kyto_core::GeneIds` in `Scan` next to the schema; cell IDs are not read; resolution is exposed as `Atlas.resolve(expression)`. `kyto_core::Resolver` (`resolve.rs`) rebuilds the tree bottom-up: `feature(...)` becomes `Column`/`Expression`/`Embedding`, every `gene_expression`/`cell_profile` without `layer` gets the default layer written in (so the stored expression names its layer in full), each column gets its kind from the schema (`ExprDtype::of_column`) and every operation is rebuilt through `Expr::binary`/`not`/`str`, so `Unknown` operands are checked now (`KY-E205`, including `~` on an integer column). Ordering an unordered `Enum` column is `KY-E205`, as pandas raises for unordered categoricals; an ordered one compares in category order (Polars' `Enum` order). `kyto_plan::Plan::resolve` resolves against the scan's schema and gene IDs and the nearest default layer; Python `Atlas.resolve(expression, /)` (`@overload`: same axis and kind in the stub) returns the resolved expression. Errors: `KY-E201` (hint and field docs updated: candidates are full names, with the kind appended where two are written alike, such as `meta:cluster (cell column)` and `meta:cluster (gene)` when a layer is named `meta`), new `KY-E210` (`ResolutionError`, unknown column, gene, empty cell ID, embedding component of the wrong axis or out of range, or `feature` name, with why for each form), new `KY-E211` (`ResolutionError`, a gene or cell read from the default layer when the plan has none; lists the layers), and `KY-E109` for a layer or embedding named but missing. A short gene name without a default layer that is also a column is `KY-E201` listing the gene in every layer (no priority for the column). Tests: `crates/kyto-core/tests/resolve.rs` (every form, clashes, kinds, `KY-E205` rechecks, unordered `Enum`, idempotence, a 1000-deep tree on a 256 KiB stack), `crates/kyto-plan/tests/resolve_in_plan.rs` and `tests/test_resolve.py` (the Done-when: a plan resolves `feature('CD3E') > 1` to `gene_expression('CD3E', layer='log1p') > 1`; the store is rewritten at the same path with a cell column `CD3E`; a new scan raises `KY-E201`, the old plan is unchanged and still resolves the same; plus every error from Python and the scan's `KY-E113`), `tests/typing/expressions.py` (overloads keep axis and kind).
  - Open: "stored in the plan" is, for now, the expression `Atlas.resolve` returns; 4.1's `Filter`/`WithColumns` must call `Plan::resolve` when built and store the result (4.2 then reports these errors with I/O counter 0; the scan now reads `genes.parquet`'s ID column, metadata only). **Resolved by 4.1:** `filter_cells`/`filter_genes` and `with_cells`/`with_genes` resolve each expression when the node is built and store the resolved one (so does `highly_variable_genes` for its `batch_key`). `cell_profile` checks only that the cell ID is not empty: the plan holds no cell IDs (millions of rows), so whether the cell exists waits for execution. Embedding resolution is tested in Rust only: the h5ad importer does not bring `obsm` over, so no Python store has an embedding. `Atlas.resolve` is public on `Atlas` but `Atlas` itself is not exported yet (7.1). Comparing an `Enum` column with a string that is not one of its categories is not checked (Polars raises when the plan runs); 3.7 or 4.2 may check it against the categories, which the plan does not hold. Messages quote the resolved expression (`gene_expression('CD3E', layer='log1p')`), not the `feature(...)` text the user wrote, when a `KY-E205` fires during resolution. Decided by the user (Nov 7, 2026): keep the error. Every non-ID string column is stored as an unordered `Enum` (format.md §5.4), so ordering any string column (`cell_column('cell_type') < 'B'`) is `KY-E205`, even one that was a plain string in the source `.h5ad`, where pandas would compare text (for such columns the writer's categories are sorted, so category order equals text order). Allowing it later is not breaking; see deviations.md. Duplicate gene IDs collapse silently in `GeneIds` (a valid store has none; `read_ids` does not re-check uniqueness). Fixed in 3.9: `read_ids` raises `KY-E113`. format.md §5.4 now states the `ordered` rule and its §13.2 open question is removed (bug-hunter: stale).
- [x] **3.6 Handles.** `Table`, `Matrix`, `Embedding`, `Graph`, `Modality`: read-only, lazy, each with `names()`, `schema()`, `estimate()` and its fixed `to_*` terminals.
  - Depends: 3.2
  - Done when: no handle exposes a method that mutates; checked by a stub test.
  - Done: Rust handles in `kyto_plan` (`handles.rs`; in kyto-plan, not kyto-core, for 3.2's reason: a handle holds an `Arc<Plan>`): `Table` (cells or genes), `Matrix` (a layer), `Embedding`, `Graph`, `Modality`, each holding the plan (shared, never changed) and the member's schema entry, checked when built (`KY-E109`, now also for kind `modality`). Read from metadata only: `names()` (Table: column names in stored order; Matrix: gene IDs in table order, so `GeneIds` now keeps the order as well as the set; Embedding: `<name>:<component>`, the full-name form of docs/feature_names.md; Graph: its edge-list columns `source`, `target`, `weight`; Modality: its layer names), `schema()` (the member's `schema.json` entry, format.md §8, as new dicts) and `estimate()` (`n_rows`, `n_columns`, `nnz`, `dense_bytes`, each `None` when not known from metadata; `dense_bytes` is a `u128` that saturates). Python: frozen pyclasses `kyto._kyto.Table`/`Matrix`/`Embedding`/`Graph`/`Modality` with no constructor, setter, `__getitem__`, `.X`/`obs`/`var` (R4) or `__array__` (R8), one-line metadata `repr`; stubs with `TypedDict`s for the returned dicts (type-checking only). Terminals, fixed per handle, with signatures, stubs and NumPy docstrings stating copy behaviour (R2): `Table.to_polars() -> pl.DataFrame`, `Table.lazy() -> pl.LazyFrame`, `Matrix.to_scipy() -> csr_matrix`, `Matrix.to_dense(*, max_bytes) -> ndarray` (keyword required, no default: R8), `Embedding.to_numpy() -> ndarray` (stored dense, so densifies nothing), `Graph.to_scipy() -> csr_matrix`, `Graph.to_edges() -> pl.DataFrame`; `Modality` has none. **All terminals are stubs**: each raises the new `KY-E601` (`NotImplementedYetError`, a `NotImplementedError`; new range KY-E6xx "not implemented yet") naming the method and task 7.3. None is real because each returns Polars, NumPy or SciPy objects, and kyto has no runtime dependency or hand-off for them yet (7.5), nor an executor (section 5). Accessors taken early from 7.3: `Atlas.cells`, `Atlas.genes` (properties), `Atlas.layer(name, /)`, `embedding`, `graph`, `modality`. `scipy` added to the dev dependencies (it types `to_scipy` in the stub; scipy ships no stubs, so its import carries a `reportMissingTypeStubs` ignore). Tests: `tests/test_handles.py`, whose Done-when test `test_stub_handles_expose_no_mutation` parses `_kyto.pyi` and fails if a handle class is not `@final`, has anything but undecorated methods (no fields, properties or setters), a dunder, or a public member other than its checked-in set (`names`, `schema`, `estimate` and its terminals); its runtime twin checks the classes against the stub, no constructor, no R4/R8 attributes, and that instances refuse `setattr`/`delattr`. Five injected mutations (a setter method, a property, a field, `__setitem__`, an extra `to_*`) each fail it. Also: values against `schema.json`, new objects per call, handles answer from the scan's snapshot, `KY-E109` per kind, `KY-E601` for every terminal, runtime and stub docstring examples (stub text equals runtime). Rust: `crates/kyto-plan/tests/handles.rs` (every handle, genes-axis embedding, a modality built by hand, `KY-E109` text, shared plan, saturation), and a compile-time check that every handle pyclass is frozen (`crates/kyto-py/src/handles.rs`).
  - Open: `estimate()` holds only what metadata gives; bytes read, peak memory, passes and device are 4.6, and `n_rows`/`nnz` become `None` once filters exist (4.x). **Resolved by 4.1:** estimates read the plan's output schema: a size not known exactly (the axis after a filter on it, every `nnz` after a filter or selection over its axis, a neighbour graph's `nnz`) is `None`. The meaning of `names()` per handle (above) is a choice to confirm at 7.3, notably `Graph.names()` (edge columns, since the plan holds no cell IDs) and `Modality.names()` (layers). `to_edges` column names and whether `source`/`target` are positions or IDs, `csr_matrix` versus `csr_array`, and whether a `Matrix` gets `to_torch(device=)` (9.5) are 7.3's; `to_torch` is not defined yet. `to_dense` takes any `max_bytes` until 7.3 checks it (the stub says `int`). Modality is minimal: no v0 store has one, so its methods have no runnable example (R12) and no terminals until the format adds modalities. Names with a lone surrogate (`atlas.layer('\ud800')`) raise PyO3's raw `UnicodeEncodeError`, and non-`str` names a plain `TypeError`, as in 3.4 (bug-hunter). Classes can be monkeypatched at class level (PyO3 heap types); instances are frozen. `ruff` drops `__repr__` from stubs, so it is runtime-only. Python has no runtime dependency on Polars, NumPy or SciPy yet; 7.5 decides.
- [x] **3.7 Missing-value rules.** Sparse absence densifies to 0; missing metadata is null; reading a gene removed by a filter raises `KY-E203` (R9).
  - Depends: 3.5
  - Done when: tests cover all three cases.
  - Done: each rule lives where later tasks inherit it. **Densify to 0:** `CellBlock::to_dense()` (row-major `n_rows x n_genes`, in the layer's dtype as `DenseValues`) and `CellBlock::gene_values(position)` (one gene for every row, as an expression reads it) in kyto-store; every absent entry is 0, stored `NaN`/`inf` are copied; `KY-E117` for a position not below `n_genes`, a size above `isize::MAX` bytes or a failed allocation (checked before allocating, never a panic). They allocate the whole result: the caller bounds it (R8, 7.3's `max_bytes`, 4.5's block sizes). **Missing metadata is null:** storage already had it (code `-1` and nullable masks become null, `NaN` is a value, format.md §5.5); expressions gain the null tests `is_null()` / `is_not_null()` (`kyto_core::NullTest`, node `ExprNode::Null`, any kind in, `Boolean` out, lowered to Polars, resolved with its input; Python methods on `CellExpr` and `GeneExpr` with stubs). Lowered expressions follow Polars' null rules (null in, null out; three-valued `&`/`|`; a filter keeps `True` rows only). 3.3's open choice is settled: `x == None` and `x != None` (and `None == x`) raise the new `KY-E212` (`NullComparison`, `ExpressionError`), naming the null test to write (`cell_column('a').is_null()`); `None` with any other operator stays `KY-E205`. **Removed genes:** `kyto_core::RemovedGenes` (gene ID to the node that removed it, upstream first) and `Resolver::with_removed_genes`; a gene read of a removed gene (`gene_expression`, or a `feature(...)` name resolving to that one gene) raises `KY-E203` (already registered by 8.1, unchanged) at plan build. A removed gene keeps its name, so a short name that also matches a column stays `KY-E201` and an unknown gene stays `KY-E210`; `cell_profile` is not affected. `Plan::removed_genes()` is wired into `Plan::resolve` and is empty: its exhaustive match over nodes makes 4.1 add `SelectGenes`. Parity: `X.toarray()` and `sc.get.obs_df` give 0 for absent entries, as kyto; anndata reads code `-1` and masked entries as `NaN`/`pd.NA`, kyto as null; reading a removed gene is a `KeyError` in anndata and scanpy, `KY-E203` here (an error in both). Three rows in deviations.md: null propagation versus pandas' "missing compares False", `NaN` in float metadata, and `== None`. Tests: `crates/kyto-store/tests/missing_values.rs` (through the tile reader over partial and unstored tiles and gene selections, `float32` and `uint32`, edge shapes, errors, a proptest), `crates/kyto-core/tests/missing_values.rs` (null tests: kinds, code, depth, resolution; evaluated on a frame with nulls and `NaN`; `KY-E203` for every gene read and its interplay with `KY-E201`/`KY-E210`/`KY-E211`), `tests/test_missing_values.py` (a converted `.h5ad` with a missing categorical, nullable integer and boolean has nulls and keeps `NaN`; null tests resolve; `KY-E212` messages), and `tests/typing/expressions.py` (null tests type as `Boolean` of the same axis). Bug-hunter: fixed `to_dense` panicking or aborting on huge blocks (only the value count was checked), the garbled `== None` message (now `KY-E212`), and two wrong pandas claims in deviations.md.
  - Open: which genes a filter on data keeps (`filter_genes(gene_column('highly_variable'))`) is known only when the plan runs, so for those `KY-E203` comes from the executor (5.x), with the same error and `RemovedGenes`; only `SelectGenes` by ID is caught at plan build. (4.1: done; `select_genes` adds the genes it leaves out to `removed_genes()`.) No Python test reaches `KY-E203` or densifying until 4.1 and 7.3 add a gene filter and a terminal. Decided by the user (Nov 7, 2026): (1) keep Polars' null rules in filters (`~(x > 1)` and `x != 'a'` drop missing rows, where pandas keeps float and categorical ones; deviations.md); (2) the `.h5ad` importer makes `NaN` in `obs`/`var` float columns null (anndata has no nullable float), so `is_null()` matches pandas' `isna` on imported data. format.md §5.5 keeps `NaN` and null apart for kyto's own writers and states the importer's conversion; `NaN` in a matrix stays a value; the 10x and mtx importers read string metadata only, so they are not affected. The `.h5ad` export still writes float nulls as `NaN`, so `h5ad -> kyto -> h5ad` equals anndata (`tests/test_sink_h5ad.py`), and a float `NaN` that kyto wrote comes back null after a round trip (`crates/kyto-store/tests/export_h5ad.rs`). A `NaN` that kyto itself writes (a computed column, later) still compares as in Polars (`NaN > 1` is `True`); revisit if one arises. The null tests have no `Examples` section, like `str.starts_with` (the methods are shared by both classes); 7.1 or 8.x may add them (R12).
- [x] **3.8 Derived-column lineage.** Stored columns record the layer and plan node they came from. Reading one after an upstream gene filter raises `KY-E301` unless `.recompute()` is called.
  - Depends: 3.5
  - Done when: `n_genes` read after `filter_genes` raises `KY-E301`.
  - Done: lineage was already in the format (`ColumnSchema.lineage`, format.md §8, 3.1) and `KY-E301` in the registry (8.1); no format change. **Gene filter (taken early from 4.1/7.2, user decision Nov 7, 2026):** `kyto_plan::FilterGenes` (`filter_genes.rs`), `Plan::filter_genes(&GeneExpr)` and Python `Atlas.filter_genes(predicate)`: the predicate is resolved against the input when the node is built (R6) and must be boolean (`KY-E205`, also for a non-`GeneExpr` from Python); the node keeps its input's schema and adds nothing to `removed_genes()` (which genes it keeps is known only at run time); `explain` line `FILTER_GENES <predicate>`. **Stale check:** `Plan::gene_axis_changed_by()` (the first gene filter from the scan) feeds `Resolver::with_gene_axis_changed_by`; a cell column whose schema records `lineage`, read by `cell_column` or a `feature(...)` name, raises `KY-E301` after it, whichever genes are kept. Columns without lineage (all imported ones) and gene-table columns are read as stored. **`.recompute()` (user decision: an expression mark, executor later):** `CellExpr/GeneExpr.recompute()` builds `Source::Recompute { name }` (`cell_column('n_genes').recompute()`), only on a plain column read (new `KY-E213`); the resolver accepts it after a gene filter, takes its dtype from the schema, and needs lineage (new `KY-E302`, class `LineageError`); it is not metadata and never lowers to Polars, so the stale stored value cannot be read by mistake. deviations.md row (scanpy/Seurat keep stale QC columns silently). Tests: `crates/kyto-plan/tests/lineage.rs`, `tests/test_lineage.py` (a converted `.h5ad` whose `schema.json` marks `n_genes` with lineage), typing in `tests/typing/expressions.py`.
  - Open: (1) Nothing computes a recomputed column yet: the executor (5.x) must compute `Source::Recompute` from `lineage` (`node` points into the plan JSON of 4.3, kept in `provenance.json`; 6.1 says QC is computed on read, so it may supply the recipe), and a writer (5.x sink) must set or drop `lineage` on columns it writes after a filter. (2) 4.1: generalise `FilterGenes` into `Filter` (cells too) and add `SelectGenes` to `gene_axis_changed_by()` and `removed_genes()`; decide whether a cell filter stales gene-table columns with lineage (per-gene stats like `n_cells`, HVG flags) and whether a node that rewrites a layer stales columns whose `lineage.layer` is that layer (the layer is not used yet). **Resolved by 4.1:** `FilterGenes` is now `Filter` (cells or genes); `SelectGenes` (and `TakeCells` for cells) feed `gene_axis_changed_by()`/`cell_axis_changed_by()`, and `SelectGenes` feeds `removed_genes()`. Decided (agent, to confirm; deviations.md): a cell filter stales stored gene columns with lineage (mirror of the gene rule), and a step that writes an existing layer again (`log1p`/`normalize_total`/`scale` with `into` an existing layer) stales stored columns of either table whose `lineage.layer` is that layer; columns computed by the plan carry no lineage and are never stale. (3) Plan building is O(depth) per resolve (`removed_genes()` and `gene_axis_changed_by()` walk the chain), so N chained `filter_genes` cost O(N^2): about 1.5 s for 3 000 (bug-hunter). Cache both per node when 4.1 reshapes the nodes. **Resolved by 4.1:** every node caches its state when built (output schema, removed genes, axis and layer changes, default layer), so building is O(N); 20 000 chained `filter_genes` build in well under a second (test `building_a_long_chain_is_linear`). (4) `Table`/`Matrix` handles after a gene filter still report the input's sizes and list stale columns (upper bounds; 4.1 decides the filtered schema). **Resolved by 4.1:** a node's output schema (`PlanSchema`) keeps sizes as upper bounds next to the exact ones; after a gene filter `Matrix.estimate()` gives `n_columns`/`nnz`/`dense_bytes` as `None`, `Table` (genes) `n_rows` as `None`, and `Matrix.names()` lists the genes that may remain (exactly the selection after `select_genes`). Stale columns stay listed by `Table.names()`: they exist, reading them raises. (5) 7.2 settles the public name (`filter` with a `GeneExpr` vs `filter_genes`).
- [x] **3.9 Gene identity.** Genes indexed by ID. `by="symbol"` must match exactly one gene or raise `KY-E204` with the candidates.
  - Depends: 3.5
  - Done when: a fixture with duplicate symbols triggers `KY-E204`.
  - Done: no format or importer change. Symbols are the gene table's reserved `symbol` column (format.md §5.2, already declared by 2.1; the 10x `.h5` and mtx importers already fill it from the feature names, an `.h5ad` brings it when `var` has a `symbol` column); genes stay keyed by `gene_id`. The scan reads that column too, only it and only when `schema.json` lists it as `Enum` (new `kyto_store::read_symbols`, metadata, R1), into `kyto_core::GeneIds::with_symbols` (symbol to the IDs carrying it, in table order; nulls carry no symbol). `by=` is on `gene_expression` only: `gene_expression(gene_id, /, *, layer=None, by='id')`, `by` in `{'id', 'symbol'}` (stub `Literal`; anything else is the new `KY-E214`, `InvalidChoice`, `ResolutionError`). `by='symbol'` builds a new leaf `Source::GeneSymbol` (`CellExpr::expression_by_symbol`, rendered `gene_expression('CD3E', by='symbol')`), which the resolver turns once, at plan build (R6), into `gene_expression(<gene_id>, layer=...)`: exactly one gene gives its ID; several give `KY-E204` listing their IDs in table order (registered by 8.1; unchanged); none, or a store without an `Enum` `symbol` column, gives `KY-E210` with kind `gene symbol` and why. Matching is exact, case included; the removed-gene (`KY-E203`) and default-layer (`KY-E211`) checks then apply to the ID. 3.5's open item fixed: `read_ids` now refuses duplicate IDs (`KY-E113`), so `GeneIds` cannot collapse them. Grammar in docs/feature_names.md §3.1. Parity: `sc.get.obs_df(gene_symbols=...)` also raises on a repeated symbol (`KeyError`, without the genes); scanpy's readers key by symbol and `var_names_make_unique` renames repeats (`TBCE-1`), which kyto never does (deviations.md row). Tests: `crates/kyto-core/tests/gene_symbols.rs`, `crates/kyto-store/tests/gene_symbols.rs`, `tests/test_gene_identity.py` with the fixture `tests/gene_identity/duplicate_symbols` (a 10x matrix directory where two gene IDs carry `TBCE`, as in GRCh38-2020-A; the Done-when test), `tests/typing/expressions.py`.
  - Open: `feature(...)` names never look up symbols (full names hold IDs); a `by=` on `feature` or symbol lookup in other places (plots, `select_genes`, 7.x) is not planned yet. An `.h5ad` whose `var` index holds symbols and IDs sit in a column (common for scanpy output) imports the symbols as `gene_id` and has no `symbol` column, so `by='symbol'` there is `KY-E210`; choosing the ID column on import (a `convert` option) would be a 2.6 change for the user to decide. A `symbol` column stored with another dtype (an integer `var` column named `symbol`) is accepted by the writers and reported only at lookup (`KY-E210` naming its dtype); format.md does not forbid it. Close-match suggestions for unknown symbols are 8.2. A symbol is matched against every gene of the store, not only the genes left after a filter (bug-hunter): `filter_genes` removes nothing at plan build yet, so a repeated symbol stays `KY-E204` after the other gene is filtered out; once 4.1's `SelectGenes` feeds `RemovedGenes`, decide whether a symbol shared with a removed gene resolves to the kept one or stays ambiguous. **Resolved by 4.1:** genes removed at plan build are not candidates, as anndata's `var_names` hold only kept genes: after `select_genes`, a symbol shared with a removed gene names the kept one; when every carrier was removed, `KY-E203` names the first. After a filter on data nothing is removed at build, so a repeated symbol stays `KY-E204`.

## 4. Lazy plan and optimizer (`kyto-plan`)

Goal: chained methods build a DAG that the optimizer rewrites into the fewest passes over the data.

- [x] **4.1 Logical nodes.** `Scan`, `Filter`, `SelectGenes`, `TakeCells`, `MapElementwise`, `CrossCell` (scale, HVG, PCA), `WithColumns`, `Neighbors`, `Cluster`, `Sink`. Each declares which axis it changes and whether it is per-cell or cross-cell.
  - Depends: 3.3
  - Done when: every node has an output-schema function with unit tests.
  - Done: `kyto_plan::Plan` has one variant per node, each `Node<P>` (input, parameters `P`, state cached at build) in its own file with its builder (`impl Plan`): `scan.rs`, `with_default_layer.rs`, `filter.rs` (`filter_cells`/`filter_genes`, replacing 3.8's `FilterGenes`; explain lines `FILTER_CELLS`/`FILTER_GENES`), `select_genes.rs`, `take_cells.rs`, `map_elementwise.rs` (`normalize_total`, `log1p`), `cross_cell.rs` (`scale`, `highly_variable_genes` with `HvgFlavor`, `pca`), `with_columns.rs` (`with_cells`/`with_genes`), `neighbors.rs`, `cluster.rs` (`ClusterMethod` Leiden/Louvain), `sink.rs` (`Sink::Zarr`/`H5ad`). Every node implements `LogicalNode` (`name`, `method`, `changes_axis() -> Option<Axis>`, `cell_scope() -> CellScope::{PerCell, CrossCell}`, `describe`) and, except the scan, `Transform::output_schema(&PlanSchema) -> PlanSchema` (Polars' `IR::schema()` shape: from the input's schema and the parameters only). `PlanSchema` (`plan_schema.rs`) holds the members and columns as an `AtlasSchema` whose `n_cells`, `n_genes` and `nnz` are upper bounds (for 4.6/5.5), the exact `n_cells()`/`n_genes()` when known, and which `nnz` are only bounds. Output schemas: filters keep everything and make their axis length and the `nnz` over that axis bounds; `select_genes`/`take_cells` set the length exactly; `log1p`/`normalize_total` add or replace a `float32` layer `into` with the input's `nnz`; `scale` a `float32` layer (zero-centred: `nnz` bound `n_cells x n_genes`); `highly_variable_genes` scanpy's gene columns per flavour (`hvg_columns`); `pca` the cell embedding `into` and the gene embedding `<into>_loadings`; `with_cells`/`with_genes` columns with no lineage and a declared dtype (`column_dtype`, Polars-like rules; the executor casts); `neighbors` the cell graphs `distances`/`connectivities` (or `<key>_...`) with bounded `nnz`; `leiden`/`louvain` a `String` cell column; `sink` its input's. Per-cell/cross-cell: cross-cell are `scale`, `highly_variable_genes`, `pca`, `neighbors`, clustering, `normalize_total` without `target_sum` (median over cells), and gene filters or gene columns reading more than gene metadata (`cell_profile`). Every node checks its parameters when built: names resolved (R6: `Filter` and `WithColumns` store resolved expressions; layers default to the default layer, written in full), `KY-E109` for a missing layer/cell embedding/cell graph, `KY-E211` for no default layer, and two new codes (class `PlanError`): `KY-E215` `InvalidArgument` (zero components, non-positive resolution, repeated IDs, more cell IDs than the plan's bound, empty or reserved names, ...) and `KY-E216` `PlanEnded` (nothing follows a sink; checked first). Stable identity: `NodeId` = position from the scan (0), depending only on the chain below. Parameters and `PlanSchema` derive serde (parameters holding expressions wait for 4.3's expression serde). Cached `NodeState` per node (`node.rs`): schema, removed genes, first node changing each axis, rewritten layers, default layer, kept gene list, so building is O(N). Handles (3.6) report a size not known exactly as `None`. Resolver (kyto-core): `with_cell_axis_changed_by`, `with_rewritten_layers`, symbol lookup over genes not removed. Python: only `Atlas.filter_genes` exists (unchanged output); new test `test_handles_after_filter_genes_report_unknown_sizes`. Tests: `crates/kyto-plan/tests/logical_nodes.rs` (28: output schema, axis, scope, explain line and errors of every node, staleness, symbols, IDs, equality, serde, a 20 000-node chain). Bug-hunter: fixed `take_cells` with more IDs than the cell bound (now `KY-E215`), `normalize_total` checking `target_sum` before `KY-E216`, reserved names applied to both tables (now per table), zero-centred `scale` `nnz` claimed exact, and `/` with a literal not adapting like `*`. deviations.md: stale columns after a cell filter or layer rewrite, repeated selection IDs, PCA outputs, scale statistics, neighbours input.
  - Open: (1) Decisions to confirm (agent, mirroring the user's 3.8 rule): a cell filter stales stored gene columns with lineage; writing an existing layer stales stored columns whose `lineage.layer` it is; columns computed by the plan carry no lineage, so one computed before a later filter is not refused (e.g. `highly_variable_genes` then `filter_cells`). (2) `take_cells` takes cell IDs (R4: no positions) in the given order; existence is checked by the executor; 7.2 fixes the public signature and whether order is kept (streaming, 4.5). `select_genes` also keeps the given order. (3) Output names and dtypes to confirm when the algorithms land: HVG columns and their dtypes (`Float64` statistics, `Int64` `highly_variable_nbatches`) and whether `highly_variable_genes` gets `into=` (7.2 says every algorithm method does), and scanpy's cutoffs (`min_mean`, ...) are not parameters yet (6.3); `<into>_loadings` and no PCA variances (6.5); `scale` writes no `mean`/`std` gene columns (6.4); cluster labels are declared `String` and become `Enum` at the sink (6.8). (4) `column_dtype` approximates Polars' supertypes (mixed ints are `Int64`, any other mix `Float64`); 4.2 or 5.x should check it against Polars' own inference. (5) Names (`into`, `key`, column names) are checked for being non-empty and not reserved only; the format's name rules (format.md §2.1, §5.6) wait for 4.2 or the sink. (6) Embedding rows after a filter are not tracked (an embedding has no row count of its own; its axis length is used). (7) `neighbors` does not refuse `n_neighbors` above the cell count (the `nnz` bound is then loose); a tiny `target_sum` may underflow to 0 in `float32`, breaking "keeps `nnz`" (6.2). (8) Each `select_genes` clones the removed-genes map (about 400 MB for 300 selections on 30 000 genes, bug-hunter); fine for real plans, an incremental structure if it matters. (9) `NodeId` as depth fits chains; a DAG with joins (none planned) would need another scheme (4.3).
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
