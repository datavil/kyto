# Memory index

Critical lessons: approaches that work, or that break things. One file per lesson.
Add a line here for every new file. Format: `- [Title](file.md): one-line hook`.

Lessons carried over from ggrust (same stack: Rust + PyO3 + maturin + uv) are marked
`(from ggrust)`. Re-check them once the kyto build exists; delete any that do not apply.

- [One build path: uv](build_single_path.md): (from ggrust) never `uv run --no-sync` after `maturin develop`; it tests a stale extension (false green).
- [Never let a failing check reach a commit](check_before_commit.md): (from ggrust) don't pipe the check into grep and then `;` commit; gate on its exit status.
- [Tests need their own seeded generator](test_seed_isolation.md): (from ggrust) a shared RNG drawn at call time makes results depend on test order.
- [Fixtures and scanpy references](fixture_references.md): canonical digests, no libm samplers, pbmc3k needs gene filter before cell_ranger HVG, scanpy only on some platforms.
- [zarrs sharding is not deterministic in parallel](zarrs_shard_determinism.md): encode shards with concurrent_target(1) or shard bytes vary between runs; zarrs pinned to 0.22.x by its version requirement.
- [Polars Parquet for the metadata tables](polars_parquet_tables.md): row_group_size merges the last group, use the batched writer; small int dtypes need features; no MSRV.
- [Polars debug info vs. link memory on arm](polars_debug_link_memory.md): arm test job died linking polars debug binaries; dependency debug = line-tables-only, verify at next CI run.
- [Counting tile reads](counting_tile_reads.md): wrap the storage, map byte ranges via the shard index; missing keys and unstored inner chunks read nothing.
- [Streaming writer and atomic temporaries](streaming_writer_atomic.md): write chunks through max-shape arrays; tell stale `.kyto-tmp-` dirs by an OS file lock, never by name alone.
- [Static HDF5 through hdf5-metno](hdf5_static_build.md): user chose hdf5-metno `static` (bundled HDF5 via CMake from PyPI, dev group and build-system requires); 6-target build unverified until next CI run.
- [Release extension misaligned LINKEDIT on macOS 27](macos_ld_linkedit_alignment.md): ld with the 27 SDK leaves the string pool 4 bytes off and `import kyto` fails for release builds; `export SDKROOT=.../MacOSX26.5.sdk` on this machine fixes it.
- [MinIO for the remote store tests](minio_native_tests.md): native Homebrew `minio` here (formula disabled 2027-02-17), docker/podman in CI; SigV4 bucket creation; `Path::parse` for zarr keys.
- [serde validation and R10](serde_errors_r10.md): no `try_from` with String errors; hand-written `Deserialize` with `missing_field`/`unknown_variant`; reject `null` optional keys for exact round trips.
- [`.h5ad` float NaN becomes null](h5ad_float_nan_null.md): importer turns obs/var float NaN into null (user, 3.7); kyto writers keep NaN; a stored NaN round-trips through `.h5ad` as null.
- [Ask Polars for result dtypes](polars_dtype_inference.md): hand-copied supertype and literal rules were wrong; `column_dtype` asks `collect_schema()` with stand-in leaf columns.
- [Exact JSON floats and deep expression trees](plan_json_floats_and_depth.md): `serde_json` needs `float_roundtrip`; NaN/inf as text; flat `ExprJson` for 1000-deep trees; `from_json` replays the builders, so a new node needs a `Step`.
- [Checking optimizer rules before an executor](optimizer_reference_evaluator.md): shared-semantics reference evaluator, mutation-check proptests (delete saved seeds), fuse before and after predicate pushdown.
- [Measuring peak memory](measuring_peak_memory.md): `peak_alloc` global allocator in its own test binary; the reader's Vec doubling, zarrs whole-chunk decodes and the 1 MiB `tile_offsets` open are real and modelled.
- [Counting allocations](counting_allocations.md): `stats_alloc` global allocator, one test per binary (it sees every thread), macOS lock allocates once, assert a bounded count with a flat tail.
- [Executor threads](executor_threads.md): one rayon pool per run via `install` (reader, zarrs and operators share it), rendezvous channel, drop unread layers, thread scaling only in release as an ignored test.
- [SIMD kernels with the same bits everywhere](simd_kernels_same_bits.md): plain Rust inside pulp dispatch, fixed `f64` lanes and tree, own `log1p`; pulp `Scalar` is NEON on aarch64; `any::<f32>()` has no NaN.
- [Counting passes over the store](counting_store_passes.md): record tile reads as (key, range) counts and compare with one scan pass times P; an output reading no layer maps nothing.
- [Peak RSS is not the heap](rss_vs_heap.md): RSS = baseline + heap + 30-95% allocator overhead; memory_limit bounds RSS via (limit - RSS) / 1.5 (user, 5.5); measure in a child process.
