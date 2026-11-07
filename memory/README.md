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
