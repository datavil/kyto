# zarrs sharding: encode shards on one thread, or bytes vary (task 2.2)

**Broke:** zarrs 0.22's `sharding_indexed` encoder encodes inner chunks in parallel
and places each at an atomic `fetch_add` offset, so inner-chunk order inside a shard
(and the shard bytes) depends on thread timing. Two writes of the same layer gave
different `indptr` and `data` shard files, breaking byte-identical stores
(format.md §10.3, R11, 8.7).

**Works:** write every chunk through `zarr_arrays::store_chunk`, which passes
`CodecOptions` with `concurrent_target(1)`: the encoder then runs inner chunks in
index order. Parallelism, when needed (2.5), goes across shards, not inside one.
`tests/it/layer_format.rs::rewrites_are_byte_identical` guards it; run it several times
after touching the writer or upgrading zarrs.

**Also:** zarrs is pinned to 0.22.x by its version requirement (`0.22.10`); moving to
0.23 (needs Rust 1.91, within the 1.95 MSRV) is a deliberate upgrade: rerun the
byte-identity test. Write metadata with `with_include_zarrs_metadata(false)` so zarr.json
carries no zarrs version attributes.

**Since 5.17 (parallel and still identical):** `LayerStream` writes layer shards through
`zarr_arrays::store_shard`, which compresses a shard's inner chunks in parallel and assembles
the shard by hand: encoded non-fill chunks in chunk order, then the index (`u64::MAX` pairs
for skipped chunks) in `[bytes(little), crc32c]`; a shard all fill value is erased. These are
exactly zarrs' one-thread bytes; `zarr_arrays.rs`'s tests compare the files on pools of 1, 3
and 8 threads. If zarrs' shard layout changes on an upgrade, those tests fail first. Since 5.20
the sink's `push` runs on the run's pool (before, on rayon's global pool).
