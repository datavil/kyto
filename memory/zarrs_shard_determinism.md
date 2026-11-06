# zarrs sharding: encode shards on one thread, or bytes vary (task 2.2)

**Broke:** zarrs 0.22's `sharding_indexed` encoder encodes inner chunks in parallel
and places each at an atomic `fetch_add` offset, so inner-chunk order inside a shard
(and the shard bytes) depends on thread timing. Two writes of the same layer gave
different `indptr` and `data` shard files, breaking byte-identical stores
(format.md §10.3, R11, 8.7).

**Works:** write every chunk through `zarr_arrays::store_chunk`, which passes
`CodecOptions` with `concurrent_target(1)`: the encoder then runs inner chunks in
index order. Parallelism, when needed (2.5), goes across shards, not inside one.
`tests/layer_format.rs::rewrites_are_byte_identical` guards it; run it several times
after touching the writer or upgrading zarrs.

**Also:** zarrs 0.23 needs Rust 1.91; the workspace MSRV is 1.85, so cargo resolves
zarrs 0.22.x. Write metadata with `with_include_zarrs_metadata(false)` so zarr.json
carries no zarrs version attributes.
