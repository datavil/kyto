# Counting tile reads: wrap the storage, map byte ranges through the shard index (task 2.4)

**Works:** `LayerReader` is generic over zarrs `ReadableStorageTraits`, so a test passes a
`CountingStore` wrapper (`crates/kyto-store/tests/it/tile_reader.rs`) that records every
`(key, ByteRange)` in `get_partial_many` (zarrs' `get` and `get_partial` default to it).
zarrs 0.22 reads one tile of a sharded array as two partial reads: the shard index
(`n_inner * 16 + 4` bytes at the end) and the inner chunk's exact `[offset, nbytes]`
from that index. Parse the index from the shard file to map each range to an inner
chunk; a read of the whole file is a read of every inner chunk.

**Watch out:** chunks of fill values are not stored (format.md §4.4). A read of a missing
shard key returns `None` (counts as a request, reads nothing), and an inner chunk whose
index entry is `u64::MAX` is skipped after the index read. Indices of a width-1 gene
block are all 0, so such chunks appear even in a dense test matrix. Reset the counter
after `open` (it reads root and array `zarr.json` and `tile_offsets`). A mutation that
reads one extra tile must fail the test; check that after changing the reader.
