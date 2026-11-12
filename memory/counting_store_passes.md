# Counting passes over the store: compare with one scan pass (task 5.4)

**Works:** to check that a plan reads the store exactly `P` times, record every tile
read as `(chunk key, byte range) -> count`, keeping only keys under
`layers/<layer>/{indptr,indices,data}/c/` (opening reads `zarr.json` and
`tile_offsets` once per run, not per pass). Record one plain scan of the layer the same
way; the plan's record must equal the scan's with every count times `P`
(`crates/kyto-exec/tests/it/multi_pass.rs`, `passes`). A count of requests alone is not
enough: one pass reads a shard index once per tile in that shard, so counts are not
uniform, and an extra partial read would hide in a total.

**Watch out:** an output that reads no layer must not open a stream that maps
anything: 5.4's first version ran the maps on cells-only blocks (`KY-E109` for the
scanned layer). Check a pass-count test by mutation (run one pass more): both the
fixture test and the median test must fail.
