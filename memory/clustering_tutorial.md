# The clustering tutorial against scanpy (task 10.7)

**Works:** run the tutorial as a script in its own process (`docs/tutorials/clustering.py`)
and compare files it writes; peak RSS per stage from `ru_maxrss` (bytes on macOS, KiB on
Linux). Cache scanpy's run (it takes 95 s cold) keyed by its script, the input hashes and
the reference packages' versions, not by kyto's commit.

**Breaks (doublets at scale):** above 8192 rows a batch scanpy's `pp.scrublet` uses
pynndescent, and its `threshold_minimum` flips between two minima with the seed (`s1d3`:
5 to 167 calls over seeds 0 to 5). Comparing kyto's calls with scanpy's at one seed looks
like a 2x disagreement; compare with `use_approx_neighbors=False` over several seeds
instead (kyto then matches: 249-255 vs 252-255 in `s1d1`).

**Breaks (cellestial 0.62.1):** `cl.dotplot` raises `DuplicateKeysError` when a gene is in
two groups (scanpy's marker dicts do this, `FCGR3A`); dedupe first.

**Also:** sklearn `trustworthiness` needs all pairwise distances; on 17k cells score a
seeded sample of 3000. The 10x samples carry antibody features: filter
`feature_type == "Gene Expression"` to match `read_10x_h5(gex_only=True)`.
