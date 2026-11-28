# Exact neighbours: an `f32` product filters, the `f64` kernel decides (task 12.4)

**Works:** brute-force exact kNN at `f64` was 25 s on 1 thread at 50k; no tuning of the
`f64` kernel could reach scanpy (~9 s floor). An `f32` matrix product (Accelerate
`cblas_sgemm`, ~600 GFLOP/s on the M4's matrix unit from one thread) gives approximate keys;
keep every row with approximate key <= k-th approximate + 2E (E a proven bound, any summation
order), rank those with the old `f64` keys and (key, row) ties. Same graph bit for bit, 25x
faster (0.99 s). Ties at the k-th key are inside the set by construction: use the threshold
rule, not "top-m candidates then prove" (9.8's GPU path), which falls back on tied data.

- Centre Euclidean rows on the mean first: the bound scales with (|q| + max|p|)^2.
- Keep a runtime check `|exact - approximate| <= E` per candidate (falls back per row), so a
  BLAS computing below `f32` precision cannot change the result silently.
- Cap candidates per query (4,096) and fall back to brute force past it: duplicates would
  otherwise hold n candidates per query.

**Profiling gotcha:** samply cannot unwind through Accelerate, so `libBLAS` samples are not
under kyto's stage frames in `hotspots.py --stage`: add the `libBLAS` self time by hand.
Accelerate's `sgemm` ran on the calling thread (no BLAS threads), so a 1-thread run stays one
thread. `Vec::insert` on a 14-element list showed up as `memmove` (a quarter of the time):
use an insertion loop.

**Process:** a bug-hunter sub agent that edits a file and restores its own backup when it is
done reverts any edit made to that file meanwhile (12.4 lost one this way, caught before the
commit). Don't edit the files a bug hunt is testing until it reports.
