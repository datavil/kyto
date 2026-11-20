# Timing the full workflow against scanpy (task 10.8)

**Works:**
- kyto per step: each plan prefix in a **fresh process**, streamed to nowhere (`_execute`),
  median of 3, steps as differences (`kyto_steps.py`). Runs repeat within 1%, and the fresh
  process gives each prefix its own peak RSS. The `write` step is `sink_zarr` of the full plan
  minus the UMAP prefix. On 10 threads the sink overlaps compute, so `write` is near 0 there.
- scanpy warm: run the workflow in the same process on more than **4,096** cells first
  (`--warmup 6000`). Below that, `pp.neighbors` uses brute force and pynndescent stays
  uncompiled. Even then the first warm run in a fresh numba cache compiled part of
  pynndescent (31.6 s against 9.8 s), so take the median of 3. Cold = empty
  `NUMBA_CACHE_DIR`, about 27 s of compilation, nearly all in neighbours.
- scanpy's peak RSS: take it from the cold run. The warm-up leaves about 220 MB behind.
- Cap scanpy's threads with environment variables (`NUMBA_NUM_THREADS`, `OMP_`,
  `OPENBLAS_`, `VECLIB_MAXIMUM_THREADS`) set **before** importing scanpy, plus
  threadpoolctl and `sc.settings.n_jobs`. pynndescent and `pp.pca` barely scale with threads
  anyway.

**Broke:**
- `measure.py` divided `ru_maxrss` by 1024 on macOS, where it is bytes, not kilobytes.
  Fixed in 10.8.
- Leiden ARI on well-separated synthetic clusters is 1.0 for any sane graph, so a lower
  ARI there is a red flag. It exposed HNSW (`search="approximate"`) giving 2% of the 50k
  cells recall 0 (task 6.17). Check recall against exact search per cell, not only the
  mean: the mean was 0.979, hiding the lost cells. The generator's true clusters can be
  rebuilt from its seeded RNG (`make_data.py` draw order).
