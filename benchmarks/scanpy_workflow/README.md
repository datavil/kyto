# kyto vs scanpy: basic workflow

The scripts behind [performance.md](../../performance.md). Not the 10.3 suite: one-off
timing scripts, run by hand.

| Script | What it does |
|---|---|
| `make_data.py N OUT.h5ad` | Seeded synthetic counts: N cells × 20,000 genes, 8 clusters, ~7% nonzero, 13 `MT-` genes |
| `run_scanpy.py IN.h5ad OUT.h5ad [--threads N] [--warmup CELLS]` | The workflow in scanpy; prints seconds and peak RSS per step as JSON. `--threads` caps numba, BLAS, OpenMP and `n_jobs`; `--warmup` first runs the workflow on that many cells in the same process (compiles numba out of the timed run) |
| `run_kyto.py IN.h5ad STORE OUT SEARCH` | `ky.convert`, then the same workflow as one plan run by `sink_zarr`; `SEARCH` is `exact` or `approximate` |
| `kyto_breakdown.py STORE OUT [--with-umap]` | Runs the kyto plan cut after each step; prints cumulative seconds |
| `kyto_steps.py STORE OUT [--threads 1,4,10] [--repeats 3] [--with-write] [--search exact\|approximate]` | Per-step seconds and peak RSS (task 10.8): each `kyto_breakdown.py` prefix streamed to nowhere (`_execute`) in a fresh process, median of the repeats, steps as differences; `write` is the full `sink_zarr` minus the UMAP prefix |
| `compare_labels.py SCANPY.h5ad KYTO_STORE` | Cells kept, Leiden clusters and the ARI between the two outputs, aligned by cell ID |
| `measure.py SCRIPT ARGS...` | Runs a script and prints its peak RSS in MB (bytes on macOS, kilobytes on Linux) |
| `pca_timing.py STORE WORK` | kyto's PCA alone (on the HVG store it writes to `WORK` once) and in the workflow, input held vs streamed, at 1, 4 and 10 threads; checks the embeddings are bit-identical (task 5.12). `--cases stage` times PCA alone with no sink (`_execute`), the stage 5.15 measures |
| `thread_scaling.py STORE WORK` | Preprocessing (streamed to nowhere), QC through PCA and PCA alone at 1, 4 and 10 threads (`--threads`); speedups and bit checks across thread counts (task 5.13) |
| `pca_scanpy.py IN.h5ad [--kyto WORK/kyto_pca.npy]` | scanpy's `pp.pca` alone on the same HVG matrix, threads capped by threadpoolctl; compares kyto's embedding |

scanpy, leidenalg and scikit-learn are in the dev environment (`uv sync`, which also
builds the release extension), so one environment runs both sides. The full workflow
per step (task 10.8; put the data on a disk with room, not `/tmp`):

```bash
uv run python make_data.py 50000 synthetic_50000.h5ad
uv run python measure.py run_kyto.py synthetic_50000.h5ad store.kyto out_kyto.kyto exact
uv run python kyto_steps.py store.kyto tmp.kyto --threads 1,4,10 --repeats 3 --with-write
uv run python kyto_steps.py store.kyto approx.kyto --threads 1,4,10 --with-write --search approximate
NUMBA_CACHE_DIR=$(mktemp -d) uv run python run_scanpy.py synthetic_50000.h5ad cold.h5ad --threads 1
uv run python run_scanpy.py synthetic_50000.h5ad out_scanpy.h5ad --threads 1 --warmup 6000
uv run python compare_labels.py out_scanpy.h5ad tmp.kyto
```

`--warmup` needs more than 4,096 cells, or scanpy's neighbours take the brute-force path
and pynndescent stays uncompiled; even so, the first warm run in a fresh numba cache
still compiled part of it (31.6 s against 9.8 s), so take the median of three.

PCA alone (task 5.12; peak memory with `/usr/bin/time -l`; before 10.8, `measure.py`
divided macOS's byte count as if it were Linux's kilobytes):

```bash
uv run python pca_timing.py store.kyto work
uv run python pca_timing.py store.kyto work --cases stage --threads 1,2,4,10 --repeats 5
uv run python thread_scaling.py store.kyto work --threads 1,2,4,10
uv run python pca_scanpy.py synthetic_50000.h5ad --kyto work/kyto_pca.npy
```

A cold scanpy run (empty numba cache) spends about 27 s compiling numba code, nearly
all of it in neighbours; report it beside the warm runs.
