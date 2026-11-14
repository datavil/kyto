# kyto vs scanpy: basic workflow

The scripts behind [performance.md](../../performance.md). Not the 10.3 suite: one-off
timing scripts, run by hand.

| Script | What it does |
|---|---|
| `make_data.py N OUT.h5ad` | Seeded synthetic counts: N cells × 20,000 genes, 8 clusters, ~7% nonzero, 13 `MT-` genes |
| `run_scanpy.py IN.h5ad OUT.h5ad` | The workflow in scanpy; prints seconds per step as JSON |
| `run_kyto.py IN.h5ad STORE OUT SEARCH` | `ky.convert`, then the same workflow as one plan run by `sink_zarr`; `SEARCH` is `exact` or `approximate` |
| `kyto_breakdown.py STORE OUT [--with-umap]` | Runs the kyto plan cut after each step; prints cumulative seconds |
| `measure.py SCRIPT ARGS...` | Runs a script and prints its peak RSS |

scanpy needs its own environment (it is not a kyto dependency):

```bash
uv venv -p 3.13 .venv-scanpy
uv pip install -p .venv-scanpy "scanpy[leiden]" polars pyarrow
uv run maturin develop --release   # kyto in the project environment

.venv-scanpy/bin/python make_data.py 50000 synthetic_50000.h5ad
.venv-scanpy/bin/python measure.py run_scanpy.py synthetic_50000.h5ad out_scanpy.h5ad
uv run python measure.py run_kyto.py synthetic_50000.h5ad store.kyto out_kyto.kyto exact
uv run python kyto_breakdown.py store.kyto tmp.kyto --with-umap
```

Run scanpy twice and report the second run: the first spends minutes compiling numba code.
`run_scanpy.py` sets `sc.settings.n_jobs = 2`; change it to your core count.
