# kyto benchmark suite (task 10.3)

One command measures kyto against scanpy (and rapids-singlecell where CUDA exists) on
the basic clustering workflow at increasing dataset sizes, per step, in time and peak
RSS, plus the Rust kernel benches (criterion). It writes a Markdown table, compares
kyto's numbers with a stored baseline and exits non-zero on a regression.

```bash
uv run python benchmarks/suite/suite.py                     # pbmc3k + 10k, 1 and 4 threads, compare
uv run python benchmarks/suite/suite.py --sizes 50k,100k --threads 1,4,10 --no-kernels
uv run python benchmarks/suite/suite.py --smoke             # the gate's check: harness works (~15 s)
uv run python benchmarks/suite/suite.py --update-baseline --provisional "why"
gh workflow run benchmarks.yml -f sizes=pbmc3k,10k          # the same command on a CI runner
```

Options: `--sizes` (`smoke`, `pbmc3k`, `10k`, `50k`, `100k`), `--threads` (comma list),
`--repeats` (default 3, median), `--tools` (`kyto,scanpy,rapids`), `--no-kernels`,
`--data-dir` (default `.cache/benchmarks`, gitignored), `--output` (default
`target/benchmarks/run-<UTC time>`), `--baseline` (default
`benchmarks/suite/baseline-<platform>.json`), `--update-baseline`, `--provisional NOTE`.
Exit status: 1 on a regression, 2 on a usage error, 3 when a benchmark command fails, 0
otherwise.

## Sizes

| Name | Data | Use |
|---|---|---|
| `smoke` | `make_data.py 500`: 500 cells x 20,000 genes | the gate (`--smoke`: kyto only, 1 thread, 1 repeat, no kernels, no baseline) |
| `pbmc3k` | the pbmc3k fixture (`tests/fixtures`): 2,700 cells x 32,738 genes | regression check (default) |
| `10k` | `make_data.py 10000` | regression check (default) |
| `50k` | `make_data.py 50000` (the 10.8 table in `performance.md`) | opt-in, about 10 minutes per thread count |
| `100k` | `make_data.py 100000`, 142M nonzeros | opt-in, the larger size: scanpy's peak RSS should be about twice the 50k run's 3.9 GB, so it needs a 16 GB machine with little else running |

`make_data.py` is seeded (8 clusters, about 7% nonzero, 13 `MT-` genes). Each size's
`.h5ad` and converted kyto store are made once under `--data-dir/<size>`; delete that
folder to rebuild them. The convert time is recorded when it runs (`convert_seconds`)
but is not a step, as in 10.8.

## Method (task 10.8's, not a copy of it)

The suite runs the scripts of [`../scanpy_workflow`](../scanpy_workflow/README.md),
each in fresh processes, and only collects what they print:

- **kyto:** `kyto_steps.py --with-write`: each plan prefix (QC + filters, + normalize
  and `log1p`, + HVG, + PCA, + neighbours, + Leiden, + UMAP) streamed to nowhere
  (`_execute`), median of `--repeats`, a step's time is its prefix's median minus the
  previous one's; `write` is the full plan's `sink_zarr` minus the UMAP prefix, and
  `total` is that full `sink_zarr` run. Peak RSS is each prefix process's peak (`total`:
  the largest). Exact neighbour search.
- **scanpy:** `run_scanpy.py --threads N --warmup CELLS` (warmed up in-process on the
  whole dataset), median of `--repeats` per step, for time. 10.8 warmed up on 6,000
  cells; at 10k cells that left part of pynndescent uncompiled and the timed neighbours
  took 46.8 s instead of 0.28 s, so the suite warms up on every cell. One cold run
  (empty `NUMBA_CACHE_DIR`) per thread count for peak RSS, since the warm-up leaves
  memory behind. `total` is read through write.
- **rapids-singlecell:** `run_rapids.py`, the same steps on the GPU, median of
  `--repeats`, host peak RSS (and the cupy pool's peak in `results.json`). It needs
  CUDA, which macOS lacks: where `rapids_singlecell` does not import or finds no CUDA
  device, the column is left out and the table says why (`> rapids-singlecell: skipped,
  ...`). Its neighbours are rapids' default (cuML brute force, exact), not scanpy's
  pynndescent; its `read` includes the copy to the GPU and its `write` the copy back. `run_rapids.py` has not yet run on a CUDA machine; check it there first.
- **Rust kernels:** `cargo bench -p kyto-exec --bench kernels` (criterion), the median
  per call of every bench this run wrote under `target/criterion`.

Golden rule 8 holds as in 10.8: same data, same steps and parameters, `float32` on both
sides, thread counts stated (scanpy capped through numba, OpenMP, BLAS and
threadpoolctl; numpy's Accelerate BLAS on macOS ignores the cap). The table names the
machine, commit and load average.

## Regression check

kyto's numbers only (scanpy's are the reference, not the code under test), per size,
thread count and step, against `baseline-<platform>.json`. Times are compared **through**
each step (the median of the plan prefix ending there, the sum of the steps up to it),
not step by step: a step is a difference of two medians and can be noise around zero,
or negative. A number regressed when it grew past both limits:

- time through a step: more than 1.25x and more than 0.1 s the baseline's,
- peak RSS: more than 1.15x and more than 25 MB the baseline's,
- a criterion kernel: more than 1.25x the baseline's median.

The absolute limits keep a 0.02 s prefix that doubles from failing the run. The thresholds live in
`benchmark_report.THRESHOLDS` (its tests check this page states them). The baseline is
compared only when it comes from the same platform, CPU model and core count; anything
else prints a notice and compares nothing (exit 0), as does a missing baseline. Numbers
the baseline lacks (a size or thread count it did not run) are counted in a notice and
not compared. A number the baseline has for a size, thread count or kernel set this run
measured, but this run lacks, is listed as `MISSING` and fails the run (exit 1): a
broken harness must not pass as "no regression". `--update-baseline` refuses a partial
run (it needs kyto, the kernels and at least `pbmc3k,10k`), so a baseline never silently
stops covering something.

`baseline-darwin-arm64.json` (Apple M4, 10 cores) is **provisional**: it was taken while
other builds ran beside it. Re-take it on a quiet machine at the start of task 12.1
with `--update-baseline` alone (no `--provisional`). A CI runner has no baseline yet: the manual workflow uploads its
`results.json`, which can be committed as that runner's baseline.

## Why not pytest-benchmark

The plan named `pytest-benchmark` for the Python side. It times a callable in one
process, so it cannot give per-step numbers from fresh processes, peak RSS per step, or
a scanpy column, and its compare checks time only. The suite reuses 10.8's scripts
instead; criterion stays for the Rust kernels.

## Files

| File | What it does |
|---|---|
| `suite.py` | the command: prepares data, runs the scripts, writes `results.json` and `results.md`, compares |
| `benchmark_report.py` | the table and the regression check (pure Python; tests in `tests/test_benchmark_suite.py`) |
| `baseline-<platform>.json` | a stored `results.json` per machine |
