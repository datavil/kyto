# The benchmark suite (task 10.3)

**Works:**
- One driver, `benchmarks/suite/suite.py`, over 10.8's scripts (`kyto_steps.py`,
  `run_scanpy.py`), each in fresh processes; it only collects their JSON. New methodology
  goes into those scripts, not into the driver, so 10.8's table and the suite never drift.
- The regression check compares **kyto's numbers only**, and only against a baseline from
  the same platform, CPU model and core count (`baseline-<platform>.json`). Two limits at
  once (ratio and absolute): kyto's per-step times are prefix differences, so a short step
  can be noise around zero or even negative; a ratio alone would flag it.
- The gate runs `--smoke` only (500 cells, kyto, 1 thread, 1 repeat, no kernels, about
  15 s). Data under `.cache/benchmarks`, output under `target/benchmarks`, both ignored,
  so the gate's clean-tree check holds.
- rapids-singlecell is probed in a subprocess (`import rapids_singlecell, cupy` and a CUDA
  device count); without it the table carries a `> rapids-singlecell: skipped, ...` line.
  `run_rapids.py` has never run on a CUDA machine.
- pytest-benchmark was not used: it times a callable in one process, so no per-step
  numbers from fresh processes, no peak RSS, no scanpy column, time-only compare.

**Broke:**
- Editing `pyproject.toml` (or any Rust file) while a suite run is going: the next
  `uv run` rebuilds and reinstalls the extension (uv `cache-keys`), stealing CPU from the
  run and possibly swapping the `.so` under it. Make such edits before or after a run.
- The darwin-arm64 baseline (Nov 22, 2026) was taken with other builds running (load
  average 12 to 17): **provisional**, re-take it on a quiet machine at the start of 12.1.
