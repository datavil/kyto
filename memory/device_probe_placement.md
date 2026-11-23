# Devices: the probe, the placement rules and the GPU features (task 9.1)

**Works:**

- The device a plan asks for is a plan step (`ON_DEVICE`, method `on`), so it is part of
  the plan JSON and provenance (R7). The last `on` decides for the whole plan
  (`NodeState::device`, read from the first output in `optimize_reuse::lower` into
  `OptimizedPlan::device_request`).
- The devices found are an argument, `PhysicalOptions::devices` (`kyto_gpu::Devices`,
  plain `Copy` data). `Default` is `Devices::CPU_ONLY`, so Rust explain/estimate
  snapshots are the same on every machine; tests inject a Mac or a CUDA box by hand.
  kyto-py probes (`terminal::devices_for`) only when the plan asks for a non-CPU device,
  so a CPU plan never waits for a GPU driver.
- Rules in `kyto-plan/src/device_placement.rs` (`place`, injectable kernel table);
  kernels per runtime in `kyto-gpu/src/backend.rs` (`CUDA_KERNELS`, `WGPU_KERNELS`,
  keyed by plan method). 9.2 registered `normalize_total` and `log1p` and dispatches
  them in `Pipeline::new` (`GpuKernels::for_node`; any other GPU node is still `KY-E501`):
  a new kernel needs both the name and the dispatch (memory `gpu_kernels_cubecl.md`).
- A step placed on a GPU shows `(within 1e-6 of cpu)` in `explain()` (`GPU_NOTE`).
- Rule 6 (task 9.10): a `with_cells` or cell filter is placed **whole**, by its leaves:
  its operations are each QC leaf's metric name (`total_counts`, ...; in the kernel
  tables like any method) plus the step's own method (no kernel) when another leaf reads
  the matrix (a gene's values) or an embedding. On a GPU only the QC row sums and nnz
  run there (`CellQc::on_gpu`, `Stage::Cells(stage, gpu)`); Polars, the mask and the
  filter stay on the host. A test that needs "a cell step without a GPU kernel" must
  read a gene's values, not a QC metric (the 9.6 alternating chains were changed so).
- Rule 5 (task 9.6): blocks live on the host between steps; a GPU step copies each
  block in and back (`Placement::moves`, `BlockMoves`, shown as `moves=cpu->metal->cpu`
  and the explain JSON key `moves`). `moves` is the route, not the count: each map of
  a fused pass makes its own round trip (9.2). So mixed CPU/GPU chains need no extra rule. If M6
  keeps blocks on a device between steps, `moves` must change with it (the executor and
  the placement say the same thing, or explain lies).
- A CPU plan prints exactly what it printed before 9.1 (no device line, no notes);
  only the explain JSON gained `optimized.device` and `device_note`.

**Gotchas:**

- cudarc with `fallback-dynamic-loading` **panics** on the first driver call when
  `libcuda` is missing (macOS, CI). Call `cudarc::driver::sys::is_culib_present()`
  (unsafe, needs a `// SAFETY:` note) before `result::init()`.
- wgpu's `InstanceFlags::default()` turns on validation in debug builds; the probe sets
  `InstanceFlags::empty()`. `enumerate_adapters` is async in wgpu 30 (`pollster`).
  Skip `DeviceType::Cpu` adapters (llvmpipe, WARP): they are not GPUs.
- Versions are CubeCL 0.11's (wgpu 30, cudarc 0.19) so 9.2 adds no second copy.
- The workspace build never turns the GPU features on, so the gate lints and tests
  `kyto-gpu --all-features` on its own lines; the Metal check is an ignored test
  (`cargo test -p kyto-gpu --features wgpu -- --ignored`), since CI's macOS VMs have no GPU.
- A Python build with GPUs: `MATURIN_PEP517_ARGS="--features wgpu" uv sync
  --reinstall-package kyto` (then rebuild without it before the gate).
