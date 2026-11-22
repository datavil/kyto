# GPU kernels in CubeCL 0.11 (task 9.2)

**Works:**

- Kernels in `kyto-gpu/src/kernels.rs` (`#[cube(launch)]` over slices `&[u32]`, `&[T]`,
  `T: Numeric` for `f32`/`u32` inputs, `f32::cast_from`, `u32::reinterpret`), host side
  in `launch.rs` (`DeviceKernels`: `create_from_slice`, `empty`, `read_one`,
  `BufferArg::from_raw_parts` is `unsafe`). One thread per row or value, fixed order, no
  atomics: the same bits run to run.
- Open a device by name, matching the probe: `cubecl::Device::metal_msl / vulkan / dx12
  (WgpuDeviceKind::default())`, `cubecl::Device::cuda(0)`, then `.client()` (it panics on
  failure: catch it). `Client` is type-erased and `Clone + Send + Sync`; launches are
  `kernel::launch::<T>(&client, CubeCount, CubeDim, args...)`.
- `CubeCount::Static(x, y, 1)` with x <= 65535 (wgpu's per-axis limit); `ABSOLUTE_POS`
  is linear across axes, so a kernel only checks `pos < len`.
- `DeviceKernels` exists in every build: without `wgpu`/`cuda` it holds an
  `Infallible` field and `open` returns `KY-E505`, so kyto-exec calls it with no cfg.
- **No `f64` on Metal.** Sums are TwoSum-compensated `f32` pairs added in `f64` on the
  host (about 1e-11 relative); `u32` counts sum exactly as integers with a carry word
  (same bits as the CPU). `log1p` is musl's `log1pf` written with selects (WGSL's `log`
  promises only an absolute error near 1, useless for small x).

**Gotchas:**

- **Metal compiles with fast math**, whatever CubeCL's own `fast_math` (off by
  default): it reassociated TwoSum's `(s + x) - s` into `x` and the compensation
  vanished (row sums 1.5e-6 off). Multiplying each TwoSum step by a runtime `one` (a
  launch argument, exact) stops it: 5e-12. Any error-free transformation on a GPU needs
  this, and a test with long rows of mixed magnitudes to catch it.
- The cloned `target` (memory `worktree-build-cache`) holds HDF5's CMake cache with the
  main checkout's path: a feature change that rebuilds `hdf5-metno-src` fails ("CMakeCache.txt
  directory ... is different"). Delete `target/*/build/hdf5-metno-src-*/out/build` dirs
  whose `CMakeCache.txt` names another checkout.

- `cubecl::Runtime`, `ScalarType`, `Handle` are not in the prelude: use
  `cubecl::Device` (above), `T: Numeric`, `cubecl::server::Handle`.
- Clippy: f32 literals must be the shortest exact form (`excessive_precision`), and the
  `#[cube]` macro wants comparisons, not `Range::contains`.
- A wait loop `pgrep -f "cargo build ..."` matches its own command line: wait on the
  backgrounded command's own notification instead.
- First build of the wgpu + CubeCL stack is long; `uv run cargo test` also rebuilds the
  Python extension: use `uv run --no-sync` for the Rust GPU tests.
