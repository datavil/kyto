# GPU matrix kernels: PCA products and exact neighbours (task 9.8)

**Works:**

- PCA on a GPU keeps the CPU's structure: only the sparse x dense products move
  (`kyto-gpu/src/kernels_matrix.rs`, host side `launch_matrix.rs`; kyto-exec
  `pca/gpu.rs`). Basis, Rayleigh-Ritz and variances stay `f64` on the host (genes x l).
- **R11 with split blocks:** 5.14 cuts a cell block into thread-dependent parts that
  continue one leaf. A GPU partial computed per fold and added in `f64` would change bits
  with the thread count. Carry the leaf's sums as compensated `f32` pairs (`PairSums`)
  uploaded and continued by each fold; convert to `f64` only in `combine`/`end_pass`
  (`settle`). Same bits held/streamed, 1/all threads, 37-cell blocks.
- Round `Omega` and `w` to `f32` *before* the products and take their column sums from
  the rounded values, so `X^T W - mu (1^T W)` stays consistent. pbmc3k on Metal: 1e-8
  from the CPU, far inside the 1e-3 tolerances.
- Exact neighbours equal to the CPU: the GPU's `f32` keys only pick `2k + 16`
  candidates; the host re-ranks with `Space::keys` and proves the floor
  (`key / (1 + 2(w+4)u)` Euclidean, `key - 2(w+16)u` cosine) of the farthest candidate
  lies above the `k`-th exact key, else searches that row on the CPU. Ties and
  duplicates just fall back. Cosine needs norms in `[1e-15, 1e15]` (else whole CPU).
- Placement notes per step: `Placement::On(Device, Agreement)` (`kyto_gpu::Agreement`:
  equal / 1e-6 / 1e-3); approximate neighbours are the operation
  `neighbors(search=approximate)`, which no GPU has.

**Gotchas:**

- kyto-plan's `devices.rs` tests used `pca` as "the step without a kernel": now `scale`.
- A cloned target's debug HDF5 rebuild needs cmake from `.venv` (`uv sync` first).
- Explain's logical plan also prints the node name: match the optimized line by
  `[device=`.
