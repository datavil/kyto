# SIMD kernels with the same bits everywhere: pulp dispatch + plain Rust (task 5.3)

**Works:** write each kernel body as plain Rust (`#[inline(always)]`) and run it inside a
`pulp::WithSimd` impl through `Arch::new().dispatch(...)`. The body is compiled once per
instruction set with its target features, and LLVM vectorizes it (checked: `fdiv.2d` on
NEON, `zmm`/`ymm` on an `x86_64` build). Rust never reorders or contracts float
operations, so the bits are those of the order written, on every set: no tolerances,
equality tests are exact. Make the order independent of the vector width yourself: a
sum uses a fixed number of `f64` lanes (`SUM_LANES` = 8, value `j` into lane `j % 8`)
and a fixed tree, never pulp's lane count or `reduce_sum`.

**Watch out:**
- `Kernels::scalar()` (pulp's `Scalar`) is still compiled for the target's baseline:
  on aarch64 that is NEON, so "scalar vs NEON" benches measure the same code. Compare
  against an independent value-at-a-time reference in the tests, with its own lane tree
  and its own zero test (a reference sharing the kernel's helpers let a mutation pass).
- `proptest`'s `any::<f32>()` generates no NaN: force NaN, `-0.0` and infinities in with
  `prop_oneof!` or NaN handling goes untested.
- libm (`f32::ln_1p`) is neither vectorizable nor the same on every platform; kyto's
  `log1p` is computed in `f64` and rounded once (correctly rounded but for 1 of 3.2e9
  inputs; 0.07% of inputs differ from macOS libm by 1 ulp). In `f64`, NEON has only 2
  lanes, so it is no faster than libm there (an `f32` version is a numeric choice for the
  user, plan 5.3 Open (1)).
- No Rosetta here (and Rosetta has no AVX-512): x86 paths can only be compiled locally
  (`rustup target add x86_64-apple-darwin`, a scratch crate with the kernel files and
  pulp; kyto-exec itself pulls hdf5 through kyto-store). They run when CI runs.
