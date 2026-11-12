# scanpy `scale` numerics (task 6.4)

- `mean_var` (fast_array_utils, CSR, axis 0): sums in `f64`, but a `float32` value is
  squared in `float32` (`value * value` before the `f64` add). Integer data is cast to
  `float64` first by `scale`, so counts square exactly.
- `std = sqrt(var * n / (n - 1))`, then `std[std == 0] = 1` (the returned `var['std']`
  holds the 1s).
- Centred on sparse input: `x -= mean` turns the CSR `float32` into a dense **`float64`**
  array; divide and clip in `float64`. kyto computes in `f64` and rounds once to `f32`:
  bit-identical to `float32(scanpy)` (deviations.md).
- Not centred on CSR: `scale_and_clip_csr`, `float32` data, `min(max_value, x / std)`:
  clipped above only, no lower clip.
- Parity trap: the 1.5 reference's input went through numpy's `float32` `log1p`, 1 ulp
  off kyto's on ~620 pbmc3k values (6.2). That alone moves means by ~4e-9 relative and a
  few hundred scaled values by up to ~6e-7. To test kyto's own steps, run scanpy live
  with a correctly rounded `log1p` (`np.log1p(x.astype(f64)).astype(f32)`) and demand
  identical bits; bound the stored reference separately (`tests/scale_parity.rs`).
