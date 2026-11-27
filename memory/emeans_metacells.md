# e-means metacells: what transferred from the paper, and how to test it

Task 6.16 (Nov 27, 2026). The paper (Bessa et al., arXiv 2611.09501) fitted its laws on
DEEP (d = 96, k >= 256); kyto runs it on 50-component PCA at k = n/50 to n/200.

- **The paper's Eq. 6/7 transfer.** A grid over batch size x first step put the best step
  at ~`0.65 B / sqrt(n k)` (Eq. 6 is ~0.73 at 3 epochs); MSE is flat over a wide band of
  batches once the step follows them. Don't refit them for single-cell data.
- **The revival threshold does not transfer.** 10% of the mean count is 20 cells at
  n/k = 200, and pbmc3k's ~14 platelets got split away every 5 batches (+2% MSE). kyto
  revives below min(10% of the mean, two cells' worth). It must stay relative to the mean
  count: the EMA counts start at 1e-10 and ramp up, so an absolute threshold revives
  everything early.
- **Seed noise at small k is large.** At k = 13 on pbmc3k both e-means and Lloyd swing
  54.5 to 58.5 MSE with the seed; ten seeds ranked them either way (+0.8% on seeds 0-9,
  -1.1% +- 0.3% over 100). Compare means over 50+ seeds, and report the standard error.
- **Bit-for-bit with NumPy is cheap when designed in**: the same SplitMix64 streams
  (multiply-shift `below`), distances summed per component in a NumPy loop over
  components (not `@`, not `.sum()`, which is pairwise), `np.add.at` for sums in batch
  order, `math.cos`/`pow`/`log2` (libm on both sides), and half-away rounding
  (`floor(x + 0.5)`, not Python's banker's `round`).
- **"Same as before the task" needs a snapshot from the old build.** Build the base commit
  in a scratch `git worktree` with an APFS clone of `target` (`cp -c -R`, then delete the
  cloned HDF5 CMake caches), write the snapshot with its `.venv`, and replace temporary
  store paths (explain text and the plan inside `provenance.json`) with `<store>`.
