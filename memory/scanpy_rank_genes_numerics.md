# scanpy `rank_genes_groups` numerics (task 6.10)

**Works (bit-identical statistics on both fixtures):** follow scanpy 1.12.5's code path
per method, not the textbook formula.
- t-test means and variances come from `aggregate(func=["mean","var"])`: Welford over
  the stored non-zeros in cell order, then merged with the group's zeros
  (`mean_var_csr`), `var = M2/n * (n/(n-1))`. Wilcoxon's means (for the fold change)
  come from `aggregate(func="mean")`: plain `f64` sums / n. So the two methods give
  slightly different log fold changes, as scanpy does.
- The rest's mean is `(sum_g count*mean - count_g*mean_g) / n_rest` over every group
  plus a "remainder" (cells of unselected groups and null labels); the rest's variance
  is Chan's leave-one-out scan (`_vars_rest`), with `M2 = var*(count-1)` and 0 for
  groups of <= 1 cell. An empty remainder is zero-filled (aggregate drops it).
- scipy takes `std = sqrt(var)` then squares it again (`ttest_ind_from_stats`); keep
  that round trip. Wilcoxon's `n*(N-n)*(N+1)/12` is exact integer arithmetic (Python
  ints) unless `tie_correct`, then left-to-right `f64`.
- Ranks are halves of integers: rank sums are exact in any order, so sparse ranking
  (zeros as one tied block, negatives before it) equals scipy's dense `rankdata`.
- scipy's normal tail underflows to 0 beyond |z| ~ 37.7 (cephes `erfc`, `MAXLOG`),
  where libm goes subnormal: return 0 there (bug-hunter).
- Group order is pandas' category order: an Enum column's categories, numbers by value
  (not as text), strings sorted (bug-hunter caught text-sorting a custom order).

**Watch out:** Student's t tail via `lgamma(a)+lgamma(b)-lgamma(a+b)` loses digits for
df above ~1e9 (1.5e-6 relative there); fine for any realistic cell count. A parity
helper must derive extra obs columns from the same matrix on both sides (a group made
from filtered-gene totals on one side and unfiltered on the other looked like a bug).
