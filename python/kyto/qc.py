"""``kyto.qc``: quality-control metrics as cell expressions (task 6.1).

What it does: re-exports the four QC constructors of the extension. Each builds a
numeric ``CellExpr`` leaf computed on read from a layer (so it never goes stale), for
``filter`` and ``with_cells`` like any other cell expression.

What it does not: read data, or hold per-gene metrics. The constructors live in this
namespace, not at top level, because ``kyto`` exports only what task 7.1 lists.
"""

from kyto._kyto import genes_detected, percent_counts_in, total_counts, total_counts_in

__all__ = [
    "genes_detected",
    "percent_counts_in",
    "total_counts",
    "total_counts_in",
]
