"""``kyto.qc``: quality-control metrics as expressions (tasks 6.1, 6.13).

What it does: re-exports the QC constructors of the extension. Four build a numeric
``CellExpr`` leaf (``total_counts``, ``genes_detected``, ``total_counts_in``,
``percent_counts_in``) and four a numeric ``GeneExpr`` leaf (``cells_detected``,
``gene_total_counts``, ``gene_mean_counts``, ``percent_dropout``), each computed on
read from a layer (so it never goes stale), for ``filter``, ``with_cells`` and
``with_genes`` like any other expression. scanpy's ``log1p_`` columns are ``.log1p()``
on these.

What it does not: read data. The constructors live in this namespace, not at top
level, because ``kyto`` exports only what task 7.1 lists.
"""

from kyto._kyto import (
    cells_detected,
    gene_mean_counts,
    gene_total_counts,
    genes_detected,
    percent_counts_in,
    percent_dropout,
    total_counts,
    total_counts_in,
)

__all__ = [
    "cells_detected",
    "gene_mean_counts",
    "gene_total_counts",
    "genes_detected",
    "percent_counts_in",
    "percent_dropout",
    "total_counts",
    "total_counts_in",
]
