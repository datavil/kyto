"""Convert fixture datasets to AnnData, for scanpy and anndata references.

What it does: :func:`to_anndata` copies a :class:`~fixtures.dataset.Dataset` into a new
AnnData (scipy CSR counts, string columns).

What it does not: run on platforms without anndata (Windows ARM64, macOS x86_64; see
pyproject.toml); import it only where scanpy is available.
"""

# pandas, scipy and parts of anndata ship without complete type information; strict
# pyright cannot see their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from fixtures.dataset import Dataset


def to_anndata(dataset: Dataset) -> AnnData:
    """Return a new AnnData with the counts of ``dataset`` in ``X`` (scipy CSR, float32).

    ``obs`` is indexed by ``cell_id`` and ``var`` by ``gene_id``; the other columns
    become string columns.

    Parameters
    ----------
    dataset : Dataset
        Source dataset.

    Returns
    -------
    AnnData
        Independent of ``dataset`` (every array is copied).
    """
    matrix = csr_matrix(
        (dataset.data.copy(), dataset.indices.copy(), dataset.indptr.copy()),
        shape=(dataset.n_cells, dataset.n_genes),
    )
    obs = pd.DataFrame(
        {key: value.copy() for key, value in dataset.cells.items() if key != "cell_id"},
        index=pd.Index(dataset.cells["cell_id"].copy(), name="cell_id"),
    )
    var = pd.DataFrame(
        {key: value.copy() for key, value in dataset.genes.items() if key != "gene_id"},
        index=pd.Index(dataset.genes["gene_id"].copy(), name="gene_id"),
    )
    return AnnData(X=matrix, obs=obs, var=var)
