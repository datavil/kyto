"""Write a seeded synthetic count matrix with cluster structure as an .h5ad file."""

import sys

import anndata
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse


def make_counts(
    n_cells: int, n_genes: int, n_clusters: int, seed: int
) -> anndata.AnnData:
    """Return clustered negative-binomial counts with mitochondrial genes.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes; the first 13 are named ``MT-*``.
    n_clusters : int
        Number of cell populations with distinct mean profiles.
    seed : int
        Random seed.

    Returns
    -------
    anndata.AnnData
        Counts in ``X`` as float32 CSR, gene IDs as ``var`` index, ``symbol`` in ``var``.
    """
    generator = np.random.default_rng(seed)
    base_mean = generator.lognormal(mean=-4.0, sigma=1.8, size=n_genes)
    cluster_fold = generator.lognormal(mean=0.0, sigma=0.8, size=(n_clusters, n_genes))
    cluster_of_cell = generator.integers(0, n_clusters, size=n_cells)
    library_size = generator.lognormal(mean=0.0, sigma=0.3, size=n_cells)
    blocks = []
    block_size = 2000
    for start in range(0, n_cells, block_size):
        stop = min(start + block_size, n_cells)
        means = (
            base_mean[None, :]
            * cluster_fold[cluster_of_cell[start:stop]]
            * library_size[start:stop, None]
        )
        dispersion = 2.0
        rates = generator.gamma(dispersion, means / dispersion)
        counts = generator.poisson(rates).astype(np.float32)
        blocks.append(sparse.csr_matrix(counts))
    matrix = sparse.vstack(blocks, format="csr")
    symbols = [
        f"MT-G{index}" if index < 13 else f"GENE{index}" for index in range(n_genes)
    ]
    # No obs/var column may repeat the index name, or ky.convert raises KY-E121.
    gene_frame = pl.DataFrame({"symbol": symbols}).to_pandas()
    gene_frame.index = [f"ENSG{index:08d}" for index in range(n_genes)]
    cell_frame = pd.DataFrame(index=[f"cell{index}" for index in range(n_cells)])
    return anndata.AnnData(X=matrix, obs=cell_frame, var=gene_frame)


if __name__ == "__main__":
    n_cells = int(sys.argv[1])
    output_path = sys.argv[2]
    annotated = make_counts(n_cells=n_cells, n_genes=20000, n_clusters=8, seed=0)
    annotated.write_h5ad(output_path)
    print(output_path, annotated.shape, annotated.X.nnz)
