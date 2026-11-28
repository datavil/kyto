"""Write a seeded synthetic count matrix with cluster structure as an .h5ad file.

Usage: ``make_data.py N OUT.h5ad [--stream]``. ``--stream`` writes block by block (12.1:
the 1M-cell file does not fit in memory); the values are the same either way.
"""

# anndata, pandas and scipy ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import sys
from collections.abc import Iterator
from typing import Any

import anndata
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse

BLOCK_SIZE = 2000


def count_blocks(
    n_cells: int, n_genes: int, n_clusters: int, seed: int
) -> Iterator[sparse.csr_matrix]:
    """Yield the clustered negative-binomial counts, 2,000 cells at a time.

    The draws (and so the values) do not depend on how the caller stores the blocks:
    ``make_counts`` and ``write_counts_streamed`` give the same matrix.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    n_clusters : int
        Number of cell populations with distinct mean profiles.
    seed : int
        Random seed.

    Yields
    ------
    scipy.sparse.csr_matrix
        The next block of counts, float32.
    """
    generator = np.random.default_rng(seed)
    base_mean = generator.lognormal(mean=-4.0, sigma=1.8, size=n_genes)
    cluster_fold = generator.lognormal(mean=0.0, sigma=0.8, size=(n_clusters, n_genes))
    cluster_of_cell = generator.integers(0, n_clusters, size=n_cells)
    library_size = generator.lognormal(mean=0.0, sigma=0.3, size=n_cells)
    for start in range(0, n_cells, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, n_cells)
        means = (
            base_mean[None, :]
            * cluster_fold[cluster_of_cell[start:stop]]
            * library_size[start:stop, None]
        )
        dispersion = 2.0
        rates = generator.gamma(dispersion, means / dispersion)
        counts = generator.poisson(rates).astype(np.float32)
        yield sparse.csr_matrix(counts)


def gene_and_cell_frames(n_cells: int, n_genes: int) -> tuple[Any, Any]:
    """Return the ``var`` frame (IDs, ``symbol``; the first 13 ``MT-*``) and ``obs``.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.

    Returns
    -------
    tuple
        ``(gene_frame, cell_frame)`` as pandas frames, as anndata stores them.
    """
    symbols = [
        f"MT-G{index}" if index < 13 else f"GENE{index}" for index in range(n_genes)
    ]
    # No obs/var column may repeat the index name, or ky.convert raises KY-E121.
    gene_frame = pl.DataFrame({"symbol": symbols}).to_pandas()
    gene_frame.index = [f"ENSG{index:08d}" for index in range(n_genes)]
    cell_frame = pd.DataFrame(index=[f"cell{index}" for index in range(n_cells)])
    return gene_frame, cell_frame


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
    blocks = list(count_blocks(n_cells, n_genes, n_clusters, seed))
    matrix = sparse.vstack(blocks, format="csr")
    gene_frame, cell_frame = gene_and_cell_frames(n_cells, n_genes)
    return anndata.AnnData(X=matrix, obs=cell_frame, var=gene_frame)


def write_counts_streamed(
    n_cells: int, n_genes: int, n_clusters: int, seed: int, output_path: str
) -> int:
    """Write ``make_counts``'s matrix to ``.h5ad`` one block at a time (task 12.1).

    For sizes whose matrix does not fit in memory (1M cells: about 1.4 billion
    nonzeros, 11 GB as CSR). ``X`` is written as anndata's ``csr_matrix`` encoding with
    ``int64`` ``indptr`` and ``int32`` indices, uncompressed, as ``write_h5ad`` writes it;
    ``obs`` and ``var`` through ``anndata.io.write_elem``. Peak memory is one block.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    n_clusters : int
        Number of cell populations.
    seed : int
        Random seed.
    output_path : str
        The ``.h5ad`` file to write (replaced if it exists).

    Returns
    -------
    int
        The number of nonzeros written.
    """
    import h5py
    from anndata.io import write_elem

    gene_frame, cell_frame = gene_and_cell_frames(n_cells, n_genes)
    with h5py.File(output_path, "w") as file:
        file.attrs["encoding-type"] = "anndata"
        file.attrs["encoding-version"] = "0.1.0"
        write_elem(file, "obs", cell_frame)
        write_elem(file, "var", gene_frame)
        for name in ("obsm", "varm", "obsp", "varp", "layers", "uns"):
            write_elem(file, name, {})
        group = file.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = (n_cells, n_genes)
        data = group.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,)
        )
        indices = group.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,)
        )
        indptr = [np.zeros(1, dtype=np.int64)]
        nnz = 0
        for block in count_blocks(n_cells, n_genes, n_clusters, seed):
            size = block.nnz
            data.resize((nnz + size,))
            indices.resize((nnz + size,))
            data[nnz:] = block.data
            indices[nnz:] = block.indices
            indptr.append(block.indptr[1:].astype(np.int64) + nnz)
            nnz += size
        group.create_dataset("indptr", data=np.concatenate(indptr))
    return nnz


if __name__ == "__main__":
    n_cells = int(sys.argv[1])
    output_path = sys.argv[2]
    if "--stream" in sys.argv[3:]:
        nnz = write_counts_streamed(n_cells, 20000, 8, 0, output_path)
        print(output_path, (n_cells, 20000), nnz)
    else:
        annotated = make_counts(n_cells=n_cells, n_genes=20000, n_clusters=8, seed=0)
        annotated.write_h5ad(output_path)
        matrix: Any = annotated.X
        print(output_path, annotated.shape, matrix.nnz)
