"""Per-cell kNN recall of scanpy's and kyto's neighbour graphs against exact (12.1).

Usage: ``neighbor_recall.py EXACT_KYTO_OUT SCANPY_OUT.h5ad [APPROXIMATE_KYTO_OUT]``.

The neighbour steps the 12.1 table compares are not equal in accuracy: scanpy's
pynndescent is approximate, kyto's ``search="exact"`` is exact and its HNSW is
approximate. This states how approximate, per cell (memory/workflow_benchmark.md: a
mean hides cells with recall 0). Graphs are aligned by cell ID; a cell's recall is the
share of its exact neighbours (the ``distances`` graph of the exact kyto run) that the
other graph also lists. Both embeddings come from the same PCA up to float rounding
(6.5's tolerance), so recall against kyto's exact graph measures the search, not PCA.
Prints one JSON object: mean, minimum and the counts of cells below 0.5 and 0.9.
"""

# scanpy and anndata ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import sys
from typing import Any

import anndata
import kyto as ky
import numpy as np
from scipy import sparse


def kyto_graph(path: str) -> tuple[list[str], Any]:
    """Cell IDs and the ``distances`` graph (CSR) of a kyto output store."""
    atlas = ky.scan(path)
    cell_ids = atlas.cells.to_polars()["cell_id"].to_list()
    return cell_ids, sparse.csr_matrix(atlas.graph("distances").to_scipy())


def recall(exact: Any, other: Any) -> dict[str, float]:
    """Per-row share of ``exact``'s neighbours that ``other`` also has."""
    values = np.empty(exact.shape[0])
    for row in range(exact.shape[0]):
        truth = exact.indices[exact.indptr[row] : exact.indptr[row + 1]]
        found = other.indices[other.indptr[row] : other.indptr[row + 1]]
        values[row] = len(np.intersect1d(truth, found)) / max(len(truth), 1)
    return {
        "mean": round(float(values.mean()), 5),
        "min": round(float(values.min()), 4),
        "below_0.5": int((values < 0.5).sum()),
        "below_0.9": int((values < 0.9).sum()),
    }


def main() -> None:
    """Compute and print the recalls."""
    exact_ids, exact = kyto_graph(sys.argv[1])
    position = {cell_id: index for index, cell_id in enumerate(exact_ids)}
    annotated = anndata.read_h5ad(sys.argv[2])
    order = np.array([position[name] for name in annotated.obs_names])
    # Put scanpy's rows and columns in kyto's cell order.
    permutation = sparse.csr_matrix(
        (np.ones(len(order)), (order, np.arange(len(order)))),
        shape=(len(order), len(order)),
    )
    scanpy = sparse.csr_matrix(
        permutation @ annotated.obsp["distances"] @ permutation.T
    )
    scanpy.sort_indices()
    result: dict[str, Any] = {"scanpy_pynndescent": recall(exact, scanpy)}
    if len(sys.argv) > 3:
        approximate_ids, approximate = kyto_graph(sys.argv[3])
        assert approximate_ids == exact_ids
        result["kyto_hnsw"] = recall(exact, approximate)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
