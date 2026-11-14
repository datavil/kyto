"""Run the basic clustering workflow with kyto and print timings.

kyto is lazy, so the whole plan runs once in ``sink_zarr``; only conversion and the
run are timed separately.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import kyto as ky
from kyto import cell_column, gene_column, qc


def run(
    input_path: str, store_path: str, output_path: str, search: str
) -> dict[str, float]:
    """Convert, then run QC → normalize → HVG → PCA → neighbours → Leiden → UMAP.

    Parameters
    ----------
    input_path : str
        Counts as ``.h5ad``.
    store_path : str
        Where the converted kyto store goes.
    output_path : str
        Where the processed kyto store goes.
    search : str
        Neighbour search, ``"exact"`` or ``"approximate"``.

    Returns
    -------
    dict[str, float]
        Seconds for ``convert`` and ``run``.
    """
    for path in (store_path, output_path):
        shutil.rmtree(path, ignore_errors=True)
    timings: dict[str, float] = {}
    start = time.perf_counter()
    ky.convert(input_path, store_path, layer="counts")
    timings["convert"] = time.perf_counter() - start

    mito = gene_column("symbol").str.starts_with("MT-")
    plan = (
        ky.scan(store_path)
        .with_default_layer("counts")
        .with_cells(
            n_genes_by_counts=qc.genes_detected(),
            pct_counts_mt=qc.percent_counts_in(mito),
        )
        .filter(cell_column("n_genes_by_counts") >= 200)
        .filter(cell_column("n_genes_by_counts") < 2500)
        .filter(cell_column("pct_counts_mt") < 5)
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .highly_variable_genes(layer="log1p", n_top_genes=2000)
        .filter(gene_column("highly_variable"))
        .pca(layer="log1p", into="X_pca", n_components=50, seed=0)
        .neighbors("X_pca", n_neighbors=15, search=search, seed=0)
        .leiden(seed=0, into="leiden")
        .umap(into="X_umap", seed=0)
    )
    start = time.perf_counter()
    plan.sink_zarr(output_path)
    timings["run"] = time.perf_counter() - start
    result = ky.scan(output_path)
    timings["n_cells_kept"] = result.count_cells()
    timings["n_clusters"] = result.cells.to_polars()["leiden"].n_unique()
    timings["store_mb"] = (
        sum(f.stat().st_size for f in Path(store_path).rglob("*") if f.is_file()) / 1e6
    )
    return timings


if __name__ == "__main__":
    print(json.dumps(run(*sys.argv[1:5])))
