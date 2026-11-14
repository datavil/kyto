"""Time kyto plan prefixes to attribute run time to steps (each prefix runs from scratch)."""

from __future__ import annotations

import json
import shutil
import sys
import time

import kyto as ky
from kyto import cell_column, gene_column, qc


def build_prefixes(store_path: str) -> dict[str, ky.Atlas]:
    """Return the workflow's plan cut after each step, in order.

    Parameters
    ----------
    store_path : str
        Converted kyto store.

    Returns
    -------
    dict[str, ky.Atlas]
        Step name to the plan ending at that step.
    """
    mito = gene_column("symbol").str.starts_with("MT-")
    prefixes: dict[str, ky.Atlas] = {}
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
    )
    prefixes["qc_filter"] = plan
    plan = plan.normalize_total(1e4, into="normalized").log1p(
        layer="normalized", into="log1p"
    )
    prefixes["normalize_log1p"] = plan
    plan = plan.highly_variable_genes(layer="log1p", n_top_genes=2000).filter(
        gene_column("highly_variable")
    )
    prefixes["hvg"] = plan
    plan = plan.pca(layer="log1p", into="X_pca", n_components=50, seed=0)
    prefixes["pca"] = plan
    plan = plan.neighbors("X_pca", n_neighbors=15, seed=0)
    prefixes["neighbors"] = plan
    plan = plan.leiden(seed=0, into="leiden")
    prefixes["leiden"] = plan
    if "--with-umap" in sys.argv:
        prefixes["umap"] = plan.umap(into="X_umap", seed=0)
    return prefixes


if __name__ == "__main__":
    store_path, output_path = sys.argv[1], sys.argv[2]
    cumulative: dict[str, float] = {}
    for step_name, plan in build_prefixes(store_path).items():
        shutil.rmtree(output_path, ignore_errors=True)
        start = time.perf_counter()
        plan.sink_zarr(output_path)
        cumulative[step_name] = round(time.perf_counter() - start, 2)
        print(step_name, cumulative[step_name], flush=True)
    print(json.dumps(cumulative))
