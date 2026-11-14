"""Run the basic clustering workflow with scanpy and print per-step timings."""

# scanpy, anndata and pandas ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
import sys
import time
from typing import Any

import scanpy as sc

sc.settings.n_jobs = 2


def run(input_path: str, output_path: str) -> dict[str, float]:
    """Run QC → normalize → HVG → PCA → neighbours → Leiden → UMAP and write h5ad.

    Parameters
    ----------
    input_path : str
        Counts as ``.h5ad``.
    output_path : str
        Where the processed ``.h5ad`` goes.

    Returns
    -------
    dict[str, float]
        Seconds per step.
    """
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(step_name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[step_name] = now - clock
        clock = now

    annotated = sc.read_h5ad(input_path)
    lap("read")
    annotated.var["mt"] = annotated.var["symbol"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        annotated, qc_vars=["mt"], percent_top=None, inplace=True
    )
    annotated = annotated[
        (annotated.obs["n_genes_by_counts"] >= 200)
        & (annotated.obs["n_genes_by_counts"] < 2500)
        & (annotated.obs["pct_counts_mt"] < 5)
    ].copy()
    lap("qc_filter")
    sc.pp.normalize_total(annotated, target_sum=1e4)
    sc.pp.log1p(annotated)
    lap("normalize_log1p")
    sc.pp.highly_variable_genes(annotated, n_top_genes=2000)
    annotated = annotated[:, annotated.var["highly_variable"]].copy()
    lap("hvg")
    sc.pp.pca(annotated, n_comps=50, random_state=0)
    lap("pca")
    sc.pp.neighbors(annotated, n_neighbors=15, use_rep="X_pca", random_state=0)
    lap("neighbors")
    sc.tl.leiden(annotated, flavor="igraph", n_iterations=2, random_state=0)
    lap("leiden")
    sc.tl.umap(annotated, random_state=0)
    lap("umap")
    annotated.write_h5ad(output_path)
    lap("write")
    timings["n_cells_kept"] = annotated.n_obs
    clusters: Any = annotated.obs["leiden"]
    timings["n_clusters"] = clusters.nunique()
    return timings


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2])))
