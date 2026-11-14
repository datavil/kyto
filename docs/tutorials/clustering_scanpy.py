"""scanpy's basic clustering tutorial, step for step, as the reference for kyto's run.

The steps and arguments are the tutorial's
(https://scanpy.scverse.org/en/stable/tutorials/basics/clustering.html); only the data
loading differs (the two samples come from kyto's verified fixture cache instead of
pooch) and nothing is plotted. ``run`` returns the processed ``AnnData`` with the
Wilcoxon ranking in ``uns`` and the wall-clock seconds per step.
"""

import time
from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import scanpy as sc

# scanpy and anndata ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

RESOLUTIONS = (0.02, 0.5, 2.0)


def run(samples: Mapping[str, Path]) -> tuple[ad.AnnData, dict[str, float]]:
    """Run the tutorial on the given 10x ``.h5`` files.

    Parameters
    ----------
    samples : Mapping[str, Path]
        Sample name to its 10x ``.h5`` file, in the tutorial's order (``s1d1``,
        ``s1d3``).

    Returns
    -------
    tuple of (anndata.AnnData, dict of str to float)
        The processed data and the seconds each step took.
    """
    timings: dict[str, float] = {}
    start = time.perf_counter()

    def lap(step: str) -> None:
        nonlocal start
        now = time.perf_counter()
        timings[step] = now - start
        start = now

    adatas: dict[str, ad.AnnData] = {}
    for sample_id, path in samples.items():
        sample_adata = sc.read_10x_h5(path)
        sample_adata.var_names_make_unique()
        adatas[sample_id] = sample_adata
    adata = ad.concat(adatas, label="sample")
    adata.obs_names_make_unique()
    lap("read and concat")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    lap("qc metrics")

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    lap("filters")

    sc.pp.scrublet(adata, batch_key="sample")
    lap("scrublet")

    counts = adata.X
    if counts is None:
        raise ValueError("the concatenated data has no matrix")
    adata.layers["counts"] = counts.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    lap("normalize and log1p")

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
    lap("highly variable genes")

    sc.tl.pca(adata)
    lap("pca")

    sc.pp.neighbors(adata)
    lap("neighbors")

    sc.tl.umap(adata)
    lap("umap")

    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    for resolution in RESOLUTIONS:
        sc.tl.leiden(
            adata,
            key_added=f"leiden_res_{resolution:4.2f}",
            resolution=resolution,
            flavor="igraph",
        )
    lap("leiden")

    sc.tl.rank_genes_groups(adata, groupby="leiden_res_0.50", method="wilcoxon")
    lap("rank genes groups")
    return adata, timings


if __name__ == "__main__":
    import resource
    import sys

    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: clustering_scanpy.py <s1d1.h5> <s1d3.h5> [out.h5ad]")
    paths = {"s1d1": Path(sys.argv[1]), "s1d3": Path(sys.argv[2])}
    total = time.perf_counter()
    result, seconds = run(paths)
    elapsed = time.perf_counter() - total
    for step, value in seconds.items():
        print(f"{step}: {value:.2f} s")
    print(f"total: {elapsed:.2f} s")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    print(f"peak_rss_mb={peak / (2**20 if sys.platform == 'darwin' else 2**10):.0f}")
    if len(sys.argv) > 3:
        result.write_h5ad(sys.argv[3])
