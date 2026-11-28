"""Run the basic clustering workflow with scanpy; print per-step seconds and peak RSS.

Usage: ``run_scanpy.py IN.h5ad OUT.h5ad [--threads N] [--warmup CELLS]``.

``--threads`` caps every pool scanpy uses (numba, BLAS and OpenMP through environment
variables set before anything imports them, threadpoolctl, ``sc.settings.n_jobs``).
``--warmup`` first runs the whole workflow in this process on the first ``CELLS``
cells, so numba's compilation is out of the timed run (warm); without it the timed
run pays any compilation numba's on-disk cache does not cover (point
``NUMBA_CACHE_DIR`` at an empty folder for a cold run).

Prints one JSON object: seconds per step, ``peak_mb`` (the process's peak RSS after
each step, cumulative), ``warmup_seconds``, the cells kept and the Leiden clusters.
"""

# scanpy, anndata and pandas ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Any

THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB.

    Returns
    -------
    float
        ``ru_maxrss`` in MB (bytes on macOS, kilobytes on Linux).
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def run(annotated: Any, output_path: str | None) -> dict[str, Any]:
    """Run QC → normalize → HVG → PCA → neighbours → Leiden → UMAP and write h5ad.

    Parameters
    ----------
    annotated : anndata.AnnData
        Counts, read already (the caller times the read).
    output_path : str or None
        Where the processed ``.h5ad`` goes; ``None`` skips the write.

    Returns
    -------
    dict[str, Any]
        ``seconds`` and ``peak_mb`` per step, cells kept and Leiden clusters.
    """
    import scanpy as sc

    seconds: dict[str, float] = {}
    peak_mb: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(step_name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        seconds[step_name] = round(now - clock, 3)
        peak_mb[step_name] = round(peak_rss_mb())
        # Progress on stderr (12.1): a run killed past the memory limit still says
        # which steps finished.
        print(step_name, seconds[step_name], peak_mb[step_name], file=sys.stderr)
        sys.stderr.flush()
        clock = now

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
    if output_path is not None:
        annotated.write_h5ad(output_path)
        lap("write")
    clusters: Any = annotated.obs["leiden"]
    return {
        "seconds": seconds,
        "peak_mb": peak_mb,
        "n_cells_kept": int(annotated.n_obs),
        "n_clusters": int(clusters.nunique()),
    }


def main() -> None:
    """Parse the arguments, cap the threads, warm up if asked, time one run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0, metavar="CELLS")
    arguments = parser.parse_args()
    if arguments.threads is not None:
        for name in THREAD_VARIABLES:
            os.environ[name] = str(arguments.threads)

    import scanpy as sc
    from threadpoolctl import threadpool_limits

    if arguments.threads is not None:
        sc.settings.n_jobs = arguments.threads
        threadpool_limits(arguments.threads)

    warmup_seconds = 0.0
    if arguments.warmup:
        start = time.perf_counter()
        # Backed, so only the warm-up's cells are loaded (12.1: 1M cells do not fit
        # twice); the same values as reading everything and slicing.
        backed = sc.read_h5ad(arguments.input_path, backed="r")
        small = backed[: arguments.warmup].to_memory()
        backed.file.close()
        run(small, None)
        del small
        warmup_seconds = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    annotated = sc.read_h5ad(arguments.input_path)
    read_seconds = round(time.perf_counter() - start, 3)
    read_peak_mb = round(peak_rss_mb())
    result = run(annotated, arguments.output_path)
    result["seconds"] = {"read": read_seconds, **result["seconds"]}
    result["peak_mb"] = {"read": read_peak_mb, **result["peak_mb"]}
    result["seconds"]["total"] = round(sum(result["seconds"].values()), 3)
    result["warmup_seconds"] = warmup_seconds
    result["threads"] = arguments.threads
    print(json.dumps(result))


if __name__ == "__main__":
    main()
