"""Run the basic clustering workflow with rapids-singlecell (task 10.3); print JSON.

Usage: ``run_rapids.py IN.h5ad OUT.h5ad [--threads N] [--warmup CELLS]``.

The same steps and parameters as ``run_scanpy.py``, on the GPU: read the ``.h5ad``
with scanpy, move the matrix to the GPU (timed as part of ``read``), run
``rapids_singlecell``'s ``pp``/``tl`` functions, move back and write. Every lap
synchronises the device first, so a step's seconds include its kernels.
``--warmup`` first runs the workflow on that many cells in this process (JIT and
CUDA context set-up out of the timed run).

Prints one JSON object: ``seconds`` and ``peak_mb`` (host peak RSS) per step, and
``peak_gpu_mb``, the largest ``cupy`` memory pool size seen at a lap (a lower bound of
the device peak; RMM or other allocators are not counted).

Needs CUDA, ``cupy`` and ``rapids-singlecell``, which the dev environment does not
install. Not yet run on a CUDA machine: check it there before trusting its numbers.
"""

# scanpy, anndata, cupy and rapids-singlecell ship without complete type information.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import importlib
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
)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (``ru_maxrss``)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def run(annotated: Any, output_path: str | None) -> dict[str, Any]:
    """Run QC → normalize → HVG → PCA → neighbours → Leiden → UMAP and write h5ad.

    Parameters
    ----------
    annotated : anndata.AnnData
        Counts already on the GPU (``rsc.get.anndata_to_GPU``).
    output_path : str or None
        Where the processed ``.h5ad`` goes; ``None`` skips the write.

    Returns
    -------
    dict[str, Any]
        ``seconds``, ``peak_mb`` and ``peak_gpu_mb`` per step, cells kept, clusters.
    """
    cupy: Any = importlib.import_module("cupy")
    rsc: Any = importlib.import_module("rapids_singlecell")
    pool = cupy.get_default_memory_pool()
    seconds: dict[str, float] = {}
    peak_mb: dict[str, float] = {}
    peak_gpu_mb = 0.0
    clock = time.perf_counter()

    def lap(step_name: str) -> None:
        nonlocal clock, peak_gpu_mb
        cupy.cuda.runtime.deviceSynchronize()
        now = time.perf_counter()
        seconds[step_name] = round(now - clock, 3)
        peak_mb[step_name] = round(peak_rss_mb())
        peak_gpu_mb = max(peak_gpu_mb, pool.total_bytes() / 1e6)
        clock = now

    annotated.var["mt"] = annotated.var["symbol"].str.startswith("MT-")
    rsc.pp.calculate_qc_metrics(annotated, qc_vars=["mt"])
    annotated = annotated[
        (annotated.obs["n_genes_by_counts"] >= 200)
        & (annotated.obs["n_genes_by_counts"] < 2500)
        & (annotated.obs["pct_counts_mt"] < 5)
    ].copy()
    lap("qc_filter")
    rsc.pp.normalize_total(annotated, target_sum=1e4)
    rsc.pp.log1p(annotated)
    lap("normalize_log1p")
    rsc.pp.highly_variable_genes(annotated, n_top_genes=2000, flavor="seurat")
    annotated = annotated[:, annotated.var["highly_variable"]].copy()
    lap("hvg")
    rsc.pp.pca(annotated, n_comps=50, random_state=0)
    lap("pca")
    rsc.pp.neighbors(annotated, n_neighbors=15, use_rep="X_pca", random_state=0)
    lap("neighbors")
    rsc.tl.leiden(annotated, n_iterations=2, random_state=0)
    lap("leiden")
    rsc.tl.umap(annotated, random_state=0)
    lap("umap")
    if output_path is not None:
        rsc.get.anndata_to_CPU(annotated)
        annotated.write_h5ad(output_path)
        lap("write")
    return {
        "seconds": seconds,
        "peak_mb": peak_mb,
        "peak_gpu_mb": round(peak_gpu_mb),
        "n_cells_kept": int(annotated.n_obs),
        "n_clusters": int(annotated.obs["leiden"].nunique()),
    }


def main() -> None:
    """Parse the arguments, cap host threads, warm up if asked, time one run."""
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

    rsc: Any = importlib.import_module("rapids_singlecell")
    if arguments.warmup:
        small = sc.read_h5ad(arguments.input_path)[: arguments.warmup].copy()
        rsc.get.anndata_to_GPU(small)
        run(small, None)
        del small
        # The timed run's pool peak must not include the warm-up's blocks.
        importlib.import_module("cupy").get_default_memory_pool().free_all_blocks()

    start = time.perf_counter()
    annotated = sc.read_h5ad(arguments.input_path)
    rsc.get.anndata_to_GPU(annotated)
    read_seconds = round(time.perf_counter() - start, 3)
    read_peak_mb = round(peak_rss_mb())
    result = run(annotated, arguments.output_path)
    result["seconds"] = {"read": read_seconds, **result["seconds"]}
    result["peak_mb"] = {"read": read_peak_mb, **result["peak_mb"]}
    result["seconds"]["total"] = round(sum(result["seconds"].values()), 3)
    result["threads"] = arguments.threads
    print(json.dumps(result))


if __name__ == "__main__":
    main()
