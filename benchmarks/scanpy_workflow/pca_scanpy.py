"""Time scanpy's PCA alone on the workflow's HVG matrix (the counterpart of pca_timing.py).

Runs ``run_scanpy.py``'s steps up to the HVG subset (not timed), then
``sc.pp.pca(n_comps=50, random_state=0)`` ``--repeats`` times on a copy, at each
``--threads`` (BLAS and OpenMP pools capped by threadpoolctl), and prints the median
seconds. With ``--kyto WORK/kyto_pca.npy`` it compares kyto's embedding: per leading
component, the largest relative difference of the column (signs follow ``svd_flip`` in
both), and the largest principal angle over the first 10 components.
"""

# scanpy, anndata and pandas ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np
import scanpy as sc
from threadpoolctl import threadpool_limits


def hvg_matrix(input_path: str) -> Any:
    """QC, filters, ``normalize_total``, ``log1p``, HVGs and the HVG subset.

    Parameters
    ----------
    input_path : str
        Counts as ``.h5ad``.

    Returns
    -------
    anndata.AnnData
        The HVG-filtered ``log1p`` matrix.
    """
    annotated = sc.read_h5ad(input_path)
    annotated.var["mt"] = annotated.var["symbol"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        annotated, qc_vars=["mt"], percent_top=None, inplace=True
    )
    annotated = annotated[
        (annotated.obs["n_genes_by_counts"] >= 200)
        & (annotated.obs["n_genes_by_counts"] < 2500)
        & (annotated.obs["pct_counts_mt"] < 5)
    ].copy()
    sc.pp.normalize_total(annotated, target_sum=1e4)
    sc.pp.log1p(annotated)
    sc.pp.highly_variable_genes(annotated, n_top_genes=2000)
    return annotated[:, annotated.var["highly_variable"]].copy()


def main() -> None:
    """Time the PCA, then compare with kyto's embedding if given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="counts as .h5ad (make_data.py)")
    parser.add_argument("--threads", default="1,4,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kyto", help="kyto's embedding (.npy, pca_timing.py)")
    arguments = parser.parse_args()
    matrix = hvg_matrix(arguments.input)
    print(
        "hvg matrix",
        matrix.shape,
        matrix.X.dtype,
        f"nnz={matrix.X.nnz}",
        flush=True,
    )
    embedding: Any = None
    for threads in (int(count) for count in arguments.threads.split(",")):
        sc.settings.n_jobs = threads
        seconds: list[float] = []
        with threadpool_limits(limits=threads):
            for _ in range(arguments.repeats):
                annotated = matrix.copy()
                start = time.perf_counter()
                sc.pp.pca(annotated, n_comps=50, random_state=0)
                seconds.append(time.perf_counter() - start)
                embedding = annotated.obsm["X_pca"]
        print(
            json.dumps(
                {
                    "tool": "scanpy",
                    "threads": threads,
                    "seconds": round(statistics.median(seconds), 3),
                }
            ),
            flush=True,
        )
    if arguments.kyto:
        kyto = np.load(arguments.kyto).astype(np.float64)
        scanpy = np.asarray(embedding, dtype=np.float64)
        columns = [
            float(
                np.max(np.abs(kyto[:, column] - scanpy[:, column]))
                / np.max(np.abs(scanpy[:, column]))
            )
            for column in range(10)
        ]
        left, _ = np.linalg.qr(kyto[:, :10])
        right, _ = np.linalg.qr(scanpy[:, :10])
        cosines = np.linalg.svd(left.T @ right, compute_uv=False)
        angle = float(np.arccos(np.clip(cosines.min(), -1.0, 1.0)))
        print(
            json.dumps(
                {
                    "first_10_columns_worst_relative": max(columns),
                    "largest_angle_first_10_rad": angle,
                }
            )
        )


if __name__ == "__main__":
    main()
