"""Task 6.16 on the 50k data: e-means metacells against Lloyd's k-means, and timings.

Runs the workflow up to a 50-component PCA on a kyto store (``run_kyto.py``'s steps),
holds it with ``collect``, then for k = n/50 and n/200 groups the cells with kyto's
``metacells`` (3 epochs) and with scikit-learn's ``KMeans`` (Lloyd, 25 iterations,
random init) over the same seeds. Prints, as JSON lines, each side's mean training MSE
and median seconds, and checks that kyto's labels and centroids are bit-identical at
1, 4 and all threads. Not part of the gate (minutes on 50k cells); the numbers go in
performance.md.

    uv run python metacells.py store.kyto [--seeds 10] [--threads 10]
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import kyto as ky
import numpy as np
from kyto import _kyto, cell_column, gene_column, qc
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits  # pyright: ignore[reportMissingTypeStubs]


def preprocessed(store: Path) -> _kyto.Atlas:
    """QC, normalize, log1p, HVG and a 50-component PCA, as ``run_kyto.py``."""
    mito = gene_column("symbol").str.starts_with("MT-")
    return (
        ky.scan(store)
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
    )


def mse(points: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Mean squared distance of each cell to its group's mean."""
    centroids = np.zeros((labels.max() + 1, points.shape[1]))
    np.add.at(centroids, labels, points)
    centroids /= np.maximum(np.bincount(labels), 1)[:, None]
    return float(((points - centroids[labels]) ** 2).sum() / len(points))


def labels_of(atlas: _kyto.Atlas, threads: int) -> NDArray[np.int64]:
    values = atlas.pull(cell_column("metacells"), threads=threads).to_numpy()
    return np.array([int(value) for value in values], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="e-means metacells against Lloyd's k-means on a kyto store"
    )
    parser.add_argument("store", type=Path)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    arguments = parser.parse_args()
    cluster: Any = importlib.import_module("sklearn.cluster")

    held = preprocessed(arguments.store).collect(threads=arguments.threads)
    points = held.embedding("X_pca").to_numpy().astype(np.float64)
    n = len(points)
    for divisor in (50, 200):
        k = n // divisor
        ours: list[float] = []
        ours_seconds: list[float] = []
        lloyd: list[float] = []
        lloyd_seconds: list[float] = []
        for seed in range(arguments.seeds):
            atlas = held.metacells("X_pca", n_metacells=k, seed=seed)
            start = time.perf_counter()
            labels = labels_of(atlas, arguments.threads)
            ours_seconds.append(time.perf_counter() - start)
            ours.append(mse(points, labels))
            with threadpool_limits(arguments.threads):
                start = time.perf_counter()
                model: Any = cluster.KMeans(
                    k,
                    init="random",
                    n_init=1,
                    max_iter=25,
                    tol=0.0,
                    algorithm="lloyd",
                    random_state=seed,
                ).fit(points)
                lloyd_seconds.append(time.perf_counter() - start)
            lloyd.append(float(model.inertia_) / n)
        atlas = held.metacells("X_pca", n_metacells=k, seed=0)
        found = [
            (
                labels_of(atlas, threads).tobytes(),
                atlas.embedding("metacells_centroids")
                .to_numpy(threads=threads)
                .tobytes(),
            )
            for threads in sorted({1, 4, os.cpu_count() or 1})
        ]
        print(
            json.dumps(
                {
                    "n": n,
                    "k": k,
                    "seeds": arguments.seeds,
                    "threads": arguments.threads,
                    "emeans_mse": float(np.mean(ours)),
                    "lloyd_mse": float(np.mean(lloyd)),
                    "relative": float(np.mean(ours) / np.mean(lloyd) - 1),
                    "emeans_seconds": float(np.median(ours_seconds)),
                    "lloyd_seconds": float(np.median(lloyd_seconds)),
                    "identical_across_threads": all(item == found[0] for item in found),
                }
            )
        )


if __name__ == "__main__":
    main()
