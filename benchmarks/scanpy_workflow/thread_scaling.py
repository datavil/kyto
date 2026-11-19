"""Time the workflow's preprocessing and PCA at several thread counts (task 5.13).

Three runs per thread count, from the converted counts store (``run_kyto.py``):

- ``preprocessing``: QC, filters, ``normalize_total``, ``log1p``, HVG and the HVG filter
  (``kyto_breakdown.py``'s ``hvg`` prefix), streamed to nowhere (``_execute``): every
  pass over the store, nothing written.
- ``pca_workflow``: the same plan with PCA, to the embedding (``pca`` prefix).
- ``pca_isolated``: PCA alone on the HVG store ``pca_timing.py`` writes to ``WORK``.

Prints one JSON line per run with seconds (median of ``--repeats``, after one warm-up
run per plan) and the speedup over the first thread count, then whether the outputs
(the HVG gene table, the cell table and each embedding) are bit-identical across
thread counts. Nothing sets the blocks in flight: the run's default does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import kyto as ky
import numpy as np
import polars as pl
from kyto_breakdown import build_prefixes

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from kyto._kyto import Atlas

    Output = pl.DataFrame | npt.NDArray[np.float32 | np.float64]


def median_seconds(run: Callable[[], object], repeats: int) -> float:
    """Median wall-clock seconds of ``repeats`` calls of ``run``.

    Parameters
    ----------
    run : callable
        The work to time.
    repeats : int
        Calls; the median is returned.

    Returns
    -------
    float
        Median seconds.
    """
    seconds: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds)


def same_bits(
    left: npt.NDArray[np.float32 | np.float64],
    right: npt.NDArray[np.float32 | np.float64],
) -> bool:
    """Whether two arrays have the same dtype, shape and bytes.

    Parameters
    ----------
    left, right : numpy.ndarray
        The arrays.

    Returns
    -------
    bool
        ``True`` if bit-identical.
    """
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )


def same_frame(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Whether two frames have the same schema and values, floats compared by bits.

    ``DataFrame.equals`` counts ``-0.0`` equal to ``0.0``; float columns are compared
    as their bytes instead.

    Parameters
    ----------
    left, right : polars.DataFrame
        The frames.

    Returns
    -------
    bool
        ``True`` if bit-identical.
    """
    if left.schema != right.schema or left.height != right.height:
        return False
    for name, dtype in left.schema.items():
        if dtype.is_float():
            if not same_bits(left[name].to_numpy(), right[name].to_numpy()):
                return False
        elif not left[name].equals(right[name]):
            return False
    return True


def same_outputs(left: list[Output], right: list[Output]) -> bool:
    """Whether two runs' outputs (frames and arrays, in order) are bit-identical.

    Parameters
    ----------
    left, right : list
        ``polars.DataFrame`` and ``numpy.ndarray`` outputs of two runs.

    Returns
    -------
    bool
        ``True`` if every pair is bit-identical.
    """
    for item, expected in zip(left, right, strict=True):
        if isinstance(item, np.ndarray) and isinstance(expected, np.ndarray):
            if not same_bits(item, expected):
                return False
        elif isinstance(item, pl.DataFrame) and isinstance(expected, pl.DataFrame):
            if not same_frame(item, expected):
                return False
        else:
            return False
    return True


def main() -> None:
    """Run every case at every thread count; print JSON lines and the bit checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("store", help="the converted counts store (run_kyto.py)")
    parser.add_argument("work", help="the directory holding pca_timing.py's HVG store")
    parser.add_argument("--threads", default="1,4,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", default="preprocessing,pca_workflow,pca_isolated")
    arguments = parser.parse_args()
    thread_counts = [int(count) for count in arguments.threads.split(",")]
    prefixes = build_prefixes(arguments.store)
    hvg_store = Path(arguments.work) / "hvg.kyto"
    if not hvg_store.exists():
        prefixes["hvg"].sink_zarr(str(hvg_store))
    isolated = ky.scan(str(hvg_store)).pca(
        layer="log1p", into="X_pca", n_components=50, seed=0
    )
    for case in arguments.cases.split(","):
        outputs: dict[int, list[Output]] = {}
        first = 0.0
        for threads in thread_counts:
            if case == "preprocessing":
                plan = prefixes["hvg"]

                def run(plan: Atlas = plan, threads: int = threads) -> object:
                    execute = plan._execute  # pyright: ignore[reportPrivateUsage]
                    return execute(threads=threads)

                outputs[threads] = [
                    plan.genes.to_polars(threads=threads),
                    plan.cells.to_polars(threads=threads),
                ]
            else:
                plan = prefixes["pca"] if case == "pca_workflow" else isolated

                def run(plan: Atlas = plan, threads: int = threads) -> object:
                    return plan.embedding("X_pca").to_numpy(threads=threads)

                outputs[threads] = [plan.embedding("X_pca").to_numpy(threads=threads)]
            run()  # warm-up
            seconds = median_seconds(run, arguments.repeats)
            first = first or seconds
            print(
                json.dumps(
                    {
                        "case": case,
                        "threads": threads,
                        "seconds": round(seconds, 3),
                        "speedup": round(first / seconds, 2),
                    }
                ),
                flush=True,
            )
        reference = outputs[thread_counts[0]]
        identical = {
            threads: same_outputs(output, reference)
            for threads, output in outputs.items()
        }
        print(case, "bit-identical:", json.dumps(identical), flush=True)


if __name__ == "__main__":
    main()
