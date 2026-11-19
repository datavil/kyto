"""Time kyto's PCA with its input held in memory and streamed (task 5.12).

Two cases, each run with the input held (the default) and streamed on every pass (a
``memory_limit`` that leaves no room to hold it, ``--isolated-limit`` and
``--workflow-limit``), at each thread count:

- ``isolated``: PCA alone, on a store of the workflow's HVG-filtered ``log1p`` layer
  (written once beforehand, not timed): one read of the stored layer and the passes.
- ``workflow``: QC, filters, ``normalize_total``, ``log1p``, HVG, the HVG filter and
  PCA from the converted counts store, as ``kyto_breakdown.py``'s ``pca`` prefix.

Prints, per run, seconds (median of ``--repeats``) and the progress total (cell rows:
a held input adds one pass, so it tells which path ran), then whether every
embedding of a case is bit-identical. Saves the isolated held embedding (most threads)
as ``WORK/kyto_pca.npy``, for ``pca_scanpy.py`` to compare.
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
from kyto_breakdown import build_prefixes

if TYPE_CHECKING:
    import numpy.typing as npt
    from kyto._kyto import Atlas


def timed(
    plan: Atlas, threads: int, memory_limit: str | None, repeats: int
) -> tuple[float, int, npt.NDArray[np.float32 | np.float64]]:
    """Run ``plan``'s ``X_pca`` embedding ``repeats`` times.

    Parameters
    ----------
    plan : Atlas
        A plan ending in ``pca(into="X_pca")``.
    threads : int
        Threads to run it on.
    memory_limit : str or None
        ``None`` holds the input; a size too small to hold it streams every pass.
    repeats : int
        Runs; the median time is reported.

    Returns
    -------
    tuple
        Median seconds, the progress total of the last run, and the embedding.
    """
    seconds: list[float] = []
    totals: list[int] = []
    embedding = np.empty(0, dtype=np.float32)
    for _ in range(repeats):
        start = time.perf_counter()
        embedding = plan.embedding("X_pca").to_numpy(
            threads=threads,
            memory_limit=memory_limit,
            progress=lambda _done, total: totals.append(total),
        )
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds), totals[-1], embedding


def main() -> None:
    """Run the cases and print one JSON line per run, then the bit checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("store", help="the converted counts store (run_kyto.py)")
    parser.add_argument("work", help="a directory for the HVG store and outputs")
    parser.add_argument("--threads", default="1,4,10")
    parser.add_argument("--repeats", type=int, default=3)
    # memory_limit values below what holding the input needs, above what streaming
    # needs (found on the 50k data, macOS, by the progress total; they also shrink the
    # blocks, so the streamed times are an upper bound on streaming at full blocks).
    parser.add_argument("--isolated-limit", default="180MB")
    parser.add_argument("--workflow-limit", default="1GB")
    parser.add_argument("--cases", default="isolated,workflow")
    arguments = parser.parse_args()
    work = Path(arguments.work)
    work.mkdir(parents=True, exist_ok=True)
    thread_counts = [int(count) for count in arguments.threads.split(",")]
    prefixes = build_prefixes(arguments.store)
    hvg_store = work / "hvg.kyto"
    if not hvg_store.exists():
        prefixes["hvg"].sink_zarr(str(hvg_store))
    plans = {
        "isolated": ky.scan(str(hvg_store)).pca(
            layer="log1p", into="X_pca", n_components=50, seed=0
        ),
        "workflow": prefixes["pca"],
    }
    for case in arguments.cases.split(","):
        plan = plans[case]
        print(case, plan.estimate(), flush=True)
        embeddings: dict[str, npt.NDArray[np.float32 | np.float64]] = {}
        stream_limit = getattr(arguments, f"{case}_limit")
        for threads in thread_counts:
            for path, limit in (("held", None), ("streamed", stream_limit)):
                seconds, total, embedding = timed(
                    plan, threads, limit, arguments.repeats
                )
                embeddings[f"{path}@{threads}"] = embedding
                print(
                    json.dumps(
                        {
                            "case": case,
                            "path": path,
                            "threads": threads,
                            "seconds": round(seconds, 3),
                            "progress_total": total,
                        }
                    ),
                    flush=True,
                )
        reference = next(iter(embeddings.values()))
        identical = {
            name: bool(
                embedding.dtype == reference.dtype
                and np.array_equal(embedding.view(np.uint8), reference.view(np.uint8))
            )
            for name, embedding in embeddings.items()
        }
        print(case, "bit-identical:", json.dumps(identical), flush=True)
        if case == "isolated":
            np.save(work / "kyto_pca.npy", embeddings[f"held@{thread_counts[-1]}"])


if __name__ == "__main__":
    main()
