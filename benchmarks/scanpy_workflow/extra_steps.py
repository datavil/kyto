"""Time the steps the workflow scripts leave out, kyto and scanpy (task 12.1).

Usage::

    extra_steps.py read H5AD WORK [--repeats 3]
    extra_steps.py rank KYTO_OUT SCANPY_OUT [--threads 1,4,10] [--repeats 3]
    extra_steps.py metacells KYTO_OUT [--threads 1,4,10] [--repeats 3]

``--kyto-only`` skips scanpy's measurements.

- ``read``: kyto's ``ky.convert`` of the ``.h5ad`` into a store under ``WORK`` against
  scanpy's ``read_h5ad`` (not equal work: kyto writes a compressed store once, scanpy
  loads the matrix into memory each session; both are each tool's entry cost).
- ``rank``: ``rank_genes_groups`` by ``leiden`` (``t-test`` and ``wilcoxon``) on each
  tool's own processed output (the workflow's ``log1p`` HVG matrix and its Leiden
  labels, which are the same partition: ARI 1.0): kyto
  ``scan(KYTO_OUT).rank_genes_groups(..., layer="log1p").collect()``, scanpy
  ``tl.rank_genes_groups`` then ``get.rank_genes_groups_df`` (the same table). scanpy is
  timed after its read and after one warm-up call in the same process (numba).
- ``metacells``: kyto's opt-in ``metacells`` (6.16) on ``X_pca``, ``n / 200`` metacells,
  3 epochs, streamed to nowhere; scanpy has no such step.

Every measurement runs in a fresh process; time is the median of ``--repeats``, peak
RSS the largest (``ru_maxrss``: the whole process, data and interpreter included).
Prints one JSON line per measurement.
"""

# scanpy, anndata and pandas ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def cap_scanpy_threads(threads: int) -> None:
    """Cap scanpy's pools as ``run_scanpy.py`` does (before numba is imported)."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    import scanpy as sc
    from threadpoolctl import threadpool_limits

    sc.settings.n_jobs = threads
    threadpool_limits(threads)


def child(task: str, threads: int, paths: list[str]) -> dict[str, Any]:
    """Run one measurement in this process and return seconds and peak RSS.

    Parameters
    ----------
    task : str
        ``kyto_convert``, ``scanpy_read``, ``kyto_rank:<method>``,
        ``scanpy_rank:<method>`` or ``kyto_metacells``.
    threads : int
        Threads of the run.
    paths : list[str]
        The task's paths (input first).

    Returns
    -------
    dict[str, Any]
        ``seconds``, ``peak_mb`` and what the task produced (rows, cells).
    """
    extra: dict[str, Any] = {}
    if task.startswith("scanpy") and threads > 0:  # 0: the read, no cap
        cap_scanpy_threads(threads)
    if task == "kyto_convert":
        import kyto as ky

        shutil.rmtree(paths[1], ignore_errors=True)
        start = time.perf_counter()
        ky.convert(paths[0], paths[1], layer="counts")
        seconds = time.perf_counter() - start
    elif task == "scanpy_read":
        import scanpy as sc

        start = time.perf_counter()
        annotated = sc.read_h5ad(paths[0])
        seconds = time.perf_counter() - start
        extra["n_cells"] = int(annotated.n_obs)
    elif task.startswith("kyto_rank:"):
        import kyto as ky

        method = task.split(":", 1)[1]
        start = time.perf_counter()
        frame = (
            ky.scan(paths[0])
            .rank_genes_groups(
                "leiden",
                method=method,  # pyright: ignore[reportArgumentType]
                layer="log1p",
                threads=threads,
            )
            .collect()
        )
        seconds = time.perf_counter() - start
        extra["rows"] = frame.height
    elif task.startswith("scanpy_rank:"):
        import scanpy as sc

        method = task.split(":", 1)[1]
        annotated = sc.read_h5ad(paths[0])
        sc.tl.rank_genes_groups(annotated, "leiden", method=method)  # pyright: ignore[reportArgumentType]  # warm-up
        start = time.perf_counter()
        sc.tl.rank_genes_groups(annotated, "leiden", method=method)  # pyright: ignore[reportArgumentType]
        frame = sc.get.rank_genes_groups_df(annotated, group=None)
        seconds = time.perf_counter() - start
        extra["rows"] = int(frame.shape[0])
    elif task == "kyto_metacells":
        import kyto as ky

        plan = ky.scan(paths[0])
        n_metacells = max(1, plan.count_cells() // 200)
        plan = plan.metacells("X_pca", n_metacells=n_metacells, seed=0, into="meta")
        start = time.perf_counter()
        rows = plan._execute(threads=threads)  # pyright: ignore[reportPrivateUsage]
        seconds = time.perf_counter() - start
        extra["n_metacells"] = n_metacells
        extra["rows"] = rows
    else:
        raise SystemExit(f"unknown task {task!r}")
    return {"seconds": seconds, "peak_mb": peak_rss_mb(), **extra}


def measure(task: str, threads: int, paths: list[str], repeats: int) -> dict[str, Any]:
    """Run ``child`` ``repeats`` times in fresh processes; median seconds, peak RSS.

    Parameters
    ----------
    task : str
        As ``child``.
    threads : int
        Threads of the run.
    paths : list[str]
        As ``child``.
    repeats : int
        Fresh-process runs.

    Returns
    -------
    dict[str, Any]
        ``task``, ``threads``, ``seconds`` (median), ``runs``, ``peak_mb`` (largest)
        and the last run's other fields.
    """
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        command = [sys.executable, __file__, "--child", task, str(threads), *paths]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        print(task, threads, round(runs[-1]["seconds"], 3), file=sys.stderr, flush=True)
    return {
        **runs[-1],
        "task": task,
        "threads": threads,
        "seconds": round(statistics.median(run["seconds"] for run in runs), 3),
        "runs": [round(run["seconds"], 3) for run in runs],
        "peak_mb": round(max(run["peak_mb"] for run in runs)),
        "load_average": [round(value, 1) for value in os.getloadavg()],
    }


def main() -> None:
    """Parse the mode and print one JSON line per measurement."""
    if sys.argv[1:2] == ["--child"]:
        result = child(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
        print(json.dumps(result))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("read", "rank", "metacells"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--threads", default="1,4,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="t-test,wilcoxon")
    parser.add_argument(
        "--kyto-only",
        action="store_true",
        help="skip scanpy (12.1: at 1M cells scanpy runs under guard.py instead)",
    )
    arguments = parser.parse_args()
    thread_counts = [int(count) for count in arguments.threads.split(",")]
    jobs: list[tuple[str, int, list[str]]] = []
    if arguments.mode == "read":
        h5ad, work = arguments.paths
        store = str(Path(work) / "convert.kyto")
        jobs = [("kyto_convert", 0, [h5ad, store]), ("scanpy_read", 0, [h5ad])]
    elif arguments.mode == "rank":
        kyto_out, scanpy_out = arguments.paths
        for threads in thread_counts:
            for method in arguments.methods.split(","):
                jobs.append((f"kyto_rank:{method}", threads, [kyto_out]))
                jobs.append((f"scanpy_rank:{method}", threads, [scanpy_out]))
    else:
        jobs = [
            ("kyto_metacells", count, arguments.paths[:1]) for count in thread_counts
        ]
    if arguments.kyto_only:
        jobs = [job for job in jobs if job[0].startswith("kyto")]
    for task, threads, paths in jobs:
        print(json.dumps(measure(task, threads, paths, arguments.repeats)), flush=True)


if __name__ == "__main__":
    main()
