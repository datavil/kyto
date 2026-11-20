"""Per-step seconds and peak RSS of kyto's workflow, at several thread counts (10.8).

Usage: ``kyto_steps.py STORE OUT [--threads 1,4,10] [--repeats 3] [--with-write]
[--search exact|approximate]``.

Each plan prefix of ``kyto_breakdown.py`` (QC and filters, + normalize and ``log1p``,
+ HVG, + PCA, + neighbours, + Leiden, + UMAP) runs in a fresh process, streamed to
nowhere (``_execute``): every pass and every embedding, nothing written. A step's
seconds are the difference between its prefix's median and the previous one's, so
they include any pass the step adds over the store. ``--with-write`` adds the full
plan run by ``sink_zarr`` to ``OUT``; ``write`` is its median minus the UMAP prefix's.
A prefix's ``peak_mb`` is its process's peak RSS: the peak of the plan up to that step.

Prints one JSON line per thread count.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import statistics
import subprocess
import sys
import time
from typing import Literal

STEPS = (
    "qc_filter",
    "normalize_log1p",
    "hvg",
    "pca",
    "neighbors",
    "leiden",
    "umap",
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


def child(
    step_name: str, threads: int, search: str, store_path: str, output_path: str
) -> None:
    """Run one prefix (``write``: the full plan to ``output_path``); print JSON.

    Parameters
    ----------
    step_name : str
        A name in ``STEPS``, or ``"write"``.
    threads : int
        Threads of the run.
    search : str
        Neighbour search, ``"exact"`` or ``"approximate"``.
    store_path : str
        Converted kyto store.
    output_path : str
        Where ``write`` puts the processed store.
    """
    sys.argv = [sys.argv[0], "--with-umap"]
    from kyto_breakdown import build_prefixes

    prefixes = build_prefixes(store_path, _search(search))
    plan = prefixes["umap" if step_name == "write" else step_name]
    shutil.rmtree(output_path, ignore_errors=True)
    start = time.perf_counter()
    if step_name == "write":
        plan.sink_zarr(output_path, threads=threads)
    else:
        plan._execute(threads=threads)  # pyright: ignore[reportPrivateUsage]
    seconds = time.perf_counter() - start
    print(json.dumps({"seconds": seconds, "peak_mb": peak_rss_mb()}))


def _search(value: str) -> Literal["exact", "approximate"]:
    """The neighbour search named on the command line."""
    if value == "exact":
        return "exact"
    if value == "approximate":
        return "approximate"
    raise SystemExit(f"search must be 'exact' or 'approximate', not {value!r}")


def measure(
    step_name: str, threads: int, search: str, store_path: str, output_path: str
) -> dict[str, float]:
    """Run ``child`` in a fresh process and return what it printed.

    Parameters
    ----------
    step_name : str
        The prefix to run.
    threads : int
        Threads of the run.
    search : str
        Neighbour search, ``"exact"`` or ``"approximate"``.
    store_path : str
        Converted kyto store.
    output_path : str
        Where ``write`` puts the processed store.

    Returns
    -------
    dict[str, float]
        ``seconds`` and ``peak_mb``.
    """
    command = [
        sys.executable,
        __file__,
        "--child",
        step_name,
        str(threads),
        search,
        store_path,
        output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    """Measure every prefix at every thread count; print one JSON line each."""
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store_path")
    parser.add_argument("output_path")
    parser.add_argument("--threads", default="1,4,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-write", action="store_true")
    parser.add_argument("--search", default="exact", type=_search)
    arguments = parser.parse_args()
    step_names = [*STEPS, "write"] if arguments.with_write else list(STEPS)
    for threads in (int(count) for count in arguments.threads.split(",")):
        cumulative: dict[str, float] = {}
        peak_mb: dict[str, float] = {}
        for step_name in step_names:
            runs = [
                measure(
                    step_name,
                    threads,
                    arguments.search,
                    arguments.store_path,
                    arguments.output_path,
                )
                for _ in range(arguments.repeats)
            ]
            cumulative[step_name] = statistics.median(run["seconds"] for run in runs)
            peak_mb[step_name] = round(max(run["peak_mb"] for run in runs))
            spread = [round(run["seconds"], 2) for run in runs]
            print(step_name, threads, spread, file=sys.stderr, flush=True)
        seconds: dict[str, float] = {}
        previous = 0.0
        for step_name in step_names:
            base = cumulative["umap"] if step_name == "write" else previous
            seconds[step_name] = round(cumulative[step_name] - base, 3)
            previous = cumulative[step_name]
        report = {
            "threads": threads,
            "search": arguments.search,
            "cumulative": {name: round(value, 3) for name, value in cumulative.items()},
            "seconds": seconds,
            "peak_mb": peak_mb,
        }
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
