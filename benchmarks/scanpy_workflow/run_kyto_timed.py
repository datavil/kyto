"""Run kyto's whole workflow once and split its time by step from provenance (12.1).

Usage: ``run_kyto_timed.py STORE OUT [--threads N] [--search exact|approximate]
[--device cpu|wgpu|auto]``.

At 1M cells ``kyto_steps.py``'s method (every plan prefix in a fresh process, three
times) would cost hours, so this runs the full plan of ``kyto_breakdown.py`` once, by
``sink_zarr``, and reads the seconds each node took from the written store's
``provenance.json`` (task 8.5: a node with its own passes or in-memory step is timed on
its own; per-cell maps and cell steps run fused in the sink's stream and are timed
under the sink node, with the writes). Prints one JSON object: ``seconds`` (the run),
``peak_mb`` (the process's peak RSS), ``nodes`` (each timed node's operation and
seconds, in plan order) and the load average at the start.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def node_operations(plan: Any) -> dict[str, str]:
    """Map each logical node ID (``n<ID>``) of a provenance plan to its operation.

    Parameters
    ----------
    plan : Any
        ``provenance.json``'s ``plan`` (the plan JSON).

    Returns
    -------
    dict[str, str]
        ``"n3"`` to ``"pca"``, ... (the step's ``method`` in the plan JSON). The sink
        node is not in the recorded plan (8.5): its ID is the one timed but unnamed.
    """
    operations: dict[str, str] = {}
    for node in plan.get("nodes", []):
        operations[f"n{node['id']}"] = str(node["step"]["method"])
    return operations


def main() -> None:
    """Run the plan once, print the timings JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store_path")
    parser.add_argument("output_path")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--search", default="exact", choices=("exact", "approximate"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "wgpu", "auto"))
    arguments = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    sys.argv = [sys.argv[0], "--with-umap"]
    from kyto_breakdown import build_prefixes

    load_average = [round(value, 1) for value in os.getloadavg()]
    plan = build_prefixes(arguments.store_path, arguments.search)["umap"]
    if arguments.device != "cpu":
        plan = plan.on(device=arguments.device)
    shutil.rmtree(arguments.output_path, ignore_errors=True)
    start = time.perf_counter()
    plan.sink_zarr(arguments.output_path, threads=arguments.threads)
    seconds = time.perf_counter() - start
    peak_mb = peak_rss_mb()
    provenance = json.loads(
        (Path(arguments.output_path) / "provenance.json").read_text()
    )
    run = provenance.get("run", {})
    print(
        json.dumps(
            {
                "threads": arguments.threads,
                "search": arguments.search,
                "device": arguments.device,
                "seconds": round(seconds, 3),
                "peak_mb": round(peak_mb),
                "load_average": load_average,
                "run": run,
                "operations": node_operations(provenance.get("plan", {})),
            }
        )
    )


if __name__ == "__main__":
    main()
