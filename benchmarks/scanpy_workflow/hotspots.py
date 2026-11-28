"""Name the hotspots of a samply profile, per workflow stage (task 12.1).

Usage: ``hotspots.py PROFILE.json.gz [--top 8] [--stage NAME=REGEX ...]``.

Reads a profile recorded with ``samply record --save-only --unstable-presymbolicate``
(the ``.syms.json`` sidecar beside it gives the function names) and weighs every
sample by the CPU time its thread used since the last one (``threadCPUDelta``), so
threads parked in rayon or waiting on a channel count for nothing. Prints, as JSON:

- ``total_cpu_seconds``;
- ``self``: the functions with the most self time (the leaf frame), all threads;
- ``stages``: for each ``--stage NAME=REGEX``, the CPU seconds of the samples with a
  frame matching ``REGEX`` (the first matching stage wins, in the order given) and the
  top self functions inside it. Stage names come from kyto's own functions
  (``kyto_exec::umap``, ``kyto_algo::neighbors``, ...), so a kernel inlined into its
  caller shows under the caller.

Release builds inline aggressively, so a "function" here is the outermost non-inlined
symbol: build with line tables (``CARGO_PROFILE_RELEASE_DEBUG=line-tables-only``) and
``--strip false`` to have symbols at all.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[dict[str, Any], dict[str, dict[int, str]]]:
    """Load the profile and, per library debug name, its address-to-name table.

    Parameters
    ----------
    path : Path
        The ``.json.gz`` profile.

    Returns
    -------
    tuple
        The profile and ``{debug_name: {address: function name}}``.
    """
    with gzip.open(path) as stream:
        profile = json.loads(stream.read())
    sidecar = path.with_name(path.name.removesuffix(".gz").removesuffix(".json"))
    symbols = json.loads(Path(f"{sidecar}.json.syms.json").read_text())
    strings: list[str] = symbols["string_table"]
    names: dict[str, dict[int, str]] = {}
    for library in symbols["data"]:
        table = library["symbol_table"]
        names[library["debug_name"]] = {
            address: strings[table[index]["symbol"]]
            for address, index in library["known_addresses"]
        }
    return profile, names


def short(name: str) -> str:
    """Drop Rust hashes and generic noise from a symbol so it reads in a table."""
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    return name if len(name) <= 140 else name[:137] + "..."


def frame_names(
    thread: dict[str, Any], libs: list[dict[str, Any]], names: dict[str, dict[int, str]]
) -> list[str]:
    """The function name of every frame of a thread (``?`` where unknown)."""
    frames = thread["frameTable"]
    functions = thread["funcTable"]
    resources = thread["resourceTable"]
    strings = thread["stringArray"]
    result: list[str] = []
    for index in range(frames["length"]):
        function = frames["func"][index]
        resource = functions["resource"][function]
        address = frames["address"][index]
        name = None
        if resource is not None and resource >= 0:
            library = str(libs[int(resources["lib"][resource])]["debugName"])
            name = names.get(library, {}).get(address)
            if name is None:
                name = f"{library}+{address:#x}"
        result.append(short(name or strings[functions["name"][function]]))
    return result


def main() -> None:
    """Aggregate the profile and print the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--stage", action="append", default=[], metavar="NAME=REGEX")
    arguments = parser.parse_args()
    stages = [
        (text.split("=", 1)[0], re.compile(text.split("=", 1)[1]))
        for text in arguments.stage
    ]
    profile, names = load(arguments.profile)
    self_time: Counter[str] = Counter()
    stage_time: Counter[str] = Counter()
    stage_self: dict[str, Counter[str]] = {name: Counter() for name, _ in stages}
    total = 0.0
    for thread in profile["threads"]:
        frame_name = frame_names(thread, profile["libs"], names)
        stacks = thread["stackTable"]
        samples = thread["samples"]
        deltas = samples.get("threadCPUDelta") or [1.0] * samples["length"]
        stage_of_stack: dict[int, str] = {}
        for stack, delta in zip(samples["stack"], deltas, strict=True):
            if stack is None or not delta:
                continue
            seconds = delta / 1e6
            total += seconds
            leaf = frame_name[int(stacks["frame"][stack])]
            self_time[leaf] += seconds
            if stack not in stage_of_stack:
                chain: list[str] = []
                node: int | None = stack
                while node is not None:
                    chain.append(frame_name[int(stacks["frame"][node])])
                    node = stacks["prefix"][node]
                stage_of_stack[stack] = next(
                    (
                        name
                        for name, pattern in stages
                        if any(pattern.search(item) for item in chain)
                    ),
                    "",
                )
            stage = stage_of_stack[stack]
            if stage:
                stage_time[stage] += seconds
                stage_self[stage][leaf] += seconds

    def top(counter: Counter[str], whole: float) -> list[str]:
        """``"41.2% 3.104s name"`` for the largest entries of ``counter``."""
        return [
            f"{100 * seconds / whole if whole else 0:.1f}% {seconds:.3f}s {name}"
            for name, seconds in counter.most_common(arguments.top)
        ]

    print(
        json.dumps(
            {
                "total_cpu_seconds": round(total, 3),
                "self": top(self_time, total),
                "stages": {
                    name: {
                        "cpu_seconds": round(stage_time[name], 3),
                        "self": top(stage_self[name], stage_time[name]),
                    }
                    for name, _ in stages
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
