"""Turn the raw results of ``measure_12_1.sh`` into performance.md's 12.1 tables.

Usage: ``report_12_1.py RAW``, where ``RAW`` is the folder ``measure_12_1.sh`` wrote
(``benchmarks/results/12.1`` keeps a copy). Prints Markdown: the time table, the peak
memory table and the two rankings (share of kyto's workflow time, gap to scanpy).

Cells read ``kyto / scanpy (ratio)``; the ratio is kyto's time over scanpy's, so
above 1 kyto is slower. At 50k cells kyto's steps come from ``kyto_steps.py`` (plan
prefixes, median of 3); at 1M from ``run_kyto_timed.py`` (one run, the store's
provenance timings), where QC, the filters, ``normalize_total`` and ``log1p`` run
fused inside the HVG pass and the sink's stream and have no time of their own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

THREADS = ("1", "4", "10")


def json_lines(path: Path) -> list[dict[str, Any]]:
    """Every JSON line of a file (none if it is missing)."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]


def number(value: float | None, digits: int = 2) -> str:
    """A time or size for a table cell; an em-free dash for a missing one."""
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def cell(kyto: float | None, scanpy: float | None) -> str:
    """``kyto / scanpy (ratio)``, or whichever side exists."""
    if kyto is None and scanpy is None:
        return "-"
    if scanpy is None:
        return f"{number(kyto)} / -"
    if kyto is None:
        return f"- / {number(scanpy)}"
    ratio = kyto / scanpy if scanpy else float("inf")
    mark = "**" if ratio > 1.2 else ""
    return f"{number(kyto)} / {number(scanpy)} ({mark}{ratio:.2f}x{mark})"


def load_50k(raw: Path) -> dict[str, Any]:
    """The 50k results by thread count: kyto (cpu exact, approximate, metal), scanpy."""
    suite = json.loads((raw / "suite-50k" / "results.json").read_text())
    threads = suite["workflow"]["50k"]["threads"]
    approximate = {
        str(r["threads"]): r for r in json_lines(raw / "kyto-50k-approximate.jsonl")
    }
    metal = {str(r["threads"]): r for r in json_lines(raw / "kyto-50k-metal.jsonl")}
    h5ad = {str(r["threads"]): r for r in json_lines(raw / "kyto-50k-sink-h5ad.jsonl")}
    extra: dict[tuple[str, str], dict[str, Any]] = {}
    for name in ("read", "rank", "metacells"):
        for record in json_lines(raw / f"extra-50k-{name}.jsonl"):
            extra[(record["task"], str(record["threads"]))] = record
    return {
        "suite": suite,
        "threads": threads,
        "approximate": approximate,
        "metal": metal,
        "h5ad": h5ad,
        "extra": extra,
    }


def load_1m(raw: Path) -> dict[str, Any]:
    """The 1M results: kyto runs by search and threads, extra steps, scanpy's attempt."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for name in ("approximate", "exact", "metal"):
        for record in json_lines(raw / f"kyto-1m-{name}.jsonl"):
            record = record.get("stdout", record)  # the metal run went through guard.py
            if "run" not in record:
                continue
            steps: dict[str, float] = {}
            for node, seconds in record["run"]["timings"].items():
                steps[record["operations"].get(node, "sink")] = seconds
            record["steps"] = steps
            runs[(name, str(record["threads"]))] = record
    guarded = json_lines(raw / "kyto-1m-metal.jsonl")
    extra: dict[tuple[str, str], dict[str, Any]] = {}
    for name in ("read", "rank", "metacells"):
        for record in json_lines(raw / f"extra-1m-{name}.jsonl"):
            extra[(record["task"], str(record["threads"]))] = record
    scanpy_path = raw / "scanpy-1m.json"
    scanpy = json.loads(scanpy_path.read_text()) if scanpy_path.exists() else None
    return {"runs": runs, "metal_guard": guarded, "extra": extra, "scanpy": scanpy}


def report(raw: Path) -> str:
    """The Markdown for performance.md."""
    small = load_50k(raw)
    large = load_1m(raw)
    lines: list[str] = []
    header = (
        "| Step | "
        + " | ".join(
            [f"50k, {t} thr" for t in THREADS] + [f"1M, {t} thr" for t in THREADS]
        )
        + " |"
    )
    rule = "|---" * (1 + 2 * len(THREADS)) + "|"

    def kyto50(t: str, step: str) -> float | None:
        return small["threads"][t]["kyto"]["seconds"].get(step)

    def scanpy50(t: str, step: str) -> float | None:
        return small["threads"][t]["scanpy"]["seconds"].get(step)

    def large_step(name: str, t: str, step: str) -> float | None:
        run = large["runs"].get((name, t))
        return None if run is None else run["steps"].get(step)

    def extra50(task: str, t: str) -> float | None:
        record = small["extra"].get((task, t))
        return None if record is None else record["seconds"]

    def extra1m(task: str, t: str) -> float | None:
        record = large["extra"].get((task, t))
        return None if record is None else record["seconds"]

    scanpy_large = large["scanpy"]
    scanpy_note = "-"
    if scanpy_large is not None and scanpy_large.get("status") != "ok":
        scanpy_note = "scanpy: did not finish"

    rows: list[tuple[str, list[str]]] = []
    convert = extra50("kyto_convert", "0")
    read = extra50("scanpy_read", "0")
    convert_1m = extra1m("kyto_convert", "0")
    rows.append(
        (
            "read (scanpy) / convert (kyto, once per dataset)",
            [cell(convert, read)] * 3 + [cell(convert_1m, None)] * 3,
        )
    )
    for step, label, large_label in (
        ("qc_filter", "QC + filters", None),
        ("normalize_log1p", "normalize + log1p", None),
        (
            "hvg",
            "HVG + keep HVGs (1M: the HVG pass, QC and normalize fused in)",
            "highly_variable_genes",
        ),
        ("pca", "PCA", "pca"),
        ("neighbors", "neighbours, kyto exact", None),
        ("leiden", "Leiden", "leiden"),
        ("umap", "UMAP", "umap"),
    ):
        cells = [cell(kyto50(t, step), scanpy50(t, step)) for t in THREADS]
        if large_label is None and step == "neighbors":
            cells += [cell(large_step("exact", t, "neighbors"), None) for t in THREADS]
        elif large_label is None:
            cells += ["(fused)"] * 3
        else:
            cells += [
                cell(large_step("approximate", t, large_label), None) for t in THREADS
            ]
        rows.append((label, cells))
        if step == "neighbors":
            rows.append(
                (
                    "neighbours, kyto approximate (HNSW)",
                    [
                        cell(
                            small["approximate"][t]["seconds"]["neighbors"],
                            scanpy50(t, step),
                        )
                        if t in small["approximate"]
                        else "-"
                        for t in THREADS
                    ]
                    + [
                        cell(large_step("approximate", t, "neighbors"), None)
                        for t in THREADS
                    ],
                )
            )
    rows.append(
        (
            "write: kyto `sink_zarr` (1M: stream + write) / scanpy `write_h5ad`",
            [cell(kyto50(t, "write"), scanpy50(t, "write")) for t in THREADS]
            + [cell(large_step("approximate", t, "sink"), None) for t in THREADS],
        )
    )
    rows.append(
        (
            "write: kyto `sink_h5ad` / scanpy `write_h5ad`",
            [
                cell(small["h5ad"][t]["seconds"]["write"], scanpy50(t, "write"))
                if t in small["h5ad"]
                else "-"
                for t in THREADS
            ]
            + ["-"] * 3,
        )
    )
    for method in ("t-test", "wilcoxon"):
        rows.append(
            (
                f"rank_genes_groups ({method})",
                [
                    cell(
                        extra50(f"kyto_rank:{method}", t),
                        extra50(f"scanpy_rank:{method}", t),
                    )
                    for t in THREADS
                ]
                + [cell(extra1m(f"kyto_rank:{method}", t), None) for t in THREADS],
            )
        )
    rows.append(
        (
            "metacells (kyto only, opt-in)",
            [cell(extra50("kyto_metacells", t), None) for t in THREADS]
            + [cell(extra1m("kyto_metacells", t), None) for t in THREADS],
        )
    )
    rows.append(
        (
            "**total**, exact (scanpy read through write; kyto the plan to `sink_zarr`)",
            [cell(kyto50(t, "total"), scanpy50(t, "total")) for t in THREADS]
            + [
                cell(large["runs"][("exact", t)]["seconds"], None)
                if ("exact", t) in large["runs"]
                else "-"
                for t in THREADS
            ],
        )
    )
    rows.append(
        (
            "**total**, kyto approximate",
            [
                cell(
                    small["approximate"][t]["cumulative"]["umap"]
                    + (kyto50(t, "write") or 0.0),
                    scanpy50(t, "total"),
                )
                if t in small["approximate"]
                else "-"
                for t in THREADS
            ]
            + [
                cell(large["runs"][("approximate", t)]["seconds"], None)
                if ("approximate", t) in large["runs"]
                else "-"
                for t in THREADS
            ],
        )
    )
    lines += [
        "Seconds, `kyto / scanpy (kyto/scanpy)`; **bold** where kyto takes more than 1.2x.",
        "",
    ]
    lines += [header, rule]
    lines += [f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows]
    if scanpy_note != "-":
        lines += ["", f"1M, {scanpy_note}: {json.dumps(scanpy_large)}"]

    # Metal rows (device="auto"), 50k.
    lines += [
        "",
        'Metal (`on(device="auto")`: QC, normalize, log1p, PCA and exact neighbours on the GPU), 50k, seconds `kyto Metal / kyto CPU`:',
        "",
    ]
    lines += ["| Step | 1 thr | 4 thr | 10 thr |", "|---|---|---|---|"]
    for step in (
        "qc_filter",
        "normalize_log1p",
        "hvg",
        "pca",
        "neighbors",
        "leiden",
        "umap",
        "write",
    ):
        cells: list[str] = []
        for t in THREADS:
            metal = small["metal"].get(t, {}).get("seconds", {}).get(step)
            cells.append(cell(metal, kyto50(t, step)).replace(" / ", " / cpu ", 1))
        lines.append(f"| {step} | " + " | ".join(cells) + " |")

    # Peak memory.
    lines += ["", "Peak RSS (MB) of the run through each step, `kyto / scanpy`:", ""]
    lines += [
        "| Step | 50k, 1 thr | 50k, 4 thr | 50k, 10 thr | 1M approx, 1 thr | 1M approx, 4 thr | 1M approx, 10 thr |",
        "|---|---|---|---|---|---|---|",
    ]
    for step in (
        "qc_filter",
        "normalize_log1p",
        "hvg",
        "pca",
        "neighbors",
        "leiden",
        "umap",
        "write",
    ):
        cells: list[str] = []
        for t in THREADS:
            kyto = small["threads"][t]["kyto"]["peak_mb"].get(step)
            scanpy = small["threads"][t]["scanpy"]["peak_mb"].get(step)
            cells.append(f"{number(kyto, 0)} / {number(scanpy, 0)}")
        if step == "write":
            for t in THREADS:
                run = large["runs"].get(("approximate", t))
                cells.append(f"{number(run['peak_mb'], 0)} / -" if run else "-")
        else:
            cells += [""] * 3
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(Path(sys.argv[1])))
