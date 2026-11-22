"""The benchmark suite's tables and regression check (task 10.3).

What it does: turns a suite result (the JSON ``suite.py`` writes) into a Markdown
table, and compares kyto's numbers in it with a stored baseline result: a time or a
peak RSS that grew past both thresholds below is a regression.

What it does not: run anything, or judge scanpy's or rapids-singlecell's numbers (they
are the reference, not the code under test). Pure Python, no third-party imports, so
its unit tests (``tests/test_benchmark_suite.py``) run everywhere.

A result is a JSON object::

    {"machine": {"platform": ..., "cpu": ..., "cpu_count": ...},
     "workflow": {SIZE: {"threads": {"1": {"kyto": RUN, "scanpy": RUN | null,
                                            "rapids": RUN | null}}}},
     "kernels": {BENCH_ID: nanoseconds},
     "notices": [...]}

where a RUN is ``{"seconds": {STEP: s}, "peak_mb": {STEP: MB}}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The workflow's steps, in table order (10.8; `read` is scanpy's alone, `total` both).
STEPS = (
    "read",
    "qc_filter",
    "normalize_log1p",
    "hvg",
    "pca",
    "neighbors",
    "leiden",
    "umap",
    "write",
    "total",
)
STEP_LABELS = {
    "read": "read (scanpy)",
    "qc_filter": "QC + filters",
    "normalize_log1p": "normalize + log1p",
    "hvg": "HVG + keep HVGs",
    "pca": "PCA",
    "neighbors": "neighbours",
    "leiden": "Leiden",
    "umap": "UMAP",
    "write": "write",
    "total": "**total**",
}


@dataclass(frozen=True)
class Threshold:
    """When a kyto number counts as a regression: past both limits at once.

    Parameters
    ----------
    ratio : float
        New value over baseline value above which it may be a regression.
    absolute : float
        Least growth (new minus baseline, in the metric's unit) that counts, so a
        0.02 s step that doubles is noise, not a regression.
    """

    ratio: float
    absolute: float


# Stated in README.md and the plan entry; change them there too.
THRESHOLDS = {
    "seconds": Threshold(ratio=1.25, absolute=0.10),
    "peak_mb": Threshold(ratio=1.15, absolute=25.0),
    "kernel_ns": Threshold(ratio=1.25, absolute=0.0),
}


@dataclass(frozen=True)
class Change:
    """One kyto number next to its baseline.

    Parameters
    ----------
    key : str
        ``SIZE/THREADSt/METRIC/STEP`` for the workflow, ``kernels/BENCH_ID`` for a
        criterion bench.
    metric : str
        ``"seconds"``, ``"peak_mb"`` or ``"kernel_ns"``.
    baseline : float
        The baseline's value.
    value : float
        This run's value.
    """

    key: str
    metric: str
    baseline: float
    value: float

    @property
    def regressed(self) -> bool:
        """Whether the value grew past both limits of its metric's threshold."""
        threshold = THRESHOLDS[self.metric]
        growth = self.value - self.baseline
        if growth <= threshold.absolute:
            return False
        if self.baseline <= 0:
            # Not reached by kyto's times (cumulative, positive) or peaks; kept so a
            # zero baseline never divides.
            return True
        return self.value / self.baseline > threshold.ratio


def kyto_numbers(result: Mapping[str, Any]) -> dict[str, tuple[str, float]]:
    """Return kyto's numbers of a result, flattened.

    Parameters
    ----------
    result : Mapping[str, Any]
        A suite result.

    Returns
    -------
    dict[str, tuple[str, float]]
        Key (see :class:`Change`) to ``(metric, value)``. ``seconds`` are the time
        **through** each step (the sum of the steps up to it: the median of that plan
        prefix, as ``kyto_steps.py`` measured it), not the step alone: a step is a
        difference of two medians and can be noise around zero, or negative. The
        ``total`` is the same as ``write``'s, so it is left out. A new dict.
    """
    numbers: dict[str, tuple[str, float]] = {}
    for size, size_result in result.get("workflow", {}).items():
        for threads, tools in size_result["threads"].items():
            run = tools.get("kyto")
            if run is None:
                continue
            through = 0.0
            for step in STEPS:
                if step != "total" and step in run["seconds"]:
                    through += float(run["seconds"][step])
                    key = f"{size}/{threads}t/seconds/{step}"
                    numbers[key] = ("seconds", round(through, 3))
            for step, value in run["peak_mb"].items():
                key = f"{size}/{threads}t/peak_mb/{step}"
                numbers[key] = ("peak_mb", float(value))
    for bench, nanoseconds in result.get("kernels", {}).items():
        numbers[f"kernels/{bench}"] = ("kernel_ns", float(nanoseconds))
    return numbers


def comparable(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> str | None:
    """Return why the two results cannot be compared, or ``None`` if they can.

    Timings from another platform, CPU or core count say nothing about a regression.

    Parameters
    ----------
    result, baseline : Mapping[str, Any]
        Suite results.

    Returns
    -------
    str or None
        A one-line reason, or ``None``.
    """
    for field in ("platform", "cpu", "cpu_count"):
        mine = result["machine"].get(field)
        theirs = baseline["machine"].get(field)
        if mine != theirs:
            return f"the baseline's {field} is {theirs!r}, this machine's is {mine!r}"
    return None


def compare(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[Change]:
    """Pair every kyto number of ``result`` with the baseline's, where it has one.

    Parameters
    ----------
    result, baseline : Mapping[str, Any]
        Suite results from the same machine (see :func:`comparable`).

    Returns
    -------
    list[Change]
        One per key in both, in ``result``'s order. Numbers the baseline lacks (a
        size it did not run) are left out. A new list.
    """
    old = kyto_numbers(baseline)
    changes: list[Change] = []
    for key, (metric, value) in kyto_numbers(result).items():
        if key in old:
            changes.append(Change(key, metric, old[key][1], value))
    return changes


def missing(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Return the baseline's kyto numbers this run should have measured but lacks.

    Only sizes and thread counts this run measured with kyto count (and the kernels
    when it ran them): a run of other sizes is not missing anything, but a step or
    bench that vanished from a size it ran is (a broken harness must not pass).

    Parameters
    ----------
    result, baseline : Mapping[str, Any]
        Suite results from the same machine.

    Returns
    -------
    list[str]
        Keys, in the baseline's order. A new list.
    """
    mine = kyto_numbers(result)
    covered = {key.rsplit("/", 2)[0] for key in mine if not key.startswith("kernels/")}
    ran_kernels = bool(result.get("kernels"))
    absent: list[str] = []
    for key in kyto_numbers(baseline):
        if key in mine:
            continue
        if (
            key.startswith("kernels/")
            and ran_kernels
            or key.rsplit("/", 2)[0] in covered
        ):
            absent.append(key)
    return absent


def _seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _megabytes(value: float | None) -> str:
    return "" if value is None else f"{value:,.0f}"


def _ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    return f"{numerator / denominator:.2f}"


def _delta(change: Change | None) -> str:
    if change is None:
        return ""
    if change.baseline <= 0:
        text = f"{change.value - change.baseline:+.2f}"
    else:
        text = f"{(change.value / change.baseline - 1) * 100:+.0f}%"
    return f"**{text}**" if change.regressed else text


def _value(run: Mapping[str, Any] | None, metric: str, step: str) -> float | None:
    if run is None:
        return None
    value = run[metric].get(step)
    return None if value is None else float(value)


def markdown(
    result: Mapping[str, Any],
    changes: list[Change] | None,
    absent: list[str] | None = None,
) -> str:
    """Render a result (and its changes against the baseline) as Markdown.

    Parameters
    ----------
    result : Mapping[str, Any]
        A suite result.
    changes : list[Change] or None
        From :func:`compare`, or ``None`` when no baseline was compared.
    absent : list[str] or None, default=None
        From :func:`missing`.

    Returns
    -------
    str
        The tables, notices first, regressions last. A new string.
    """
    by_key = {change.key: change for change in changes or []}
    machine = result["machine"]
    lines = [
        "# kyto benchmark suite",
        "",
        (
            f"Machine: {machine.get('cpu')}, {machine.get('cpu_count')} cores, "
            f"{machine.get('platform')}; kyto `{machine.get('commit')}`; "
            f"load average at start {machine.get('load_average')}. "
            f"Repeats: {result.get('settings', {}).get('repeats')} (median)."
        ),
        "",
    ]
    lines += [f"> {notice}" for notice in result.get("notices", [])]
    if result.get("notices"):
        lines.append("")
    for size, size_result in result.get("workflow", {}).items():
        shape = size_result.get("shape", {})
        for threads, tools in size_result["threads"].items():
            kyto = tools.get("kyto")
            scanpy = tools.get("scanpy")
            rapids = tools.get("rapids")
            lines += [
                (
                    f"## {size}: {shape.get('n_cells', '?'):,} cells x "
                    f"{shape.get('n_genes', '?'):,} genes, {threads} thread(s)"
                ),
                "",
            ]
            header = (
                "| Step | scanpy s | kyto s | kyto/scanpy | kyto through step vs baseline "
                "| scanpy MB | kyto MB | kyto MB vs baseline |"
            )
            rule = "|---|---|---|---|---|---|---|---|"
            if rapids is not None:
                header += " rapids s | rapids MB |"
                rule += "---|---|"
            lines += [header, rule]
            for step in STEPS:
                values = {
                    (tool, metric): _value(run, metric, step)
                    for tool, run in (
                        ("kyto", kyto),
                        ("scanpy", scanpy),
                        ("rapids", rapids),
                    )
                    for metric in ("seconds", "peak_mb")
                }
                if all(value is None for value in values.values()):
                    continue
                kyto_seconds = values["kyto", "seconds"]
                scanpy_seconds = values["scanpy", "seconds"]
                cells = [
                    STEP_LABELS[step],
                    _seconds(scanpy_seconds),
                    _seconds(kyto_seconds),
                    _ratio(kyto_seconds, scanpy_seconds),
                    _delta(by_key.get(f"{size}/{threads}t/seconds/{step}")),
                    _megabytes(values["scanpy", "peak_mb"]),
                    _megabytes(values["kyto", "peak_mb"]),
                    _delta(by_key.get(f"{size}/{threads}t/peak_mb/{step}")),
                ]
                if rapids is not None:
                    cells += [
                        _seconds(values["rapids", "seconds"]),
                        _megabytes(values["rapids", "peak_mb"]),
                    ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    kernels: Mapping[str, float] = result.get("kernels", {})
    if kernels:
        lines += [
            "## Rust kernels (criterion, median per call)",
            "",
            "| Bench | µs | vs baseline |",
            "|---|---|---|",
        ]
        for bench, nanoseconds in kernels.items():
            change = by_key.get(f"kernels/{bench}")
            lines.append(f"| {bench} | {nanoseconds / 1e3:,.1f} | {_delta(change)} |")
        lines.append("")
    if changes is not None:
        regressions = [change for change in changes if change.regressed]
        lines.append(
            f"Compared {len(changes)} kyto numbers with the baseline; "
            f"{len(regressions)} regressed (time through a step: > "
            f"{THRESHOLDS['seconds'].ratio:.2f}x and > "
            f"{THRESHOLDS['seconds'].absolute} s; peak RSS: > "
            f"{THRESHOLDS['peak_mb'].ratio:.2f}x and > "
            f"{THRESHOLDS['peak_mb'].absolute:.0f} MB; kernels: > "
            f"{THRESHOLDS['kernel_ns'].ratio:.2f}x)."
        )
        lines += [
            f"- REGRESSION {change.key}: {change.baseline:.3f} -> {change.value:.3f}"
            for change in regressions
        ]
        lines += [
            f"- MISSING {key}: in the baseline, not in this run" for key in absent or []
        ]
        lines.append("")
    return "\n".join(lines)
