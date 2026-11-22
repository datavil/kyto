"""kyto's benchmark suite against scanpy and rapids-singlecell (task 10.3).

Usage (from the repo root, see README.md)::

    uv run python benchmarks/suite/suite.py                  # pbmc3k and 10k, compare
    uv run python benchmarks/suite/suite.py --sizes 50k,100k --threads 1,4,10
    uv run python benchmarks/suite/suite.py --smoke          # the gate's check (seconds)
    uv run python benchmarks/suite/suite.py --update-baseline

What it does: for each dataset size, prepares the data once (``.h5ad`` and a converted
kyto store, cached under ``--data-dir``), then measures the basic workflow per step,
time and peak RSS, with task 10.8's scripts (``benchmarks/scanpy_workflow``), each run
in fresh processes:

- kyto: ``kyto_steps.py`` (each plan prefix streamed to nowhere, median of the repeats,
  steps as differences; ``write`` is the full ``sink_zarr`` minus the UMAP prefix).
- scanpy: ``run_scanpy.py`` warm (warmed up in-process on the whole dataset, median of
  the repeats) for time,
  plus one cold run (empty ``NUMBA_CACHE_DIR``) for peak RSS, as 10.8 reports it.
- rapids-singlecell: ``run_rapids.py``, only where it imports and finds a CUDA device;
  otherwise skipped with a notice in the table.
- the criterion kernel benches (``cargo bench -p kyto-exec --bench kernels``), unless
  ``--no-kernels``.

It writes ``results.json`` and ``results.md`` to ``--output``, prints the table, and
compares kyto's numbers with the baseline (``baseline-<platform>.json`` here) when the
baseline comes from the same platform, CPU and core count. Exit status 1 when a kyto
number regressed past the thresholds of ``benchmark_report.THRESHOLDS``, 2 on a usage
error, 3 when a benchmark command fails, 0 otherwise (also when no comparable baseline
exists: the table says so).

What it does not: optimise anything, judge scanpy's numbers, or run in the default gate
beyond ``--smoke`` (a 500-cell kyto-only run that checks the harness works).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark_report import comparable, compare, kyto_numbers, markdown, missing

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
WORKFLOW = ROOT / "benchmarks" / "scanpy_workflow"

# Name -> synthetic cell count (make_data.py, 20,000 genes), or None for pbmc3k.
SIZES: dict[str, int | None] = {
    "smoke": 500,
    "pbmc3k": None,
    "10k": 10_000,
    "50k": 50_000,
    "100k": 100_000,
}
DEFAULT_SIZES = "pbmc3k,10k"
TOOLS = ("kyto", "scanpy", "rapids")


def machine() -> dict[str, Any]:
    """Describe this machine and checkout, for the table and the baseline check.

    Returns
    -------
    dict[str, Any]
        ``platform``, ``cpu``, ``cpu_count``, ``commit``, ``load_average``, ``python``.
    """
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = _output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    commit = _output(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    if _output(["git", "status", "--porcelain", "--untracked-files=no"]):
        commit += "+dirty"
    return {
        "platform": f"{sys.platform}-{platform.machine()}",
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "commit": commit,
        "load_average": [round(value, 1) for value in os.getloadavg()],
        "python": platform.python_version(),
    }


def _output(command: list[str]) -> str:
    """Return a command's stripped stdout, or ``""`` if it fails."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    """Return the JSON objects a script printed, one per line starting with ``{``."""
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def _run(command: list[str], environment: dict[str, str] | None = None) -> str:
    """Run a command (stderr shown live) and return its stdout; fail loudly."""
    print("$", " ".join(command), file=sys.stderr, flush=True)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        print(
            f"benchmark command failed (exit {completed.returncode}): "
            f"{' '.join(command)}\n{completed.stdout[-2000:]}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return completed.stdout


def prepare(size: str, data_dir: Path) -> dict[str, Any]:
    """Write the size's ``.h5ad`` and kyto store once; return their paths and shape.

    Parameters
    ----------
    size : str
        A key of ``SIZES``.
    data_dir : Path
        Cache folder; the files go to ``data_dir / size``. Delete it to rebuild.

    Returns
    -------
    dict[str, Any]
        ``h5ad``, ``store`` (paths), ``shape`` (``n_cells``, ``n_genes``, ``nnz``)
        and ``convert_seconds`` (``None`` when the store was cached).
    """
    folder = data_dir / size
    folder.mkdir(parents=True, exist_ok=True)
    h5ad = folder / "counts.h5ad"
    store = folder / "store.kyto"
    shape_file = folder / "shape.json"
    if not shape_file.exists():
        h5ad.unlink(missing_ok=True)
        shutil.rmtree(store, ignore_errors=True)
        n_cells = SIZES[size]
        if n_cells is None:
            _write_pbmc3k(h5ad)
        else:
            script = str(WORKFLOW / "make_data.py")
            _run([sys.executable, script, str(n_cells), str(h5ad)])
        shape_file.write_text(json.dumps(_shape(h5ad)))
    convert_seconds: float | None = None
    if not store.exists():
        import kyto as ky

        # Convert next to the store and rename, so an interrupted convert never
        # leaves a half-written store that later runs would take as cached.
        partial = folder / "store.kyto.partial"
        shutil.rmtree(partial, ignore_errors=True)
        start = time.perf_counter()
        ky.convert(str(h5ad), str(partial), layer="counts")
        convert_seconds = round(time.perf_counter() - start, 3)
        partial.rename(store)
    return {
        "h5ad": h5ad,
        "store": store,
        "shape": json.loads(shape_file.read_text()),
        "convert_seconds": convert_seconds,
    }


def _write_pbmc3k(path: Path) -> None:
    """Write the pbmc3k fixture's counts (tests/fixtures) as ``.h5ad``."""
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures.interop import to_anndata
    from fixtures.pbmc import load_pbmc3k

    to_anndata(load_pbmc3k()).write_h5ad(path)


def _shape(h5ad: Path) -> dict[str, int]:
    """Cells, genes and nonzeros of an ``.h5ad`` (read backed, the matrix not loaded)."""
    command = [
        sys.executable,
        "-c",
        (
            "import json, sys, anndata; a = anndata.read_h5ad(sys.argv[1], backed='r'); "
            "print(json.dumps({'n_cells': a.n_obs, 'n_genes': a.n_vars, "
            "'nnz': int(a.file['X']['data'].shape[0])}))"
        ),
        str(h5ad),
    ]
    return _json_lines(_run(command))[-1]


def run_kyto(
    data: dict[str, Any], work: Path, threads: list[int], repeats: int
) -> dict[str, dict[str, Any]]:
    """Measure kyto per step with ``kyto_steps.py``, one result per thread count.

    Parameters
    ----------
    data : dict[str, Any]
        From :func:`prepare`.
    work : Path
        Scratch folder for the written store.
    threads : list[int]
        Thread counts.
    repeats : int
        Runs per prefix (median).

    Returns
    -------
    dict[str, dict[str, Any]]
        Thread count (as text) to ``{"seconds": ..., "peak_mb": ...}``, with
        ``seconds["total"]`` the full plan to ``sink_zarr``.
    """
    command = [
        sys.executable,
        str(WORKFLOW / "kyto_steps.py"),
        str(data["store"]),
        str(work / "kyto_out.kyto"),
        "--threads",
        ",".join(str(count) for count in threads),
        "--repeats",
        str(repeats),
        "--with-write",
    ]
    runs: dict[str, dict[str, Any]] = {}
    for report in _json_lines(_run(command)):
        seconds = dict(report["seconds"])
        seconds["total"] = round(report["cumulative"]["write"], 3)
        peak_mb = dict(report["peak_mb"])
        peak_mb["total"] = max(peak_mb.values())
        runs[str(report["threads"])] = {"seconds": seconds, "peak_mb": peak_mb}
    return runs


def run_scanpy(
    data: dict[str, Any], work: Path, threads: int, repeats: int, data_dir: Path
) -> dict[str, Any]:
    """Measure scanpy per step with ``run_scanpy.py``: warm time, cold peak RSS.

    Parameters
    ----------
    data : dict[str, Any]
        From :func:`prepare`.
    work : Path
        Scratch folder for the written ``.h5ad``.
    threads : int
        Thread cap (``--threads``).
    repeats : int
        Warm runs (median per step).
    data_dir : Path
        Where the cold run's empty numba cache goes (removed afterwards).

    Returns
    -------
    dict[str, Any]
        ``seconds`` (warm medians), ``peak_mb`` (the cold run's), ``cold_seconds``,
        ``n_cells_kept``, ``n_clusters``.
    """
    script = str(WORKFLOW / "run_scanpy.py")
    output = str(work / "scanpy_out.h5ad")
    # Warm up on the whole dataset: a 6,000-cell warm-up (10.8's) left numba code of
    # pynndescent uncompiled, and 10k cells' neighbours then took 46.8 s against 0.28 s
    # (task 10.3, memory/workflow_benchmark.md).
    warmup = data["shape"]["n_cells"]
    warm_command = [sys.executable, script, str(data["h5ad"]), output]
    warm_command += ["--threads", str(threads), "--warmup", str(warmup)]
    warm = [_json_lines(_run(warm_command))[-1] for _ in range(repeats)]
    cache = Path(tempfile.mkdtemp(prefix="numba-cold-", dir=data_dir))
    try:
        environment = {**os.environ, "NUMBA_CACHE_DIR": str(cache)}
        cold_command = [sys.executable, script, str(data["h5ad"]), output]
        cold_command += ["--threads", str(threads)]
        cold = _json_lines(_run(cold_command, environment))[-1]
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    seconds = {
        step: round(statistics.median(run["seconds"][step] for run in warm), 3)
        for step in warm[0]["seconds"]
    }
    peak_mb = dict(cold["peak_mb"])
    peak_mb["total"] = max(peak_mb.values())
    return {
        "seconds": seconds,
        "peak_mb": peak_mb,
        "cold_seconds": cold["seconds"],
        "n_cells_kept": warm[0]["n_cells_kept"],
        "n_clusters": warm[0]["n_clusters"],
    }


def rapids_unavailable() -> str | None:
    """Return why rapids-singlecell cannot run here, or ``None`` if it can."""
    probe = "import rapids_singlecell, cupy; print(cupy.cuda.runtime.getDeviceCount())"
    completed = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        last = (completed.stderr.strip().splitlines() or ["no output"])[-1]
        return f"needs rapids-singlecell with CUDA ({last})"
    if completed.stdout.strip() in ("", "0"):
        return "rapids-singlecell imports but finds no CUDA device"
    return None


def run_rapids(
    data: dict[str, Any], work: Path, threads: int, repeats: int
) -> dict[str, Any]:
    """Measure rapids-singlecell per step with ``run_rapids.py`` (median of warm runs).

    Parameters
    ----------
    data : dict[str, Any]
        From :func:`prepare`.
    work : Path
        Scratch folder for the written ``.h5ad``.
    threads : int
        Host thread cap.
    repeats : int
        Runs (median per step; peak RSS is the largest).

    Returns
    -------
    dict[str, Any]
        ``seconds``, ``peak_mb`` (host RSS) and ``peak_gpu_mb``.
    """
    command = [
        sys.executable,
        str(WORKFLOW / "run_rapids.py"),
        str(data["h5ad"]),
        str(work / "rapids_out.h5ad"),
        "--threads",
        str(threads),
        "--warmup",
        str(data["shape"]["n_cells"]),
    ]
    runs = [_json_lines(_run(command))[-1] for _ in range(repeats)]
    seconds = {
        step: round(statistics.median(run["seconds"][step] for run in runs), 3)
        for step in runs[0]["seconds"]
    }
    peak_mb = {
        step: max(run["peak_mb"][step] for run in runs) for step in runs[0]["peak_mb"]
    }
    peak_mb["total"] = max(peak_mb.values())
    return {
        "seconds": seconds,
        "peak_mb": peak_mb,
        "peak_gpu_mb": max(run["peak_gpu_mb"] for run in runs),
    }


def run_kernels() -> dict[str, float]:
    """Run the criterion kernel benches; return each bench's median, in nanoseconds.

    Returns
    -------
    dict[str, float]
        Criterion's ``full_id`` to the median time per iteration, for the benches this
        run wrote (older results under ``target/criterion`` are ignored).
    """
    started = time.time()
    _run(["cargo", "bench", "-p", "kyto-exec", "--bench", "kernels", "--locked"])
    metadata = _output(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    target = Path(json.loads(metadata)["target_directory"])
    kernels: dict[str, float] = {}
    for description in sorted((target / "criterion").rglob("new/benchmark.json")):
        estimates = description.parent / "estimates.json"
        # Two seconds of slack for file systems with coarse modification times; the
        # bench itself runs for minutes, so no earlier run's result is that recent.
        if estimates.stat().st_mtime < started - 2:
            continue
        full_id = json.loads(description.read_text())["full_id"]
        median = json.loads(estimates.read_text())["median"]["point_estimate"]
        kernels[full_id] = round(float(median), 1)
    if not kernels:
        print("cargo bench ran, but wrote no criterion estimates", file=sys.stderr)
        raise SystemExit(3)
    return kernels


def main() -> int:
    """Run the suite; write and print the table; return the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tools", default=",".join(TOOLS))
    parser.add_argument("--no-kernels", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".cache" / "benchmarks")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--provisional",
        default=None,
        metavar="NOTE",
        help="with --update-baseline: mark the baseline provisional, saying why",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="500 cells, kyto only, 1 thread, 1 repeat, no kernels, no baseline",
    )
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.sizes, arguments.threads, arguments.repeats = "smoke", "1", 1
        arguments.tools, arguments.no_kernels = "kyto", True
    sizes = arguments.sizes.split(",")
    tools = arguments.tools.split(",")
    for name, accepted in ((sizes, SIZES), (tools, TOOLS)):
        unknown = [value for value in name if value not in accepted]
        if unknown:
            parser.error(f"unknown {unknown}; accepted: {', '.join(accepted)}")
    try:
        threads = [int(count) for count in arguments.threads.split(",")]
    except ValueError:
        parser.error(f"--threads must be a comma list of integers: {arguments.threads}")
    if arguments.update_baseline and (
        arguments.smoke
        or "kyto" not in tools
        or arguments.no_kernels
        or not set(DEFAULT_SIZES.split(",")) <= set(sizes)
    ):
        parser.error(
            "--update-baseline needs a full run: kyto, the kernels and at least the "
            f"sizes {DEFAULT_SIZES} (a partial baseline would stop checking the rest)"
        )
    baseline_path: Path = arguments.baseline or SUITE / (
        f"baseline-{sys.platform}-{platform.machine()}.json"
    )
    baseline: dict[str, Any] | None = None
    if not arguments.smoke and not arguments.update_baseline:
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text())
            except json.JSONDecodeError as error:
                parser.error(f"baseline {baseline_path} is not valid JSON: {error}")
        elif arguments.baseline is not None:
            parser.error(f"no baseline at {baseline_path}")
    if arguments.repeats < 1 or any(count < 1 for count in threads):
        parser.error("--repeats and every --threads count must be at least 1")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    output: Path = arguments.output or ROOT / "target" / "benchmarks" / (
        "smoke" if arguments.smoke else f"run-{stamp}"
    )
    output.mkdir(parents=True, exist_ok=True)
    data_dir: Path = arguments.data_dir.resolve()

    result: dict[str, Any] = {
        "created": stamp,
        "machine": machine(),
        "settings": {
            "sizes": sizes,
            "threads": threads,
            "repeats": arguments.repeats,
            "tools": tools,
        },
        "workflow": {},
        "kernels": {},
        "notices": [],
    }
    if arguments.provisional is not None:
        result["provisional"] = arguments.provisional
    notices: list[str] = result["notices"]
    rapids_reason = rapids_unavailable() if "rapids" in tools else None
    if rapids_reason is not None:
        notices.append(f"rapids-singlecell: skipped, {rapids_reason}.")
    for size in sizes:
        data = prepare(size, data_dir)
        work = data_dir / size / "work"
        work.mkdir(exist_ok=True)
        size_result: dict[str, Any] = {
            "shape": data["shape"],
            "convert_seconds": data["convert_seconds"],
            "threads": {str(count): {} for count in threads},
        }
        if "kyto" in tools:
            for count, run in run_kyto(data, work, threads, arguments.repeats).items():
                size_result["threads"][count]["kyto"] = run
        for count in threads:
            tools_result = size_result["threads"][str(count)]
            if "scanpy" in tools:
                tools_result["scanpy"] = run_scanpy(
                    data, work, count, arguments.repeats, data_dir
                )
            if "rapids" in tools and rapids_reason is None:
                tools_result["rapids"] = run_rapids(
                    data, work, count, arguments.repeats
                )
        result["workflow"][size] = size_result
    if not arguments.no_kernels:
        result["kernels"] = run_kernels()

    changes = None
    absent: list[str] = []
    if arguments.smoke:
        notices.append("Smoke run: the harness works; no baseline comparison.")
    elif arguments.update_baseline:
        notices.append(f"Wrote this run as the baseline {baseline_path.name}.")
    elif baseline is None:
        notices.append(
            f"No baseline {baseline_path.name}: nothing compared "
            "(record one with --update-baseline)."
        )
    else:
        reason = comparable(result, baseline)
        if reason is not None:
            notices.append(f"Baseline not compared: {reason}.")
        else:
            changes = compare(result, baseline)
            absent = missing(result, baseline)
            unpaired = len(kyto_numbers(result)) - len(changes)
            if unpaired:
                notices.append(
                    f"{unpaired} kyto numbers have no baseline counterpart (a size, "
                    "thread count or bench the baseline did not run): not compared."
                )
            if baseline.get("provisional"):
                notices.append(
                    f"The baseline is provisional: {baseline['provisional']}"
                )
    table = markdown(result, changes, absent)
    (output / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (output / "results.md").write_text(table + "\n")
    if arguments.update_baseline:
        baseline_path.write_text(json.dumps(result, indent=1) + "\n")
    print(table)
    print(f"Wrote {output / 'results.md'} and results.json.", file=sys.stderr)
    regressed = changes is not None and any(change.regressed for change in changes)
    return 1 if regressed or absent else 0


if __name__ == "__main__":
    sys.exit(main())
