"""Run a command under a memory and time watchdog; print what it reached (task 12.1).

Usage: ``guard.py [--rss-mb 14500] [--swap-mb 3000] [--timeout 3600] -- COMMAND...``.

For runs that may not fit in this machine's memory (scanpy at 1M cells on 16 GB). macOS
does not stop a process at a memory limit: it compresses and swaps, and the run slows
to a crawl instead of failing. So the watchdog samples the command's resident memory
(``ps -o rss``, the command and its children) and the system's swap in use
(``sysctl vm.swapusage``) twice a second, and kills the command when its RSS passes
``--rss-mb``, when swap grows by more than ``--swap-mb`` over its value at the start, or
after ``--timeout`` seconds. That is recorded as the result, never hidden.

Prints one JSON object: ``status`` (``"ok"``, ``"failed"`` with the exit code, or
``"killed"`` with the reason), ``seconds``, ``peak_rss_mb`` (sampled, so it can miss a
peak shorter than half a second), ``peak_swap_growth_mb``, the command's last
``stderr`` lines (``run_scanpy.py`` prints each finished step there) and its last
``stdout`` line parsed as JSON when it is one.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


def swap_used_mb() -> float:
    """The system's swap in use, in MB (``sysctl vm.swapusage``; 0 if unreadable).

    Returns
    -------
    float
        Swap used, MB.
    """
    completed = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    )
    match = re.search(r"used = ([0-9.]+)M", completed.stdout)
    return float(match.group(1)) if match else 0.0


def tree_rss_mb(pid: int) -> float:
    """Resident memory of ``pid`` and its descendants, in MB (``ps``; KB on macOS).

    Parameters
    ----------
    pid : int
        The root process.

    Returns
    -------
    float
        Sum of the tree's RSS, MB.
    """
    completed = subprocess.run(
        ["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True, check=False
    )
    parent: dict[int, int] = {}
    rss: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            parent[int(fields[0])] = int(fields[1])
            rss[int(fields[0])] = int(fields[2])
    total = 0
    for candidate, kilobytes in rss.items():
        node = candidate
        while node not in (0, 1) and node != pid:
            node = parent.get(node, 0)
        if node == pid:
            total += kilobytes
    return total / 1e3


def main() -> None:
    """Run the command under the watchdog and print the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rss-mb", type=float, default=14500.0)
    parser.add_argument("--swap-mb", type=float, default=3000.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else None
    if not command:
        parser.error("give the command after --")

    swap_start = swap_used_mb()
    peak_rss = 0.0
    peak_swap = 0.0
    reason: str | None = None
    start = time.perf_counter()
    with (
        tempfile.TemporaryFile("w+") as stdout,
        tempfile.TemporaryFile("w+") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        while process.poll() is None:
            peak_rss = max(peak_rss, tree_rss_mb(process.pid))
            peak_swap = max(peak_swap, swap_used_mb() - swap_start)
            elapsed = time.perf_counter() - start
            if peak_rss > arguments.rss_mb:
                reason = f"RSS passed {arguments.rss_mb:.0f} MB"
            elif peak_swap > arguments.swap_mb:
                reason = f"swap grew by more than {arguments.swap_mb:.0f} MB"
            elif elapsed > arguments.timeout:
                reason = f"no result after {arguments.timeout:.0f} s"
            if reason is not None:
                subprocess.run(["pkill", "-9", "-P", str(process.pid)], check=False)
                process.kill()
                process.wait()
                break
            time.sleep(0.5)
        seconds = time.perf_counter() - start
        stdout.seek(0)
        stderr.seek(0)
        out_lines = stdout.read().strip().splitlines()
        err_lines = stderr.read().strip().splitlines()
    result: dict[str, Any] = {
        "status": "killed"
        if reason
        else ("ok" if process.returncode == 0 else "failed"),
        "reason": reason,
        "exit_code": process.returncode,
        "seconds": round(seconds, 1),
        "peak_rss_mb": round(peak_rss),
        "peak_swap_growth_mb": round(peak_swap),
        "stderr_tail": err_lines[-15:],
    }
    if out_lines and out_lines[-1].startswith("{"):
        result["stdout"] = json.loads(out_lines[-1])
    print(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
