"""Run the Rust tests with cargo-nextest when it is installed, else cargo test (1.7).

Does: with `cargo nextest` on PATH, run `cargo nextest run ARGS` (every test in its
own process, binaries in parallel), then `cargo test --doc ARGS`, because nextest
runs no doctests. Without it, run `cargo test ARGS`, which runs both. Either way the
same tests run; stops at the first failing command and returns its exit status.
`.config/nextest.toml` runs the binaries that measure their own memory alone.

Does not: install cargo-nextest (`cargo install cargo-nextest --locked` does), or
retry failing tests.

Usage (from the repo root; `uv run` so PyO3 sees the .venv python)::

    uv run python scripts/rust_tests.py --workspace --locked
    uv run python scripts/rust_tests.py --no-nextest --workspace --locked
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def has_nextest() -> bool:
    """Return whether `cargo nextest` runs here."""
    try:
        result = subprocess.run(
            ["cargo", "nextest", "--version"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def commands(arguments: list[str], nextest: bool) -> list[list[str]]:
    """Return the cargo commands that run every Rust test, in order.

    Parameters
    ----------
    arguments : list[str]
        Arguments for every command (package selection, `--locked`).
    nextest : bool
        Run the tests through cargo-nextest, then the doctests through cargo test.

    Returns
    -------
    list[list[str]]
        One argument vector per command. Returns a new list.
    """
    if nextest:
        return [
            ["cargo", "nextest", "run", *arguments],
            ["cargo", "test", "--doc", *arguments],
        ]
    return [["cargo", "test", *arguments]]


def main(argv: list[str] | None = None) -> int:
    """Run the Rust tests; return the first non-zero exit status, else 0."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    nextest = has_nextest()
    if "--no-nextest" in arguments:
        arguments.remove("--no-nextest")
        nextest = False
    elif not nextest:
        print(
            "cargo-nextest not found (cargo install cargo-nextest --locked):"
            " running cargo test",
            flush=True,
        )
    for command in commands(arguments, nextest):
        print("$ " + " ".join(command), flush=True)
        try:
            status = subprocess.call(command, cwd=ROOT)
        except OSError as error:
            print(f"FAILED: cannot run {command[0]}: {error}", file=sys.stderr)
            return 1
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
