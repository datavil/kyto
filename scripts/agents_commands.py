"""Run the commands listed in AGENTS.md, so CI and the docs never drift (task 1.6).

Does: read the one ```bash block under the `### Run by CI` heading of the
`## Commands` section, run each line in order with bash (pipefail), stop at the first
failure and exit non-zero. With `--check-clean` it also fails if the commands left
the git working tree dirty (a formatter or `--fix` changed a tracked file, or a
command wrote a file that is neither tracked nor ignored). With `--only` it runs just
the commands that contain one of the given texts, for a partial rerun under the rules
in AGENTS.md ("Rerunning part of the gate"); a full run still comes last before every
commit.

Does not: install anything itself, or run the "On purpose only" commands.

Usage (from the repo root)::

    uv run --no-project python scripts/agents_commands.py --list
    uv run --no-project python scripts/agents_commands.py --check-clean
    uv run --no-project python scripts/agents_commands.py --only "uv sync" "cargo build" stubs.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = "## Commands"
SUBSECTION = "### Run by CI"
# A trailing comment starts at a `#` with whitespace on both sides.
_COMMENT = re.compile(r"\s#(\s|$)")


def extract_commands(text: str) -> list[str]:
    """Return the command lines of the CI block in an AGENTS.md text.

    Parameters
    ----------
    text : str
        The full Markdown text of AGENTS.md.

    Returns
    -------
    list[str]
        One entry per non-blank line of the block, in order, comments kept (bash
        ignores them). Returns a new list.

    Raises
    ------
    ValueError
        If the section, the subsection or exactly one ```bash block in it is
        missing, the block is empty, uses ~~~ fences or continues a line with a
        backslash.
    """
    lines = text.splitlines()
    section = _between_headings(lines, SECTION, level=2)
    subsection = _between_headings(section, SUBSECTION, level=3)
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in subsection:
        stripped = line.strip()
        if current is None and stripped.startswith("~~~"):
            raise ValueError(f"{SUBSECTION}: use ``` fences, not ~~~")
        if current is None and stripped.startswith("```"):
            if stripped != "```bash":
                raise ValueError(f"{SUBSECTION}: code fences must be ```bash")
            current = []
        elif current is not None and stripped == "```":
            blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        raise ValueError(f"{SUBSECTION}: unclosed ```bash block")
    if len(blocks) != 1:
        raise ValueError(
            f"{SUBSECTION}: expected exactly one ```bash block, found {len(blocks)}"
        )
    commands = [line.strip() for line in blocks[0] if line.strip()]
    for command in commands:
        if command.endswith("\\"):
            raise ValueError(
                f"{SUBSECTION}: one command per line, no '\\' continuation: {command}"
            )
    if not commands:
        raise ValueError(f"{SUBSECTION}: the ```bash block is empty")
    return commands


def _between_headings(lines: list[str], heading: str, level: int) -> list[str]:
    """Return the lines after `heading` up to the next heading of `level` or above."""
    start: int | None = None
    in_fence = False
    for index, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and line.rstrip() == heading:
            start = index
            break
    if start is None:
        raise ValueError(f"no '{heading}' heading outside code blocks")
    stop = re.compile(rf"^#{{1,{level}}} ")
    body: list[str] = []
    in_fence = False
    for line in lines[start + 1 :]:
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence and stop.match(line):
            break
        body.append(line)
    return body


def select_commands(commands: list[str], steps: list[str]) -> list[str]:
    """Return the commands that contain any of `steps`, in their listed order.

    Parameters
    ----------
    commands : list[str]
        The commands, as `extract_commands` returns them.
    steps : list[str]
        Texts to look for in each command, its trailing `# comment` left out
        (`"cargo build"`, `"stubs.py"`).

    Returns
    -------
    list[str]
        The matching commands, each once. Returns a new list.

    Raises
    ------
    ValueError
        If a step is blank or matches no command.
    """
    # Match the command itself, not its comment ("# fmt, clippy, ruff, pyright").
    bodies = [_COMMENT.split(command, maxsplit=1)[0] for command in commands]
    for step in steps:
        if not step.strip():
            raise ValueError("--only: a step must not be blank")
        if not any(step in body for body in bodies):
            raise ValueError(
                f"--only: no command contains {step!r} (see --list for the commands)"
            )
    return [
        command
        for command, body in zip(commands, bodies, strict=True)
        if any(step in body for step in steps)
    ]


def run(commands: list[str], cwd: Path) -> int:
    """Run each command with bash in `cwd`; return the first non-zero exit status."""
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] $ {command}", flush=True)
        started = time.monotonic()
        status = subprocess.call(["bash", "-o", "pipefail", "-c", command], cwd=cwd)
        elapsed = time.monotonic() - started
        if status != 0:
            print(
                f"\nFAILED (exit {status}, {elapsed:.0f}s): {command}\n"
                "A command listed in AGENTS.md (Commands, Run by CI) failed.",
                file=sys.stderr,
            )
            return status
        print(f"ok ({elapsed:.0f}s)", flush=True)
    return 0


def check_clean(cwd: Path) -> int:
    """Return 1 and print the changes if the git working tree is dirty, else 0."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"\nFAILED: cannot check the working tree, git status: {result.stderr}",
            file=sys.stderr,
        )
        return 1
    status = result.stdout
    if not status.strip():
        return 0
    print(
        "\nFAILED: the AGENTS.md commands changed the working tree. Run them locally"
        " and commit the result (unformatted code, a stale lock file, or a generated"
        " file missing from .gitignore):\n" + status,
        file=sys.stderr,
    )
    subprocess.call(["git", "--no-pager", "diff", "--stat"], cwd=cwd)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, then list or run the commands; return the exit status."""
    parser = argparse.ArgumentParser(
        description="Run the commands listed in AGENTS.md (Commands, Run by CI)."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "AGENTS.md",
        help="Markdown file to read (default: AGENTS.md at the repo root)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the commands, do not run them"
    )
    parser.add_argument(
        "--check-clean",
        action="store_true",
        help="also fail if the commands leave the git working tree dirty",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="TEXT",
        help="run only the commands containing one of these texts (a partial rerun;"
        " the full list still runs last before a commit)",
    )
    arguments = parser.parse_args(argv)
    path: Path = arguments.file
    try:
        commands = extract_commands(path.read_text(encoding="utf-8"))
        selected = (
            commands
            if arguments.only is None
            else select_commands(commands, arguments.only)
        )
    except ValueError as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 2
    if arguments.list:
        print("\n".join(selected))
        return 0
    status = run(selected, cwd=ROOT)
    if status == 0 and arguments.check_clean:
        status = check_clean(cwd=ROOT)
    if status == 0 and len(selected) < len(commands):
        print(
            f"\nPartial run: {len(selected)} of {len(commands)} commands passed."
            " Run the full list before the commit (AGENTS.md, Commands).",
            flush=True,
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
