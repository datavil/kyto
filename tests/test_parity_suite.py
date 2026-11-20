"""Task 10.2: the parity suite and its one tolerance table.

Done when: every 6.x task links its parity test. The tests that compare kyto with
scanpy and anndata on the 1.5 fixtures stay next to the code they check (Rust
``crates/*/tests/it/*_parity.rs``, Python ``tests/test_*.py``); this module checks what
holds them together:

- the table ``tests/parity/tolerances.toml`` is well formed: every entry has a value,
  a kind, its task, the tests that apply it and a one-line reason;
- the Rust loader's line scan (``tests/parity/tolerances.rs``) reads the same values
  as a TOML parser;
- every ``tolerance("...")`` call names an entry, and the entry lists that file;
- no parity test keeps a tolerance of its own (a numeric ``const`` in a Rust parity
  test);
- every 6.x task, done or open, has a ``Parity:`` line whose links resolve, and the
  done ones link at least one test, including every test that applies a tolerance of
  that task.
"""

import re
from pathlib import Path
from typing import cast

import pytest
import tomllib
from parity import TABLE, entries, tolerance

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"relative", "absolute", "floor", "margin", "ulp"}
KEYS = {"value", "kind", "task", "tests", "reason"}
PLANS = [ROOT / "Implementation Plan.md", ROOT / "Implementation Plan - Done.md"]
CALL = re.compile(r'tolerance\("([^"]+)"\)')
LINK = re.compile(r"\]\(([^)]+)\)")
TASK = re.compile(r"^- \[(.)\] \*\*(6\.\d+) ")


def _test_files() -> list[Path]:
    """Every Rust integration test and Python test file in the repo (but this one)."""
    rust = sorted(ROOT.glob("crates/*/tests/it/*.rs"))
    python = sorted(
        path
        for path in (ROOT / "tests").glob("test_*.py")
        if path.name != Path(__file__).name
    )
    return [*rust, *python]


def _parity_lines() -> dict[str, tuple[bool, str]]:
    """Each 6.x task's ``Parity:`` line (text after the label), with whether it is done.

    A task without one maps to an empty line.
    """
    found: dict[str, tuple[bool, str]] = {}
    for plan in PLANS:
        task: str | None = None
        for line in plan.read_text().splitlines():
            match = TASK.match(line)
            if match is not None:
                done, task = match.group(1) == "x", str(match.group(2))
                found[task] = (done, "")
            elif line.startswith(("- ", "#")):
                task = None
            elif task is not None and line.startswith("  - Parity: "):
                found[task] = (found[task][0], line.removeprefix("  - Parity: "))
                task = None
    return found


def _tests_of(entry: dict[str, object]) -> list[str]:
    """An entry's ``tests``, checked to be a list of strings."""
    tests = entry["tests"]
    assert isinstance(tests, list)
    return [str(test) for test in cast("list[object]", tests)]


def test_entries_are_complete() -> None:
    """Every entry has exactly the documented keys, with sensible values."""
    assert entries(), TABLE
    for name, entry in entries().items():
        assert set(entry) == KEYS, (name, sorted(entry))
        value = entry["value"]
        assert isinstance(value, float), (name, value)
        assert value >= 0.0, (name, value)
        assert entry["kind"] in KINDS, (name, entry["kind"])
        task = entry["task"]
        assert isinstance(task, str), (name, task)
        assert re.fullmatch(r"\d+\.\d+", task), (name, task)
        reason = entry["reason"]
        assert isinstance(reason, str), (name, reason)
        assert reason.strip(), name
        assert "\n" not in reason, name
        assert "—" not in reason, name  # no em dashes (AGENTS.md)
        assert all(
            isinstance(test, str) for test in cast("list[object]", entry["tests"])
        )
        tests = _tests_of(entry)
        assert tests, name
        for test in tests:
            assert (ROOT / test).is_file(), (name, test)


def test_rust_line_scan_reads_the_toml_values() -> None:
    """The Rust loader's scan (a ``[section]`` header, then ``value = <float>``) agrees
    with ``tomllib`` on every entry, and finds one value per section."""
    scanned: dict[str, float] = {}
    section: str | None = None
    for line in TABLE.read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif line.replace(" ", "").startswith("value="):
            assert line.startswith("value = "), line
            assert section is not None, line
            assert section not in scanned, section
            text = line.removeprefix("value = ")
            # The plain form Rust's `str::parse::<f64>` reads: no `_`, no comment.
            assert re.fullmatch(r"[0-9][0-9.]*(e-?[0-9]+)?", text), (section, text)
            scanned[section] = float(text)
    expected = {name: tolerance(name) for name in entries()}
    assert scanned == expected
    with TABLE.open("rb") as file:
        assert tomllib.load(file)  # a real TOML document


def test_every_call_names_an_entry_that_lists_its_file() -> None:
    """``tolerance("x")`` in a test file: ``x`` exists and names that file; every entry
    is used by each file it lists."""
    used: dict[str, set[str]] = {}
    for path in _test_files():
        relative = path.relative_to(ROOT).as_posix()
        for name in CALL.findall(path.read_text()):
            assert name in entries(), (relative, name)
            used.setdefault(name, set()).add(relative)
    for name, entry in entries().items():
        listed = set(_tests_of(entry))
        assert used.get(name, set()) == listed, (name, sorted(used.get(name, set())))


def test_no_tolerance_constants_in_parity_tests() -> None:
    """Rust parity tests take their tolerances from the table, not from a ``const``."""
    constant = re.compile(r"^const \w+: f(32|64) = ", re.MULTILINE)
    for path in sorted(ROOT.glob("crates/*/tests/it/*_parity.rs")):
        assert constant.search(path.read_text()) is None, path


def _task_number(task: str) -> int:
    """The number after ``6.``, so 6.10 sorts after 6.9."""
    return int(task.removeprefix("6."))


@pytest.mark.parametrize("task", sorted(_parity_lines(), key=_task_number))
def test_every_algorithm_task_links_its_parity_test(task: str) -> None:
    """6.x tasks: a ``Parity:`` line with resolving links; a done task links a test,
    and every test applying one of its tolerances."""
    done, line = _parity_lines()[task]
    assert line, f"6.x task {task} has no '  - Parity:' line"
    links = LINK.findall(line)
    for link in links:
        assert (ROOT / link).exists(), (task, link)
    if done:
        tests = [link for link in links if link != "tests/parity/tolerances.toml"]
        assert tests, (task, line)
        for name, entry in entries().items():
            if entry["task"] == task:
                for test in _tests_of(entry):
                    assert test in links, (task, name, test)
