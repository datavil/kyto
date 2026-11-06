"""AGENTS.md commands are what CI runs, and a failing command fails the run (task 1.6)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agents_commands.py"

# The script runs commands with bash; CI runs it on Linux only.
needs_bash = pytest.mark.skipif(
    sys.platform == "win32", reason="the commands run with bash (Linux CI job)"
)


def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _agents_file(tmp_path: Path, block: str) -> Path:
    path = tmp_path / "AGENTS.md"
    path.write_text(
        "# AGENTS.md\n\n## Commands\n\nIntro.\n\n### Run by CI\n\n"
        f"```bash\n{block}```\n\n### On purpose only\n\n```bash\nfalse\n```\n\n"
        "## Next section\n",
        encoding="utf-8",
    )
    return path


def test_lists_the_real_agents_md_block() -> None:
    result = _run_script("--list")
    assert result.returncode == 0, result.stderr
    commands = result.stdout.splitlines()
    assert commands[0].startswith("uv sync")
    assert any(command.startswith("uv run pytest ") for command in commands)
    assert any(
        command.startswith("uv run pre-commit run --all-files") for command in commands
    )
    # The on-purpose-only command is not run by CI.
    assert not any("make_reference" in command for command in commands)


def test_ci_runs_the_script_with_a_clean_tree_check() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/agents_commands.py --check-clean" in workflow


@needs_bash
def test_a_failing_command_fails_the_run(tmp_path: Path) -> None:
    marker = tmp_path / "after"
    path = _agents_file(tmp_path, f"true  # passes\nexit 3\ntouch '{marker}'\n")
    result = _run_script("--file", str(path))
    assert result.returncode == 3
    assert "FAILED (exit 3" in result.stderr
    assert not marker.exists(), "commands after a failure must not run"


@needs_bash
def test_a_failing_pipeline_stage_fails_the_run(tmp_path: Path) -> None:
    path = _agents_file(tmp_path, "false | cat\n")
    assert _run_script("--file", str(path)).returncode != 0


@needs_bash
def test_passing_commands_succeed(tmp_path: Path) -> None:
    path = _agents_file(tmp_path, "true\n\ntrue && true  # comment\n")
    result = _run_script("--file", str(path))
    assert result.returncode == 0, result.stderr
    assert "[2/2]" in result.stdout


def test_missing_or_ambiguous_block_is_an_error(tmp_path: Path) -> None:
    empty = _agents_file(tmp_path, "")
    assert _run_script("--file", str(empty), "--list").returncode == 2
    two = tmp_path / "two.md"
    two.write_text(
        "## Commands\n\n### Run by CI\n\n```bash\ntrue\n```\n\n```bash\ntrue\n```\n",
        encoding="utf-8",
    )
    result = _run_script("--file", str(two), "--list")
    assert result.returncode == 2
    assert "exactly one" in result.stderr


def test_continuation_lines_and_tilde_fences_are_errors(tmp_path: Path) -> None:
    continued = _agents_file(tmp_path, "echo a \\\n  && false\n")
    result = _run_script("--file", str(continued), "--list")
    assert result.returncode == 2
    assert "continuation" in result.stderr
    tilde = tmp_path / "tilde.md"
    tilde.write_text(
        "## Commands\n\n### Run by CI\n\n```bash\ntrue\n```\n\n~~~bash\nfalse\n~~~\n",
        encoding="utf-8",
    )
    assert _run_script("--file", str(tilde), "--list").returncode == 2


def test_headings_inside_code_blocks_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "fenced.md"
    path.write_text(
        "```bash\n## Commands\n### Run by CI\n```\n\n"
        "## Commands\n\n### Run by CI\n\n```bash\necho real\n```\n",
        encoding="utf-8",
    )
    result = _run_script("--file", str(path), "--list")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["echo real"]
