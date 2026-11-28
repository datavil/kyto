"""AGENTS.md commands are what CI runs, and a failing command fails the run (1.6, 1.7)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
    # 1.7: one sync, the extension rebuilt through cargo's incremental build.
    assert commands[0].startswith("uv sync --reinstall-package kyto ")
    assert sum(command.startswith("uv sync") for command in commands) == 1
    assert any("scripts/rust_tests.py --workspace" in command for command in commands)
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


@needs_bash
def test_only_runs_the_matching_commands_in_order(tmp_path: Path) -> None:
    alpha, beta, gamma = (tmp_path / name for name in ("alpha", "beta", "gamma"))
    path = _agents_file(
        tmp_path,
        f"touch '{alpha}'  # gamma\ntouch '{beta}'\ntouch '{gamma}'  # alpha\n",
    )
    result = _run_script("--file", str(path), "--only", "gamma", "alpha")
    assert result.returncode == 0, result.stderr
    assert alpha.exists() and gamma.exists() and not beta.exists()
    assert "[1/2]" in result.stdout and "[2/2]" in result.stdout
    assert "Partial run: 2 of 3 commands passed" in result.stdout
    listed = _run_script("--file", str(path), "--list", "--only", "beta")
    assert listed.stdout.splitlines() == [f"touch '{beta}'"]
    # A text only in a comment selects nothing.
    path = _agents_file(tmp_path, "true  # gamma\n")
    assert _run_script("--file", str(path), "--only", "gamma").returncode == 2


def test_only_with_an_unknown_step_is_an_error(tmp_path: Path) -> None:
    path = _agents_file(tmp_path, "echo one\n")
    result = _run_script("--file", str(path), "--only", "one", "nothing-like-it")
    assert result.returncode == 2
    assert "no command contains 'nothing-like-it'" in result.stderr
    assert _run_script("--file", str(path), "--only", " ").returncode == 2


@needs_bash
def test_a_full_run_does_not_say_partial(tmp_path: Path) -> None:
    path = _agents_file(tmp_path, "echo one\n")
    result = _run_script("--file", str(path), "--only", "one")
    assert result.returncode == 0, result.stderr
    assert "Partial run" not in result.stdout


def test_only_ignores_the_comments() -> None:
    result = _run_script("--list", "--only", "pyright")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        line
        for line in _run_script("--list").stdout.splitlines()
        if line.startswith("uv run pyright ")
    ]
    rebase = _run_script(
        "--list",
        "--only",
        "uv sync",
        "cargo build",
        "error_registry.py",
        "stubs.py",
        "llms_txt.py",
    )
    assert len(rebase.stdout.splitlines()) == 5, rebase.stdout


def _rust_tests() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rust_tests", ROOT / "scripts" / "rust_tests.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nextest_runs_the_doctests_through_cargo_test() -> None:
    commands = _rust_tests().commands
    arguments = ["--workspace", "--locked"]
    assert commands(arguments, True) == [
        ["cargo", "nextest", "run", "--workspace", "--locked"],
        ["cargo", "test", "--doc", "--workspace", "--locked"],
    ]
    assert commands(arguments, False) == [["cargo", "test", "--workspace", "--locked"]]


def test_the_gpu_features_are_linted_while_their_tests_build() -> None:
    commands = _run_script("--list").stdout.splitlines()
    gpu = [command for command in commands if "-p kyto-gpu" in command]
    assert len(gpu) == 1, gpu
    assert "RUSTC_WORKSPACE_WRAPPER=clippy-driver" in gpu[0]
    assert "CLIPPY_ARGS=-D__CLIPPY_HACKERY__warnings__CLIPPY_HACKERY__" in gpu[0]
    assert "--all-targets --all-features" in gpu[0]
    assert "--doc --all-features" in gpu[0]


def test_every_own_process_test_binary_runs_alone_under_nextest() -> None:
    # A `tests/<name>.rs` outside `tests/it/` exists because it needs the process to
    # itself (AGENTS.md); nextest must not run other tests beside it (.config/nextest.toml).
    config = (ROOT / ".config" / "nextest.toml").read_text(encoding="utf-8")
    binaries = sorted(path.stem for path in ROOT.glob("crates/*/tests/*.rs"))
    assert binaries, "no own-process test binaries found"
    missing = [name for name in binaries if f"binary({name})" not in config]
    assert not missing, f"add to the filter in .config/nextest.toml: {missing}"
    assert 'threads-required = "num-test-threads"' in config
