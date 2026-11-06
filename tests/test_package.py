"""Smoke tests for the installed package: import and version."""

import re
from pathlib import Path

import kyto

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_workspace_cargo_toml() -> None:
    # A regex, not tomllib: tomllib needs Python 3.11 and kyto supports 3.10.
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', text, re.M | re.S
    )
    assert match is not None
    assert kyto.__version__ == match.group(1)


def test_version_is_pep440_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", kyto.__version__)


def test_extension_is_compiled() -> None:
    from kyto import _kyto

    assert _kyto.__file__ is not None
    assert Path(_kyto.__file__).suffix in {".so", ".pyd"}
