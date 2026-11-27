"""Check the release version before wheels are published (task 10.6).

Does: read the one version source (``[workspace.package] version`` in ``Cargo.toml``),
check that it is a semantic version ``MAJOR.MINOR.PATCH`` (no pre-release or build
suffix: it is also the PyPI version), that ``CHANGELOG.md`` has a dated section for it,
that the licence texts the wheels must carry exist (``LICENSE-MIT``,
``LICENSE-APACHE``: ``pyproject.toml`` says ``MIT OR Apache-2.0``), and, given
``--tag``, that the tag is exactly ``v<version>``. Prints the version.

Does not: change any file, create a tag or publish anything.

Usage (from the repo root; ``.github/workflows/release.yml`` runs it first)::

    python scripts/release_version.py
    python scripts/release_version.py --tag v0.1.0
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
LICENCE_FILES = ("LICENSE-APACHE", "LICENSE-MIT")


def workspace_version(cargo_toml: str) -> str:
    """Return ``[workspace.package] version`` from a ``Cargo.toml`` text.

    A regex, not ``tomllib``: ``tomllib`` needs Python 3.11 and kyto supports 3.10.

    Parameters
    ----------
    cargo_toml : str
        The text of the workspace ``Cargo.toml``.

    Returns
    -------
    str
        The version string.

    Raises
    ------
    ValueError
        If the section or its ``version`` key is missing.
    """
    in_section = False
    for line in cargo_toml.splitlines():
        if line.startswith("["):
            in_section = line.split("#", 1)[0].strip() == "[workspace.package]"
        elif in_section:
            match = re.match(r'version\s*=\s*"([^"]+)"', line)
            if match is not None:
                return match.group(1)
    raise ValueError("Cargo.toml has no [workspace.package] version")


def problems(
    version: str, changelog: str, tag: str | None, licence_files: list[str]
) -> list[str]:
    """List what stops ``version`` from being released; empty if nothing does.

    Parameters
    ----------
    version : str
        The workspace version.
    changelog : str
        The text of ``CHANGELOG.md``.
    tag : str or None
        The pushed git tag (``GITHUB_REF_NAME``), or ``None`` for a run without one.
    licence_files : list of str
        The names of the files at the repo root.

    Returns
    -------
    list of str
        One sentence per problem.
    """
    found: list[str] = []
    if SEMVER.fullmatch(version) is None:
        found.append(
            f"version '{version}' is not MAJOR.MINOR.PATCH (no leading zeros, "
            "no pre-release or build suffix)"
        )
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", re.MULTILINE
    )
    dated = heading.search(changelog)
    if dated is not None:
        try:
            datetime.date.fromisoformat(dated.group(1))
        except ValueError:
            found.append(f"CHANGELOG.md dates {version} '{dated.group(1)}', no date")
    else:
        found.append(
            f"CHANGELOG.md has no section '## [{version}] - YYYY-MM-DD'; move the "
            "Unreleased entries under it"
        )
    if tag is not None and tag != f"v{version}":
        found.append(
            f"tag '{tag}' does not match the version: tag the commit 'v{version}', "
            "or change [workspace.package] version in Cargo.toml"
        )
    missing = [name for name in LICENCE_FILES if name not in licence_files]
    if missing:
        found.append(
            f"no {' or '.join(missing)} at the repo root: the wheels and the sdist "
            "must carry the licence texts (add them and `license-files` in "
            "pyproject.toml)"
        )
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the checks, print the version or the problems, return the exit status.

    Parameters
    ----------
    argv : list of str or None, default=None
        The arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 if the version can be released, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "kyto").splitlines()[0])
    parser.add_argument("--tag", help="the pushed tag, for example v0.1.0")
    arguments = parser.parse_args(argv)
    version = workspace_version((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    root_files = sorted(path.name for path in ROOT.iterdir())
    found = problems(version, changelog, arguments.tag, root_files)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
