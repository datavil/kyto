"""Release rules (task 10.6): the version check, the store read policy, the workflow.

The read policy (docs/release.md, "Reading older stores"): before 1.0 kyto reads
exactly the format versions it lists and refuses every other one, older or newer, with
KY-E115 before reading anything.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import kyto as ky
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_version = _load_script("release_version")
wheel_smoke = _load_script("wheel_smoke")


def _supported_format_versions() -> list[str]:
    """The versions in `SUPPORTED_FORMAT_VERSIONS`, read from the Rust source."""
    source = (ROOT / "crates/kyto-store/src/store_version.rs").read_text("utf-8")
    written = re.search(r'pub const FORMAT_VERSION: &str = "([^"]+)";', source)
    listed = re.search(
        r"pub const SUPPORTED_FORMAT_VERSIONS: &\[&str\] = &\[(.*)\];", source
    )
    assert written is not None and listed is not None
    return [
        written.group(1)
        if item.strip() == "FORMAT_VERSION"
        else item.strip().strip('"')
        for item in listed.group(1).split(",")
        if item.strip()
    ]


# --- version check -----------------------------------------------------------------

CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n## [0.3.1] - 2027-01-05\n\n- Fixed.\n"
LICENCES = ["Cargo.toml", "LICENSE-APACHE", "LICENSE-MIT"]


def test_release_version_reads_the_workspace_version() -> None:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert release_version.workspace_version(text) == ky.__version__


def test_release_version_accepts_a_dated_section_and_matching_tag() -> None:
    assert release_version.problems("0.3.1", CHANGELOG, None, LICENCES) == []
    assert release_version.problems("0.3.1", CHANGELOG, "v0.3.1", LICENCES) == []


@pytest.mark.parametrize("tag", ["0.3.1", "v0.3.2", "v0.3.1-rc1", "release-0.3.1"])
def test_release_version_refuses_a_tag_that_is_not_v_version(tag: str) -> None:
    found = release_version.problems("0.3.1", CHANGELOG, tag, LICENCES)
    assert len(found) == 1
    assert f"tag '{tag}' does not match" in found[0]
    assert "'v0.3.1'" in found[0]


@pytest.mark.parametrize(
    "version", ["0.3", "01.3.1", "0.3.1-rc.1", "0.3.1+build", "v0.3.1"]
)
def test_release_version_refuses_a_non_semver_version(version: str) -> None:
    changelog = f"## [{version}] - 2027-01-05\n"
    found = release_version.problems(version, changelog, None, LICENCES)
    expected = (
        f"version '{version}' is not MAJOR.MINOR.PATCH (no leading zeros, "
        "no pre-release or build suffix)"
    )
    assert found == [expected]


@pytest.mark.parametrize(
    "changelog",
    [
        "## [Unreleased]\n",
        "## [0.3.1]\n",
        "## [0.3.1] - soon\n",
        "## [0.3.10] - 2027-01-05\n",
        "### [0.3.1] - 2027-01-05\n",
    ],
)
def test_release_version_needs_a_dated_changelog_section(changelog: str) -> None:
    found = release_version.problems("0.3.1", changelog, "v0.3.1", LICENCES)
    assert len(found) == 1
    assert "CHANGELOG.md has no section '## [0.3.1] - YYYY-MM-DD'" in found[0]


def test_release_version_reads_version_past_arrays_and_comments() -> None:
    text = (
        '[workspace]\nmembers = ["crates/*"]\nversion = "9.9.9"\n\n'
        '[workspace.package]  # shared\nauthors = ["a [b]"]\nversion = "1.2.3"\n'
        '[profile.dev]\nversion = "0.0.0"\n'
    )
    assert release_version.workspace_version(text) == "1.2.3"
    with pytest.raises(ValueError, match="no \\[workspace.package\\] version"):
        release_version.workspace_version('[workspace]\nversion = "1.0.0"\n')


def test_release_version_refuses_an_impossible_date() -> None:
    found = release_version.problems(
        "0.3.1", "## [0.3.1] - 2027-13-40\n", None, LICENCES
    )
    assert found == ["CHANGELOG.md dates 0.3.1 '2027-13-40', no date"]


def test_release_version_needs_the_licence_texts() -> None:
    found = release_version.problems("0.3.1", CHANGELOG, None, ["LICENSE-MIT"])
    assert len(found) == 1
    assert found[0].startswith("no LICENSE-APACHE at the repo root")


def test_release_version_main_prints_problems_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert release_version.main(["--tag", "v999.0.0"]) == 1
    assert "tag 'v999.0.0' does not match" in capsys.readouterr().err


# --- store read policy -------------------------------------------------------------


@pytest.fixture(scope="module")
def written_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real store written by this kyto (from the smoke test's synthetic matrix)."""
    directory = tmp_path_factory.mktemp("release_store")
    wheel_smoke._write_synthetic(directory / "matrix")
    store = directory / "current.kyto"
    ky.convert(directory / "matrix", store)
    return store


def _store_at_version(source: Path, target: Path, version: str) -> Path:
    """Copy ``source`` and set the copy's root ``format_version`` to ``version``."""
    shutil.copytree(source, target)
    root = target / "zarr.json"
    metadata = json.loads(root.read_text(encoding="utf-8"))
    metadata["attributes"]["kyto"]["format_version"] = version
    root.write_text(json.dumps(metadata), encoding="utf-8")
    return target


def test_this_kyto_writes_the_newest_version_it_reads(written_store: Path) -> None:
    metadata = json.loads((written_store / "zarr.json").read_text(encoding="utf-8"))
    written = metadata["attributes"]["kyto"]["format_version"]
    assert written == _supported_format_versions()[-1]
    assert ky.scan(written_store).count_cells() == wheel_smoke.N_CELLS


# "0.0" is older than any version kyto has written, "0.2" newer; "1.0" and "1.1" are
# what a 1.x kyto writes. Before 1.0 every one of them is refused.
@pytest.mark.parametrize("version", ["0.0", "0.2", "1.0", "1.1", "0.01", "1"])
def test_unread_format_versions_are_refused_with_ky_e115(
    written_store: Path, tmp_path: Path, version: str
) -> None:
    if version in _supported_format_versions():
        pytest.skip(f"{version} is read by this kyto")
    store = _store_at_version(written_store, tmp_path / "other.kyto", version)
    supported = ", ".join(f"'{v}'" for v in _supported_format_versions())
    with pytest.raises(ky.FormatVersionError) as caught:
        ky.scan(store)
    assert caught.value.code == "KY-E115"
    assert isinstance(caught.value, ValueError)
    message = str(caught.value)
    assert f"has format_version '{version}'" in message
    assert f"this kyto reads only {supported}" in message
    # Refusing is read-only: the store is left as it was (docs/release.md).
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert root["attributes"]["kyto"]["format_version"] == version


def test_release_md_reads_table_matches_the_store_reader() -> None:
    text = (ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (\d+\.\d+)\.x \| ([\d., ]+) \| ([\d., ]+) \|$", text, re.MULTILINE
    )
    assert rows, "docs/release.md has no 'kyto | Writes | Reads' rows"
    series, writes, reads = rows[-1]
    supported = _supported_format_versions()
    assert ky.__version__.startswith(f"{series}.")
    assert writes.strip() == supported[-1]
    assert [v.strip() for v in reads.split(",")] == supported


def test_changelog_names_the_format_versions_of_the_next_release() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    top = text.split("\n## [", 2)[1]
    supported = _supported_format_versions()
    reads = ", ".join(f'"{v}"' for v in supported)
    assert f'writes `format_version` "{supported[-1]}", reads {reads}' in top


# --- release workflow --------------------------------------------------------------


def test_release_workflow_publishes_every_ci_wheel() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    wheels_job = ci.split("\n  wheels:\n", 1)[1].split("\n  gpu:\n", 1)[0]
    targets = re.findall(r"- \{ os: [\w.-]+, target: \w+", wheels_job)
    assert len(targets) == 6
    assert f'cp310-abi3-*.whl | wc -l)" -eq {len(targets)}' in release
    # CI is called by the release (tags) and runs by hand, never on its own on a tag.
    assert "  workflow_call:\n" in ci and "tags:" not in ci.split("\njobs:", 1)[0]
    assert "uses: ./.github/workflows/ci.yml" in release


def test_release_workflow_publishes_to_pypi_only_from_a_tag() -> None:
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    publish = release.split("\n  publish:\n", 1)[1]
    assert "id-token: write" in publish
    assert "password:" not in publish and "secrets." not in release
    pypi, testpypi = publish.split("- name: publish to PyPI\n", 1)[1].split(
        "- name: publish to TestPyPI\n"
    )
    assert "if: github.event_name == 'push'" in pypi
    assert "repository-url" not in pypi
    assert "if: github.event_name == 'workflow_dispatch'" in testpypi
    assert "repository-url: https://test.pypi.org/legacy/" in testpypi
    assert 'tags: ["v*"]' in release.split("\njobs:", 1)[0]


def test_pyproject_declares_the_runtime_dependencies() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("\n[project]\n", 1)[1].split("\n[", 1)[0]
    dependencies = re.search(
        r"^dependencies = \[(.*?)^\]", project, re.MULTILINE | re.DOTALL
    )
    assert dependencies is not None
    names = re.findall(r'"([a-z]+)>=', dependencies.group(1))
    # What crates/kyto-py imports at run time (polars, numpy, scipy.sparse).
    assert names == ["numpy", "polars", "scipy"]
    bindings = "".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "crates/kyto-py/src").rglob("*.rs")
    )
    for name in names:
        assert re.search(rf'import\("{name}[".]', bindings), name
