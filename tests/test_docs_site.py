"""The docs site's generated and hand-written pages against the real package (10.5).

The migration table (``docs/migration.md``) names kyto calls: every name must exist and
every call must bind to its signature, and the page's workflow must build a plan. The
design rules page must hold R1 to R12 as the plan and AGENTS.md list them, and the API
reference must show every public item of ``llms.txt``. ``mkdocs build --strict`` itself
runs in the local gate (AGENTS.md, Commands).
"""

from __future__ import annotations

import ast
import gzip
import importlib.util
import inspect
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import kyto
import kyto.qc
import pytest
from kyto import _kyto

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "docs" / "migration.md"


def _script(name: str) -> ModuleType:
    """A module of ``scripts/``, imported by path."""
    spec = importlib.util.spec_from_file_location(
        f"kyto_docs_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The migration table.

# What a bare name in a kyto cell stands for.
NAMES: dict[str, object] = {"ky": kyto, "qc": kyto.qc, "atlas": _kyto.Atlas}
# The public classes of the extension, by name: what a call may return.
CLASSES: dict[str, type] = {
    name: value
    for name, value in vars(_kyto).items()
    if isinstance(value, type) and not name.startswith("_")
}


def _kyto_cells() -> list[str]:
    """Every code span in the kyto column of the page's tables, in order."""
    snippets: list[str] = []
    for line in MIGRATION.read_text().splitlines():
        if not line.startswith("| ") or line.startswith(("| ---", "| AnnData")):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        assert len(cells) == 3, line
        snippets += re.findall(r"`([^`]+)`", cells[1])
    return snippets


CELLS = _kyto_cells()


def _returns() -> dict[str, set[str]]:
    """The type names each public item returns, from its stub signatures."""
    returns: dict[str, set[str]] = {}
    for section in _script("llms_txt").sections():
        for entry in section.entries:
            for signature in entry.signatures:
                if signature.startswith("class "):
                    continue
                if ") -> " in signature:
                    returned = signature.rsplit(") -> ", 1)[1]
                else:  # a property: `Class.name: Type`
                    returned = signature.split(": ", 1)[1]
                returns.setdefault(entry.name, set()).add(returned.split("[")[0])
    return returns


RETURNS = _returns()


def _prefix(owner: object) -> str:
    """How ``llms.txt`` qualifies a member of ``owner``."""
    if owner is kyto:
        return ""
    if owner is kyto.qc:
        return "qc."
    assert isinstance(owner, type)
    return f"{owner.__name__}."


def _returned(qualified: str) -> object | None:
    """The kyto class a member returns, or ``None`` for any other value."""
    names = RETURNS.get(qualified, set())
    classes = {CLASSES[name] for name in names if name in CLASSES}
    return classes.pop() if len(classes) == 1 and len(names) == 1 else None


def _member(owner: object | None, attribute: str, snippet: str) -> tuple[object, str]:
    """``owner.attribute`` and its qualified name; fails if it does not exist."""
    assert owner is not None, f"{snippet}: .{attribute} on a value kyto does not type"
    assert hasattr(owner, attribute), f"{snippet}: {owner!r} has no {attribute!r}"
    return getattr(owner, attribute), _prefix(owner) + attribute


def _type(node: ast.expr, snippet: str) -> object | None:
    """Check every name and call under ``node``; return the class (or module) of
    its value, as the stubs type it, or ``None`` for a value that is not kyto's."""
    if isinstance(node, ast.Call):
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            _type(argument, snippet)
        if isinstance(node.func, ast.Name):
            assert node.func.id in kyto.__all__, f"{snippet}: unknown {node.func.id}"
            function, qualified = getattr(kyto, node.func.id), node.func.id
            method = False
        else:
            assert isinstance(node.func, ast.Attribute), snippet
            owner = _type(node.func.value, snippet)
            function, qualified = _member(owner, node.func.attr, snippet)
            method = isinstance(owner, type)
        assert callable(function) and _binds(function, node, method), (
            f"{snippet}: {ast.unparse(node)} does not bind to {qualified}"
        )
        return _returned(qualified)
    if isinstance(node, ast.Attribute):
        _, qualified = _member(_type(node.value, snippet), node.attr, snippet)
        return _returned(qualified)
    if isinstance(node, ast.Name):
        assert node.id in NAMES, f"{snippet}: unknown name {node.id!r}"
        return NAMES[node.id]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _type(child, snippet)
    return None


def _binds(function: Callable[..., object], call: ast.Call, method: bool) -> bool:
    """Whether the call's arguments bind to the function's signature."""
    signature = inspect.signature(function)
    arguments: list[object] = [None] * (len(call.args) + (1 if method else 0))
    keywords: dict[str, object] = {
        keyword.arg: None for keyword in call.keywords if keyword.arg is not None
    }
    try:
        signature.bind(*arguments, **keywords)
    except TypeError:
        return False
    return True


def test_migration_table_has_the_common_calls() -> None:
    assert len(CELLS) >= 30
    text = " ".join(CELLS)
    for name in ("normalize_total", "pca", "neighbors", "leiden", "sink_h5ad"):
        assert f".{name}(" in text


@pytest.mark.parametrize("snippet", CELLS)
def test_migration_names_exist_and_calls_bind(snippet: str) -> None:
    _type(ast.parse(snippet, mode="eval").body, snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "atlas.cells.to_scipy()",
        "atlas.layer('x').sink_h5ad('o')",
        "ky.scan('x').count_cells().to_numpy()",
        "atlas.pca(into='pca', n_comps=50, seed=0)",
        "atlas.umapp(into='umap', seed=0)",
        "atlas.filter(cell_columnn('x') > 1)",
        "ky.scan()",
    ],
)
def test_migration_check_rejects_wrong_calls(snippet: str) -> None:
    with pytest.raises(AssertionError):
        _type(ast.parse(snippet, mode="eval").body, snippet)


def _store(folder: Path) -> Path:
    """A 3-cell, 3-gene store with a counts layer and gene symbols."""
    source = folder / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 3 6\n1 1 5\n3 1 1\n2 2 7\n1 2 2\n2 3 3\n3 3 4\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\nAAAT-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            b"ENSG01\tCD3E\tGene Expression\nENSG02\tMT-CO1\tGene Expression\n"
            b"ENSG03\tMS4A1\tGene Expression\n"
        )
    )
    store = folder / "pbmc3k.kyto"
    kyto.convert(source, store, layer="counts")
    return store


def test_migration_workflow_builds_a_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocks = re.findall(r"```python\n(.*?)```", MIGRATION.read_text(), flags=re.DOTALL)
    assert len(blocks) == 1
    store = _store(tmp_path)
    code = blocks[0].replace('"pbmc3k.kyto"', repr(str(store)))
    namespace: dict[str, Any] = {}
    exec(compile(code, str(MIGRATION), "exec"), namespace)  # noqa: S102
    plan = namespace["plan"]
    assert isinstance(plan, _kyto.Atlas)
    printed = capsys.readouterr().out
    for step in ("pca", "neighbors", "umap", "leiden"):
        assert step in printed.lower()


# The generated pages.


def test_design_rules_page_is_the_plans_list() -> None:
    hooks = _script("docs_hooks")
    page = hooks.design_rules((ROOT / "Implementation Plan.md").read_text())
    rules = [line for line in page.splitlines() if line.startswith("- **R")]
    assert [rule.split()[1] for rule in rules] == [
        f"**R{number}" for number in range(1, 13)
    ]
    # AGENTS.md copies the list from the plan; the two must not drift.
    agents = (ROOT / "AGENTS.md").read_text().splitlines()
    assert rules == [line for line in agents if re.match(r"- \*\*R\d+ ", line)]


def test_design_rules_page_needs_all_twelve() -> None:
    hooks = _script("docs_hooks")
    with pytest.raises(ValueError, match="R1 to R12"):
        hooks.design_rules("**Design rules**\n\n- **R1 Lazy.** x\n- **R3 Other.** y\n")
    with pytest.raises(ValueError, match="no \\*\\*Design rules\\*\\*"):
        hooks.design_rules("# Plan\n")


def test_api_reference_shows_every_public_item() -> None:
    hooks = _script("docs_hooks")
    llms_txt = _script("llms_txt")
    pages = "\n".join(hooks.api_pages().values())
    shown = set(re.findall(r"^::: (\S+)$", pages, flags=re.MULTILINE))
    for section in llms_txt.sections():
        for entry in section.entries:
            parts = entry.name.split(".")
            owner = (
                parts[0] if parts[0] not in ("qc", "dtypes") else ".".join(parts[:2])
            )
            assert any(
                target == f"kyto.{owner}" or target.endswith(f".{owner}")
                for target in shown
            ), entry.name


def test_links_outside_docs_point_at_the_repository(tmp_path: Path) -> None:
    hooks = _script("docs_hooks")
    url = "https://github.com/datavil/kyto"
    docs = hooks.DOCS
    tutorials = docs / "tutorials"
    assert (
        hooks._rewrite("../../AGENTS.md#design-rules", tutorials, tutorials, url)
        == f"{url}/blob/main/AGENTS.md#design-rules"
    )
    assert (
        hooks._rewrite("Implementation%20Plan.md", hooks.ROOT, docs, url)
        == f"{url}/blob/main/Implementation%20Plan.md"
    )
    assert hooks._rewrite("docs/format.md#v0", hooks.ROOT, docs, url) == (
        "format.md#v0"
    )
    assert hooks._rewrite("../format.md", tutorials, tutorials, url) == "../format.md"
    assert hooks._rewrite("https://scanpy.org", docs, docs, url) == "https://scanpy.org"
    assert hooks._rewrite("#ky-e101", docs, docs, url) == "#ky-e101"
    # A target that exists nowhere is left as written, for mkdocs to report.
    assert hooks._rewrite("../missing.md", docs, docs, url) == "../missing.md"


def test_api_reference_renders_every_member() -> None:
    """griffe, as mkdocstrings runs it, finds every item and member of llms.txt."""
    import griffe

    hooks = _script("docs_hooks")
    package = griffe.load(
        "kyto",
        search_paths=[str(ROOT / "python")],
        allow_inspection=False,
        extensions=griffe.load_extensions(hooks.OverloadOnlyMembers),
    )
    pages = "\n".join(hooks.api_pages().values())
    targets = re.findall(r"^::: kyto\.(\S+)$", pages, flags=re.MULTILINE)
    roots = {target.rsplit(".", 1)[-1]: target for target in targets}
    for section in _script("llms_txt").sections():
        for entry in section.entries:
            parts = entry.name.split(".")
            if parts[0] in ("qc", "dtypes"):
                parts = parts[1:]
            path = ".".join([roots[parts[0]], *parts[1:]])
            assert package[path].name == parts[-1], entry.name


def test_link_forms_are_rewritten_outside_code() -> None:
    from types import SimpleNamespace

    hooks = _script("docs_hooks")
    page = SimpleNamespace(file=SimpleNamespace(src_uri="index.md"))
    config = SimpleNamespace(repo_url="https://github.com/datavil/kyto")
    blob = "https://github.com/datavil/kyto/blob/main"
    markdown = (
        '[t](AGENTS.md "title") [s](<Implementation Plan.md>) [d](docs/format.md)\n'
        "`[c](AGENTS.md)` ``[c](AGENTS.md)``\n"
        "[r]: deviations.md\n"
        "```\n[f](AGENTS.md)\n```\n"
    )
    assert hooks.on_page_markdown(markdown, page, config, None) == (
        f'[t]({blob}/AGENTS.md "title") [s](<{blob}/Implementation%20Plan.md>) '
        "[d](format.md)\n"
        "`[c](AGENTS.md)` ``[c](AGENTS.md)``\n"
        f"[r]: {blob}/deviations.md\n"
        "```\n[f](AGENTS.md)\n```\n"
    )
