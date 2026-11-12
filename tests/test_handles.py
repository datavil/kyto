"""Task 3.6: the handles ``Table``, ``Matrix``, ``Embedding``, ``Graph``, ``Modality``.

Done when: no handle exposes a method that mutates, checked by a stub test
(``test_stub_handles_expose_no_mutation``, which reads ``_kyto.pyi``, and its run-time
twin ``test_runtime_handles_match_stub``). Each handle is read-only and lazy:
``names()``, ``schema()`` and ``estimate()`` answer from metadata; the ``to_*``
terminals raise ``KY-E601`` until task 7.3. The accessors on ``Atlas`` are 7.3's, taken
early. Rust tests: ``crates/kyto-plan/tests/it/handles.rs``; frozen pyclasses:
``crates/kyto-py/src/handles.rs``.

No store kyto writes holds an embedding or a graph yet (the importers bring none), and
``scan`` reads only ``schema.json`` and the gene IDs, so ``store_with_members`` lists an
embedding and a graph in ``schema.json`` without their arrays. That is enough for
handles, which never open a member.
"""

import ast
import doctest
import gzip
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kyto
import pytest
from kyto import _kyto

HANDLES: dict[str, type] = {
    "Table": _kyto.Table,
    "Matrix": _kyto.Matrix,
    "Embedding": _kyto.Embedding,
    "Graph": _kyto.Graph,
    "Modality": _kyto.Modality,
}

# The fixed surface of each handle. Every handle has names(), schema() and estimate();
# the rest are its terminals. A new member must be added here on purpose, and must be
# read-only (see ALLOWED_PUBLIC).
INSPECTION = {"names", "schema", "estimate"}
TERMINALS = {
    "Table": {"to_polars", "lazy"},
    "Matrix": {"to_scipy", "to_dense"},
    "Embedding": {"to_numpy"},
    "Graph": {"to_scipy", "to_edges"},
    "Modality": set[str](),
}
# The only dunders a handle defines. No __setattr__/__delattr__/__setitem__/
# __delitem__ (mutation), no __getitem__ (R4), no __array__ (R8), no __iter__.
ALLOWED_DUNDERS = {"__repr__"}


def _allowed_public(name: str) -> bool:
    """A public member is an inspection method, ``lazy`` or a ``to_*`` terminal."""
    return name in INSPECTION or name == "lazy" or name.startswith("to_")


def _store(folder: Path) -> Path:
    """A 2-cell, 3-gene store with one layer ``counts`` (as ``tests/test_atlas.py``)."""
    source = folder / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n"
            b"ENSG03\tMS4A1\tGene Expression\n"
        )
    )
    store = folder / "pbmc.kyto"
    kyto.convert(source, store, layer="counts")
    return store


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


@pytest.fixture
def store_with_members(tmp_path: Path) -> Path:
    """The 2-cell store, with an embedding ``X_pca`` and a graph listed in its schema."""
    folder = tmp_path / "members"
    folder.mkdir()
    store = _store(folder)
    copy = folder / "members.kyto"
    shutil.copytree(store, copy)
    schema_path = copy / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["embeddings"] = [
        {"name": "X_pca", "axis": "cells", "width": 2, "dtype": "float32"}
    ]
    schema["graphs"] = [
        {"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 3}
    ]
    schema_path.write_text(json.dumps(schema, indent=2))
    return copy


def _schema_json(store: Path) -> dict[str, Any]:
    return json.loads((store / "schema.json").read_text())


# Inspection: names(), schema(), estimate() from metadata.


def test_tables(store: Path) -> None:
    atlas = _kyto.scan(store)
    stored = _schema_json(store)
    assert atlas.cells.names() == ["cell_id"]
    assert atlas.genes.names() == ["gene_id", "symbol", "feature_type"]
    assert atlas.cells.schema() == stored["cells"]
    assert atlas.genes.schema() == stored["genes"]
    assert atlas.cells.estimate() == {
        "n_rows": 2,
        "n_columns": 1,
        "nnz": None,
        "dense_bytes": None,
    }
    assert atlas.genes.estimate()["n_rows"] == 3
    assert repr(atlas.genes) == "<Table genes: 3 rows x 3 columns>"


def test_matrix(store: Path) -> None:
    matrix = _kyto.scan(store).layer("counts")
    assert matrix.names() == ["ENSG01", "ENSG02", "ENSG03"]
    assert matrix.schema() == _schema_json(store)["layers"][0]
    assert matrix.estimate() == {
        "n_rows": 2,
        "n_columns": 3,
        "nnz": 3,
        "dense_bytes": 2 * 3 * 4,
    }
    assert repr(matrix) == "<Matrix 'counts': 2 cells x 3 genes, nnz 3, uint32>"


def test_embedding_and_graph(store_with_members: Path) -> None:
    atlas = _kyto.scan(store_with_members)
    stored = _schema_json(store_with_members)
    embedding = atlas.embedding("X_pca")
    assert embedding.names() == ["X_pca:0", "X_pca:1"]
    assert embedding.schema() == stored["embeddings"][0]
    assert embedding.estimate() == {
        "n_rows": 2,
        "n_columns": 2,
        "nnz": None,
        "dense_bytes": 16,
    }
    graph = atlas.graph("connectivities")
    assert graph.names() == ["source", "target", "weight"]
    assert graph.schema() == stored["graphs"][0]
    assert graph.estimate() == {
        "n_rows": 2,
        "n_columns": 2,
        "nnz": 3,
        "dense_bytes": 16,
    }


def test_embedding_names_resolve_as_features(store_with_members: Path) -> None:
    atlas = _kyto.scan(store_with_members)
    for name in atlas.embedding("X_pca").names():
        assert repr(atlas.resolve(kyto.feature(name))).startswith("cell_embedding(")


def test_results_are_new_objects(store: Path) -> None:
    atlas = _kyto.scan(store)
    matrix = atlas.layer("counts")
    names = matrix.names()
    names.append("ENSG99")
    schema = atlas.genes.schema()
    schema["columns"].clear()
    estimate = matrix.estimate()
    estimate["nnz"] = 0
    assert matrix.names() == ["ENSG01", "ENSG02", "ENSG03"]
    assert len(atlas.genes.schema()["columns"]) == 3
    assert matrix.estimate()["nnz"] == 3


def test_handles_leave_the_plan_unchanged(store: Path) -> None:
    atlas = _kyto.scan(store).with_default_layer("counts")
    before = str(atlas)
    _ = (atlas.cells, atlas.genes, atlas.layer("counts"))
    assert str(atlas) == before
    assert atlas.cells is not atlas.cells


def test_handles_follow_the_plan_snapshot(store: Path) -> None:
    """A handle answers from the schema read at scan time, not the store now (R6)."""
    atlas = _kyto.scan(store)
    matrix = atlas.layer("counts")
    schema_path = store / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["layers"][0]["nnz"] = 99
    schema_path.write_text(json.dumps(schema))
    assert matrix.estimate()["nnz"] == 3
    assert atlas.layer("counts").schema()["nnz"] == 3


# Errors.


@pytest.mark.parametrize("kind", ["layer", "embedding", "graph", "modality"])
def test_missing_member(store: Path, kind: str) -> None:
    accessor: Callable[[str], object] = getattr(_kyto.scan(store), kind)
    with pytest.raises(kyto.MissingMemberError) as raised:
        accessor("Counts")
    assert raised.value.code == "KY-E109"
    assert raised.value.fields["kind"] == kind
    assert raised.value.fields["name"] == "Counts"
    assert f"no {kind} named 'Counts'" in str(raised.value)


def _terminal_calls(atlas: _kyto.Atlas) -> list[tuple[str, Callable[[], object]]]:
    matrix = atlas.layer("counts")
    embedding = atlas.embedding("X_pca")
    graph = atlas.graph("connectivities")
    return [
        ("Table.to_polars", atlas.cells.to_polars),
        ("Table.lazy", atlas.genes.lazy),
        ("Matrix.to_scipy", matrix.to_scipy),
        ("Matrix.to_dense", lambda: matrix.to_dense(max_bytes=1 << 20)),
        ("Embedding.to_numpy", embedding.to_numpy),
        ("Graph.to_scipy", graph.to_scipy),
        ("Graph.to_edges", graph.to_edges),
    ]


def test_terminals_are_not_implemented_yet(store_with_members: Path) -> None:
    calls = _terminal_calls(_kyto.scan(store_with_members))
    # Every terminal of every constructible handle is covered.
    covered = {method for method, _ in calls}
    assert covered == {
        f"{handle}.{terminal}"
        for handle, terminals in TERMINALS.items()
        for terminal in terminals
    }
    for method, call in calls:
        with pytest.raises(kyto.NotImplementedYetError) as raised:
            call()
        assert isinstance(raised.value, NotImplementedError)
        assert raised.value.code == "KY-E601"
        assert raised.value.fields["method"] == method
        assert raised.value.fields["task"] == "7.3"
        assert "names(), schema() and estimate()" in str(raised.value)


def test_to_dense_needs_max_bytes(store: Path) -> None:
    matrix = _kyto.scan(store).layer("counts")
    with pytest.raises(TypeError):
        matrix.to_dense()  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        matrix.to_dense(1024)  # type: ignore[misc]  # pyright: ignore[reportCallIssue]


# Read-only: the stub test (Done-when) and its run-time twin.


def _stub_classes() -> dict[str, ast.ClassDef]:
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in HANDLES
    }


def _decorator_names(function: ast.FunctionDef) -> list[str]:
    return [ast.unparse(decorator) for decorator in function.decorator_list]


def test_stub_handles_expose_no_mutation() -> None:
    classes = _stub_classes()
    assert set(classes) == set(HANDLES)
    for name, node in classes.items():
        assert _decorator_names_of_class(node) == ["final"], name
        members: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ):
                continue  # the docstring
            # Only plain methods: no fields (settable attributes), no properties or
            # setters, no class-level assignments.
            assert isinstance(statement, ast.FunctionDef), (
                f"{name}: {ast.unparse(statement)[:60]!r} is not a plain method"
            )
            assert _decorator_names(statement) == [], (
                f"{name}.{statement.name} is decorated: {_decorator_names(statement)}"
            )
            members.add(statement.name)
        public = {member for member in members if not member.startswith("_")}
        dunders = members - public
        assert all(_allowed_public(member) for member in public), (name, public)
        assert public == INSPECTION | TERMINALS[name], name
        assert dunders <= ALLOWED_DUNDERS, (name, dunders)


def _decorator_names_of_class(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(decorator) for decorator in node.decorator_list]


def test_runtime_handles_match_stub() -> None:
    classes = _stub_classes()
    for name, cls in HANDLES.items():
        stub_members = {
            statement.name
            for statement in classes[name].body
            if isinstance(statement, ast.FunctionDef)
        }
        runtime = set(vars(cls)) - {"__doc__", "__module__"}
        # ruff's stub rules drop __repr__ from .pyi files, so it is runtime-only.
        assert runtime - stub_members <= ALLOWED_DUNDERS, name
        assert stub_members <= runtime, name
        # Every member is a method (no data descriptor that could be set).
        for member in runtime:
            assert callable(getattr(cls, member)), f"{name}.{member}"
        with pytest.raises(TypeError, match="cannot create"):
            cls()
        assert not hasattr(cls, "__dict__") or "__dict__" not in vars(cls)
        for absent in ("X", "obs", "var", "__getitem__", "__array__", "__setitem__"):
            assert not hasattr(cls, absent), f"{name}.{absent}"


def test_instances_are_frozen(store_with_members: Path) -> None:
    atlas = _kyto.scan(store_with_members)
    instances = [
        atlas.cells,
        atlas.layer("counts"),
        atlas.embedding("X_pca"),
        atlas.graph("connectivities"),
    ]
    for instance in instances:
        for attribute in ("names", "schema", "plan", "layer", "__class__"):
            with pytest.raises((AttributeError, TypeError)):
                setattr(instance, attribute, None)
            if attribute != "__class__":
                with pytest.raises(AttributeError):
                    delattr(instance, attribute)


# Examples (R12): runtime and stub docstrings.


def _docstrings_runtime() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for name, cls in HANDLES.items():
        found.append((name, cls.__doc__ or ""))
        for member in sorted(set(vars(cls)) - {"__doc__", "__module__", "__repr__"}):
            found.append((f"{name}.{member}", getattr(cls, member).__doc__ or ""))
    for member in ("cells", "genes", "layer", "embedding", "graph", "modality"):
        found.append((f"Atlas.{member}", getattr(_kyto.Atlas, member).__doc__ or ""))
    return found


def _docstrings_stub() -> list[tuple[str, str]]:
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    accessors = {"cells", "genes", "layer", "embedding", "graph", "modality"}
    found: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in {*HANDLES, "Atlas"}:
            continue
        if node.name != "Atlas":
            found.append((node.name, ast.get_docstring(node) or ""))
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if node.name == "Atlas" and statement.name not in accessors:
                continue
            if statement.name == "__repr__":
                continue
            found.append(
                (f"{node.name}.{statement.name}", ast.get_docstring(statement) or "")
            )
    return found


def _run_examples(
    docstrings: list[tuple[str, str]], store: Path, store_with_members: Path
) -> tuple[int, list[str]]:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    without_examples: list[str] = []
    for name, docstring in docstrings:
        globs = {
            "scan": _kyto.scan,
            "store": store,
            "store_with_members": store_with_members,
        }
        test = parser.get_doctest(docstring, globs, name, None, 0)
        if not test.examples:
            without_examples.append(name)
        runner.run(test)
    assert runner.failures == 0
    return runner.tries, without_examples


# Modality cannot be built from a v0 store, so its methods have no runnable example.
NO_EXAMPLE = {"Modality.names", "Modality.schema", "Modality.estimate"}


def test_runtime_docstring_examples_run(store: Path, store_with_members: Path) -> None:
    docstrings = _docstrings_runtime()
    tries, without = _run_examples(docstrings, store, store_with_members)
    assert set(without) == NO_EXAMPLE
    assert tries >= len(docstrings) - len(NO_EXAMPLE)


def test_stub_docstrings_match_runtime(store: Path, store_with_members: Path) -> None:
    runtime = dict(_docstrings_runtime())
    stub = dict(_docstrings_stub())
    assert stub.keys() == runtime.keys()
    for name, text in stub.items():
        assert text == runtime[name].strip(), name
    _run_examples(list(stub.items()), store, store_with_members)
