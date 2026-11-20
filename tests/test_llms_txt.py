"""Task 8.6: every public item of kyto is in ``llms.txt``, and its example passes.

``llms.txt`` is generated from the stubs by ``scripts/llms_txt.py`` (the gate runs it
with ``--check``, so the file is what the docstrings say). These tests hold it to the
built package: every public item has an entry (the names of ``kyto``, ``kyto.qc`` and
``kyto.dtypes``, every public class of the extension and its public methods and
properties), every entry has an example (Modality's methods aside, as no store can
build one yet), and every example runs as a doctest, from the text an agent reads. A
new method without an example fails here.

Each example runs in a fresh folder, with the namespace ``llms.txt`` states: every name
of ``kyto`` and ``kyto.qc``, ``store`` (2 cells x 3 genes, layer ``counts``) and
``store_with_members`` (the same, listing an embedding and a graph). Small stores keep
the whole file under a few seconds.
"""

import doctest
import gzip
import inspect
import json
import re
import shutil
from pathlib import Path

import kyto
import kyto.dtypes
import kyto.qc
import pytest
from kyto import _kyto

_LLMS = Path(__file__).resolve().parents[1] / "llms.txt"

# Items with no runnable example, and why (llms.txt says so in their entry).
NO_EXAMPLE = {
    # Store format v0 holds no modalities, so no store can build a Modality yet.
    "Modality.names",
    "Modality.schema",
    "Modality.estimate",
}


def public_surface() -> set[str]:
    """The qualified names of every public item of the built package."""
    names: set[str] = set(kyto.__all__)
    names |= {f"qc.{name}" for name in kyto.qc.__all__}
    names |= {f"dtypes.{name}" for name in kyto.dtypes.__all__}
    for name, value in vars(_kyto).items():
        if name.startswith("_") or not inspect.isclass(value):
            continue
        names.add(name)
        names |= {
            f"{name}.{attribute}"
            for attribute in vars(value)
            if not attribute.startswith("_")
        }
    return names


def entries() -> dict[str, str]:
    """``llms.txt`` entries by name: the text under each ``### name`` heading."""
    text = _LLMS.read_text(encoding="utf-8")
    parts = re.split(r"^### (\S+)$", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def example(entry: str) -> str | None:
    """The example of an entry: its fenced block that starts with ``>>>``."""
    for block in re.findall(r"^```python\n(.*?)^```$", entry, re.MULTILINE | re.DOTALL):
        if block.startswith(">>>"):
            return block
    return None


_ENTRIES = entries()


def test_every_public_item_has_an_entry() -> None:
    surface = public_surface()
    assert len(surface) > 130, "the surface walk found almost nothing"
    assert sorted(surface - _ENTRIES.keys()) == [], "missing from llms.txt"
    assert sorted(_ENTRIES.keys() - surface) == [], "in llms.txt, not public"


def test_every_entry_has_an_example() -> None:
    without = {name for name, entry in _ENTRIES.items() if example(entry) is None}
    assert without == NO_EXAMPLE
    for name in NO_EXAMPLE:
        assert "No example: store format v0 holds no modalities" in _ENTRIES[name]


def _store(folder: Path) -> Path:
    """The 2-cell, 3-gene store of the examples (as ``tests/test_atlas.py``)."""
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
    shutil.rmtree(source)
    return store


def _with_members(store: Path, folder: Path) -> Path:
    """A copy of ``store`` whose schema lists an embedding and a graph."""
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


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Both stores, built once; each example gets its own copy."""
    folder = tmp_path_factory.mktemp("stores")
    _with_members(_store(folder), folder)
    return folder


@pytest.mark.parametrize(
    "name", sorted(name for name in _ENTRIES if name not in NO_EXAMPLE)
)
def test_example_runs(
    name: str, stores: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = example(_ENTRIES[name])
    assert source is not None, f"{name} has no example"
    shutil.copytree(stores, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)  # an example writing a relative path writes here
    namespace: dict[str, object] = {
        **{name: getattr(kyto, name) for name in kyto.__all__},
        **{name: getattr(kyto.qc, name) for name in kyto.qc.__all__},
        "store": tmp_path / "pbmc.kyto",
        "store_with_members": tmp_path / "members.kyto",
    }
    test = doctest.DocTestParser().get_doctest(source, namespace, name, None, 0)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    report: list[str] = []
    runner.run(test, out=report.append)
    assert runner.tries > 0, f"{name}: no example ran"
    assert runner.failures == 0, "".join(report)
