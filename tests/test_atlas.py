"""Task 3.2: ``Atlas`` is an immutable plan; every method returns a new ``Atlas`` (R2).

Done when: the original plan is unchanged after 10 chained calls. The stores are built
with ``kyto.convert`` from a tiny Matrix Market directory and opened with the private
``kyto._kyto.scan`` (the public ``kyto.scan`` is task 7.1). Rust tests:
``crates/kyto-plan/tests/it/plan_chain.rs``.
"""

import doctest
import gzip
from pathlib import Path

import kyto
import pytest
from kyto import _kyto


def _store(folder: Path, layer: str = "counts") -> Path:
    """A 2-cell, 3-gene store with one layer named ``layer``."""
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
    kyto.convert(source, store, layer=layer)
    return store


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


def test_original_unchanged_after_ten_chained_calls(store: Path) -> None:
    original = _kyto.scan(store)
    before = repr(original)
    plans = [original]
    for _ in range(10):
        plans.append(plans[-1].with_default_layer("counts"))
    texts = [str(plan) for plan in plans]

    assert repr(original) == before
    assert str(original) == f'SCAN "{store}" (2 cells x 3 genes)'
    for depth, (plan, text) in enumerate(zip(plans, texts, strict=True)):
        assert plan is not plans[depth - 1] or depth == 0
        assert str(plan) == text
        assert len(text.splitlines()) == depth + 1
    assert texts[-1].splitlines()[0] == 'WITH_DEFAULT_LAYER "counts"'


def test_branches_do_not_affect_each_other(tmp_path: Path) -> None:
    base = _kyto.scan(_store(tmp_path, layer="raw"))
    first = base.with_default_layer("raw")
    second = base.with_default_layer("raw")
    assert first is not second and first is not base
    assert str(first) == str(second)
    assert len(str(base).splitlines()) == 1


def test_atlas_is_frozen_and_has_no_matrix_accessors(store: Path) -> None:
    atlas = _kyto.scan(store)
    with pytest.raises(AttributeError):
        atlas.plan = None  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        _kyto.Atlas()  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    # R4: no indexing, no .X, no obs/var.
    for name in ("__getitem__", "__setitem__", "X", "obs", "var", "__dict__"):
        assert not hasattr(atlas, name), name
    public = {name for name in dir(_kyto.Atlas) if not name.startswith("_")}
    # The handle accessors (task 3.6, taken early from 7.3) are read-only too.
    assert public == {
        "resolve",
        "with_default_layer",
        "filter_genes",
        "cells",
        "genes",
        "layer",
        "embedding",
        "graph",
        "modality",
        "to_json",
        "explain",
        "estimate",
    }


def test_unknown_layer_is_ky_e109(store: Path) -> None:
    atlas = _kyto.scan(store)
    with pytest.raises(kyto.MissingMemberError) as raised:
        atlas.with_default_layer("Counts")
    assert raised.value.code == "KY-E109"
    assert "'Counts'" in str(raised.value)
    assert str(store) in str(raised.value)
    assert isinstance(raised.value, kyto.KytoError)
    assert str(atlas) == f'SCAN "{store}" (2 cells x 3 genes)'


def test_scan_of_a_missing_store_is_coded(tmp_path: Path) -> None:
    with pytest.raises(kyto.CorruptStoreError) as raised:
        _kyto.scan(tmp_path / "missing.kyto")
    assert raised.value.code == "KY-E116"
    assert not (tmp_path / "missing.kyto").exists()


def test_docstring_examples_run(store: Path) -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for name, docstring in (
        ("Atlas", _kyto.Atlas.__doc__),
        ("Atlas.with_default_layer", _kyto.Atlas.with_default_layer.__doc__),
        ("Atlas.resolve", _kyto.Atlas.resolve.__doc__),
        ("Atlas.filter_genes", _kyto.Atlas.filter_genes.__doc__),
        ("Atlas.to_json", _kyto.Atlas.to_json.__doc__),
        ("Atlas.explain", _kyto.Atlas.explain.__doc__),
        ("Atlas.estimate", _kyto.Atlas.estimate.__doc__),
    ):
        assert docstring is not None
        globs = {
            "scan": _kyto.scan,
            "store": store,
            "feature": kyto.feature,
            "gene_column": kyto.gene_column,
            "from_json": kyto.from_json,
        }
        runner.run(parser.get_doctest(docstring, globs, name, None, 0))
    assert runner.failures == 0
    assert runner.tries == 21
