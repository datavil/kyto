"""Task 4.3: the plan JSON round trip is exact.

Done when: ``from_json(atlas.to_json())`` gives back an equal plan for every plan in the
test suite. From Python, the plans the suite builds are scans of a store (with and
without gene symbols), ``with_default_layer`` and ``filter_genes`` with the gene
expressions the suite uses; each one here writes the same text again after the round
trip and prints the same tree. The Rust side (``crates/kyto-plan/tests/plan_json.rs``)
covers every node kind, every expression form, stores without symbols and random
chains. Then the refusals:
``KY-E217`` for broken text, ``KY-E218`` for another version, and the builders' own
errors for text edited into a plan that does not build; and reading needs no store.
"""

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import kyto
import pytest
from kyto import (
    _kyto,
    cell_profile,
    from_json,
    gene_column,
)


def _store(folder: Path) -> Path:
    """A 2-cell, 3-gene store with a ``symbol`` and a ``feature_type`` gene column."""
    source = folder / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    features = (
        b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n"
        b"ENSG03\tMS4A1\tAntibody Capture\n"
    )
    (source / "features.tsv.gz").write_bytes(gzip.compress(features))
    store = folder / "pbmc.kyto"
    kyto.convert(source, store, layer="counts")
    return store


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


def _plans(store: Path) -> list[_kyto.Atlas]:
    atlas = _kyto.scan(store)
    layered = atlas.with_default_layer("counts")
    gene_expressions = (
        gene_column("feature_type") == "Gene Expression",
        gene_column("gene_id").str.starts_with("ENSG0"),
        gene_column("gene_id").str.ends_with("1"),
        ~gene_column("symbol").is_null(),
        gene_column("symbol").is_not_null() & (gene_column("symbol") != "CD4"),
        cell_profile("AAAC-1", layer="counts") > 0,
        (cell_profile("AAAG-1") * 2.5 + 1) / 3 >= 0.1 + 0.2,
        (cell_profile("AAAG-1") > float("inf")) | (cell_profile("AAAG-1") < -1e300),
        cell_profile("AAAG-1") != float("nan"),  # noqa: PLW0177 (builds an expression)
        (gene_column("feature_type") == "Gene Expression") == True,
    )
    plans = [atlas, layered]
    plans += [layered.filter_genes(expression) for expression in gene_expressions]
    chained = layered
    for expression in gene_expressions[:4]:
        chained = chained.filter_genes(expression)
    plans.append(chained)
    plans.append(layered.with_default_layer("counts"))
    return plans


def test_every_plan_round_trips_exactly(store: Path) -> None:
    for atlas in _plans(store):
        text = atlas.to_json()
        back = from_json(text)
        assert isinstance(back, _kyto.Atlas)
        assert back is not atlas
        assert back.to_json() == text
        assert str(back) == str(atlas)
        assert back.genes.names() == atlas.genes.names()
        assert back.layer("counts").schema() == atlas.layer("counts").schema()


def test_the_text_is_versioned_with_stable_ids(store: Path) -> None:
    atlas = (
        _kyto.scan(store)
        .with_default_layer("counts")
        .filter_genes(cell_profile("AAAC-1") > 0)
    )
    document: dict[str, Any] = json.loads(atlas.to_json())
    assert document["format"] == "kyto-plan"
    assert document["version"] == 1
    nodes: list[dict[str, Any]] = document["nodes"]
    assert [node["id"] for node in nodes] == [0, 1, 2]
    assert [node["input"] for node in nodes] == [None, 0, 1]
    assert [node["step"]["method"] for node in nodes] == [
        "scan",
        "with_default_layer",
        "filter_genes",
    ]
    assert nodes[0]["step"]["source"] == str(store)
    assert nodes[0]["step"]["gene_ids"] == ["ENSG01", "ENSG02", "ENSG03"]
    assert nodes[0]["step"]["symbols"] == ["CD3E", "CD4", "MS4A1"]
    # Names are stored resolved (R6): the default layer is written in full.
    assert nodes[2]["step"]["predicate"][0] == {
        "op": "cell_profile",
        "layer": "counts",
        "cell_id": "AAAC-1",
    }


def test_reading_needs_no_store(store: Path) -> None:
    atlas = _kyto.scan(store).filter_genes(
        gene_column("feature_type") == "Gene Expression"
    )
    text = atlas.to_json()
    shutil.rmtree(store)
    back = from_json(text)
    assert back.to_json() == text
    assert str(back) == str(atlas)


def test_broken_text_is_ky_e217(store: Path) -> None:
    text = _kyto.scan(store).with_default_layer("counts").to_json()
    document: dict[str, Any] = json.loads(text)
    document["nodes"][1]["step"]["method"] = "explode"
    for broken, expected in (
        ("", "line 1 column 0"),
        ("[]", "line 1 column"),
        ('{"format":"kyto-plan","version":1,"nodes":[]}', "at least one node"),
        (json.dumps(document), "unknown variant `explode`"),
    ):
        with pytest.raises(kyto.PlanError) as raised:
            from_json(broken)
        assert raised.value.code == "KY-E217", str(raised.value)
        assert expected in str(raised.value), str(raised.value)
        assert isinstance(raised.value, ValueError)


def test_another_version_is_ky_e218() -> None:
    with pytest.raises(kyto.PlanError) as raised:
        from_json('{"format":"kyto-plan","version":7,"nodes":"?"}')
    assert raised.value.code == "KY-E218"
    assert "version 7" in str(raised.value)


def test_text_edited_into_an_invalid_plan_raises_the_builders_error(
    store: Path,
) -> None:
    text = _kyto.scan(store).with_default_layer("counts").to_json()
    document: dict[str, Any] = json.loads(text)
    document["nodes"][1]["step"]["layer"] = "raw"
    with pytest.raises(kyto.MissingMemberError) as raised:
        from_json(json.dumps(document))
    assert raised.value.code == "KY-E109"
    assert "'raw'" in str(raised.value)


def test_expressions_of_the_other_axis_are_refused(store: Path) -> None:
    atlas = _kyto.scan(store).filter_genes(cell_profile("AAAC-1", layer="counts") > 0)
    document: dict[str, Any] = json.loads(atlas.to_json())
    document["nodes"][1]["step"]["predicate"][0] = {
        "op": "gene_expression",
        "layer": "counts",
        "gene_id": "ENSG01",
    }
    with pytest.raises(kyto.PlanError) as raised:
        from_json(json.dumps(document))
    assert raised.value.code == "KY-E217"
    assert "node 1 (filter_genes), predicate entry 0" in str(raised.value)
