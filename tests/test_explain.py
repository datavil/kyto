"""Tasks 4.5 and 4.7: ``explain()`` shows the logical and optimized trees, with the
device and pass count of every optimized node, as text or (``format="json"``) a dict.

From Python, the plans that can be built are scans, ``with_default_layer`` and
``filter_genes`` (on gene metadata, or on ``cell_profile``, which needs a statistics
pass over all cells). Each optimized node line ends with its device and passes.
Nothing is read: the store is deleted before ``explain()`` runs. Every node kind and
rule is covered by the Rust tests (``crates/kyto-plan/tests/it/physical_plan.rs``), and the
text and JSON snapshots of 5 reference pipelines by ``crates/kyto-plan/tests/it/explain.rs``.
"""

import gzip
import shutil
from pathlib import Path

import kyto
import pytest
from kyto import _kyto, cell_profile, gene_column


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 2-cell, 3-gene store with a ``feature_type`` gene column."""
    source = tmp_path / "source"
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
    target = tmp_path / "pbmc.kyto"
    kyto.convert(source, target, layer="counts")
    return target


def _node_lines(atlas: _kyto.Atlas) -> list[str]:
    """The optimized node lines of ``explain()`` (after the output line), unindented."""
    text = atlas.explain()
    assert text.startswith("logical plan\n")
    _, optimized = text.split("\noptimized plan\n")
    lines = optimized.splitlines()
    assert lines[0].startswith("  output #")
    return [line.removeprefix("  ") for line in lines[1:]]


def test_every_node_shows_device_and_passes(store: Path) -> None:
    atlas = _kyto.scan(store)
    plans = [
        atlas,
        atlas.with_default_layer("counts"),
        atlas.filter_genes(gene_column("feature_type") == "Gene Expression"),
        atlas.with_default_layer("counts").filter_genes(cell_profile("AAAC-1") > 0),
    ]
    shutil.rmtree(store)  # explain() reads nothing
    for plan in plans:
        lines = _node_lines(plan)
        assert len(lines) == len(str(plan).splitlines())
        for line in lines:
            assert " [device=cpu passes=" in line, line


def test_passes_per_node(store: Path) -> None:
    atlas = _kyto.scan(store).with_default_layer("counts")
    metadata = atlas.filter_genes(gene_column("feature_type") == "Gene Expression")
    profile = atlas.filter_genes(cell_profile("AAAC-1") > 0)
    assert _node_lines(metadata) == [
        (
            "#2 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- #1 (node 2)"
            " [device=cpu passes=1 blocks=4096 cells]"
        ),
        '#1 WITH_DEFAULT_LAYER "counts" <- #0 (node 1) [device=cpu passes=0 no data]',
        _node_lines(atlas)[-1],
    ]
    # A gene filter on the data: a statistics pass over all cells, then the stream.
    assert _node_lines(profile)[0].endswith(
        "[device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]"
    )
    assert _node_lines(atlas)[-1].endswith("[device=cpu passes=1 blocks=4096 cells]")


def test_explain_leaves_the_plan_unchanged(store: Path) -> None:
    atlas = _kyto.scan(store).filter_genes(
        gene_column("feature_type") == "Gene Expression"
    )
    before = atlas.to_json()
    assert atlas.explain() == atlas.explain()
    assert atlas.to_json() == before


def test_text_snapshot(store: Path) -> None:
    atlas = (
        _kyto.scan(store)
        .with_default_layer("counts")
        .filter_genes(cell_profile("AAAC-1") > 0)
    )
    text = atlas.explain().replace(str(store), "<store>")
    assert text == (
        "logical plan\n"
        "  node 2 FILTER_GENES cell_profile('AAAC-1', layer='counts') > 0 <- node 1\n"
        '  node 1 WITH_DEFAULT_LAYER "counts" <- node 0\n'
        '  node 0 SCAN "<store>" (2 cells x 3 genes)\n'
        "optimized plan\n"
        "  output #2 reads [layer:counts, genes:feature_type, genes:symbol]\n"
        "  #2 FILTER_GENES cell_profile('AAAC-1', layer='counts') > 0 <- #1 (node 2)"
        " [device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells]\n"
        '  #1 WITH_DEFAULT_LAYER "counts" <- #0 (node 1) [device=cpu passes=0 no data]\n'
        '  #0 SCAN "<store>" (2 cells x 3 genes) reads [layer:counts, genes:feature_type,'
        " genes:symbol] (node 0) [device=cpu passes=1 blocks=4096 cells]"
    )


def test_json_is_the_same_as_a_dict(store: Path) -> None:
    atlas = _kyto.scan(store).filter_genes(
        gene_column("feature_type") == "Gene Expression"
    )
    explained = atlas.explain(format="json")
    path = str(store)
    assert explained == {
        "logical": [
            {
                "id": 1,
                "name": "FILTER_GENES",
                "parameters": "gene_column('feature_type') == 'Gene Expression'",
                "input": 0,
                "note": None,
            },
            {
                "id": 0,
                "name": "SCAN",
                "parameters": f'"{path}" (2 cells x 3 genes)',
                "input": None,
                "note": None,
            },
        ],
        "optimized": {
            "outputs": [
                {
                    "node": 1,
                    "reads": ["layer:counts", "genes:feature_type", "genes:symbol"],
                }
            ],
            "nodes": [
                {
                    "index": 1,
                    "name": "FILTER_GENES",
                    "parameters": "gene_column('feature_type') == 'Gene Expression'",
                    "reads": None,
                    "input": 0,
                    "origin": [1],
                    "device": "cpu",
                    "passes": 1,
                    "statistics_passes": 0,
                    "apply_pass": True,
                    "blocks": "cells",
                    "block_cells": 4096,
                },
                {
                    "index": 0,
                    "name": "SCAN",
                    "parameters": f'"{path}" (2 cells x 3 genes)',
                    "reads": ["layer:counts", "genes:feature_type", "genes:symbol"],
                    "input": None,
                    "origin": [0],
                    "device": "cpu",
                    "passes": 1,
                    "statistics_passes": 0,
                    "apply_pass": True,
                    "blocks": "cells",
                    "block_cells": 4096,
                },
            ],
        },
    }
    # Each entry's name and parameters start its text line, in the same order.
    text = atlas.explain(format="text")
    assert text == atlas.explain()
    for step in explained["logical"]:
        assert f"  node {step['id']} {step['name']} {step['parameters']}" in text
    # A new dict each call: changing one leaves the next alone.
    explained["logical"].clear()
    assert len(atlas.explain(format="json")["logical"]) == 2


def test_unknown_format_is_ky_e215(store: Path) -> None:
    atlas = _kyto.scan(store)
    with pytest.raises(kyto.PlanError) as raised:
        atlas.explain(format="xml")  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert raised.value.code == "KY-E215"
    assert isinstance(raised.value, ValueError)
    message = str(raised.value)
    assert "explain(..., format='xml')" in message
    assert "'text' or 'json'" in message
    with pytest.raises(TypeError):
        atlas.explain("json")  # pyright: ignore[reportCallIssue]
