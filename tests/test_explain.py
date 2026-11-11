"""Task 4.5: ``explain()`` shows device and pass count for every node.

From Python, the plans that can be built are scans, ``with_default_layer`` and
``filter_genes`` (on gene metadata, or on ``cell_profile``, which needs a statistics
pass over all cells). Each node line of ``explain()`` ends with its device and passes.
Nothing is read: the store is deleted before ``explain()`` runs. Every node kind and
rule is covered by the Rust tests (``crates/kyto-plan/tests/physical_plan.rs``).
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
    """The node lines of ``explain()`` (after the output line)."""
    text = atlas.explain()
    assert text.startswith("output #")
    return text.splitlines()[1:]


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
