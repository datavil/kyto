"""Task 7.7: ``repr(atlas)`` is one line and never triggers compute.

``<Atlas plan: 7 nodes · cells: ? · genes: 2000 · layers: counts, log1p>``: the node
count, the sizes (``?`` when only the data can tell, after a filter) and the layers.
Done when: a test asserts ``repr()`` performs zero I/O. Here the store is deleted
before ``repr()`` runs and the folder is still empty afterwards, so nothing can have
been read or written. The text itself comes from ``Plan::summary`` (Rust tests:
``crates/kyto-plan/tests/it/summary.rs``), which is given the plan alone.
"""

import gzip
import shutil
from pathlib import Path

import kyto
import pytest
from kyto import cell_column, gene_column
from kyto._kyto import Atlas


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 2-cell, 3-gene store with the layer ``counts``, alone in its folder."""
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
    folder = tmp_path / "stores"
    folder.mkdir()
    target = folder / "pbmc.kyto"
    kyto.convert(source, target, layer="counts")
    return target


def _plans(atlas: Atlas) -> list[tuple[Atlas, str]]:
    """Plans and their expected ``repr``."""
    genes = atlas.filter(gene_column("feature_type") == "Gene Expression")
    pipeline = (
        genes.normalize_total(1e4, layer="counts", into="norm")
        .log1p(layer="norm", into="log1p")
        .filter(cell_column("cell_id") != "AAAC-1")
        .pca(layer="log1p", into="X_pca", n_components=1, seed=0)
    )
    return [
        (atlas, "<Atlas plan: 1 node · cells: 2 · genes: 3 · layers: counts>"),
        (
            atlas.with_default_layer("counts"),
            "<Atlas plan: 2 nodes · cells: 2 · genes: 3 · layers: counts>",
        ),
        (genes, "<Atlas plan: 2 nodes · cells: 2 · genes: ? · layers: counts>"),
        (
            pipeline,
            "<Atlas plan: 6 nodes · cells: ? · genes: ? · layers: counts, norm, log1p>",
        ),
        (
            atlas.select_genes(["ENSG01", "ENSG03"]).take_cells(["AAAG-1"]),
            "<Atlas plan: 3 nodes · cells: 1 · genes: 2 · layers: counts>",
        ),
    ]


def test_repr_is_one_line_with_nodes_sizes_and_layers(store: Path) -> None:
    for plan, expected in _plans(kyto.scan(store)):
        assert repr(plan) == expected
        assert "\n" not in repr(plan)
        assert repr([plan]) == f"[{expected}]"


def test_repr_performs_zero_io(store: Path) -> None:
    plans = _plans(kyto.scan(store))
    shutil.rmtree(store)  # nothing left to read
    for plan, expected in plans:
        assert repr(plan) == expected
        assert f"{plan!r:>80}" == f"{expected:>80}"
    # Nothing was written or created either.
    assert list(store.parent.iterdir()) == []


def test_repr_stays_one_line_on_a_long_chain(store: Path) -> None:
    atlas = kyto.scan(store)
    for _ in range(500):
        atlas = atlas.with_default_layer("counts")
    assert (
        repr(atlas) == "<Atlas plan: 501 nodes · cells: 2 · genes: 3 · layers: counts>"
    )


def test_str_is_still_the_plan_node_by_node(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    assert str(atlas).splitlines() == [
        'WITH_DEFAULT_LAYER "counts"',
        f'SCAN "{store}" (2 cells x 3 genes)',
    ]
