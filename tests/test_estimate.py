"""Task 4.6: ``Atlas.estimate()`` gives passes, bytes read, peak memory and device.

From Python, the plans that can be built are scans, ``with_default_layer`` and
``filter_genes``. Nothing is read: the store is deleted before ``estimate()`` runs.
The model and every node kind are covered by the Rust tests
(``crates/kyto-plan/tests/it/estimate.rs``), and the check against measured memory on
the fixtures by ``crates/kyto-exec/tests/estimate_fixtures.rs`` (the executor, task 5.5).
"""

import gzip
import shutil
from pathlib import Path

import kyto
import pytest
from kyto import _kyto, cell_profile, gene_column

# One pass over the 3-entry `uint32` layer: entries (8 bytes each), one tile's
# `indptr` (4 097 x 4) and `tile_offsets` (2 x 8).
_PASS_BYTES = 3 * 8 + 4097 * 4 + 2 * 8
# Opening the layer decodes one 65 536-value `tile_offsets` chunk and copies it; its
# 2 `tile_offsets` values stay held.
_PEAK = 2 * 65536 * 8 + 2 * 8
# A gene filter reads the gene table (task 5.10), held until the run ends: `gene_id`
# at 40 bytes a String value, and the Enum columns `symbol` (3 categories) and
# `feature_type` (2) at a 4-byte code per gene and 160 bytes per category.
_GENE_TABLE = 3 * 40 + (3 * 4 + 3 * 160) + (3 * 4 + 2 * 160)
# Finding a profile's cell reads the cell table: `cell_id` of 2 cells.
_CELL_TABLE = 2 * 40


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


def test_estimates_read_nothing(store: Path) -> None:
    atlas = _kyto.scan(store)
    plans = [
        atlas,
        atlas.with_default_layer("counts"),
        atlas.filter(gene_column("feature_type") == "Gene Expression"),
        atlas.with_default_layer("counts").filter(cell_profile("AAAC-1") > 0),
    ]
    shutil.rmtree(store)  # estimate() reads nothing
    # The `cell_profile` filter makes a statistics pass and keeps 8 bytes per gene.
    expected = [
        (1, _PEAK),
        (1, _PEAK),
        (1, _PEAK + _GENE_TABLE),
        (2, _PEAK + _GENE_TABLE + _CELL_TABLE + 3 * 8),
    ]
    for plan, (passes, peak) in zip(plans, expected, strict=True):
        assert plan.estimate() == {
            "passes": passes,
            "bytes_read": passes * _PASS_BYTES,
            "peak_memory": peak,
            "held_memory": 0,
            "passes_saved": 0,
            "device": "cpu",
        }


def test_estimate_types(store: Path) -> None:
    estimate = _kyto.scan(store).estimate()
    assert list(estimate) == [
        "passes",
        "bytes_read",
        "peak_memory",
        "held_memory",
        "passes_saved",
        "device",
    ]
    assert all(
        type(estimate[key]) is int
        for key in (
            "passes",
            "bytes_read",
            "peak_memory",
            "held_memory",
            "passes_saved",
        )
    )
    # A new dict each call.
    assert estimate is not _kyto.scan(store).estimate()


def test_pca_holds_its_input(store: Path) -> None:
    """Task 5.12: ``pca`` holds its input; ``explain()`` and ``estimate()`` say so."""
    atlas = _kyto.scan(store).log1p(layer="counts", into="log1p")
    plan = atlas.pca(layer="log1p", into="X_pca", n_components=2, seed=0)
    shutil.rmtree(store)  # estimate() and explain() read nothing
    pca_line = next(
        line for line in plan.explain().splitlines() if line.lstrip().startswith("#")
    )
    assert pca_line.endswith("input held]"), pca_line
    nodes = plan.explain(format="json")["optimized"]["nodes"]
    assert [node["holds_input"] for node in nodes] == [True, False, False]
    estimate = plan.estimate()
    # The 3 `float32` entries and `u32` indices, 2 cells (16 bytes each), 1 block.
    assert estimate["held_memory"] == 3 * 8 + 2 * 16 + 512
    # 11 statistics passes, read once; one final pass for the layers.
    assert estimate["passes_saved"] == 10
    assert estimate["passes"] == 2
    assert atlas.estimate()["held_memory"] == 0
