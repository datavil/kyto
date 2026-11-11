"""Task 4.6: ``Atlas.estimate()`` gives passes, bytes read, peak memory and device.

From Python, the plans that can be built are scans, ``with_default_layer`` and
``filter_genes``. Nothing is read: the store is deleted before ``estimate()`` runs.
The model and every node kind are covered by the Rust tests
(``crates/kyto-plan/tests/estimate.rs``), and the check against measured memory on
the fixtures by ``crates/kyto-plan/tests/estimate_fixtures.rs``.
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
        atlas.filter_genes(gene_column("feature_type") == "Gene Expression"),
        atlas.with_default_layer("counts").filter_genes(cell_profile("AAAC-1") > 0),
    ]
    shutil.rmtree(store)  # estimate() reads nothing
    # The `cell_profile` filter makes a statistics pass and keeps 8 bytes per gene.
    expected = [(1, _PEAK), (1, _PEAK), (1, _PEAK), (2, _PEAK + 3 * 8)]
    for plan, (passes, peak) in zip(plans, expected, strict=True):
        assert plan.estimate() == {
            "passes": passes,
            "bytes_read": passes * _PASS_BYTES,
            "peak_memory": peak,
            "device": "cpu",
        }


def test_estimate_types(store: Path) -> None:
    estimate = _kyto.scan(store).estimate()
    assert list(estimate) == ["passes", "bytes_read", "peak_memory", "device"]
    assert all(
        type(estimate[key]) is int for key in ("passes", "bytes_read", "peak_memory")
    )
    # A new dict each call.
    assert estimate is not _kyto.scan(store).estimate()
