"""Build every fixture, check its digest, its edge cases and the stored references."""

import doctest
import importlib.util
import shutil
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from fixtures import dataset, hashing, pbmc, reference, synthetic
from fixtures.dataset import Dataset
from fixtures.hashing import array_digest
from fixtures.reference import load_manifest, load_reference, reference_keys
from fixtures.registry import EXPECTED_DIGESTS, FIXTURE_NAMES, FixtureName, load_fixture
from fixtures.synthetic import MITOCHONDRIAL_SYMBOLS, SyntheticSpec, make_synthetic

HAS_SCANPY = importlib.util.find_spec("scanpy") is not None


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_matches_expected_digest(name: FixtureName) -> None:
    assert load_fixture(name).digest() == EXPECTED_DIGESTS[name]


def test_synthetic_has_the_edge_cases_later_tasks_need() -> None:
    data = load_fixture("synthetic")
    spec = SyntheticSpec()
    dense = data.to_dense()
    assert dense.shape == (spec.n_cells, spec.n_genes)
    assert np.all(dense == np.round(dense)) and dense.min() == 0
    # All-zero cells and genes, and a gene seen in exactly one cell.
    assert np.count_nonzero(dense.sum(axis=1) == 0) == spec.n_empty_cells
    assert np.count_nonzero(dense.sum(axis=0) == 0) == spec.n_empty_genes
    assert np.any(np.count_nonzero(dense, axis=0) == 1)
    # Batches and cell types all present.
    assert sorted(set(data.cells["batch"].tolist())) == [
        "batch_0",
        "batch_1",
        "batch_2",
    ]
    assert len(set(data.cells["cell_type"].tolist())) == spec.n_cell_types
    # Mitochondrial genes: all 13, expressed, some cells dominated by them.
    symbols = data.genes["symbol"]
    is_mito = np.char.startswith(symbols, "MT-")
    assert sorted(symbols[is_mito].tolist()) == sorted(MITOCHONDRIAL_SYMBOLS)
    assert np.all(dense[:, is_mito].sum(axis=0) > 0)
    totals = dense.sum(axis=1)
    mito_fraction = dense[:, is_mito].sum(axis=1) / np.maximum(totals, 1)
    assert np.count_nonzero(mito_fraction > 0.5) >= spec.n_high_mito_cells
    # Repeated symbols with unique IDs (task 3.9).
    _, symbol_counts = np.unique(symbols, return_counts=True)
    assert np.count_nonzero(symbol_counts == 2) == spec.n_duplicate_symbols
    assert np.unique(data.genes["gene_id"]).shape[0] == spec.n_genes
    # Sparse enough to exercise sparse paths, dense enough to have structure.
    assert 0.05 < data.data.shape[0] / dense.size < 0.3


def test_synthetic_depends_only_on_its_seed() -> None:
    spec = SyntheticSpec(seed=7, n_cells=60, n_genes=80)
    first = make_synthetic(spec).digest()
    # Drawing from numpy's global generator in between must not change anything.
    np.random.seed(123)
    np.random.random(1000)
    assert make_synthetic(spec).digest() == first
    assert (
        make_synthetic(SyntheticSpec(seed=8, n_cells=60, n_genes=80)).digest() != first
    )


def test_synthetic_rejects_a_spec_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="n_genes=20 is too small"):
        SyntheticSpec(n_genes=20)
    with pytest.raises(ValueError, match="n_cells=10 is too small"):
        SyntheticSpec(n_cells=10)
    with pytest.raises(ValueError, match="n_empty_cells=-1 must be zero or positive"):
        SyntheticSpec(n_empty_cells=-1)


def test_pbmc3k_replaces_a_corrupt_cached_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pbmc.fetch_pbmc3k()
    corrupt = tmp_path / pbmc.PBMC3K_ARCHIVE
    corrupt.write_bytes(b"trunc")

    # Stand in for the network with the verified archive.
    def copy_verified(path: Path) -> None:
        shutil.copyfile(source, path)

    monkeypatch.setattr(pbmc, "_download", copy_verified)
    assert pbmc.fetch_pbmc3k(tmp_path) == corrupt
    assert corrupt.stat().st_size == source.stat().st_size


def test_pbmc3k_contents() -> None:
    data = load_fixture("pbmc3k")
    assert (data.n_cells, data.n_genes, data.data.shape[0]) == (2700, 32738, 2286884)
    assert data.cells["cell_id"][0] == "AAACATACAACCAC-1"
    assert data.genes["gene_id"][0] == "ENSG00000243485"
    symbols = data.genes["symbol"]
    assert np.count_nonzero(np.char.startswith(symbols, "MT-")) == 13
    _, symbol_counts = np.unique(symbols, return_counts=True)
    assert np.count_nonzero(symbol_counts > 1) == 91
    assert float(data.data.sum()) == 6390631.0


def _tiny(indices: list[int], data: list[float]) -> Dataset:
    return Dataset(
        name="tiny",
        indptr=np.array([0, len(indices)], dtype=np.int64),
        indices=np.array(indices, dtype=np.int32),
        data=np.array(data, dtype=np.float32),
        cells={"cell_id": np.array(["c0"])},
        genes={"gene_id": np.array(["g0", "g1"]), "symbol": np.array(["A", "A"])},
    )


def test_dataset_rejects_broken_csr() -> None:
    assert _tiny([0, 1], [1.0, 2.0]).n_genes == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        _tiny([1, 0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-zero"):
        _tiny([0, 1], [1.0, 0.0])
    with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
        _tiny([0, 2], [1.0, 2.0])
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            _tiny([0, 1], [1.0, bad])
    with pytest.raises(ValueError, match="integer counts"):
        _tiny([0, 1], [1.0, 1.5])


def test_dataset_arrays_are_read_only() -> None:
    data = load_fixture("synthetic")
    with pytest.raises(ValueError, match="read-only"):
        data.data[0] = 5
    with pytest.raises(TypeError):
        data.cells["extra"] = data.cells["cell_id"]  # pyright: ignore[reportIndexIssue]
    assert data.digest() == EXPECTED_DIGESTS["synthetic"]


def test_digest_is_canonical() -> None:
    values = np.array([1.5, np.nan, -0.0])
    reference_digest = array_digest({"x": values})
    assert array_digest({"x": values.astype(">f8")}) == reference_digest
    assert array_digest({"x": values.astype(np.float32)}) == reference_digest
    negative_nan = np.array([1.5, -np.nan, -0.0])
    assert array_digest({"x": negative_nan}) == reference_digest
    assert array_digest({"x": np.array([1.5, np.nan, 0.0])}) != reference_digest
    assert array_digest({"x": values.reshape(3, 1)}) != reference_digest
    assert array_digest({"y": values}) != reference_digest
    # Strings are length-prefixed: ["ab", "c"] differs from ["a", "bc"].
    assert array_digest({"s": np.array(["ab", "c"])}) != array_digest(
        {"s": np.array(["a", "bc"])}
    )


def test_digest_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match="accepted kinds"):
        array_digest({"x": np.array([object()])})


@pytest.mark.parametrize("key", reference_keys())
def test_reference_matches_manifest(key: str) -> None:
    fixture, step = key.split("/")
    assert fixture in FIXTURE_NAMES
    arrays = load_reference(fixture, step)  # pyright: ignore[reportArgumentType]
    assert arrays


def test_reference_rejects_a_changed_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(reference.EXPECTED_DIGESTS, "synthetic", "0" * 64)
    with pytest.raises(ValueError, match="generated from fixture digest"):
        load_reference("synthetic", "qc")


def test_reference_covers_every_fixture() -> None:
    fixtures = {key.split("/")[0] for key in reference_keys()}
    assert fixtures == set(FIXTURE_NAMES)


@pytest.mark.skipif(not HAS_SCANPY, reason="scanpy is not installed on this platform")
def test_reference_versions_match_installed_packages() -> None:
    recorded = cast(dict[str, str], load_manifest()["versions"])
    for package, recorded_version in recorded.items():
        assert version(package) == recorded_version, (
            f"{package} is {version(package)} but the references were made with "
            f"{recorded_version}; regenerate them (fixtures/make_reference.py)."
        )


@pytest.mark.skipif(not HAS_SCANPY, reason="scanpy is not installed on this platform")
@pytest.mark.parametrize(
    "step",
    ["qc", "normalize_log1p", "hvg", "scale", "pca", "neighbors", "rank_genes"],
)
def test_synthetic_reference_regenerates(step: str) -> None:
    from fixtures.make_reference import STEPS

    _, fresh = STEPS["synthetic"][step]("synthetic")
    stored = load_reference("synthetic", step)
    assert sorted(fresh) == sorted(stored)
    for name, expected in stored.items():
        actual = fresh[name]
        assert actual.dtype == expected.dtype, name
        if expected.dtype.kind == "f":
            # Same scanpy, other BLAS/CPU: allow last-bit differences only.
            np.testing.assert_allclose(
                cast(NDArray[np.floating], actual),
                cast(NDArray[np.floating], expected),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
                err_msg=name,
            )
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=name)


@pytest.mark.parametrize("module", [hashing, dataset, synthetic, reference])
def test_docstring_examples(module: ModuleType) -> None:
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
    if module is not dataset:
        assert result.attempted > 0
