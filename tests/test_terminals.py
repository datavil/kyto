"""Task 7.3: the handle terminals return one type each (R3), a copy the caller owns.

Done when: each terminal's return type is asserted in tests (R3). ``Table.to_polars``,
``Table.lazy``, ``Matrix.to_scipy`` and ``Matrix.to_dense`` run here, on a plain scan
(read in place), on a plan with steps (run into a temporary store first) and on a
collected ``Atlas``; ``Embedding.to_numpy``, ``Graph.to_scipy`` and ``Graph.to_edges``
on ``pca`` and ``neighbors`` plans (task 5.11 writes them into stores).
``Embedding.to_torch`` runs only where ``torch`` is installed (it is not a dependency);
its declared return type is checked in the stub here. Also 7.3's: ``sink_zarr``'s
``overwrite`` and the public ``Atlas.sink_h5ad`` (7.2 Open (5)).

Task 7.5 (the hand-off): frames reach Polars without a copy (the Rust buffers, through
the Arrow C stream interface), arrays reach NumPy and SciPy without a copy (the memory
a Rust ``Vec`` filled), dtypes are kept, and changing any result changes no later read.
"""

import ast
import gc
import gzip
from pathlib import Path
from typing import Protocol, cast

import kyto
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
import scipy.sparse  # pyright: ignore[reportMissingTypeStubs]
from kyto import _kyto, gene_column
from kyto.qc import total_counts, total_counts_in

# The cells x genes counts of the store below (Matrix Market is genes x cells).
COUNTS = np.array([[5, 0, 1], [0, 7, 0]], dtype=np.uint32)

# The return type each terminal declares in the stub (R3: one type).
STUB_RETURNS = {
    ("Table", "to_polars"): "pl.DataFrame",
    ("Table", "lazy"): "pl.LazyFrame",
    ("Matrix", "to_scipy"): "scipy.sparse.csr_matrix",
    ("Matrix", "to_dense"): "npt.NDArray[np.float32 | np.uint32]",
    ("Embedding", "to_numpy"): "npt.NDArray[np.float32 | np.float64]",
    ("Embedding", "to_torch"): "Any",
    ("Graph", "to_scipy"): "scipy.sparse.csr_matrix",
    ("Graph", "to_edges"): "pl.DataFrame",
}


class _Csr(Protocol):
    """The parts of ``scipy.sparse.csr_matrix`` these tests read (SciPy has no stubs)."""

    shape: tuple[int, int]
    dtype: np.dtype[np.generic]
    indptr: npt.NDArray[np.generic]
    indices: npt.NDArray[np.generic]
    data: npt.NDArray[np.generic]
    nnz: int
    has_sorted_indices: bool

    def toarray(self) -> npt.NDArray[np.number]: ...


def _csr(matrix: object) -> _Csr:
    """``matrix``, checked to be exactly a ``csr_matrix`` (R3)."""
    assert type(matrix) is scipy.sparse.csr_matrix
    return cast(_Csr, matrix)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 2-cell, 3-gene store with one layer ``counts`` (as ``tests/test_handles.py``)."""
    source = tmp_path / "source"
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
    store = tmp_path / "pbmc.kyto"
    kyto.convert(source, store, layer="counts")
    return store


def _with_steps(store: Path) -> _kyto.Atlas:
    """A plan with steps: keep the second cell, normalize it, add a cell column."""
    return (
        kyto.scan(store)
        .filter(total_counts(layer="counts") >= 7)
        .normalize_total(10.0, layer="counts", into="normalized")
        .with_cells(total=total_counts(layer="counts"))
    )


# Return types (the Done-when).


def test_table_terminals_return_polars(store: Path) -> None:
    atlas = kyto.scan(store)
    for table in (atlas.cells, atlas.genes):
        frame = table.to_polars()
        assert type(frame) is pl.DataFrame
        assert frame.columns == table.names()
        lazy = table.lazy()
        assert type(lazy) is pl.LazyFrame
        assert lazy.collect().equals(frame)
    genes = atlas.genes.to_polars()
    assert genes["gene_id"].to_list() == ["ENSG01", "ENSG02", "ENSG03"]
    assert genes.schema["gene_id"] == pl.String
    assert isinstance(genes.schema["symbol"], pl.Enum)
    assert genes["symbol"].to_list() == ["CD3E", "CD4", "MS4A1"]
    assert atlas.cells.to_polars()["cell_id"].to_list() == ["AAAC-1", "AAAG-1"]


def test_matrix_to_scipy_returns_csr_matrix(store: Path) -> None:
    matrix = _csr(kyto.scan(store).layer("counts").to_scipy())
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.uint32
    assert matrix.indptr.dtype == np.int32
    assert matrix.indices.dtype == np.int32
    assert matrix.has_sorted_indices
    assert matrix.nnz == 3
    np.testing.assert_array_equal(matrix.toarray(), COUNTS)


def test_matrix_to_dense_returns_ndarray(store: Path) -> None:
    dense = kyto.scan(store).layer("counts").to_dense(max_bytes=24)
    assert type(dense) is np.ndarray
    assert dense.dtype == np.uint32
    assert dense.flags.c_contiguous
    assert dense.flags.writeable
    np.testing.assert_array_equal(dense, COUNTS)


def test_float_layer(store: Path) -> None:
    """A computed ``float32`` layer comes back as ``float32``, sparse and dense."""
    atlas = kyto.scan(store).log1p(layer="counts", into="log1p")
    expected = np.log1p(COUNTS.astype(np.float32))
    matrix = _csr(atlas.layer("log1p").to_scipy())
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=1e-6)
    dense = atlas.layer("log1p").to_dense(max_bytes=24)
    assert type(dense) is np.ndarray
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense, matrix.toarray())


def test_stub_declares_one_return_type_per_terminal() -> None:
    """Every terminal of the stub returns the one type listed here: no ``Union`` or
    ``Optional`` beyond the dtype of an array (R3)."""
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    if not stub.exists():
        stub = Path(kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text())
    found: dict[tuple[str, str], str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "Table",
            "Matrix",
            "Embedding",
            "Graph",
            "Modality",
        }:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (
                    item.name == "lazy" or item.name.startswith("to_")
                ):
                    assert item.returns is not None
                    found[(node.name, item.name)] = ast.unparse(item.returns)
    assert found == STUB_RETURNS


# Plans with steps, collected plans, and edge cases.


def test_plan_with_steps(store: Path) -> None:
    atlas = _with_steps(store)
    cells = atlas.cells.to_polars()
    assert type(cells) is pl.DataFrame
    assert cells.columns == ["cell_id", "total"]
    assert cells["cell_id"].to_list() == ["AAAG-1"]
    assert cells["total"].to_list() == [7.0]
    assert atlas.genes.to_polars()["gene_id"].to_list() == [
        "ENSG01",
        "ENSG02",
        "ENSG03",
    ]
    matrix = _csr(atlas.layer("normalized").to_scipy())
    np.testing.assert_array_equal(matrix.toarray(), [[0.0, 10.0, 0.0]])
    # The size after a filter is known only once the plan ran: checked then.
    assert atlas.layer("normalized").estimate()["dense_bytes"] is None
    dense = atlas.layer("normalized").to_dense(max_bytes=12)
    np.testing.assert_array_equal(dense, [[0.0, 10.0, 0.0]])
    with pytest.raises(kyto.MemoryLimitError) as raised:
        atlas.layer("normalized").to_dense(max_bytes=11)
    assert raised.value.code == "KY-E402"
    assert raised.value.fields["required_bytes"] == 12
    assert raised.value.fields["n_rows"] == 1


def test_selected_genes(store: Path) -> None:
    """Genes come back in the order ``select_genes`` gave them, in the table and the
    matrix alike."""
    atlas = kyto.scan(store).select_genes(["ENSG03", "ENSG01"])
    assert atlas.genes.to_polars()["gene_id"].to_list() == ["ENSG03", "ENSG01"]
    expected = COUNTS[:, [2, 0]]
    np.testing.assert_array_equal(
        atlas.layer("counts").to_dense(max_bytes=16), expected
    )
    np.testing.assert_array_equal(
        _csr(atlas.layer("counts").to_scipy()).toarray(), expected
    )


def test_no_cells_left(store: Path) -> None:
    atlas = kyto.scan(store).filter(total_counts(layer="counts") > 100)
    cells = atlas.cells.to_polars()
    assert type(cells) is pl.DataFrame
    assert cells.height == 0
    matrix = _csr(atlas.layer("counts").to_scipy())
    assert matrix.shape == (0, 3)
    dense = atlas.layer("counts").to_dense(max_bytes=0)
    assert type(dense) is np.ndarray
    assert dense.shape == (0, 3)


def test_collected_atlas_keeps_its_store(store: Path) -> None:
    """Handles of a collected ``Atlas`` read its temporary store after the ``Atlas``
    itself is gone (7.2 Open (4))."""
    matrix = _with_steps(store).collect(threads=1).layer("normalized")
    cells = _with_steps(store).collect(threads=1).cells
    gc.collect()
    np.testing.assert_array_equal(matrix.to_dense(max_bytes=12), [[0.0, 10.0, 0.0]])
    assert cells.to_polars()["cell_id"].to_list() == ["AAAG-1"]


def test_results_are_copies(store: Path) -> None:
    """Changing a returned object changes no later read (R2, 7.5's Done-when)."""
    handle = kyto.scan(store).layer("counts")
    dense = handle.to_dense(max_bytes=24)
    dense[:] = 99
    matrix = _csr(handle.to_scipy())
    matrix.data[:] = 99
    frame = kyto.scan(store).cells.to_polars()
    frame = frame.with_columns(pl.lit("x").alias("cell_id"))
    np.testing.assert_array_equal(handle.to_dense(max_bytes=24), COUNTS)
    np.testing.assert_array_equal(_csr(handle.to_scipy()).toarray(), COUNTS)
    assert kyto.scan(store).cells.to_polars()["cell_id"].to_list() == [
        "AAAC-1",
        "AAAG-1",
    ]
    assert handle.to_dense(max_bytes=24) is not handle.to_dense(max_bytes=24)


# Hand-off (task 7.5): what is copied, what is not, and the dtypes.


def _rust_owned(array: npt.NDArray[np.generic]) -> bool:
    """Whether ``array`` views memory a Rust ``Vec`` filled (no copy in Python).

    The ``numpy`` crate hands a ``Vec`` over as an array whose base is its
    ``PySliceContainer``; SciPy may wrap that array in views, never a copy here.
    """
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    return type(base).__name__ == "PySliceContainer"


def test_frames_are_handed_over_without_a_copy() -> None:
    """Polars reads the Rust buffers themselves (Arrow C stream), dtypes kept."""
    frame, x, n = _kyto.handoff_example()
    assert type(frame) is pl.DataFrame
    assert frame.schema == pl.Schema({"x": pl.Float64, "n": pl.UInt32, "s": pl.String})
    assert frame["x"].to_numpy(allow_copy=False).ctypes.data == x
    assert frame["n"].to_numpy(allow_copy=False).ctypes.data == n
    assert frame["x"].sum() == sum(range(1000))
    assert frame["s"].unique().to_list() == ["a"]


def test_table_dtypes_after_the_handoff(store: Path) -> None:
    genes = kyto.scan(store).genes.to_polars()
    assert genes.dtypes[0] == pl.String
    assert genes.schema["symbol"] == pl.Enum(["CD3E", "CD4", "MS4A1"])
    cells = _with_steps(store).cells.to_polars()
    assert cells.schema == pl.Schema({"cell_id": pl.String, "total": pl.Float64})
    assert cells.rows() == [("AAAG-1", 7.0)]
    lazy = _with_steps(store).cells.lazy()
    assert lazy.collect().equals(cells)


def test_arrays_are_handed_over_without_a_copy(store: Path) -> None:
    """Every NumPy array (also inside SciPy) is the memory Rust filled, writeable."""
    layer = kyto.scan(store).layer("counts")
    matrix = _csr(layer.to_scipy())
    atlas = _with_graph(store)
    graph = _csr(atlas.graph("distances").to_scipy())
    arrays: list[npt.NDArray[np.generic]] = [
        layer.to_dense(max_bytes=24),
        atlas.embedding("X_pca").to_numpy(),
        matrix.data,
        matrix.indices,
        matrix.indptr,
        graph.data,
        graph.indices,
        graph.indptr,
    ]
    for array in arrays:
        assert _rust_owned(array), array
        assert array.flags.writeable and array.flags.c_contiguous
    assert [array.dtype for array in arrays] == [
        np.uint32,
        np.float32,
        np.uint32,
        np.int32,
        np.int32,
        np.float32,
        np.int32,
        np.int32,
    ]


def test_mutating_any_result_changes_no_later_read(store: Path) -> None:
    """R2 for every terminal (7.5's Done-when, until 7.4's ``pull``)."""
    atlas = _with_graph(store)
    embedding = atlas.embedding("X_pca").to_numpy()
    expected = embedding.copy()
    embedding[:] = 99
    graph = _csr(atlas.graph("distances").to_scipy())
    expected_graph = graph.toarray()
    graph.data[:] = 99
    graph.indices[:] = 0
    matrix = _csr(kyto.scan(store).layer("counts").to_scipy())
    matrix.indptr[:] = 0
    np.testing.assert_array_equal(atlas.embedding("X_pca").to_numpy(), expected)
    np.testing.assert_array_equal(
        _csr(atlas.graph("distances").to_scipy()).toarray(), expected_graph
    )
    np.testing.assert_array_equal(
        _csr(kyto.scan(store).layer("counts").to_scipy()).toarray(), COUNTS
    )
    # Polars frames are immutable; their NumPy views are read-only.
    edges = atlas.graph("distances").to_edges()
    assert not edges["weight"].to_numpy(allow_copy=False).flags.writeable
    collected = atlas.collect()
    first = collected.embedding("X_pca").to_numpy()
    first[:] = -1
    np.testing.assert_array_equal(collected.embedding("X_pca").to_numpy(), expected)


def test_mutating_a_returned_array_never_changes_a_later_pull(store: Path) -> None:
    """7.5's Done-when, with 7.4's ``pull`` and ``select``."""
    atlas = kyto.scan(store).with_default_layer("counts")
    dense = atlas.layer("counts").to_dense(max_bytes=24)
    dense[:] = 99
    matrix = _csr(atlas.layer("counts").to_scipy())
    matrix.data[:] = 99
    pulled = atlas.pull("ENSG01")
    view = pulled.to_numpy(allow_copy=False)
    assert not view.flags.writeable  # a frame's buffer cannot be changed through NumPy
    writable = pulled.to_numpy(writable=True)
    writable[:] = 99
    assert atlas.pull("ENSG01").to_list() == [5, 0]
    assert atlas.pull("ENSG01").dtype == pl.UInt32
    assert atlas.select("ENSG01", "ENSG03").rows() == [
        ("AAAC-1", 5, 1),
        ("AAAG-1", 0, 0),
    ]


def test_rank_genes_groups_dtypes_after_the_handoff(tmp_path: Path) -> None:
    """The DE frame reaches Polars with the dtypes its deferred schema declares."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n2 4 6\n"
        "1 1 5\n2 1 1\n1 2 6\n1 3 1\n2 3 4\n2 4 5\n"
    )
    (source / "barcodes.tsv").write_text("A-1\nB-1\nC-1\nD-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n")
    )
    store = tmp_path / "four.kyto"
    kyto.convert(source, store, layer="counts")
    first = gene_column("symbol") == "CD3E"
    lazy = (
        kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(cd3e=total_counts_in(first, layer="counts") > 2)
        .rank_genes_groups("cd3e")
    )
    frame = lazy.collect()
    assert frame.schema == lazy.collect_schema()
    assert frame.schema == pl.Schema(
        {
            "group": pl.String,
            "gene_id": pl.String,
            "statistic": pl.Float64,
            "log_fold_change": pl.Float64,
            "p_value": pl.Float64,
            "adjusted_p_value": pl.Float64,
        }
    )
    assert sorted(frame["group"].unique().to_list()) == ["false", "true"]
    assert frame.height == 4


def test_progress_and_threads(store: Path) -> None:
    calls: list[tuple[int, int]] = []
    frame = _with_steps(store).cells.to_polars(
        threads=1, progress=lambda done, total: calls.append((done, total))
    )
    assert frame.height == 1
    assert calls and calls[-1][0] == calls[-1][1]
    # A plain scan runs nothing: no progress, but the arguments are still checked.
    plain: list[tuple[int, int]] = []
    kyto.scan(store).cells.to_polars(progress=lambda d, t: plain.append((d, t)))
    assert plain == []


# Errors.


@pytest.mark.parametrize("max_bytes", [-1, 1.5, True, "24", None])
def test_to_dense_max_bytes_is_checked(store: Path, max_bytes: object) -> None:
    with pytest.raises(kyto.PlanError) as raised:
        kyto.scan(store).layer("counts").to_dense(max_bytes=max_bytes)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert raised.value.code == "KY-E215"
    assert raised.value.fields["parameter"] == "max_bytes"


def test_to_dense_too_large_before_running(store: Path) -> None:
    """A size known from metadata is refused before the plan runs."""
    atlas = kyto.scan(store).log1p(layer="counts", into="log1p")
    with pytest.raises(kyto.MemoryLimitError) as raised:
        atlas.layer("log1p").to_dense(max_bytes=23, progress=lambda d, t: 1 / 0)
    assert isinstance(raised.value, MemoryError)
    assert raised.value.code == "KY-E402"
    assert raised.value.fields == {
        "method": "Matrix.to_dense",
        "n_rows": 2,
        "n_columns": 3,
        "required_bytes": 24,
        "problem": "max_bytes is 23",
    }
    assert "to_scipy()" in str(raised.value)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("threads", 0), ("memory_limit", "lots"), ("progress", 3)],
)
def test_run_arguments_are_checked(store: Path, argument: str, value: object) -> None:
    for atlas in (kyto.scan(store), _with_steps(store)):
        with pytest.raises(kyto.PlanError) as raised:
            atlas.cells.to_polars(**{argument: value})  # pyright: ignore[reportArgumentType]
        assert raised.value.code == "KY-E215"
        assert raised.value.fields["function"] == "Table.to_polars"


def _with_graph(store: Path) -> _kyto.Atlas:
    """A plan with an embedding (``pca``) and its neighbour graphs."""
    return (
        kyto.scan(store)
        .log1p(layer="counts", into="log1p")
        .pca(layer="log1p", into="X_pca", n_components=1, seed=0)
        .neighbors("X_pca", n_neighbors=2)
    )


def test_embedding_to_numpy_returns_ndarray(store: Path) -> None:
    atlas = _with_graph(store)
    array = atlas.embedding("X_pca").to_numpy(threads=1)
    assert type(array) is np.ndarray
    assert array.shape == (2, 1)
    assert array.dtype == np.float32
    assert array.flags.c_contiguous and array.flags.writeable
    # Two cells: the one component separates them, centred.
    np.testing.assert_allclose(array[:, 0].sum(), 0.0, atol=1e-6)
    assert array[0, 0] != array[1, 0]
    # The same values from a collected Atlas (read in place).
    collected = atlas.collect(threads=1).embedding("X_pca").to_numpy()
    np.testing.assert_array_equal(collected, array)


def test_embedding_to_torch(store: Path) -> None:
    handle = _with_graph(store).embedding("X_pca")
    try:
        import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            handle.to_torch(device="cpu")
        return
    tensor = handle.to_torch(device="cpu")
    assert type(tensor).__name__ == "Tensor"
    assert tuple(tensor.shape) == (2, 1)


def test_graph_terminals(store: Path) -> None:
    atlas = _with_graph(store)
    for name in ("distances", "connectivities"):
        graph = _csr(atlas.graph(name).to_scipy())
        assert graph.shape == (2, 2)
        assert graph.dtype == np.float32
        edges = atlas.graph(name).to_edges()
        assert type(edges) is pl.DataFrame
        assert edges.columns == atlas.graph(name).names()
        assert edges.schema == pl.Schema(
            {"source": pl.UInt32, "target": pl.UInt32, "weight": pl.Float32}
        )
        assert edges.height == graph.nnz
        # The edge list is the CSR row by row.
        dense = np.zeros((2, 2), dtype=np.float32)
        dense[edges["source"].to_numpy(), edges["target"].to_numpy()] = edges[
            "weight"
        ].to_numpy()
        np.testing.assert_array_equal(dense, graph.toarray())
    distances = _csr(atlas.graph("distances").to_scipy()).toarray()
    assert distances[0, 0] == 0 and distances[0, 1] == distances[1, 0] > 0


def test_graph_after_a_cell_filter(store: Path) -> None:
    """A graph past a cell filter is re-indexed to the kept cells (5.11)."""
    atlas = _with_graph(store).filter(total_counts(layer="counts") >= 7)
    assert atlas.cells.to_polars()["cell_id"].to_list() == ["AAAG-1"]
    graph = _csr(atlas.graph("distances").to_scipy())
    assert graph.shape == (1, 1)
    assert atlas.embedding("X_pca").to_numpy().shape == (1, 1)


def test_store_changed_after_scan(store: Path, tmp_path: Path) -> None:
    """A plain scan is read in place, but never as a store replaced since the scan
    (the plan answers from its snapshot; bug-hunter)."""
    atlas = kyto.scan(store)
    source = tmp_path / "other"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 3\n"
    )
    (source / "barcodes.tsv").write_text("A-1\nB-1\n")
    (source / "features.tsv").write_text(
        "G1\tX\tGene Expression\nG2\tY\tGene Expression\n"
    )
    kyto.convert(source, store, layer="counts", overwrite=True)
    for call in (atlas.cells.to_polars, atlas.layer("counts").to_scipy):
        with pytest.raises(kyto.CorruptStoreError) as raised:
            call()
        assert raised.value.code == "KY-E127"


# Sinks: overwrite and sink_h5ad.


def test_sink_zarr_overwrite(store: Path, tmp_path: Path) -> None:
    target = tmp_path / "out.kyto"
    kyto.scan(store).sink_zarr(target)
    with pytest.raises(kyto.StoreInputError) as raised:
        _with_steps(store).sink_zarr(target)
    assert raised.value.code == "KY-E118"
    _with_steps(store).sink_zarr(target, overwrite=True)
    written = kyto.scan(target)
    assert written.cells.to_polars()["cell_id"].to_list() == ["AAAG-1"]
    assert written.cells.names() == ["cell_id", "total"]
    # Never over the store the plan reads, overwrite or not.
    with pytest.raises(kyto.StoreInputError) as raised:
        kyto.scan(store).sink_zarr(store, overwrite=True)
    assert raised.value.code == "KY-E118"
    np.testing.assert_array_equal(
        kyto.scan(store).layer("counts").to_dense(max_bytes=24), COUNTS
    )


def test_sink_h5ad(store: Path, tmp_path: Path) -> None:
    target = tmp_path / "out.h5ad"
    atlas = _with_steps(store).with_default_layer("normalized")
    assert atlas.sink_h5ad(target) is None
    assert target.is_file()
    with pytest.raises(kyto.StoreInputError) as raised:
        atlas.sink_h5ad(target)
    assert raised.value.code == "KY-E123"
    kyto.scan(store).with_default_layer("counts").sink_h5ad(target, overwrite=True)
    anndata = pytest.importorskip("anndata")
    adata = anndata.read_h5ad(target)
    assert list(adata.obs_names) == ["AAAC-1", "AAAG-1"]
    assert list(adata.var_names) == ["ENSG01", "ENSG02", "ENSG03"]
    np.testing.assert_array_equal(adata.X.toarray(), COUNTS)


def test_sink_h5ad_of_steps_matches_terminals(store: Path, tmp_path: Path) -> None:
    anndata = pytest.importorskip("anndata")
    target = tmp_path / "steps.h5ad"
    atlas = _with_steps(store).with_default_layer("normalized")
    atlas.sink_h5ad(target, threads=1)
    adata = anndata.read_h5ad(target)
    assert list(adata.obs_names) == atlas.cells.to_polars()["cell_id"].to_list()
    assert adata.obs["total"].to_list() == [7.0]
    np.testing.assert_array_equal(
        adata.X.toarray(), atlas.layer("normalized").to_dense(max_bytes=12)
    )
    np.testing.assert_array_equal(
        adata.layers["counts"].toarray(), atlas.layer("counts").to_dense(max_bytes=12)
    )
