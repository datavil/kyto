"""``kyto.convert`` from Python: arguments, duplicate IDs, empty inputs, coded errors.

Values against anndata for every fixture and input format are checked by the Rust test
``crates/kyto-store/tests/import_parity.rs`` (it reads the store with the tile reader).
These tests cover the Python surface: the doctest, ``on_duplicate_ids``, 0-cell and
0-gene inputs, and the coded error of each failure. Tests that write ``.h5ad`` files
need anndata or h5py and skip without them.
"""

# h5py, pandas and anndata ship without complete type information; strict pyright
# cannot see their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import doctest
import gzip
import json
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import polars as pl
import pytest
from kyto import _convert


def _mtx_dir(
    folder: Path,
    entries: list[tuple[int, int, str]],
    barcodes: list[str],
    gene_ids: list[str],
    field: str = "integer",
    gzipped: bool = False,
) -> Path:
    """Write a 10x matrix directory (genes x cells, 1-based entries)."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"%%MatrixMarket matrix coordinate {field} general", "% a comment"]
    lines.append(f"{len(gene_ids)} {len(barcodes)} {len(entries)}")
    lines += [f"{gene} {cell} {value}" for gene, cell, value in entries]
    texts = {
        "matrix.mtx": "\n".join(lines) + "\n",
        "barcodes.tsv": "".join(f"{barcode}\n" for barcode in barcodes),
        "features.tsv": "".join(
            f"{gene_id}\tSYM{position}\tGene Expression\n"
            for position, gene_id in enumerate(gene_ids)
        ),
    }
    for name, text in texts.items():
        if gzipped:
            (folder / f"{name}.gz").write_bytes(gzip.compress(text.encode()))
        else:
            (folder / name).write_text(text)
    return folder


def _schema(store: Path) -> dict[str, Any]:
    return json.loads((store / "schema.json").read_text())


def _error(raised: pytest.ExceptionInfo[kyto.KytoError], code: str) -> str:
    assert raised.value.code == code
    return str(raised.value)


def test_docstring_example_runs() -> None:
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for test in finder.find(_convert.convert, "convert", globs={}):
        runner.run(test)
    assert runner.failures == 0
    assert runner.tries > 0


def test_mtx_values_tables_and_layer_name(tmp_path: Path) -> None:
    source = _mtx_dir(
        tmp_path / "in",
        [(3, 2, "4"), (1, 1, "2"), (1, 1, "3"), (2, 2, "0")],
        ["c1", "c2"],
        ["g1", "g2", "g3"],
        gzipped=True,
    )
    store = tmp_path / "out.kyto"
    kyto.convert(source / "matrix.mtx.gz", store, layer="raw")
    schema = _schema(store)
    # The repeated (1, 1) entry is summed and the explicit 0 dropped, as scipy does.
    assert schema["layers"] == [{"name": "raw", "dtype": "uint32", "nnz": 2}]
    genes = pl.read_parquet(store / "genes.parquet")
    assert genes.columns == ["gene_id", "symbol", "feature_type"]
    assert genes["gene_id"].to_list() == ["g1", "g2", "g3"]
    provenance = json.loads((store / "provenance.json").read_text())
    assert provenance["source"]["kind"] == "mtx"


def test_duplicate_ids_error_before_writing(tmp_path: Path) -> None:
    source = _mtx_dir(tmp_path / "in", [(1, 1, "1")], ["a", "a", "b"], ["g1"])
    store = tmp_path / "out.kyto"
    with pytest.raises(kyto.DuplicateIdError) as raised:
        kyto.convert(source, store)
    message = _error(raised, "KY-E101")
    assert "cells" in message and "'a'" in message
    assert not store.exists()


def test_duplicate_ids_suffix_matches_anndata(tmp_path: Path) -> None:
    barcodes = ["a", "a", "b", "a-1", "a"]
    source = _mtx_dir(tmp_path / "in", [(1, 1, "1")], barcodes, ["g", "g"])
    store = tmp_path / "out.kyto"
    kyto.convert(source, store, on_duplicate_ids="suffix")
    cell_ids = pl.read_parquet(store / "cells.parquet")["cell_id"].to_list()
    gene_ids = pl.read_parquet(store / "genes.parquet")["gene_id"].to_list()
    # anndata.utils.make_index_unique gives exactly these (checked below if installed).
    assert cell_ids == ["a", "a-2", "b", "a-1", "a-3"]
    assert gene_ids == ["g", "g-1"]
    anndata_utils = pytest.importorskip("anndata.utils")
    import pandas as pd

    expected = anndata_utils.make_index_unique(pd.Index(barcodes, dtype=object))
    assert cell_ids == list(expected)


@pytest.mark.parametrize(("n_cells", "n_genes"), [(0, 3), (2, 0), (0, 0)])
def test_empty_mtx_inputs(tmp_path: Path, n_cells: int, n_genes: int) -> None:
    barcodes = [f"c{position}" for position in range(n_cells)]
    gene_ids = [f"g{position}" for position in range(n_genes)]
    source = _mtx_dir(tmp_path / "in", [], barcodes, gene_ids)
    store = tmp_path / "out.kyto"
    kyto.convert(source, store)
    schema = _schema(store)
    assert (schema["n_cells"], schema["n_genes"]) == (n_cells, n_genes)
    assert schema["layers"] == [{"name": "counts", "dtype": "uint32", "nnz": 0}]


@pytest.mark.parametrize(("n_cells", "n_genes"), [(0, 3), (2, 0)])
def test_empty_h5ad_inputs(tmp_path: Path, n_cells: int, n_genes: int) -> None:
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    path = tmp_path / "empty.h5ad"
    adata = anndata.AnnData(
        csr_matrix((n_cells, n_genes), dtype=np.float32),
        obs=pd.DataFrame(index=[f"c{position}" for position in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{position}" for position in range(n_genes)]),
    )
    adata.write_h5ad(path)
    store = tmp_path / "out.kyto"
    kyto.convert(path, store)
    schema = _schema(store)
    assert (schema["n_cells"], schema["n_genes"]) == (n_cells, n_genes)
    assert schema["layers"] == [{"name": "counts", "dtype": "float32", "nnz": 0}]


def test_overwrite_and_existing_target(tmp_path: Path) -> None:
    source = _mtx_dir(tmp_path / "in", [(1, 1, "1")], ["c"], ["g"])
    store = tmp_path / "out.kyto"
    kyto.convert(source, store)
    with pytest.raises(kyto.StoreInputError) as raised:
        kyto.convert(source, store)
    _error(raised, "KY-E118")
    kyto.convert(source, store, overwrite=True)
    assert _schema(store)["n_cells"] == 1


@pytest.mark.parametrize(
    ("entries", "field", "location"),
    [
        ([(1, 1, "-3")], "integer", "cell 0, gene 0"),
        ([(1, 1, "2.5")], "integer", "line 4"),
        ([(4, 1, "1")], "integer", "line 4"),
        ([(1, 1, "x")], "real", "line 4"),
    ],
)
def test_invalid_mtx_values(
    tmp_path: Path, entries: list[tuple[int, int, str]], field: str, location: str
) -> None:
    source = _mtx_dir(tmp_path / "in", entries, ["c"], ["g1", "g2", "g3"], field)
    store = tmp_path / "out.kyto"
    with pytest.raises(kyto.InputFileError) as raised:
        kyto.convert(source, store)
    assert isinstance(raised.value, ValueError)
    assert location in _error(raised, "KY-E121")
    assert not store.exists()


def test_unsupported_inputs(tmp_path: Path) -> None:
    other = tmp_path / "counts.csv"
    other.write_text("a,b\n")
    with pytest.raises(kyto.InputFileError) as raised:
        kyto.convert(other, tmp_path / "out.kyto")
    assert ".h5ad" in _error(raised, "KY-E120")
    symmetric = _mtx_dir(tmp_path / "symmetric", [], ["c"], ["g"])
    text = (symmetric / "matrix.mtx").read_text().replace("general", "symmetric")
    (symmetric / "matrix.mtx").write_text(text)
    with pytest.raises(kyto.InputFileError) as raised:
        kyto.convert(symmetric, tmp_path / "out.kyto")
    _error(raised, "KY-E120")
    not_hdf5 = tmp_path / "broken.h5ad"
    not_hdf5.write_text("not hdf5")
    with pytest.raises(kyto.InputFileError) as raised:
        kyto.convert(not_hdf5, tmp_path / "out.kyto")
    _error(raised, "KY-E120")


def test_missing_source_and_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(kyto.StoreIOError) as raised:
        kyto.convert(tmp_path / "missing.h5ad", tmp_path / "out.kyto")
    _error(raised, "KY-E110")
    source = _mtx_dir(tmp_path / "in", [], ["c"], ["g"])
    with pytest.raises(kyto.StoreInputError) as raised:
        kyto.convert(source, tmp_path / "out.kyto", on_duplicate_ids="rename")  # pyright: ignore[reportArgumentType]
    _error(raised, "KY-E104")
    with pytest.raises(kyto.StoreInputError) as raised:
        kyto.convert(source, tmp_path / "out.kyto", layer="a/b")
    _error(raised, "KY-E102")


def _h5ad_with_x(path: Path, write_x: Any) -> Path:
    """An .h5ad from anndata whose X h5py then replaces with ``write_x(file)``."""
    anndata = pytest.importorskip("anndata")
    h5py = pytest.importorskip("h5py")
    import pandas as pd

    adata = anndata.AnnData(
        np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    adata.write_h5ad(path)
    with h5py.File(path, "r+") as file:
        del file["X"]
        write_x(file)
    return path


def test_unsupported_h5ad_encodings(tmp_path: Path) -> None:
    def awkward(file: Any) -> None:
        group = file.create_group("X")
        group.attrs["encoding-type"] = "awkward-array"
        group.attrs["encoding-version"] = "0.1.0"

    def future_version(file: Any) -> None:
        dataset = file.create_dataset("X", data=np.ones((2, 3), np.float32))
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "9.0.0"

    def no_encoding(file: Any) -> None:
        file.create_dataset("X", data=np.ones((2, 3), np.float32))

    def strings(file: Any) -> None:
        dataset = file.create_dataset("X", data=np.full((2, 3), b"x"))
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"

    for position, write_x in enumerate([awkward, future_version, no_encoding, strings]):
        path = _h5ad_with_x(tmp_path / f"bad{position}.h5ad", write_x)
        with pytest.raises(kyto.InputFileError) as raised:
            kyto.convert(path, tmp_path / "out.kyto")
        assert f"{path}:/X" in _error(raised, "KY-E120")
    assert not (tmp_path / "out.kyto").exists()


def test_invalid_h5ad_values(tmp_path: Path) -> None:
    def negative_integers(file: Any) -> None:
        values = np.array([[1, 0, -2], [0, 0, 0]], dtype=np.int32)
        dataset = file.create_dataset("X", data=values)
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"

    def index_out_of_range(file: Any) -> None:
        group = file.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.array([2, 3])
        group["indptr"] = np.array([0, 1, 1])
        group["indices"] = np.array([5], dtype=np.int32)
        group["data"] = np.array([1.0], dtype=np.float32)

    def wrong_shape(file: Any) -> None:
        dataset = file.create_dataset("X", data=np.ones((3, 2), np.float32))
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"

    def beyond_float32(file: Any) -> None:
        values = np.array([[1.0, 0, 1e300], [0, 0, 0]], dtype=np.float64)
        dataset = file.create_dataset("X", data=values)
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"

    cases = [
        (negative_integers, "cell 0, gene 2"),
        (beyond_float32, "cell 0, gene 2"),
        (index_out_of_range, "cell 0, entry 0"),
        (wrong_shape, "[3, 2]"),
    ]
    for position, (write_x, needle) in enumerate(cases):
        path = _h5ad_with_x(tmp_path / f"bad{position}.h5ad", write_x)
        with pytest.raises(kyto.InputFileError) as raised:
            kyto.convert(path, tmp_path / "out.kyto")
        assert needle in _error(raised, "KY-E121")
    assert not (tmp_path / "out.kyto").exists()


def test_h5ad_duplicate_gene_ids(tmp_path: Path) -> None:
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    path = tmp_path / "dup.h5ad"
    adata = anndata.AnnData(
        np.eye(2, 3, dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g", "g", "h"]),
    )
    adata.write_h5ad(path)
    with pytest.raises(kyto.DuplicateIdError) as raised:
        kyto.convert(path, tmp_path / "out.kyto")
    assert "genes" in _error(raised, "KY-E101")
    kyto.convert(path, tmp_path / "out.kyto", on_duplicate_ids="suffix")
    gene_ids = pl.read_parquet(tmp_path / "out.kyto" / "genes.parquet")["gene_id"]
    adata.var_names_make_unique()
    assert gene_ids.to_list() == list(adata.var_names)


def test_invalid_utf8_ids(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    def keep_x(file: Any) -> None:
        file.create_dataset("X", data=np.ones((2, 3), np.float32))
        file["X"].attrs["encoding-type"] = "array"
        file["X"].attrs["encoding-version"] = "0.2.0"

    path = _h5ad_with_x(tmp_path / "utf8.h5ad", keep_x)
    with h5py.File(path, "r+") as file:
        del file["obs/_index"]
        dataset = file["obs"].create_dataset(
            "_index", data=[b"ok", b"\xff\xfe"], dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["encoding-type"] = "string-array"
        dataset.attrs["encoding-version"] = "0.2.0"
        file["obs"].attrs["_index"] = "_index"
    with pytest.raises(kyto.InputFileError) as raised:
        kyto.convert(path, tmp_path / "out.kyto")
    assert "element 1" in _error(raised, "KY-E121")
    assert not (tmp_path / "out.kyto").exists()
