"""Task 2.9 Done when: two fixtures combined equal ``anndata.concat``.

Each fixture is split into two samples (cells in halves, overlapping gene subsets in
different orders; the synthetic samples also differ in cell columns, categories and
one gene column). ``kyto.concat`` combines them (one ``.h5ad`` input, one kyto store),
the result is exported with the private ``kyto._kyto.sink_h5ad`` and read back by
anndata, and must equal ``anndata.concat([a, b], label="sample", keys=["a", "b"],
join=join, merge="same")`` on the same two files: shape, cell and gene tables, ``X``
and the layers. Differences are exactly the rows of ``deviations.md``. Skips where
anndata has no wheels.

What it does not: measure memory (``crates/kyto-store/tests/concat_memory.rs``) or
test remapping mechanics in isolation (``crates/kyto-store/tests/it/concat.rs``).
"""

# pandas, scipy and anndata ship without complete type information; strict pyright
# cannot see their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any, Literal

import kyto
import numpy as np
import pytest
from fixtures.registry import FIXTURE_NAMES, FixtureName, load_fixture
from kyto import _kyto

anndata = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

Join = Literal["inner", "outer"]


def _base(name: FixtureName) -> Any:
    """The fixture as anndata holds it; synthetic gets every column encoding."""
    from fixtures.import_inputs import (
        _with_extra_columns,  # pyright: ignore[reportPrivateUsage]
    )
    from fixtures.interop import to_anndata

    adata: Any = to_anndata(load_fixture(name))
    if name == "synthetic":
        adata = _with_extra_columns(adata)
        adata.layers["counts_int"] = adata.X.astype(np.int32)
        adata.layers["dense_float64"] = adata.X.toarray().astype(np.float64)
    return adata


def _samples(name: FixtureName, folder: Path) -> tuple[Path, Path]:
    """Write samples ``a.h5ad`` and ``b.h5ad`` of the fixture: halves of the cells,
    genes without every 5th (a) or every 7th (b, in reverse order)."""
    base = _base(name)
    n_cells, n_genes = base.shape
    half = n_cells // 2
    genes = np.arange(n_genes)
    a = base[:half, genes[genes % 5 != 0]].copy()
    b = base[half:, genes[genes % 7 != 0][::-1]].copy()
    if name == "synthetic":
        del b.obs["float32"]
        b.obs["only_b"] = np.arange(b.n_obs, dtype=np.int32)
        a.obs["donor"] = pd.Categorical(
            [["d10", "d2"][row % 2] for row in range(a.n_obs)], categories=["d10", "d2"]
        )
        b.obs["donor"] = pd.Categorical(
            [["d1", "d2"][row % 2] for row in range(b.n_obs)], categories=["d2", "d1"]
        )
        # Not equal across samples: anndata's merge="same" drops it.
        b.var["dispersion"] = b.var["dispersion"] + 1
    paths = folder / "a.h5ad", folder / "b.h5ad"
    a.write_h5ad(paths[0])
    b.write_h5ad(paths[1])
    return paths


def _export(store: Path, folder: Path) -> Any:
    target = folder / "export.h5ad"
    _kyto.sink_h5ad(str(store), str(target), x_layer="counts", overwrite=True)
    return anndata.read_h5ad(target)


def _assert_csr_equal(actual: Any, expected: Any, dtype: str) -> None:
    """Exact CSR equality after canonicalising ``expected`` (sorted, no zeros)."""
    assert isinstance(actual, sparse.csr_matrix)
    assert actual.dtype == np.dtype(dtype)
    want = sparse.csr_matrix(expected, copy=True)
    want.sum_duplicates()
    want.eliminate_zeros()
    want.sort_indices()
    assert actual.shape == want.shape
    np.testing.assert_array_equal(actual.indptr, want.indptr)
    np.testing.assert_array_equal(actual.indices, want.indices)
    np.testing.assert_array_equal(actual.data, want.data.astype(dtype))


def _values(series: Any) -> list[Any]:
    """Values with every missing value (None, NaN, NA) as None."""
    return [None if pd.isna(value) else value for value in series.astype(object)]


def _assert_frames_equal(actual: Any, expected: Any) -> None:
    assert list(actual.columns) == list(expected.columns)
    assert list(actual.index) == list(expected.index)
    for name in expected.columns:
        want, got = expected[name], actual[name]
        if isinstance(want.dtype, pd.CategoricalDtype):
            if want.cat.categories.dtype.kind in "iuf":
                # deviations.md "Numeric categoricals": a plain numeric column.
                want = want.astype(want.cat.categories.dtype)
            else:
                assert isinstance(got.dtype, pd.CategoricalDtype), name
                assert list(got.cat.categories) == list(want.cat.categories), name
                assert got.cat.ordered == want.cat.ordered, name
                assert _values(got) == _values(want), name
                continue
        if isinstance(got.dtype, pd.CategoricalDtype) and not isinstance(
            want.dtype, pd.CategoricalDtype
        ):
            # Text columns are stored as categoricals (kyto.convert).
            assert _values(got) == _values(want), name
            continue
        if want.dtype == np.float64 and pd.api.types.is_integer_dtype(got.dtype):
            # deviations.md "Concatenated integer columns with missing values".
            assert got.isna().any(), name
            assert _values(got) == _values(want), name
            continue
        pd.testing.assert_series_equal(
            got, want, check_exact=True, check_index=False, obj=name
        )


def _anndata_concat(adatas: list[Any], **arguments: Any) -> Any:
    """``anndata.concat(..., merge="same")`` comparing gene columns by value.

    deviations.md "Gene columns kept by value": anndata compares categorical gene
    columns with their categories, so samples whose categoricals hold the same values
    but other categories lose the column; kyto keeps it. The gene categoricals are
    compared as text here.
    """
    copies = [adata.copy() for adata in adatas]
    for adata in copies:
        for name in adata.var.columns:
            if isinstance(adata.var[name].dtype, pd.CategoricalDtype):
                adata.var[name] = adata.var[name].astype(object)
    return anndata.concat(copies, merge="same", **arguments)


def _assert_combined_equal(actual: Any, expected: Any) -> None:
    assert actual.shape == expected.shape
    _assert_frames_equal(actual.obs, expected.obs)
    _assert_frames_equal(actual.var, expected.var)
    _assert_csr_equal(actual.X, expected.X, "float32")
    keys = sorted(key for key in expected.layers if key is not None)
    assert sorted(key for key in actual.layers if key is not None) == keys
    for key in keys:
        want = expected.layers[key]
        if isinstance(want, np.ndarray):
            # deviations.md "Outer join of dense layers": absent values are 0, not NaN.
            want = np.nan_to_num(want, nan=0.0)
        integer = np.issubdtype(want.dtype, np.integer)
        _assert_csr_equal(actual.layers[key], want, "uint32" if integer else "float32")


@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_concat_equals_anndata(tmp_path: Path, name: FixtureName, join: Join) -> None:
    a_path, b_path = _samples(name, tmp_path)
    b_store = tmp_path / "b.kyto"
    kyto.convert(b_path, b_store)
    target = tmp_path / "combined.kyto"
    # Mixed sources: an input and a store; default keys are the names "a" and "b".
    kyto.concat([a_path, b_store], target, join=join)

    a, b = anndata.read_h5ad(a_path), anndata.read_h5ad(b_path)
    expected = _anndata_concat([a, b], label="sample", keys=["a", "b"], join=join)
    _assert_combined_equal(_export(target, tmp_path), expected)
    if name == "synthetic":
        # The test exercises what it claims: genes and columns differ by join.
        if join == "outer":
            assert expected.n_vars > max(a.n_vars, b.n_vars)
        else:
            assert expected.n_vars < min(a.n_vars, b.n_vars)
        assert ("only_b" in expected.obs) == (join == "outer")
        assert ("symbol" in expected.var) == (join == "inner")
        assert "dispersion" not in expected.var


def test_duplicate_cells_suffix_or_error(tmp_path: Path) -> None:
    a_path, _ = _samples("synthetic", tmp_path)
    target = tmp_path / "twice.kyto"
    with pytest.raises(kyto.DuplicateIdError) as duplicate:
        kyto.concat([a_path, a_path], target, keys=["x", "y"])
    assert duplicate.value.code == "KY-E101"
    assert not target.exists()

    kyto.concat([a_path, a_path], target, keys=["x", "y"], on_duplicate_ids="suffix")
    a = anndata.read_h5ad(a_path)
    expected = _anndata_concat([a, a], label="batch_key", keys=["x", "y"])
    expected.obs = expected.obs.rename(columns={"batch_key": "sample"})
    expected.obs_names_make_unique()
    _assert_combined_equal(_export(target, tmp_path), expected)


def test_errors_are_coded(tmp_path: Path) -> None:
    a_path, b_path = _samples("synthetic", tmp_path)
    target = tmp_path / "out.kyto"
    cases: list[tuple[dict[str, Any], str, str]] = [
        ({"keys": ["only one"]}, "KY-E104", "one key per source"),
        ({"keys": ["s", "s"]}, "KY-E104", "different key"),
        ({"join": "left"}, "KY-E104", "'inner' or 'outer'"),
        ({"label": "cell_id"}, "KY-E104", "other than 'cell_id'"),
    ]
    for arguments, code, text in cases:
        with pytest.raises(kyto.StoreInputError) as error:
            kyto.concat([a_path, b_path], target, **arguments)
        assert error.value.code == code
        assert text in str(error.value)
    with pytest.raises(kyto.StoreInputError) as empty:
        kyto.concat([], target)
    assert empty.value.code == "KY-E104"
    # Two sources named alike need keys.
    (tmp_path / "other").mkdir()
    same_name = tmp_path / "other" / "a.h5ad"
    same_name.write_bytes(a_path.read_bytes())
    with pytest.raises(kyto.StoreInputError) as repeated:
        kyto.concat([a_path, same_name], target)
    assert "pass keys" in str(repeated.value)
    # The target may not be a source.
    store = tmp_path / "a.kyto"
    kyto.convert(a_path, store)
    # No layer name in common (the input's main matrix named "raw"): no silent store
    # without a matrix.
    plain = anndata.read_h5ad(b_path)
    plain.layers.clear()
    plain_path = tmp_path / "plain.h5ad"
    plain.write_h5ad(plain_path)
    plain_store = tmp_path / "plain.kyto"
    kyto.convert(plain_path, plain_store)
    with pytest.raises(kyto.StoreInputError) as no_layer:
        kyto.concat(
            [plain_store, plain_path],
            target,
            keys=["s", "i"],
            layer="raw",
            on_duplicate_ids="suffix",
        )
    assert no_layer.value.code == "KY-E104"
    assert "layer" in str(no_layer.value)
    with pytest.raises(kyto.StoreInputError) as is_source:
        kyto.concat([store, b_path], store, overwrite=True)
    assert is_source.value.code == "KY-E118"
    assert not target.exists()


def test_empty_source_and_single_key(tmp_path: Path) -> None:
    a_path, _ = _samples("synthetic", tmp_path)
    a = anndata.read_h5ad(a_path)
    # A source without cells whose column has another dtype adds no values.
    empty = a[:0].copy()
    empty.obs["int16"] = empty.obs["int16"].astype(str)
    empty_path = tmp_path / "empty.h5ad"
    empty.write_h5ad(empty_path)
    target = tmp_path / "with_empty.kyto"
    kyto.concat([a_path, empty_path], target)
    expected = _anndata_concat([a, a[:0].copy()], label="sample", keys=["a", "empty"])
    _assert_combined_equal(_export(target, tmp_path), expected)
    # A bare string is one key (here one source), not one key per character.
    single = tmp_path / "single.kyto"
    kyto.concat([a_path], single, keys="ab")
    exported = _export(single, tmp_path)
    assert list(exported.obs["sample"].cat.categories) == ["ab"]
