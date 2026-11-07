"""Task 2.7 Done when: ``anndata.read_h5ad`` on the export equals the source fixture.

Each fixture (synthetic, with one ``obs``/``var`` column of every encoding anndata writes
and two extra layers; pbmc3k) is written by anndata, converted to a store
(``kyto.convert``, 2.6), exported with the private ``kyto._kyto.sink_h5ad`` and read
back by anndata. The export must equal what anndata reads from the source file: same
shape, index, columns and values; differences are exactly the rows of
``deviations.md`` (layer dtypes ``float32``/``uint32``, dense layers as CSR, numeric
categoricals as plain columns). Skips where anndata has no wheels.

What it does not: test the public surface; ``sink_h5ad()`` becomes a plan terminal in
7.2/7.3 (user decision, Nov 7, 2026). Rust tests: ``crates/kyto-store/tests/export_h5ad.rs``.
"""

# pandas, scipy and anndata ship without complete type information; strict pyright
# cannot see their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from fixtures.registry import FIXTURE_NAMES, FixtureName, load_fixture
from kyto import _kyto

anndata = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")


def _source(name: FixtureName) -> Any:
    """The fixture as anndata writes it; synthetic gets every column encoding."""
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


def _assert_frames_equal(actual: Any, expected: Any) -> None:
    assert list(actual.columns) == list(expected.columns)
    assert actual.index.equals(expected.index)
    assert actual.index.name == expected.index.name
    for name in expected.columns:
        want, got = expected[name], actual[name]
        if (
            isinstance(want.dtype, pd.CategoricalDtype)
            and want.cat.categories.dtype.kind in "iuf"
        ):
            # deviations.md "Numeric categoricals": a plain numeric column.
            want = want.astype(want.cat.categories.dtype)
        pd.testing.assert_series_equal(got, want, check_exact=True, obj=name)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_export_equals_source_fixture(tmp_path: Path, name: FixtureName) -> None:
    source_path = tmp_path / "source.h5ad"
    _source(name).write_h5ad(source_path)
    source: Any = anndata.read_h5ad(source_path)
    store = tmp_path / "store.kyto"
    kyto.convert(source_path, store)
    target = tmp_path / "export.h5ad"
    _kyto.sink_h5ad(str(store), str(target), x_layer="counts", overwrite=False)
    exported: Any = anndata.read_h5ad(target)

    assert exported.shape == source.shape
    _assert_frames_equal(exported.obs, source.obs)
    _assert_frames_equal(exported.var, source.var)
    _assert_csr_equal(exported.X, source.X, "float32")
    source_layers = sorted(key for key in source.layers if key is not None)
    exported_layers = sorted(key for key in exported.layers if key is not None)
    assert exported_layers == source_layers
    # deviations.md "Layer value dtype": integer layers come back uint32, float layers
    # float32; "Dense layers": every layer comes back as CSR.
    for key in source_layers:
        integer = np.issubdtype(source.layers[key].dtype, np.integer)
        dtype = "uint32" if integer else "float32"
        _assert_csr_equal(exported.layers[key], source.layers[key], dtype)
    for member in ("obsm", "varm", "obsp", "varp", "uns"):
        assert len(getattr(exported, member)) == len(getattr(source, member)) == 0
    assert exported.raw is None


def test_errors_are_coded(tmp_path: Path) -> None:
    source_path = tmp_path / "source.h5ad"
    _source("synthetic").write_h5ad(source_path)
    store = tmp_path / "store.kyto"
    kyto.convert(source_path, store)
    target = tmp_path / "export.h5ad"

    with pytest.raises(kyto.MissingMemberError) as missing:
        _kyto.sink_h5ad(str(store), str(target), x_layer="count", overwrite=False)
    assert missing.value.code == "KY-E109"
    assert not target.exists()

    _kyto.sink_h5ad(str(store), str(target), x_layer=None, overwrite=False)
    with pytest.raises(kyto.StoreInputError) as exists:
        _kyto.sink_h5ad(str(store), str(target), x_layer=None, overwrite=False)
    assert exists.value.code == "KY-E123"
    assert "overwrite=True" in str(exists.value)
    _kyto.sink_h5ad(str(store), str(target), x_layer="counts", overwrite=True)
    exported: Any = anndata.read_h5ad(target)
    assert exported.X is not None

    with pytest.raises(kyto.CorruptStoreError) as not_a_store:
        _kyto.sink_h5ad(
            str(tmp_path), str(tmp_path / "x.h5ad"), x_layer=None, overwrite=False
        )
    assert not_a_store.value.code == "KY-E116"
