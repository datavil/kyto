"""Task 5.11: embeddings and graphs reach a store and an ``.h5ad`` that anndata reads.

On the synthetic fixture, ``normalize_total -> log1p -> pca -> neighbors -> umap ->
leiden`` runs from Python and sinks to a store (``sink_zarr``), which 2.7's export
writes as an ``.h5ad``; anndata reads ``obsm['X_pca']``, ``varm['X_pca_loadings']``,
``obsm['X_umap']`` and ``obsp`` back. Checked against what they mean: each row of
``distances`` holds the ``n_neighbors - 1`` nearest other cells of ``X_pca`` at their
Euclidean distances (numpy, brute force), ``connectivities`` is symmetric, the loadings
are orthonormal, and a cell filter before the sink keeps the rows of the kept cells.
The bit-exact checks against the executor's statistics are Rust's
(``crates/kyto-exec/tests/it/embeddings_graphs.rs``).
"""

# scipy and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from fixtures.interop import to_anndata
from fixtures.registry import load_fixture
from kyto import _kyto, cell_column

anndata = pytest.importorskip("anndata")

N_NEIGHBORS = 8


def _store(tmp_path: Path) -> Path:
    source = tmp_path / "source.h5ad"
    to_anndata(load_fixture("synthetic")).write_h5ad(source)
    store = tmp_path / "store.kyto"
    kyto.convert(source, store)
    return store


def _analysis(store: Path) -> _kyto.Atlas:
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .normalize_total(1e4, into="norm")
        .log1p(layer="norm", into="log1p")
        .pca(layer="log1p", into="X_pca", n_components=10, seed=0)
        .neighbors("X_pca", n_neighbors=N_NEIGHBORS)
        .umap(into="X_umap", n_epochs=30, seed=0)
        .leiden(seed=0, into="leiden")
    )


def _export(store: Path, target: Path) -> Any:
    _kyto.sink_h5ad(str(store), str(target), x_layer=None, overwrite=False)
    return anndata.read_h5ad(target)


def _check_neighbours(adata: Any) -> None:
    x_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    distances = adata.obsp["distances"]
    assert distances.shape == (adata.n_obs, adata.n_obs)
    squared = ((x_pca[:, None, :] - x_pca[None, :, :]) ** 2).sum(axis=2)
    for row in range(adata.n_obs):
        start, end = distances.indptr[row], distances.indptr[row + 1]
        columns = distances.indices[start:end]
        assert row not in columns
        expected = np.sqrt(squared[row, columns])
        np.testing.assert_allclose(
            distances.data[start:end], expected, rtol=1e-4, atol=1e-4
        )
        others = np.delete(np.sqrt(squared[row]), row)
        assert (
            distances.data[start:end].max() <= np.sort(others)[end - start - 1] + 1e-4
        )
    connectivities = adata.obsp["connectivities"]
    assert abs(connectivities - connectivities.T).max() < 1e-6


def test_a_sink_carries_embeddings_and_graphs_to_h5ad(tmp_path: Path) -> None:
    store = _store(tmp_path)
    written = tmp_path / "analysed.kyto"
    _analysis(store).sink_zarr(written, threads=2)
    adata = _export(written, tmp_path / "analysed.h5ad")

    n_cells = adata.n_obs
    assert adata.obsm["X_pca"].shape == (n_cells, 10)
    assert adata.obsm["X_pca"].dtype == np.float32
    assert adata.obsm["X_umap"].shape == (n_cells, 2)
    loadings = np.asarray(adata.varm["X_pca_loadings"], dtype=np.float64)
    assert loadings.shape == (adata.n_vars, 10)
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(10), atol=1e-5)
    assert sorted(adata.obsp.keys()) == ["connectivities", "distances"]
    assert (np.diff(adata.obsp["distances"].indptr) == N_NEIGHBORS - 1).all()
    _check_neighbours(adata)
    assert adata.obs["leiden"].nunique() >= 1


def test_a_cell_filter_before_the_sink_keeps_the_kept_cells_rows(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    full = tmp_path / "full.kyto"
    _analysis(store).sink_zarr(full, threads=1)
    everything = _export(full, tmp_path / "full.h5ad")
    kept = tmp_path / "kept.kyto"
    keep = ~cell_column("cell_id").str.ends_with("0")
    _analysis(store).filter(keep).sink_zarr(kept, threads=1)
    subset = _export(kept, tmp_path / "kept.h5ad")

    rows = [everything.obs_names.get_loc(name) for name in subset.obs_names]
    assert 0 < len(rows) < everything.n_obs
    np.testing.assert_array_equal(subset.obsm["X_pca"], everything.obsm["X_pca"][rows])
    np.testing.assert_array_equal(
        subset.obsm["X_umap"], everything.obsm["X_umap"][rows]
    )
    expected = everything.obsp["connectivities"][rows][:, rows]
    assert abs(subset.obsp["connectivities"] - expected).max() == 0
