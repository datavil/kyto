"""Task 7.8: plotting through cellestial.

Done when: the documented examples run in the doctests and produce each plot from a
kyto result. ``docs/plotting.md`` (one example per plot of scanpy's clustering
tutorial) runs here as a doctest on two halves of the pbmc3k fixture, and every plot it
makes is built (ggrust's ``build``: scales trained, layer data computed; no pixels
compared). Also ``Atlas.to_anndata``, the hand-off cellestial reads: scanpy's names
(``X_``-prefixed ``obsm``, genes by a text column made unique, categoricals), coded
errors before running, a copy the caller owns, its docstring examples (runtime and
stub, equal) and its declared return type. Skips without anndata or cellestial.
"""

import ast
import doctest
import gzip
import tempfile
import warnings
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from kyto import _kyto

# anndata, pandas, cellestial and ggrust ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false
anndata = pytest.importorskip("anndata")
cellestial = pytest.importorskip("cellestial")
pd = pytest.importorskip("pandas")

DOCS = Path(__file__).parents[1] / "docs" / "plotting.md"

# The plots docs/plotting.md makes, by the name it binds each to.
PLOTS = (
    "qc_violins",
    "qc_scatter",
    "umap_sample",
    "umap_cluster",
    "umap_gene",
    "markers_dotplot",
)


def _store(
    folder: Path, symbols: tuple[bytes, ...] = (b"CD3E", b"CD4", b"MS4A1")
) -> Path:
    """A 2-cell, 3-gene store with one ``counts`` layer (as ``tests/test_chain.py``)."""
    source = folder / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    rows = b"".join(
        b"ENSG0%d\t%s\tGene Expression\n" % (index + 1, symbol)
        for index, symbol in enumerate(symbols)
    )
    (source / "features.tsv.gz").write_bytes(gzip.compress(rows))
    store = folder / "pbmc.kyto"
    kyto.convert(source, store, layer="counts")
    return store


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


def _built(plot: Any) -> list[dict[str, Any]]:
    """Build a ggrust plot (or each panel of a grid) and return the builds."""
    if hasattr(plot, "figures"):
        return [figure.build() for figure in plot.figures if figure is not None]
    return [plot.build()]


# Atlas.to_anndata


def test_to_anndata_hands_over_the_plan_output(store: Path) -> None:
    atlas = (
        kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(n_counts=kyto.qc.total_counts())
        .normalize_total(into="normalized")
    )
    adata = atlas.to_anndata(threads=1)
    assert isinstance(adata, anndata.AnnData)
    assert list(adata.obs_names) == ["AAAC-1", "AAAG-1"]
    assert list(adata.var_names) == ["ENSG01", "ENSG02", "ENSG03"]
    matrix: Any = adata.X
    np.testing.assert_array_equal(matrix.toarray(), [[5, 0, 1], [0, 7, 0]])
    layers = list(adata.layers.keys())
    assert [key for key in layers if key is not None] == ["normalized"]
    assert adata.obs["n_counts"].tolist() == [6.0, 7.0]
    assert list(adata.var["symbol"]) == ["CD3E", "CD4", "MS4A1"]
    assert dict(adata.uns) == {}
    # A copy the caller owns (R2): changing it changes no later read.
    matrix[0, 0] = 100
    again: Any = atlas.to_anndata().X
    assert again[0, 0] == 5


def test_cell_embeddings_take_scanpy_obsm_keys(store: Path) -> None:
    atlas = (
        kyto.scan(store)
        .with_default_layer("counts")
        .pca(into="pca", n_components=1, seed=0)
        .pca(into="X_other", n_components=1, seed=0)
    )
    adata = atlas.to_anndata()
    assert sorted(adata.obsm) == ["X_other", "X_pca"]
    np.testing.assert_array_equal(
        adata.obsm["X_pca"], atlas.embedding("pca").to_numpy()
    )
    # Gene embeddings keep their names (varm), as cell graphs (obsp).
    assert sorted(adata.varm) == ["X_other_loadings", "pca_loadings"]


def test_two_embeddings_with_one_obsm_key_are_ky_e103(store: Path) -> None:
    atlas = (
        kyto.scan(store)
        .with_default_layer("counts")
        .pca(into="pca", n_components=1, seed=0)
        .pca(into="X_pca", n_components=1, seed=0)
    )
    with pytest.raises(kyto.StoreInputError) as raised:
        atlas.to_anndata(progress=lambda done, total: pytest.fail("ran"))
    assert raised.value.code == "KY-E103"
    assert "'pca'" in str(raised.value) and "'X_pca'" in str(raised.value)


def test_var_names_by_symbol_are_unique_and_keep_gene_ids(tmp_path: Path) -> None:
    store = _store(tmp_path, (b"CD3E", b"CD3E", b"MS4A1"))
    adata = kyto.scan(store).to_anndata(var_names="symbol")
    assert list(adata.var_names) == ["CD3E", "CD3E-1", "MS4A1"]
    assert adata.var.index.name is None
    assert list(adata.var.columns[:2]) == ["gene_id", "symbol"]
    assert list(adata.var["gene_id"]) == ["ENSG01", "ENSG02", "ENSG03"]
    # The ID column itself changes nothing.
    by_id = kyto.scan(store).to_anndata(var_names="gene_id")
    assert list(by_id.var_names) == ["ENSG01", "ENSG02", "ENSG03"]
    assert "gene_id" not in by_id.var.columns


def test_suffixes_skip_names_genes_already_have(tmp_path: Path) -> None:
    """Bug-hunter: anndata numbers past a symbol that already ends in ``-1``, and its
    warning about it reached the caller."""
    store = _store(tmp_path, (b"A", b"A", b"A-1"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        adata = kyto.scan(store).to_anndata(var_names="symbol")
    assert list(adata.var_names) == ["A", "A-2", "A-1"]


def test_a_gene_without_a_symbol_keeps_its_id(tmp_path: Path) -> None:
    var = pd.DataFrame(
        {"symbol": pd.Categorical(["CD3E", None, "CD4"])},
        index=pd.Index(["ENSG01", "ENSG02", "ENSG03"], name="gene_id"),
    )
    source = anndata.AnnData(
        np.ones((2, 3), np.float32),
        obs=pd.DataFrame(index=pd.Index(["c1", "c2"], name="cell_id")),
        var=var,
    )
    source.write_h5ad(tmp_path / "source.h5ad")
    kyto.convert(tmp_path / "source.h5ad", tmp_path / "x.kyto", layer="counts")
    adata = kyto.scan(tmp_path / "x.kyto").to_anndata(var_names="symbol")
    assert list(adata.var_names) == ["CD3E", "ENSG02", "CD4"]


@pytest.mark.parametrize(
    ("column", "problem"),
    [
        ("Symbol", "the plan's gene table has no column 'Symbol'"),
        ("is_cd3e", "gene column 'is_cd3e' holds Boolean, not text"),
    ],
)
def test_var_names_not_a_text_gene_column_is_ky_e215(
    store: Path, column: str, problem: str
) -> None:
    atlas = kyto.scan(store).with_genes(is_cd3e=kyto.gene_column("symbol") == "CD3E")
    with pytest.raises(kyto.PlanError) as raised:
        atlas.to_anndata(var_names=column)
    assert raised.value.code == "KY-E215"
    assert problem in str(raised.value)
    assert "'gene_id', 'symbol'" in str(raised.value)


def test_errors_leave_no_temporary_file(store: Path) -> None:
    before = set(Path(tempfile.gettempdir()).glob("kyto-anndata-*"))
    # A cell that does not exist is found only when the plan runs.
    atlas = kyto.scan(store).take_cells(["AAAC-1", "missing"])
    with pytest.raises(kyto.ResolutionError):
        atlas.to_anndata()
    kyto.scan(store).to_anndata()
    assert set(Path(tempfile.gettempdir()).glob("kyto-anndata-*")) == before


def _stub_docstring() -> str | None:
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    atlas = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Atlas"
    )
    method = next(
        item
        for item in atlas.body
        if isinstance(item, ast.FunctionDef) and item.name == "to_anndata"
    )
    assert method.returns is not None
    assert ast.unparse(method.returns) == "anndata.AnnData"  # R3: one type
    return ast.get_docstring(method)


@pytest.mark.parametrize("source", ["runtime", "stub"])
def test_docstring_examples_run(tmp_path: Path, source: str) -> None:
    runtime = _kyto.Atlas.to_anndata.__doc__
    stub = _stub_docstring()
    assert runtime is not None and stub is not None
    assert stub == runtime.strip()
    docstring = runtime if source == "runtime" else stub
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs: dict[str, object] = {"scan": _kyto.scan, "store": _store(tmp_path)}
    runner.run(parser.get_doctest(docstring, globs, "Atlas.to_anndata", None, 0))
    assert runner.tries > 0
    assert runner.failures == 0


# docs/plotting.md


def test_documented_plots_build_from_a_kyto_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fixtures.interop import to_anndata
    from fixtures.registry import load_fixture

    # Two samples, as the tutorial's: the two halves of pbmc3k.
    pbmc = to_anndata(load_fixture("pbmc3k"))
    half = pbmc.n_obs // 2
    for name, cells in (("s1d1", slice(0, half)), ("s1d3", slice(half, None))):
        pbmc[cells].copy().write_h5ad(tmp_path / f"{name}.h5ad")
        kyto.convert(
            tmp_path / f"{name}.h5ad", tmp_path / f"{name}.kyto", layer="counts"
        )
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, Any] = {}
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    test = parser.get_doctest(DOCS.read_text(), namespace, "plotting.md", str(DOCS), 0)
    runner.run(test, clear_globs=False)
    assert runner.failures == 0
    namespace = test.globs

    adata = namespace["adata"]
    assert adata.obs["leiden"].dtype == "category"
    assert adata.n_obs == int(
        (namespace["qc_data"].obs["n_genes_by_counts"] >= 100).sum()
    )
    for name in PLOTS:
        builds = _built(namespace[name])
        assert builds, name
        for build in builds:
            assert build["layers"] and build["panels"], name
    # The violins: one panel per QC metric, every cell in each.
    violins = _built(namespace["qc_violins"])
    assert len(violins) == 3
    # The UMAPs: one point per cell, coloured by what was asked.
    for name, colour in (
        ("umap_sample", "sample"),
        ("umap_cluster", "leiden"),
        ("umap_gene", "CD14"),
    ):
        frame = namespace[name].data
        assert frame.height == adata.n_obs, name
        assert colour in frame.columns, name
    # The dotplot: one row per marker gene and cluster.
    markers = namespace["marker_genes"]
    n_genes = sum(len(genes) for genes in markers.values())
    n_clusters = len(adata.obs["leiden"].cat.categories)
    assert namespace["markers_dotplot"].data.height == n_genes * n_clusters
