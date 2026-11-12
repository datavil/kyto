"""Task 3.5: the feature resolver, from Python through ``Atlas.resolve``.

Done when: adding a column named ``CD3E`` after a plan is built does not change that
plan's output. The stores are ``.h5ad`` files converted with ``kyto.convert`` (they need
anndata and skip without it); the grammar is ``docs/feature_names.md``. Every case is
also covered without a store by ``crates/kyto-core/tests/it/resolve.rs`` and
``crates/kyto-plan/tests/it/resolve_in_plan.rs``.
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import shutil
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import polars as pl
import pytest
from kyto import _kyto, dtypes


def _store(path: Path, *, cd3e_column: bool = False) -> Path:
    """A 3-cell store with genes CD3E, CD4, MS4A1, layers counts and log1p, and cell
    columns ``n_genes`` and ``cluster`` (and ``CD3E``). The importer does not bring
    ``obsm`` over yet, so embeddings are tested in Rust only."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0]], dtype=np.float32))
    obs = pd.DataFrame(
        {
            "n_genes": np.array([2, 1, 1], dtype=np.int32),
            "cluster": pd.Categorical(["T", "B", "T"]),
        },
        index=["AAAC-1", "AAAG-1", "AACT-1"],
    )
    if cd3e_column:
        obs["CD3E"] = np.array([0.5, 0.0, 1.5], dtype=np.float32)
    adata = anndata.AnnData(
        counts,
        obs=obs,
        var=pd.DataFrame(index=["CD3E", "CD4", "MS4A1"]),
        layers={"log1p": csr_matrix(np.log1p(counts.toarray()))},
    )
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path, overwrite=True)
    return path


def _code(raised: pytest.ExceptionInfo[kyto.KytoError], code: str) -> str:
    assert raised.value.code == code
    return str(raised.value).splitlines()[0]


def test_adding_a_column_named_cd3e_does_not_change_a_built_plan(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "pbmc.kyto")
    plan = _kyto.scan(store).with_default_layer("log1p")
    high = kyto.feature("CD3E") > 1
    resolved = plan.resolve(high)
    assert repr(resolved) == "gene_expression('CD3E', layer='log1p') > 1"
    plan_text = str(plan)

    # The store at the same path is rewritten with a cell column named CD3E.
    shutil.rmtree(store)
    _store(store, cd3e_column=True)
    rebuilt = _kyto.scan(store).with_default_layer("log1p")
    with pytest.raises(kyto.ResolutionError) as raised:
        rebuilt.resolve(high)
    assert _code(raised, "KY-E201") == (
        "KY-E201: ambiguous name 'CD3E': it matches 'meta:CD3E', 'log1p:CD3E'"
    )

    # The plan built before is unchanged and resolves exactly as it did.
    assert str(plan) == plan_text
    assert repr(plan.resolve(high)) == repr(resolved)
    assert repr(rebuilt.resolve(resolved)) == repr(resolved)
    assert repr(high) == "feature('CD3E') > 1"


@pytest.fixture
def atlas(tmp_path: Path) -> _kyto.Atlas:
    return _kyto.scan(_store(tmp_path / "pbmc.kyto")).with_default_layer("counts")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (kyto.feature("CD4"), "gene_expression('CD4', layer='counts')"),
        (kyto.feature("log1p:CD4"), "gene_expression('CD4', layer='log1p')"),
        (kyto.feature("n_genes") > 1, "cell_column('n_genes') > 1"),
        (kyto.feature("meta:cluster") == "T", "cell_column('cluster') == 'T'"),
        (kyto.gene_expression("MS4A1"), "gene_expression('MS4A1', layer='counts')"),
        (kyto.cell_profile("AAAC-1"), "cell_profile('AAAC-1', layer='counts')"),
        (kyto.gene_column("gene_id").str.ends_with("4"), None),
    ],
)
def test_resolves_to_full_names(
    atlas: _kyto.Atlas,
    expression: _kyto.CellExpr[Any] | _kyto.GeneExpr[Any],
    expected: str | None,
) -> None:
    resolved = atlas.resolve(expression)
    assert type(resolved) is type(expression)
    assert repr(resolved) == (expected or repr(expression))


def test_resolved_kinds_come_from_the_schema(atlas: _kyto.Atlas) -> None:
    assert atlas.resolve(kyto.cell_column("n_genes")).dtype is dtypes.Numeric
    assert atlas.resolve(kyto.feature("cluster")).dtype is dtypes.String
    assert kyto.cell_column("n_genes").dtype is dtypes.Unknown


@pytest.mark.parametrize(
    ("expression", "code", "message"),
    [
        (
            kyto.feature("CD3"),
            "KY-E210",
            (
                "unknown feature 'CD3' in feature('CD3'): it is not a column of the cell table "
                "or a gene ID"
            ),
        ),
        (
            kyto.cell_column(""),
            "KY-E210",
            (
                "unknown cell column '' in cell_column(''): the cell table has no column with "
                "this name"
            ),
        ),
        (kyto.gene_column("n_genes"), "KY-E210", "unknown gene column 'n_genes'"),
        (kyto.gene_expression(""), "KY-E210", "unknown gene ''"),
        (kyto.cell_profile(""), "KY-E210", "cell IDs are never empty"),
        (kyto.cell_embedding("X_umap", 0), "KY-E109", "no embedding named 'X_umap'"),
        (kyto.gene_expression("CD4", layer="raw"), "KY-E109", "no layer named 'raw'"),
        (~kyto.cell_column("n_genes"), "KY-E205", "got a numeric expression"),
        (
            kyto.cell_column("cluster") < "T",
            "KY-E205",
            "an unordered categorical column",
        ),
    ],
)
def test_names_are_checked(
    atlas: _kyto.Atlas,
    expression: _kyto.CellExpr[Any] | _kyto.GeneExpr[Any],
    code: str,
    message: str,
) -> None:
    with pytest.raises(kyto.KytoError) as raised:
        atlas.resolve(expression)
    assert message in _code(raised, code)


def test_short_gene_names_need_a_default_layer(tmp_path: Path) -> None:
    atlas = _kyto.scan(_store(tmp_path / "pbmc.kyto"))
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.feature("CD3E"))
    assert _code(raised, "KY-E211") == (
        "KY-E211: feature('CD3E') reads the default layer, but the plan has none"
    )
    assert "'counts', 'log1p'" in raised.value.hint
    assert repr(atlas.resolve(kyto.feature("counts:CD3E"))) == (
        "gene_expression('CD3E', layer='counts')"
    )


def test_resolve_rejects_other_objects(atlas: _kyto.Atlas) -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.resolve("CD3E")  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert "'resolve' needs a CellExpr or a GeneExpr, got a str" in _code(
        raised, "KY-E205"
    )


def test_scan_reads_gene_ids_and_checks_them(tmp_path: Path) -> None:
    store = _store(tmp_path / "pbmc.kyto")
    genes = pl.read_parquet(store / "genes.parquet")
    genes.head(2).write_parquet(store / "genes.parquet")
    with pytest.raises(kyto.CorruptStoreError) as raised:
        _kyto.scan(store)
    assert "it has 2 rows, but schema.json says 3" in _code(raised, "KY-E113")


def test_resolve_names_the_wrong_type_with_its_article(atlas: _kyto.Atlas) -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.resolve(5)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert "got an int: 5" in _code(raised, "KY-E205")
