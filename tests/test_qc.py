"""Tasks 6.1 and 6.13: QC metrics as expressions, from Python.

The constructors build numeric ``CellExpr`` leaves; ``Atlas.resolve`` names their
layer and checks the gene set when the plan is built (R6). The values are checked
against ``scanpy.pp.calculate_qc_metrics`` in Rust
(``crates/kyto-algo/tests/it/qc_parity.rs``): no Python terminal reads a computed column
yet (plan 5.2 Open (1), 7.x).
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any

import kyto
import numpy as np
import polars as pl
import pytest
from kyto import _kyto, cell_column, dtypes, gene_column
from kyto.qc import (
    cells_detected,
    gene_mean_counts,
    gene_total_counts,
    genes_detected,
    percent_counts_in,
    percent_dropout,
    total_counts,
    total_counts_in,
)
from parity import tolerance

MITOCHONDRIAL = gene_column("symbol").str.starts_with("MT-")


def _store(path: Path) -> Path:
    """A 2-cell store over genes with symbols MT-CO1, CD3E and MT-ND1, layer counts."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.array([[1, 2, 3], [0, 4, 0]], dtype=np.float32))
    var = pd.DataFrame(
        {"symbol": ["MT-CO1", "CD3E", "MT-ND1"], "n_cells": [2, 2, 1]},
        index=["G1", "G2", "G3"],
    )
    adata = anndata.AnnData(counts, var=var, obs=pd.DataFrame(index=["C1", "C2"]))
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path, overwrite=True)
    return path


def test_constructors_build_numeric_cell_expressions() -> None:
    assert repr(total_counts()) == "total_counts()"
    assert repr(genes_detected(layer="counts")) == "genes_detected(layer='counts')"
    assert (
        repr(total_counts_in(MITOCHONDRIAL))
        == "total_counts_in(gene_column('symbol').str.starts_with('MT-'))"
    )
    percent = percent_counts_in(MITOCHONDRIAL, layer="counts")
    assert repr(percent < 5) == (
        "percent_counts_in(gene_column('symbol').str.starts_with('MT-'), "
        "layer='counts') < 5"
    )
    assert percent.dtype is dtypes.Numeric
    assert isinstance(total_counts(), _kyto.CellExpr)


@pytest.mark.parametrize(
    "genes", [cell_column("mito"), "MT-", gene_column("n_cells") + 1]
)
def test_a_gene_set_must_be_a_boolean_gene_expression(genes: Any) -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        percent_counts_in(genes)
    assert raised.value.code == "KY-E205"
    assert "percent_counts_in" in str(raised.value)


def test_resolving_names_the_layer_and_checks_the_gene_set(tmp_path: Path) -> None:
    atlas = _kyto.scan(_store(tmp_path / "qc.kyto"))
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(total_counts())
    assert raised.value.code == "KY-E211"
    resolved = atlas.with_default_layer("counts").resolve(
        percent_counts_in(MITOCHONDRIAL)
    )
    assert repr(resolved) == (
        "percent_counts_in(gene_column('symbol').str.starts_with('MT-'), "
        "layer='counts')"
    )
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.resolve(total_counts_in(gene_column("n_cells"), layer="counts"))
    assert raised.value.code == "KY-E205"
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(total_counts_in(gene_column("mito"), layer="counts"))
    assert raised.value.code == "KY-E210"


def test_gene_constructors_build_numeric_gene_expressions() -> None:
    """Task 6.13: the per-gene QC constructors and ``log1p``."""
    assert repr(cells_detected()) == "cells_detected()"
    assert (
        repr(gene_total_counts(layer="counts")) == "gene_total_counts(layer='counts')"
    )
    assert repr(gene_mean_counts().log1p()) == "gene_mean_counts().log1p()"
    assert repr(percent_dropout(layer="counts") < 99) == (
        "percent_dropout(layer='counts') < 99"
    )
    for expr in (cells_detected(), gene_total_counts().log1p()):
        assert isinstance(expr, _kyto.GeneExpr)
        assert expr.dtype == dtypes.Numeric
    assert repr(total_counts().log1p()) == "total_counts().log1p()"
    with pytest.raises(kyto.ExpressionError) as raised:
        (cells_detected() >= 3).log1p()  # pyright: ignore[reportAttributeAccessIssue]
    assert raised.value.code == "KY-E205"
    assert "log1p" in str(raised.value)
    # Typed axes (R5): a gene metric does not mix with a cell metric.
    with pytest.raises(kyto.errors.AxisError) as mixed:
        _ = cells_detected() + total_counts()  # pyright: ignore[reportOperatorIssue]
    assert mixed.value.code == "KY-E202"


def test_gene_metrics_and_log1p_columns_match_scanpy(tmp_path: Path) -> None:
    """Task 6.13: the per-gene columns and the ``log1p_`` columns equal scanpy's
    ``calculate_qc_metrics(log1p=True)`` over the cells a filter kept, and
    ``filter(cells_detected() >= 2)`` is ``filter_genes(min_cells=2)``."""
    sc = pytest.importorskip("scanpy")
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    generator = np.random.default_rng(613)
    dense = generator.poisson(0.6, size=(60, 25)).astype(np.float32)
    dense[:, 3] = 0  # a gene no cell has
    dense[5, :] = 0  # a cell without counts
    adata = anndata.AnnData(
        csr_matrix(dense),
        obs=pd.DataFrame(index=[f"C{cell}" for cell in range(60)]),
        var=pd.DataFrame(index=[f"G{gene}" for gene in range(25)]),
    )
    source = tmp_path / "counts.h5ad"
    adata.write_h5ad(source)
    store = tmp_path / "counts.kyto"
    kyto.convert(source, store)

    sc.pp.filter_cells(adata, min_genes=8)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=True, inplace=True)
    kept, _ = sc.pp.filter_genes(adata, min_cells=20, inplace=False)
    assert 0 < kept.sum() < adata.n_vars

    columns = (
        kyto.scan(store)
        .with_default_layer("counts")
        .filter(genes_detected() >= 8)
        .with_genes(
            n_cells_by_counts=cells_detected(),
            total_counts=gene_total_counts(),
            mean_counts=gene_mean_counts(),
            pct_dropout_by_counts=percent_dropout(),
            log1p_mean_counts=gene_mean_counts().log1p(),
            log1p_total_counts=gene_total_counts().log1p(),
        )
        .with_cells(
            log1p_total_counts=total_counts().log1p(),
            log1p_n_genes_by_counts=genes_detected().log1p(),
        )
    )
    genes = columns.genes.to_polars()
    assert genes["gene_id"].to_list() == list(adata.var_names)
    assert genes.schema["n_cells_by_counts"] == pl.UInt32
    assert (
        genes["n_cells_by_counts"].to_list() == adata.var["n_cells_by_counts"].tolist()
    )
    for name in (
        "total_counts",
        "mean_counts",
        "pct_dropout_by_counts",
        "log1p_mean_counts",
        "log1p_total_counts",
    ):
        assert genes.schema[name] == pl.Float64, name
        np.testing.assert_allclose(
            genes[name].to_numpy(),
            adata.var[name].to_numpy(),
            rtol=tolerance("qc.values"),
            err_msg=name,
        )
    cells = columns.cells.to_polars()
    assert cells["cell_id"].to_list() == list(adata.obs_names)
    for name in ("log1p_total_counts", "log1p_n_genes_by_counts"):
        np.testing.assert_allclose(
            cells[name].to_numpy(),
            adata.obs[name].to_numpy(),
            rtol=tolerance("qc.values"),
            err_msg=name,
        )

    filtered = columns.filter(cells_detected() >= 20)
    assert filtered.gene_ids().to_list() == list(adata.var_names[kept])
    assert filtered.count_cells() == adata.n_obs
    # The metric is over the cells below it: a later cell filter does not change it.
    later = filtered.filter(total_counts() >= 1_000_000)
    assert later.count_cells() == 0
    assert later.gene_ids().to_list() == list(adata.var_names[kept])
