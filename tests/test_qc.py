"""Task 6.1: QC metrics as expressions, from Python.

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
import pytest
from kyto import (
    _kyto,
    cell_column,
    dtypes,
    gene_column,
    genes_detected,
    percent_counts_in,
    total_counts,
    total_counts_in,
)

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
