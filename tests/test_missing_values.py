"""Task 3.7: the missing-value rules of R9, from Python.

Done when: tests cover all three cases.

- An absent sparse value densifies to 0: no Python terminal reads the matrix yet
  (``KY-E601`` until 7.3), so the rule is tested where it lives, on ``CellBlock`` in
  ``crates/kyto-store/tests/missing_values.rs``, through the tile reader. Here, the
  stored layer of a converted ``.h5ad`` holds no null or ``NaN`` for its zeros.
- Missing metadata is null: an ``.h5ad`` categorical code ``-1`` and masked nullable
  entries become Polars nulls in the store, ``NaN`` stays a value (format.md §5.5),
  and expressions test for nulls with ``is_null()`` / ``is_not_null()``; ``x == None``
  is ``KY-E212``, naming the null test to write. How nulls flow through operations is tested in
  ``crates/kyto-core/tests/missing_values.rs``.
- A gene removed by a filter raises ``KY-E203``: no plan node removes genes before
  task 4.1, so the resolver check is tested in Rust (``missing_values.rs`` in
  kyto-core); here, the error class and its message.
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import math
import operator
from collections.abc import Callable
from pathlib import Path

import kyto
import numpy as np
import polars as pl
import pytest
from kyto import _kyto, cell_column, dtypes, gene_column, gene_expression
from kyto.dtypes import Unknown


def _store(path: Path) -> Path:
    """A 4-cell store whose cell columns have a missing value in row 1 (and a ``NaN``
    score in row 2), converted from an ``.h5ad`` written by anndata."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(
        np.array([[1, 0, 2], [0, 0, 0], [0, 3, 0], [4, 0, 0]], dtype=np.float32)
    )
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["B", None, "T", "T"]),
            "n_genes": pd.array([2, None, 1, 1], dtype="Int32"),
            "is_doublet": pd.array([False, None, True, False], dtype="boolean"),
            "score": np.array([0.5, 1.0, np.nan, 3.0]),
        },
        index=["AAAC-1", "AAAG-1", "AACT-1", "AAGT-1"],
    )
    adata = anndata.AnnData(
        counts, obs=obs, var=pd.DataFrame(index=["CD3E", "CD4", "MS4A1"])
    )
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path, overwrite=True)
    return path


def test_missing_metadata_is_null_and_nan_is_a_value(tmp_path: Path) -> None:
    store = _store(tmp_path / "pbmc.kyto")
    frame = pl.read_parquet(store / "cells.parquet")
    for name in ["cell_type", "n_genes", "is_doublet"]:
        # Categorical code -1 and masked nullable entries: null, as anndata's pd.NA.
        assert frame[name].is_null().to_list() == [False, True, False, False], name
    score = frame["score"].to_list()
    assert frame["score"].null_count() == 0
    assert math.isnan(score[2])
    assert frame["cell_type"].dtype == pl.Enum(["B", "T"])


def test_absent_matrix_entries_are_not_stored_as_null_or_nan(tmp_path: Path) -> None:
    store = _store(tmp_path / "pbmc.kyto")
    schema = _kyto.scan(store).layer("counts").schema()
    # Four stored values: every zero of the 4 x 3 matrix is absent, which reads as 0.
    assert schema["nnz"] == 4


@pytest.mark.parametrize("constructor", [cell_column, gene_column])
def test_null_tests_are_boolean_expressions(
    constructor: Callable[[str], _kyto.CellExpr[Unknown] | _kyto.GeneExpr[Unknown]],
) -> None:
    name = constructor.__name__
    expr = constructor("cell_type")
    assert repr(expr.is_null()) == f"{name}('cell_type').is_null()"
    assert repr(expr.is_not_null()) == f"{name}('cell_type').is_not_null()"
    assert expr.is_null().dtype is dtypes.Boolean
    assert repr(~expr.is_null()) == f"~{name}('cell_type').is_null()"
    assert repr((expr == "B").is_null()) == f"({name}('cell_type') == 'B').is_null()"


def test_null_tests_resolve_against_the_schema(tmp_path: Path) -> None:
    atlas = _kyto.scan(_store(tmp_path / "pbmc.kyto")).with_default_layer("counts")
    keep = cell_column("n_genes").is_not_null() & (cell_column("n_genes") > 1)
    resolved = atlas.resolve(keep)
    assert repr(resolved) == repr(keep)
    assert resolved.dtype is dtypes.Boolean
    # A matrix value is never null; the test is allowed (and always False).
    assert repr(atlas.resolve(gene_expression("CD3E").is_null())) == (
        "gene_expression('CD3E', layer='counts').is_null()"
    )
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(cell_column("missing").is_null())
    assert raised.value.code == "KY-E210"


@pytest.mark.parametrize(
    ("compare", "operation", "test", "reversed_operands"),
    [
        (operator.eq, "==", "is_null", False),
        (operator.ne, "!=", "is_not_null", False),
        (operator.eq, "==", "is_null", True),
    ],
)
def test_comparing_with_none_names_the_null_test(
    compare: Callable[[object, object], object],
    operation: str,
    test: str,
    reversed_operands: bool,
) -> None:
    expr = cell_column("cell_type")
    with pytest.raises(kyto.ExpressionError) as raised:
        # `None == expr` falls back to `expr == None`.
        compare(None, expr) if reversed_operands else compare(expr, None)
    assert raised.value.code == "KY-E212"
    assert isinstance(raised.value, TypeError)
    message, hint = str(raised.value).splitlines()[:2]
    assert message == (
        f"KY-E212: '{operation}' with None does not test for missing values: "
        f"cell_column('cell_type') {operation} None"
    )
    assert hint == (
        f"hint: write cell_column('cell_type').{test}() instead; "
        "missing values are null (R9)"
    )


def test_comparing_an_operation_with_none_keeps_its_parentheses() -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        _ = (cell_column("n_genes") > 1) == None
    message, hint = str(raised.value).splitlines()[:2]
    assert message.endswith(": (cell_column('n_genes') > 1) == None")
    assert "write (cell_column('n_genes') > 1).is_null() instead" in hint


def test_other_operators_with_none_are_ky_e205() -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        _ = cell_column("n_genes") + None  # pyright: ignore[reportOperatorIssue]
    assert raised.value.code == "KY-E205"
    assert "got a NoneType: None" in str(raised.value)


def test_removed_gene_error() -> None:
    error = kyto.RemovedFeatureError("KY-E203", gene="CD3E", removed_by="select_genes")
    assert isinstance(error, ValueError)
    assert error.code == "KY-E203"
    assert "gene 'CD3E' was removed by 'select_genes'" in str(error)
