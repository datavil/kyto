"""Task 3.8: derived-column lineage, from Python.

Done when: ``n_genes`` read after ``filter_genes`` raises ``KY-E301``. The store is an
``.h5ad`` converted with ``kyto.convert`` (it needs anndata and skips without it); no
kyto writer computes derived columns yet (6.1), so the test marks ``n_genes`` as
derived by adding ``lineage`` to its ``schema.json`` entry, as format.md section 8
defines it. Every case is also covered by ``crates/kyto-plan/tests/it/lineage.rs``.
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import doctest
import json
from pathlib import Path

import kyto
import numpy as np
import pytest
from kyto import _kyto, cell_column, dtypes, gene_column


def _store(path: Path) -> Path:
    """A 3-cell, 3-gene store whose cell column ``n_genes`` records lineage."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0]], dtype=np.float32))
    adata = anndata.AnnData(
        counts,
        obs=pd.DataFrame(
            {
                "n_genes": np.array([2, 1, 1], dtype=np.int32),
                "batch": pd.Categorical(["a", "b", "a"]),
            },
            index=["AAAC-1", "AAAG-1", "AACT-1"],
        ),
        var=pd.DataFrame(
            {"highly_variable": [True, False, True]}, index=["CD3E", "CD4", "MS4A1"]
        ),
    )
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path, layer="counts")
    schema_path = path / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for column in schema["cells"]["columns"]:
        if column["name"] == "n_genes":
            column["lineage"] = {"layer": "counts", "node": "n3"}
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path / "pbmc.kyto")


def test_n_genes_read_after_filter_genes_is_ky_e301(store: Path) -> None:
    atlas = _kyto.scan(store)
    assert (
        repr(atlas.resolve(cell_column("n_genes") > 1)) == "cell_column('n_genes') > 1"
    )

    filtered = atlas.filter(gene_column("highly_variable"))
    with pytest.raises(kyto.StaleColumnError) as raised:
        filtered.resolve(cell_column("n_genes") > 1)
    error = raised.value
    assert error.code == "KY-E301"
    assert isinstance(error, ValueError)
    assert str(error).splitlines()[0] == (
        "KY-E301: column 'n_genes' was computed before 'filter_genes' changed its inputs"
    )
    assert ".recompute()" in str(error)
    with pytest.raises(kyto.StaleColumnError):
        filtered.resolve(kyto.feature("n_genes"))
    # The plan it was built from is unchanged (R2).
    assert str(atlas) == f'SCAN "{store}" (3 cells x 3 genes)'
    assert repr(atlas.resolve(cell_column("n_genes"))) == "cell_column('n_genes')"


def test_recompute_is_accepted_after_the_filter(store: Path) -> None:
    filtered = _kyto.scan(store).filter(gene_column("highly_variable"))
    resolved = filtered.resolve(cell_column("n_genes").recompute() > 1)
    assert repr(resolved) == "cell_column('n_genes').recompute() > 1"
    assert resolved.dtype is dtypes.Boolean
    # Columns without lineage are read as stored.
    assert repr(filtered.resolve(cell_column("batch"))) == "cell_column('batch')"
    with pytest.raises(kyto.LineageError) as raised:
        filtered.resolve(cell_column("batch").recompute())
    assert raised.value.code == "KY-E302"
    assert "'batch'" in str(raised.value)


def test_recompute_needs_a_plain_column() -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        (cell_column("n_genes") + 1).recompute()
    assert raised.value.code == "KY-E213"
    assert isinstance(raised.value, TypeError)
    assert str(raised.value).splitlines()[0] == (
        "KY-E213: '.recompute()' needs a plain column read such as "
        "cell_column('n_genes').recompute(), got cell_column('n_genes') + 1"
    )
    assert (
        repr(gene_column("n_cells").recompute()) == "gene_column('n_cells').recompute()"
    )


def test_gene_filter_checks_its_predicate(store: Path) -> None:
    atlas = _kyto.scan(store)
    filtered = atlas.filter(gene_column("highly_variable"))
    assert (
        str(filtered).splitlines()[0] == "FILTER_GENES gene_column('highly_variable')"
    )
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.filter(1)  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert raised.value.code == "KY-E205"
    assert "'filter' needs a boolean CellExpr or GeneExpr, got an int" in str(
        raised.value
    )
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.filter(gene_column("gene_id"))
    assert raised.value.code == "KY-E205"
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.filter(gene_column("hvg"))
    assert raised.value.code == "KY-E210"


def test_docstring_examples_run() -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs = {
        "cell_column": cell_column,
        "gene_column": gene_column,
    }
    # Atlas.filter runs with the other Atlas examples (tests/test_atlas.py).
    docstring = _kyto.CellExpr.recompute.__doc__
    assert docstring is not None
    runner.run(parser.get_doctest(docstring, globs, "CellExpr.recompute", None, 0))
    assert runner.failures == 0
    assert runner.tries == 1


def test_handles_after_filter_genes_report_unknown_sizes(store: Path) -> None:
    """Task 4.1: a gene filter makes the gene count and every nnz unknown until the
    plan runs; the cell count stays exact and the names list the genes that may remain.
    """
    atlas = _kyto.scan(store)
    filtered = atlas.filter(gene_column("highly_variable"))
    layer = "counts"
    estimate = filtered.layer(layer).estimate()
    assert estimate["n_rows"] == 3
    assert estimate["n_columns"] is None
    assert estimate["nnz"] is None
    assert estimate["dense_bytes"] is None
    assert filtered.layer(layer).names() == atlas.layer(layer).names()
    assert filtered.genes.estimate()["n_rows"] is None
    assert filtered.cells.estimate()["n_rows"] == 3
