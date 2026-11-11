"""Task 4.2: every error of the 2xx range (names, axes, expressions) fires from a plan
that has read no data.

Done when: each code is raised while a plan or an expression is built, with an I/O
counter of 0. Here the counter is the store itself: it is converted from an ``.h5ad``,
scanned (which reads ``schema.json`` and the gene IDs and symbols, metadata only),
and then deleted, so any read of the matrix or the tables would fail with a storage
error (a 1xx code) instead of the expected code. The codes no Python method reaches
yet (``KY-E203`` needs ``select_genes``, ``KY-E215`` and ``KY-E216`` the other plan
steps, 7.2) are covered in ``crates/kyto-plan/tests/build_validation.rs``, which also
checks every code of the registry.
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import kyto
import numpy as np
import pytest
import tomllib
from kyto import (
    _kyto,
    cell_column,
    cell_embedding,
    feature,
    gene_column,
    gene_expression,
)

REGISTRY = Path(__file__).parent.parent / "errors.toml"

# Raised only by plan steps that have no Python method yet (see the module docstring).
RUST_ONLY = {"KY-E203", "KY-E215", "KY-E216"}


def _store(path: Path) -> Path:
    """3 cells; cell column ``CD3E`` and gene ``CD3E``; two genes with symbol TBCE."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0]], dtype=np.float32))
    adata = anndata.AnnData(
        counts,
        obs=pd.DataFrame(
            {
                "n_genes": np.array([2, 1, 1], dtype=np.int32),
                "CD3E": np.array([0.5, 1.5, 2.5], dtype=np.float32),
            },
            index=["AAAC-1", "AAAG-1", "AACT-1"],
        ),
        var=pd.DataFrame(
            {
                "highly_variable": [True, False, True],
                "symbol": pd.Categorical(["CD3E", "TBCE", "TBCE"]),
            },
            index=["CD3E", "ENSG01", "ENSG02"],
        ),
    )
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path, layer="counts")
    return path


def _deep() -> object:
    expression: Any = cell_column("n_genes")
    for _ in range(1000):
        expression = expression + 1
    return expression


def test_every_ky_e2xx_code_fires_without_reading_the_store(tmp_path: Path) -> None:
    store = _store(tmp_path / "pbmc.kyto")
    bare = _kyto.scan(store)
    atlas = bare.with_default_layer("counts")
    shutil.rmtree(store)
    assert not store.exists()

    loose: Any = cell_column("n_genes")
    cases: dict[str, Callable[[], object]] = {
        "KY-E201": lambda: atlas.resolve(feature("CD3E")),
        "KY-E202": lambda: loose & gene_column("highly_variable"),
        "KY-E204": lambda: atlas.resolve(gene_expression("TBCE", by="symbol")),
        "KY-E205": lambda: atlas.filter_genes(cast(Any, gene_column("symbol"))),
        "KY-E206": lambda: bool(cell_column("n_genes") > 1),
        "KY-E207": _deep,
        "KY-E208": lambda: feature("log1p:"),
        "KY-E209": lambda: cell_embedding("X_pca", -1),
        "KY-E210": lambda: atlas.resolve(cell_column("n_gene")),
        "KY-E211": lambda: bare.resolve(gene_expression("CD3E")),
        "KY-E212": lambda: loose == None,
        "KY-E213": lambda: (cell_column("n_genes") + 1).recompute(),
        "KY-E214": lambda: gene_expression("CD3E", by=cast(Any, "name")),
        "KY-E217": lambda: kyto.from_json(atlas.to_json()[:-1]),
        "KY-E218": lambda: kyto.from_json(
            atlas.to_json().replace('"version":1', '"version":2', 1)
        ),
    }
    for code, build in cases.items():
        with pytest.raises(kyto.KytoError) as raised:
            build()
        assert raised.value.code == code, (code, str(raised.value))

    registry = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    registered = {
        entry["code"]
        for entry in registry["errors"]
        if entry["code"].removeprefix("KY-E").startswith("2")
    }
    assert set(cases) | RUST_ONLY == registered
    assert not set(cases) & RUST_ONLY
    # The plans still work after all that: nothing was read, nothing was cached.
    assert repr(atlas.resolve(feature("meta:CD3E") > 1)) == "cell_column('CD3E') > 1"
    assert not store.exists()
