"""Task 3.9: gene identity. Genes are indexed by ID; ``by="symbol"`` must match exactly
one gene or raise ``KY-E204`` listing the candidate IDs.

Done when: a fixture with duplicate symbols triggers ``KY-E204``. The fixture is
``tests/gene_identity/duplicate_symbols``, a 10x matrix directory whose
``features.tsv`` gives ``TBCE`` to two gene IDs (as the 10x GRCh38 reference does).
Symbols come from the gene table's ``symbol`` column (format.md §5.2): the 10x and mtx
importers fill it, an ``.h5ad`` brings it when ``var`` has a ``symbol`` column. Every
case is also covered without a store by ``crates/kyto-core/tests/it/gene_symbols.rs``.
"""

# pandas and anndata ship without complete type information; strict pyright cannot see
# their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any, cast

import kyto
import numpy as np
import pytest
from kyto import _kyto, dtypes

FIXTURE = Path(__file__).parent / "gene_identity" / "duplicate_symbols"


@pytest.fixture
def atlas(tmp_path: Path) -> _kyto.Atlas:
    store = tmp_path / "duplicate_symbols.kyto"
    kyto.convert(FIXTURE, store)
    return _kyto.scan(store).with_default_layer("counts")


def _first_line(raised: pytest.ExceptionInfo[kyto.KytoError], code: str) -> str:
    assert raised.value.code == code
    return str(raised.value).splitlines()[0]


def test_duplicate_symbol_raises_ky_e204_with_the_candidates(
    atlas: _kyto.Atlas,
) -> None:
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.gene_expression("TBCE", by="symbol"))
    assert isinstance(raised.value, ValueError)
    assert _first_line(raised, "KY-E204") == (
        "KY-E204: symbol 'TBCE' matches more than one gene: "
        "'ENSG00000284770', 'ENSG00000285053'"
    )
    assert raised.value.fields["symbol"] == "TBCE"
    assert raised.value.fields["gene_ids"] == ("ENSG00000284770", "ENSG00000285053")
    assert "by='id'" in str(raised.value)
    # Inside a larger expression too, when the plan is built.
    with pytest.raises(kyto.KytoError) as raised:
        atlas.resolve(kyto.gene_expression("TBCE", by="symbol") + 1 > 2)
    assert raised.value.code == "KY-E204"


def test_unique_symbol_resolves_to_the_gene_id(atlas: _kyto.Atlas) -> None:
    expression = kyto.gene_expression("CD3E", by="symbol") > 1
    assert repr(expression) == "gene_expression('CD3E', by='symbol') > 1"
    resolved = atlas.resolve(expression)
    assert repr(resolved) == "gene_expression('ENSG00000198851', layer='counts') > 1"
    assert repr(atlas.resolve(resolved)) == repr(resolved)
    # Each TBCE gene is still reachable by ID.
    by_id = atlas.resolve(kyto.gene_expression("ENSG00000285053"))
    assert repr(by_id) == "gene_expression('ENSG00000285053', layer='counts')"
    assert kyto.gene_expression("CD3E", by="symbol").dtype is dtypes.Numeric


def test_symbols_are_not_ids_and_ids_are_not_symbols(atlas: _kyto.Atlas) -> None:
    with pytest.raises(kyto.KytoError) as raised:
        atlas.resolve(kyto.gene_expression("CD3E"))
    assert _first_line(raised, "KY-E210").startswith("KY-E210: unknown gene 'CD3E'")
    # Close symbols are suggested (task 8.2), case-insensitively; TBCE only once.
    suggested = {"ENSG00000198851": "", "cd3e": "CD3E", "TBCE-1": "TBCE"}
    for symbol, suggestion in suggested.items():
        with pytest.raises(kyto.KytoError) as raised:
            atlas.resolve(kyto.gene_expression(symbol, by="symbol"))
        did_you_mean = f"; did you mean '{suggestion}'?" if suggestion else ""
        assert _first_line(raised, "KY-E210") == (
            f"KY-E210: unknown gene symbol '{symbol}' in "
            f"gene_expression('{symbol}', by='symbol'): "
            f"no gene of the store has this symbol{did_you_mean}"
        )


def test_invalid_by_is_ky_e214() -> None:
    with pytest.raises(kyto.ResolutionError) as raised:
        kyto.gene_expression("CD3E", by=cast(Any, "name"))
    assert _first_line(raised, "KY-E214") == (
        "KY-E214: gene_expression(..., by='name') is not accepted"
    )
    assert raised.value.fields["accepted"] == ("id", "symbol")
    assert raised.value.fields["function"] == "gene_expression"
    with pytest.raises(TypeError):
        kyto.gene_expression("CD3E", by=cast(Any, None))
    with pytest.raises(TypeError):
        cast(Any, kyto.gene_expression)("CD3E", "symbol")  # keyword-only


def _h5ad_store(path: Path, var: dict[str, Any] | None) -> _kyto.Atlas:
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.array([[1, 0], [0, 3]], dtype=np.float32))
    adata = anndata.AnnData(
        counts,
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(var or {}, index=["ENSG01", "ENSG02"]),
    )
    source = path.with_suffix(".h5ad")
    adata.write_h5ad(source)
    kyto.convert(source, path)
    return _kyto.scan(path).with_default_layer("counts")


def test_h5ad_symbol_column_is_used(tmp_path: Path) -> None:
    atlas = _h5ad_store(tmp_path / "symbols.kyto", {"symbol": ["CD3E", "CD3E"]})
    with pytest.raises(kyto.KytoError) as raised:
        atlas.resolve(kyto.gene_expression("CD3E", by="symbol"))
    assert raised.value.code == "KY-E204"
    assert raised.value.fields["gene_ids"] == ("ENSG01", "ENSG02")


def test_store_without_symbols_says_why(tmp_path: Path) -> None:
    atlas = _h5ad_store(tmp_path / "plain.kyto", None)
    with pytest.raises(kyto.KytoError) as raised:
        atlas.resolve(kyto.gene_expression("CD3E", by="symbol"))
    assert "the gene table has no 'symbol' column" in _first_line(raised, "KY-E210")
    numbers = _h5ad_store(tmp_path / "numbers.kyto", {"symbol": np.array([1, 2])})
    with pytest.raises(kyto.KytoError) as raised:
        numbers.resolve(kyto.gene_expression("CD3E", by="symbol"))
    assert "'symbol' column is Int64, not a string (Enum) column" in _first_line(
        raised, "KY-E210"
    )
