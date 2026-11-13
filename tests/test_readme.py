"""Task 7.2 Done when: the README pipeline runs end to end on a fixture.

Runs the ``python`` block of ``README.md`` as written, in a folder holding the pbmc3k
fixture as ``pbmc3k.kyto`` (converted from the fixture's ``.h5ad``), then checks what
it wrote against scanpy's run of the same steps: the kept cells, the highly variable
genes, and the ``log1p`` layer. Skips where anndata or scanpy has no wheels.
"""

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import polars as pl
import pytest
from kyto import _kyto

# anndata and scanpy ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
anndata = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

README = Path(__file__).parents[1] / "README.md"


def _pipeline() -> str:
    """The README's ``python`` block (there is exactly one)."""
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert len(blocks) == 1
    return blocks[0]


def _scanpy(source: Path) -> Any:
    """scanpy's tutorial steps, as the README runs them."""
    adata = anndata.read_h5ad(source)
    adata.var["mt"] = adata.var["symbol"].astype(str).str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["n_genes_by_counts"] < 2500)
        & (adata.obs["pct_counts_mt"] < 5)
    ].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    return adata


def test_readme_pipeline_runs_and_matches_scanpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fixtures.interop import to_anndata
    from fixtures.registry import load_fixture

    source = tmp_path / "pbmc3k.h5ad"
    to_anndata(load_fixture("pbmc3k")).write_h5ad(source)
    kyto.convert(source, tmp_path / "pbmc3k.kyto", layer="counts")
    output = io.StringIO()
    namespace: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(output):
        exec(compile(_pipeline(), str(README), "exec"), namespace)  # noqa: S102

    printed = output.getvalue()
    assert "optimized plan" in printed
    assert "'device': 'cpu'" in printed
    markers = namespace["markers"]
    assert isinstance(markers, pl.DataFrame)
    assert markers.shape == (10, 6)
    assert markers["group"].unique().sort().to_list() == ["false", "true"]

    expected = _scanpy(source)
    hvg = expected[:, expected.var["highly_variable"]]
    # Read what the pipeline wrote through the 2.7 export.
    written = tmp_path / "written.h5ad"
    _kyto.sink_h5ad(
        str(tmp_path / "pbmc3k_hvg.kyto"),
        str(written),
        x_layer="log1p",
        overwrite=False,
    )
    actual = anndata.read_h5ad(written)
    assert list(actual.obs_names) == list(hvg.obs_names)
    assert set(actual.var_names) == set(hvg.var_names)
    assert len(actual.var_names) == 2000
    reference = hvg[:, actual.var_names]
    np.testing.assert_allclose(
        actual.obs["pct_counts_mt"], reference.obs["pct_counts_mt"], rtol=1e-6
    )
    np.testing.assert_allclose(
        actual.X.toarray(), reference.X.toarray(), rtol=1e-6, atol=1e-7
    )
