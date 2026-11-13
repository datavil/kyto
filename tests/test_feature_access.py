"""Task 7.4: feature access (``pull``, ``select``, ``pull_genes``, ``features``,
``cell_ids``, ``gene_ids``, ``count_cells``, ``count_genes``, ``metadata``).

The Done-when: every row of the AnnData migration table (``docs/anndata_migration.md``,
for 10.5's docs site) runs both sides on pbmc3k and gives the same values, and kyto's
side returns the type the table states (R3). The rest checks what a run reads (only the
layers a feature reads: the other layers are deleted first), results after filters,
selections and computed columns, the counts the plan knows without running, coded
errors raised before anything runs, and the runtime and stub examples (R12).
"""

import ast
import doctest
import gzip
import re
import shutil
from pathlib import Path
from typing import Any

import kyto
import kyto.qc
import numpy as np
import polars as pl
import pytest
from kyto import _kyto, cell_column, cell_embedding, gene_column, gene_expression
from kyto.errors import (
    CorruptStoreError,
    ExpressionError,
    PlanError,
    ResolutionError,
)
from kyto.qc import total_counts

# anndata and scanpy ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

MIGRATION = Path(__file__).parents[1] / "docs" / "anndata_migration.md"
METHODS = (
    "pull",
    "select",
    "pull_genes",
    "features",
    "cell_ids",
    "gene_ids",
    "count_cells",
    "count_genes",
    "metadata",
)


def _store(folder: Path) -> Path:
    """A 2-cell, 3-gene store with one ``counts`` layer (as ``tests/test_chain.py``)."""
    source = folder / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n"
            b"ENSG03\tMS4A1\tGene Expression\n"
        )
    )
    store = folder / "pbmc.kyto"
    kyto.convert(source, store, layer="counts")
    return store


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


@pytest.fixture
def two_layers(tmp_path: Path) -> Path:
    """The small store with a second layer ``doubled`` (counts * 2, as float32)."""
    source = _store(tmp_path)
    target = tmp_path / "two.kyto"
    kyto.scan(source).normalize_total(14.0, layer="counts", into="doubled").sink_zarr(
        target
    )
    return target


# The migration table (the Done-when).


def _setup_block() -> str:
    blocks = re.findall(r"```python\n(.*?)```", MIGRATION.read_text(), flags=re.DOTALL)
    assert len(blocks) == 1
    return blocks[0]


def _rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in MIGRATION.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        assert len(cells) == 3, line
        reference, ours, returns = cells
        rows.append((reference.strip("`"), ours.strip("`"), returns))
    return rows


ROWS = _rows()


@pytest.fixture(scope="module")
def migration(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    from fixtures.interop import to_anndata
    from fixtures.registry import load_fixture

    folder = tmp_path_factory.mktemp("migration")
    h5ad = folder / "pbmc3k.h5ad"
    to_anndata(load_fixture("pbmc3k")).write_h5ad(h5ad)
    kyto.convert(h5ad, folder / "pbmc3k.kyto", layer="counts")
    setup = (
        _setup_block()
        .replace('"pbmc3k.h5ad"', repr(str(h5ad)))
        .replace('"pbmc3k.kyto"', repr(str(folder / "pbmc3k.kyto")))
    )
    namespace: dict[str, Any] = {}
    exec(compile(setup, str(MIGRATION), "exec"), namespace)  # noqa: S102
    return namespace


def _plain(value: Any) -> Any:
    """A result as plain Python values, so both sides compare with ``==``."""
    import pandas as pd

    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, pl.Series):
        return value.to_list()
    if isinstance(value, pl.DataFrame):
        return {name: value[name].to_list() for name in value.columns}
    if isinstance(value, pd.DataFrame):
        frame: dict[str, Any] = {"cell_id": [str(name) for name in value.index]}
        frame.update({str(name): value[name].tolist() for name in value.columns})
        return frame
    if isinstance(value, pd.Index):
        return [str(name) for name in value]
    if isinstance(value, pd.Series):
        return value.astype(object).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise AssertionError(f"no comparison for {type(value)}")


def test_migration_table_has_rows() -> None:
    assert len(ROWS) >= 10


@pytest.mark.parametrize(
    ("reference", "ours", "returns"), ROWS, ids=[r[1] for r in ROWS]
)
def test_migration_row(
    migration: dict[str, Any], reference: str, ours: str, returns: str
) -> None:
    expected = eval(reference, migration)
    actual = eval(ours, migration)
    kind = returns.split()[0].strip("`,")
    assert (
        type(actual)
        is {"int": int, "pl.Series": pl.Series, "pl.DataFrame": pl.DataFrame}[kind]
    )
    assert _plain(actual) == _plain(expected)


def test_migration_symbol_values_are_cd3e(migration: dict[str, Any]) -> None:
    """The symbol rows read the gene the symbol names, not a neighbour."""
    atlas = migration["atlas"]
    by_symbol = atlas.pull_genes(["CD3E"], by="symbol")["CD3E"]
    by_id = atlas.pull("ENSG00000198851")  # CD3E in GRCh38
    assert by_symbol.to_list() == by_id.to_list()
    assert by_symbol.sum() > 0


# Return types and values on the small store.


def test_return_types(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    assert type(atlas.pull("ENSG01")) is pl.Series
    assert type(atlas.pull(gene_column("symbol"))) is pl.Series
    assert type(atlas.select("ENSG01")) is pl.DataFrame
    assert type(atlas.pull_genes(["ENSG01"])) is pl.DataFrame
    assert type(atlas.features()) is pl.DataFrame
    assert type(atlas.cell_ids()) is pl.Series
    assert type(atlas.gene_ids()) is pl.Series
    assert type(atlas.count_cells()) is int
    assert type(atlas.count_genes()) is int
    assert type(atlas.metadata()) is dict


def test_values_names_and_dtypes(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    series = atlas.pull("ENSG01")
    assert (series.name, series.dtype, series.to_list()) == (
        "ENSG01",
        pl.UInt32,
        [5, 0],
    )
    full = atlas.pull("counts:ENSG03")
    assert (full.name, full.to_list()) == ("counts:ENSG03", [1, 0])
    expression = atlas.pull(gene_expression("ENSG02") + 1)
    assert expression.name == "gene_expression('ENSG02') + 1"
    assert expression.to_list() == [1.0, 8.0]
    frame = atlas.pull_genes(["MS4A1", "CD4"], by="symbol")
    assert frame.columns == ["cell_id", "MS4A1", "CD4"]
    assert frame.schema["cell_id"] == pl.String
    assert frame.rows() == [("AAAC-1", 1, 0), ("AAAG-1", 0, 7)]
    assert atlas.cell_ids().name == "cell_id"
    assert atlas.pull("cell_id").to_list() == ["AAAC-1", "AAAG-1"]
    assert atlas.gene_ids().name == "gene_id"


def test_select_orders_positional_then_named(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    frame = atlas.select("ENSG02", "ENSG01", total=total_counts(), again="ENSG03")
    assert frame.columns == ["cell_id", "ENSG02", "ENSG01", "total", "again"]
    assert frame.rows() == [("AAAC-1", 0, 5, 6.0, 1), ("AAAG-1", 7, 0, 7.0, 0)]


def test_results_follow_the_plan(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    kept = atlas.filter(total_counts() >= 7)
    assert kept.cell_ids().to_list() == ["AAAG-1"]
    assert kept.count_cells() == 1
    assert kept.select("ENSG02").rows() == [("AAAG-1", 7)]
    taken = atlas.take_cells(["AAAG-1"])
    assert taken.pull("ENSG02").to_list() == [7]
    normalized = atlas.normalize_total(10.0, into="normalized")
    assert normalized.pull("normalized:ENSG01").to_list() == pytest.approx(
        [10 * 5 / 6, 0.0]
    )
    columns = atlas.with_cells(doubled=total_counts() * 2)
    assert columns.pull("doubled").to_list() == [12.0, 14.0]
    selected = atlas.select_genes(["ENSG03", "ENSG01"])
    assert selected.gene_ids().to_list() == ["ENSG03", "ENSG01"]
    assert selected.pull(gene_column("symbol")).to_list() == ["MS4A1", "CD3E"]
    filtered = atlas.filter(gene_column("symbol") != "CD4")
    assert filtered.gene_ids().to_list() == ["ENSG01", "ENSG03"]
    assert filtered.count_genes() == 2


def test_no_cells_left(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts").filter(total_counts() > 100)
    assert atlas.count_cells() == 0
    assert atlas.cell_ids().to_list() == []
    series = atlas.pull("ENSG01")
    assert (len(series), series.dtype) == (0, pl.UInt32)
    assert atlas.select("ENSG01").columns == ["cell_id", "ENSG01"]


def test_counts_known_without_running(store: Path) -> None:
    """The plan knows these: nothing runs (progress is never called)."""
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))

    atlas = kyto.scan(store).select_genes(["ENSG02"])
    assert atlas.count_cells(progress=progress) == 2
    assert atlas.count_genes(progress=progress) == 1
    assert atlas.gene_ids(progress=progress).to_list() == ["ENSG02"]
    assert atlas.cell_ids(progress=progress).to_list() == ["AAAC-1", "AAAG-1"]
    assert calls == []
    filtered = atlas.filter(total_counts(layer="counts") >= 7)
    assert filtered.count_cells(progress=progress, threads=1) == 1
    assert calls[-1][0] == calls[-1][1] > 0


def test_reads_only_the_layers_the_features_read(two_layers: Path) -> None:
    """R1: the layer the feature does not read is deleted first; the run never opens it."""
    atlas = kyto.scan(two_layers)
    expected = atlas.pull("doubled:ENSG02").to_list()
    shutil.rmtree(two_layers / "layers" / "counts")
    assert atlas.pull("doubled:ENSG02").to_list() == expected
    assert atlas.select(d=gene_expression("ENSG02", layer="doubled")).height == 2
    assert atlas.filter(cell_column("cell_id") == "AAAG-1").count_cells() == 1


def test_collected_atlas_and_embeddings(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    collected = atlas.with_cells(total=total_counts()).collect(threads=1)
    assert collected.pull("total").to_list() == [6.0, 7.0]
    pca = atlas.pca(into="X_pca", n_components=1, seed=0)
    component = pca.pull(cell_embedding("X_pca", 0))
    assert component.to_list() == pytest.approx(
        pca.embedding("X_pca").to_numpy()[:, 0].tolist()
    )
    assert pca.pull("X_pca:0").to_list() == component.to_list()


def test_features_lists_full_names(store: Path) -> None:
    atlas = kyto.scan(store).normalize_total(10.0, layer="counts", into="normalized")
    frame = atlas.features()
    assert frame.columns == ["feature", "kind", "symbol"]
    assert frame.filter(pl.col("kind") == "gene").height == 6
    assert "normalized:ENSG03" in frame["feature"].to_list()
    selected = kyto.scan(store).select_genes(["ENSG02"]).features()
    assert selected["feature"].to_list() == ["meta:cell_id", "counts:ENSG02"]
    pca = kyto.scan(store).pca(layer="counts", into="X_pca", n_components=2, seed=0)
    components = pca.features().filter(pl.col("kind") == "embedding component")
    assert components["feature"].to_list() == ["X_pca:0", "X_pca:1"]
    # Every listed name resolves.
    for name in frame["feature"].to_list():
        atlas.resolve(kyto.feature(name))


def test_metadata_is_the_provenance(store: Path) -> None:
    metadata = kyto.scan(store).metadata()
    assert metadata["plan"] is None
    assert metadata["source"]["kind"] == "mtx"
    assert {"kyto_version", "dependencies", "seeds", "device", "run"} <= metadata.keys()


# Errors (R10), raised before anything runs.


def test_errors_are_coded(store: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    with pytest.raises(ResolutionError, match=r"KY-E210.*'CD3E \(gene symbol\)'"):
        atlas.pull("CD3E")
    with pytest.raises(ResolutionError, match="KY-E211"):
        kyto.scan(store).pull("ENSG01")
    with pytest.raises(ResolutionError, match="KY-E210"):
        atlas.pull_genes(["CD3"], by="symbol")
    with pytest.raises(ResolutionError, match="KY-E214"):
        atlas.pull_genes(["CD3E"], by="name")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ExpressionError, match="KY-E205"):
        atlas.pull(1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ExpressionError, match="KY-E205.*select\\(name=expression\\)"):
        atlas.select(total_counts())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ExpressionError, match="KY-E205"):
        atlas.select(x=1)  # pyright: ignore[reportArgumentType]
    for call, problem in (
        (atlas.select, "no feature is given"),
        (lambda: atlas.select("ENSG01", "ENSG01"), "given twice"),
        (lambda: atlas.select(cell_id="ENSG01"), "'cell_id' is the ID column"),
        (lambda: atlas.pull_genes([]), r"genes=\(\).*no feature is given"),
        (lambda: atlas.pull_genes(["ENSG01", "ENSG01"]), "genes='ENSG01'"),
        (lambda: atlas.pull_genes("ENSG01"), "KY-E215"),
        (lambda: atlas.pull("ENSG01", threads=0), "threads"),
        (lambda: atlas.count_cells(threads=0), "threads"),
        (lambda: atlas.cell_ids(progress=1), "progress"),  # pyright: ignore[reportArgumentType]
    ):
        with pytest.raises(PlanError, match=problem):
            call()


def test_store_replaced_after_scan_is_refused(store: Path, tmp_path: Path) -> None:
    atlas = kyto.scan(store).with_default_layer("counts")
    other = tmp_path / "other"
    other.mkdir()
    replacement = _store(other)
    kyto.scan(replacement).normalize_total(
        10.0, layer="counts", into="normalized"
    ).sink_zarr(store, overwrite=True)
    with pytest.raises(CorruptStoreError, match="KY-E127"):
        atlas.cell_ids()
    with pytest.raises(CorruptStoreError, match="KY-E127"):
        atlas.select("ENSG01")


# Examples (R12): runtime and stub docstrings.


def _docstrings() -> dict[str, tuple[str | None, str | None]]:
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    atlas = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Atlas"
    )
    stubs = {
        item.name: ast.get_docstring(item)
        for item in atlas.body
        if isinstance(item, ast.FunctionDef)
    }
    return {
        name: (getattr(_kyto.Atlas, name).__doc__, stubs.get(name)) for name in METHODS
    }


def test_stub_docstrings_equal_runtime() -> None:
    for name, (runtime, stub) in _docstrings().items():
        assert runtime is not None and stub is not None, name
        assert stub == runtime.strip(), name


@pytest.mark.parametrize("source", ["runtime", "stub"])
def test_docstring_examples_run(tmp_path: Path, source: str) -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for name, (runtime, stub) in _docstrings().items():
        docstring = runtime if source == "runtime" else stub
        assert docstring is not None and "Examples" in docstring, name
        folder = tmp_path / source / name
        folder.mkdir(parents=True)
        globs: dict[str, object] = {
            "scan": _kyto.scan,
            "store": _store(folder),
            "cell_column": cell_column,
            "gene_column": gene_column,
            "gene_expression": gene_expression,
            "total_counts": kyto.qc.total_counts,
        }
        test = parser.get_doctest(docstring, globs, f"Atlas.{name}", None, 0)
        before = runner.tries
        runner.run(test)
        assert runner.tries > before, f"{name} has no example"
    assert runner.failures == 0
