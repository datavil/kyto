"""Task 7.2: the chain methods of ``Atlas`` build the plan lazily; the terminals run it.

Every chain method returns a new ``Atlas`` and reads nothing (R1, R2): the plans are
built on a store whose matrix is deleted first. Arguments are checked when the plan is
built, with coded errors (R10). ``collect`` and ``sink_zarr`` run the plan through
``run_plan`` (5.7) with a memory limit (5.5). The Done-when (the README pipeline on a
fixture) is ``tests/test_readme.py``. The executor's results are checked in Rust
(``crates/kyto-exec/tests/it/``); here, that the terminals hand them over unchanged.
"""

import ast
import doctest
import gc
import gzip
import shutil
from pathlib import Path

import kyto
import kyto.qc
import polars as pl
import pytest
from kyto import _kyto, cell_column, gene_column


def _store(folder: Path) -> Path:
    """A 2-cell, 3-gene store with one ``counts`` layer (as ``tests/test_atlas.py``)."""
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


def _chain(atlas: _kyto.Atlas) -> _kyto.Atlas:
    """Every chain method once (no terminal)."""
    return (
        atlas.with_default_layer("counts")
        .with_cells(n_counts=kyto.qc.total_counts())
        .with_genes(is_cd3e=gene_column("symbol") == "CD3E")
        .filter(cell_column("n_counts") > 0)
        .filter(gene_column("feature_type") == "Gene Expression")
        .select_genes(["ENSG01", "ENSG02", "ENSG03"])
        .take_cells(["AAAC-1", "AAAG-1"])
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .scale(layer="log1p", into="scaled", max_value=10)
        .highly_variable_genes(layer="log1p", n_top_genes=2)
        .pca(layer="log1p", into="pca", n_components=1, seed=0)
        .neighbors("pca", n_neighbors=2)
        .leiden(into="leiden", seed=0)
        .louvain(resolution=0.5, seed=1, into="louvain")
        .umap(into="X_umap", seed=0)
        .on(device="cpu")
    )


def test_chain_methods_read_nothing_and_leave_the_input_unchanged(store: Path) -> None:
    atlas = _kyto.scan(store)
    before = str(atlas)
    # R1: no layer is read while the plan is built.
    shutil.rmtree(store / "layers")
    chained = _chain(atlas)
    assert str(atlas) == before
    lines = str(chained).splitlines()
    assert lines[0].startswith('UMAP "connectivities" -> "X_umap"')
    assert lines[-1] == before
    assert len(lines) == 17  # `on(device="cpu")` adds no step
    frame = atlas.with_default_layer("counts").rank_genes_groups("cell_id")
    assert isinstance(frame, pl.LazyFrame)
    assert frame.collect_schema() == pl.Schema(
        {
            "group": pl.String,
            "gene_id": pl.String,
            "statistic": pl.Float64,
            "log_fold_change": pl.Float64,
            "p_value": pl.Float64,
            "adjusted_p_value": pl.Float64,
        }
    )


def test_filter_takes_cell_and_gene_predicates(store: Path) -> None:
    atlas = _kyto.scan(store)
    cells = atlas.filter(kyto.qc.total_counts(layer="counts") >= 6)
    genes = atlas.filter(gene_column("symbol") == "CD4")
    assert str(cells).splitlines()[0].startswith("FILTER_CELLS ")
    assert str(genes).splitlines()[0].startswith("FILTER_GENES ")
    for wrong in (1, "symbol", cell_column("cell_id")):
        with pytest.raises(kyto.ExpressionError) as raised:
            atlas.filter(wrong)  # pyright: ignore[reportArgumentType]
        assert raised.value.code == "KY-E205"


def test_selections_and_columns_are_checked_when_built(store: Path) -> None:
    atlas = _kyto.scan(store)
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.select_genes(["ENSG04"])
    assert raised.value.code == "KY-E210"
    assert "ENSG0" in str(raised.value)
    with pytest.raises(kyto.PlanError) as raised:
        atlas.select_genes(["ENSG01", "ENSG01"])
    assert raised.value.code == "KY-E215"
    with pytest.raises(kyto.RemovedFeatureError) as raised:
        atlas.select_genes(["ENSG01"]).select_genes(["ENSG02"])
    assert raised.value.code == "KY-E203"
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.take_cells([""])
    assert raised.value.code == "KY-E210"
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.with_cells(n=gene_column("symbol"))  # pyright: ignore[reportArgumentType]
    assert raised.value.code == "KY-E205"
    assert "'with_cells' needs a CellExpr" in str(raised.value)
    with pytest.raises(kyto.ExpressionError) as raised:
        atlas.with_genes(n=3)  # pyright: ignore[reportArgumentType]
    assert raised.value.code == "KY-E205"
    with pytest.raises(kyto.PlanError) as raised:
        atlas.with_cells(cell_id=cell_column("cell_id"))
    assert raised.value.code == "KY-E215"
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.with_cells(n=cell_column("n_counts"))
    assert raised.value.code == "KY-E210"


def test_algorithm_arguments_are_coded(store: Path) -> None:
    atlas = _kyto.scan(store)
    counts = atlas.with_default_layer("counts")
    cases: list[tuple[type[kyto.KytoError], str, object]] = [
        (kyto.ResolutionError, "KY-E211", lambda: atlas.log1p(into="log1p")),
        (kyto.MissingMemberError, "KY-E109", lambda: atlas.log1p(layer="x", into="y")),
        (kyto.PlanError, "KY-E215", lambda: counts.normalize_total(-1, into="n")),
        (kyto.PlanError, "KY-E215", lambda: counts.scale(into="s", max_value=0)),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.highly_variable_genes(flavor="pearson"),  # pyright: ignore[reportArgumentType]
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.highly_variable_genes(n_top_genes=-1),
        ),
        (kyto.PlanError, "KY-E215", lambda: counts.pca(into="p", seed=-1)),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.pca(into="p", n_components=0, seed=0),
        ),
        (kyto.MissingMemberError, "KY-E109", lambda: counts.neighbors("pca")),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.pca(into="p", n_components=1, seed=0).neighbors(
                "p",
                metric="manhattan",  # pyright: ignore[reportArgumentType]
            ),
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.pca(into="p", n_components=1, seed=0).neighbors(
                "p",
                search="hnsw",  # pyright: ignore[reportArgumentType]
            ),
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.rank_genes_groups("cell_id", method="logreg"),  # pyright: ignore[reportArgumentType]
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.rank_genes_groups("cell_id", n_genes=0),
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: counts.rank_genes_groups("cell_id", threads=0),
        ),
        (
            kyto.MissingMemberError,
            "KY-E109",
            lambda: counts.rank_genes_groups("cluster"),
        ),
        (
            kyto.ResolutionError,
            "KY-E211",
            lambda: atlas.rank_genes_groups("cell_id"),
        ),
        (
            kyto.NotImplementedYetError,
            "KY-E601",
            lambda: atlas.with_embedding("X_scvi", None),  # pyright: ignore[reportArgumentType]
        ),
        (
            kyto.NotImplementedYetError,
            "KY-E601",
            lambda: atlas.on(device="cuda"),
        ),
        (
            kyto.PlanError,
            "KY-E215",
            lambda: atlas.on(device="gpu"),  # pyright: ignore[reportArgumentType]
        ),
    ]
    for error, code, call in cases:
        assert callable(call)
        with pytest.raises(error) as raised:
            call()
        assert raised.value.code == code, raised.value


def test_sink_zarr_writes_the_plan_output(store: Path, tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    plan = (
        _kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(n_counts=kyto.qc.total_counts())
        .filter(cell_column("n_counts") >= 7)
        .normalize_total(10, into="normalized")
    )
    target = tmp_path / "out.kyto"
    result = plan.sink_zarr(
        target,
        memory_limit="64MiB",
        threads=1,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert result is None
    assert calls[-1][0] == calls[-1][1] > 0
    written = _kyto.scan(target)
    assert str(written).endswith("(1 cells x 3 genes)")
    assert written.cells.names() == ["cell_id", "n_counts"]
    assert [layer["name"] for layer in written.cells.schema()["columns"]] == [
        "cell_id",
        "n_counts",
    ]
    assert written.layer("normalized").schema()["dtype"] == "float32"
    # Never over an existing target (overwrite is 7.3's).
    with pytest.raises(kyto.StoreInputError) as raised:
        plan.sink_zarr(target)
    assert raised.value.code == "KY-E118"
    # A sink cannot be chained, and the plan itself is unchanged.
    assert not str(plan).startswith("SINK")


def test_terminal_arguments_are_coded(store: Path, tmp_path: Path) -> None:
    atlas = _kyto.scan(store)
    target = tmp_path / "out.kyto"
    for call in (
        lambda: atlas.sink_zarr(target, threads=0),
        lambda: atlas.sink_zarr(target, progress=3),  # pyright: ignore[reportArgumentType]
        lambda: atlas.sink_zarr(target, memory_limit="lots"),
        lambda: atlas.sink_zarr(target, memory_limit=0),
        lambda: atlas.sink_zarr(target, memory_limit=True),
        lambda: atlas.collect(memory_limit=-5),
        lambda: atlas.collect(memory_limit=1.5),  # pyright: ignore[reportArgumentType]
    ):
        with pytest.raises(kyto.PlanError) as raised:
            call()
        assert raised.value.code == "KY-E215"
    with pytest.raises(kyto.MemoryLimitError) as raised:
        atlas.sink_zarr(target, memory_limit=16)
    assert raised.value.code == "KY-E401"
    assert not target.exists()
    with pytest.raises(kyto.NotImplementedYetError) as raised:
        atlas.with_default_layer("counts").pca(
            into="pca", n_components=1, seed=0
        ).sink_zarr(target)
    assert raised.value.code == "KY-E601"
    assert "5.11" in str(raised.value)
    assert not target.exists()


def test_collect_materializes_into_a_temporary_store(store: Path) -> None:
    plan = (
        _kyto.scan(store)
        .with_default_layer("counts")
        .filter(kyto.qc.total_counts() >= 7)
        .log1p(into="log1p")
    )
    collected = plan.collect(threads=1)
    lines = str(collected).splitlines()
    assert lines[0] == 'WITH_DEFAULT_LAYER "counts"'
    assert lines[1].endswith('atlas.kyto" (1 cells x 3 genes)')
    folder = Path(lines[1].split('"')[1]).parent
    assert folder.is_dir()
    # Plans built on it keep the store alive; it goes with the last one.
    derived = collected.select_genes(["ENSG01"])
    del collected
    gc.collect()
    assert folder.is_dir()
    assert derived.layer("log1p").schema()["dtype"] == "float32"
    del derived
    gc.collect()
    assert not folder.exists()


def test_rank_genes_groups_runs_only_when_collected(store: Path) -> None:
    atlas = _kyto.scan(store).with_default_layer("counts")
    # A cell that does not exist is found only when the plan runs.
    lazy = atlas.take_cells(["AAAC-1", "missing"]).rank_genes_groups("cell_id")
    with pytest.raises(kyto.ResolutionError) as raised:
        lazy.collect()
    assert raised.value.code == "KY-E210"
    # Two groups of one cell each: a group needs two cells (scanpy's rule).
    with pytest.raises(kyto.PlanError) as raised:
        atlas.rank_genes_groups("cell_id").collect()
    assert raised.value.code == "KY-E215"


def _docstrings() -> dict[str, tuple[str | None, str | None]]:
    """Runtime and stub docstrings of the methods 7.2 adds."""
    methods = (
        "filter",
        "select_genes",
        "take_cells",
        "with_cells",
        "with_genes",
        "with_embedding",
        "normalize_total",
        "log1p",
        "scale",
        "highly_variable_genes",
        "pca",
        "neighbors",
        "leiden",
        "louvain",
        "umap",
        "rank_genes_groups",
        "on",
        "collect",
        "sink_zarr",
    )
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
        name: (getattr(_kyto.Atlas, name).__doc__, stubs.get(name)) for name in methods
    }


@pytest.mark.parametrize("source", ["runtime", "stub"])
def test_docstring_examples_run(tmp_path: Path, source: str) -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for name, (runtime, stub) in _docstrings().items():
        docstring = runtime if source == "runtime" else stub
        assert docstring is not None, name
        assert "Examples" in docstring, name
        folder = tmp_path / source / name
        folder.mkdir(parents=True)
        globs: dict[str, object] = {
            "scan": _kyto.scan,
            "store": _store(folder),
            "cell_column": cell_column,
            "gene_column": gene_column,
            "total_counts": kyto.qc.total_counts,
        }
        test = parser.get_doctest(docstring, globs, f"Atlas.{name}", None, 0)
        before = runner.tries
        runner.run(test)
        assert runner.tries > before, f"{name} has no example"
    assert runner.failures == 0


def test_python_values_of_the_wrong_kind_are_coded(store: Path) -> None:
    """Bug-hunter: PyO3's own conversions raised uncoded TypeError/OverflowError."""
    atlas = _kyto.scan(store).with_default_layer("counts")
    pca = atlas.pca(into="p", n_components=1, seed=0)
    calls = [
        lambda: atlas.select_genes("ENSG01"),
        lambda: atlas.select_genes(None),  # pyright: ignore[reportArgumentType]
        lambda: atlas.select_genes([1]),  # pyright: ignore[reportArgumentType]
        lambda: atlas.take_cells("AAAC-1"),
        lambda: atlas.pca(into="p", n_components=2**70, seed=0),
        lambda: atlas.pca(into="p", n_components=True, seed=0),
        lambda: atlas.pca(into="p", n_components=1.5, seed=0),  # pyright: ignore[reportArgumentType]
        lambda: atlas.pca(into="p", seed=2**64),
        lambda: atlas.highly_variable_genes(n_top_genes=True),
        lambda: atlas.highly_variable_genes(n_top_genes=0),
        lambda: pca.neighbors("p", n_neighbors=2**63 * 2),
        lambda: pca.neighbors("p", seed=-1),
        lambda: atlas.normalize_total("a", into="n"),  # pyright: ignore[reportArgumentType]
        lambda: atlas.normalize_total(True, into="n"),
        lambda: atlas.scale(into="s", max_value="1"),  # pyright: ignore[reportArgumentType]
        lambda: atlas.rank_genes_groups("cell_id", groups="AAAC-1"),
        lambda: atlas.rank_genes_groups("cell_id", groups=[1]),  # pyright: ignore[reportArgumentType]
        lambda: atlas.rank_genes_groups("cell_id", n_genes=2**70),
        lambda: atlas.rank_genes_groups("cell_id", threads=True),
        lambda: atlas.collect(threads=2**70),
        lambda: atlas.collect(threads=1.0),  # pyright: ignore[reportArgumentType]
        lambda: atlas.with_cells(),
        lambda: atlas.with_genes(),
        lambda: pca.neighbors("p", n_neighbors=2).leiden(seed=True, into="c"),
        lambda: pca.neighbors("p", n_neighbors=2).louvain(
            resolution=0, seed=0, into="c"
        ),
        lambda: pca.neighbors("p", n_neighbors=2).umap(into="u", seed=0, init="pca"),  # pyright: ignore[reportArgumentType]
        lambda: pca.neighbors("p", n_neighbors=2).umap(
            into="u", seed=0, n_components=101
        ),
        lambda: pca.neighbors("p", n_neighbors=2).umap(into="u", seed=0, min_dist="0"),  # pyright: ignore[reportArgumentType]
    ]
    for number, call in enumerate(calls):
        with pytest.raises(kyto.PlanError) as raised:
            call()
        assert raised.value.code == "KY-E215", number


def test_a_frame_keeps_a_collected_store_alive(store: Path) -> None:
    """Bug-hunter: the deferred DE held only the plan, not the temporary store."""
    atlas = _kyto.scan(store).with_default_layer("counts")
    frame = atlas.collect().rank_genes_groups("cell_id")
    gc.collect()
    # The plan runs (two one-cell groups are refused by the test, not by a lost store).
    with pytest.raises(kyto.PlanError) as raised:
        frame.collect()
    assert raised.value.code == "KY-E215"
