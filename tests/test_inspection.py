"""Task 8.3: ``schema()``, ``peek(n)``, ``features()``, ``estimate()`` and
``explain()`` read metadata only.

Done when: the I/O counter shows zero matrix-tile reads for each call. Two checks.
The executor test ``crates/kyto-exec/tests/it/inspection.rs`` counts every key
``peek``'s run reads through the storage: none under ``layers/``, ``embeddings/`` or
``graphs/``. Here, at the Python surface, the store's ``layers``, ``embeddings`` and
``graphs`` folders are deleted before the calls, so no matrix tile exists to be read,
and every call still gives what it gave on the whole store (a call that does read
the matrix then fails, which shows the deletion is felt).

``Atlas.schema()`` and ``Atlas.peek(n)`` are new with this task: the plan's output
schema as a dict, and the first ``n`` rows of its cell table from the store's tables
alone (``KY-E219`` when the cell table needs the matrix).
"""

import shutil
from pathlib import Path

import kyto
import polars as pl
import pytest
from kyto import _kyto, cell_column, gene_column
from kyto.errors import KytoError, PlanError, StaleColumnError
from kyto.qc import total_counts

N_CELLS = 30
N_GENES = 8


def _count(cell: int, gene: int) -> int:
    """The count of ``gene`` in ``cell``; 0 for most pairs."""
    return (cell * 7 + gene * 3) % 11 if (cell + 2 * gene) % 3 == 0 else 0


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A store with three layers, two cell columns, an embedding and two graphs."""
    source = tmp_path / "source"
    source.mkdir()
    entries = [
        f"{gene + 1} {cell + 1} {_count(cell, gene)}"
        for cell in range(N_CELLS)
        for gene in range(N_GENES)
        if _count(cell, gene) > 0
    ]
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{N_GENES} {N_CELLS} {len(entries)}\n" + "\n".join(entries) + "\n"
    )
    (source / "barcodes.tsv").write_text(
        "".join(f"cell_{cell:02d}\n" for cell in range(N_CELLS))
    )
    (source / "features.tsv").write_text(
        "".join(
            f"ENSG{gene:02d}\tSYM{gene}\tGene Expression\n" for gene in range(N_GENES)
        )
    )
    raw = tmp_path / "raw.kyto"
    kyto.convert(source, raw, layer="counts")
    target = tmp_path / "analysed.kyto"
    (
        kyto.scan(raw)
        .with_default_layer("counts")
        .with_cells(depth=total_counts(), deep=total_counts() >= 10)
        .normalize_total(1e4, into="norm")
        .log1p(layer="norm", into="log1p")
        .pca(layer="log1p", into="X_pca", n_components=2, seed=0)
        .neighbors("X_pca", n_neighbors=3)
        .sink_zarr(target)
    )
    return target


def _plans(atlas: _kyto.Atlas) -> list[_kyto.Atlas]:
    """Plans whose cell table needs metadata only: a scan, metadata steps, and steps
    on the matrix that change no cell column."""
    deep = atlas.filter(cell_column("deep")).with_cells(
        twice=cell_column("depth") * 2, shallow=~cell_column("deep")
    )
    return [
        atlas,
        deep,
        deep.take_cells(deep.peek(9)["cell_id"].to_list()[1:]),
        deep.filter(gene_column("symbol") != "SYM3"),
        atlas.with_default_layer("counts")
        .normalize_total(1e4, into="again")
        .scale(layer="log1p", into="scaled")
        .pca(layer="log1p", into="X_other", n_components=2, seed=1)
        .neighbors("X_other", n_neighbors=3, into="other")
        .filter(cell_column("depth") > 0),
    ]


def _inspect(atlas: _kyto.Atlas) -> list[object]:
    """What every inspection call gives."""
    return [
        atlas.schema(),
        atlas.peek(4).to_dict(as_series=False),
        atlas.peek(4).schema,
        atlas.features().rows(),
        atlas.estimate(),
        atlas.explain(),
        atlas.explain(format="json"),
        repr(atlas),
    ]


def test_inspection_reads_no_matrix_data(store: Path) -> None:
    built = _plans(kyto.scan(store))
    whole = [_inspect(atlas) for atlas in built]
    assert len(whole[0][1]["cell_id"]) == 4  # type: ignore[index]

    for member in ("layers", "embeddings", "graphs"):
        assert any((store / member).iterdir()), member
        shutil.rmtree(store / member)

    # Plans built before and after the matrix data went answer the same.
    assert [_inspect(atlas) for atlas in built] == whole
    assert [_inspect(atlas) for atlas in _plans(kyto.scan(store))] == whole

    # The deletion is felt: what reads the matrix now fails.
    atlas = kyto.scan(store)
    with pytest.raises(KytoError):
        atlas.pull(total_counts(layer="counts"))
    with pytest.raises(KytoError):
        atlas.embedding("X_pca").to_numpy()


def test_peek_is_the_head_of_the_cell_table(store: Path) -> None:
    for atlas in _plans(kyto.scan(store)):
        table = atlas.cells.to_polars()
        assert table.height > 6
        for n in (0, 1, 6, table.height, table.height + 5):
            peeked = atlas.peek(n)
            assert isinstance(peeked, pl.DataFrame)
            assert peeked.columns == table.columns
            expected = table.head(n)
            assert peeked.height == min(n, table.height)
            assert peeked.to_dict(as_series=False) == expected.to_dict(as_series=False)
            if n > 0:
                assert peeked.schema == expected.schema
        assert atlas.peek().equals(atlas.peek(5))
        assert atlas.peek().height == 5


def test_peek_of_a_plan_keeping_no_cell(store: Path) -> None:
    atlas = kyto.scan(store).filter(cell_column("depth") < 0)
    peeked = atlas.peek(3)
    assert peeked.height == 0
    assert peeked.columns == ["cell_id", "depth", "deep"]


def test_peek_leaves_out_a_stale_column_as_the_cell_table_does(store: Path) -> None:
    # `depth` and `deep` record their layer (lineage); a gene filter makes them stale,
    # so a sink and `cells.to_polars()` drop them and `select` refuses them (KY-E301).
    atlas = kyto.scan(store).filter(gene_column("symbol") != "SYM3")
    fresh = atlas.with_cells(batch=cell_column("cell_id") != "cell_00")
    assert atlas.peek(2).columns == atlas.cells.to_polars().columns == ["cell_id"]
    assert fresh.peek(2).columns == fresh.cells.to_polars().columns
    assert fresh.peek(2).rows() == [("cell_00", False), ("cell_01", True)]
    with pytest.raises(StaleColumnError):
        atlas.select("depth")


def test_peek_keeps_the_checks_a_run_makes_at_its_end(store: Path) -> None:
    # `take_cells` of a cell that an earlier filter removed is known only when the
    # run ends (KY-E210): `peek` must not answer for a plan that cannot run.
    atlas = kyto.scan(store)
    deep = atlas.filter(cell_column("deep"))
    kept = deep.cell_ids().to_list()
    gone = [cell for cell in atlas.cell_ids().to_list() if cell not in kept]
    plan = deep.take_cells([*kept[:3], gone[-1]])
    with pytest.raises(KytoError) as collected:
        plan.cells.to_polars()
    with pytest.raises(KytoError) as peeked:
        plan.peek(1)
    assert peeked.value.code == collected.value.code == "KY-E210"


def test_peek_returns_a_copy(store: Path) -> None:
    atlas = kyto.scan(store)
    first = atlas.peek(2)
    first = first.with_columns(pl.lit(0.0).alias("depth"))
    assert atlas.peek(2)["depth"].to_list() != [0.0, 0.0]


def test_peek_after_collect(store: Path) -> None:
    collected = (
        kyto.scan(store)
        .filter(cell_column("deep"))
        .with_cells(total=total_counts(layer="counts"))
        .collect()
    )
    peeked = collected.peek(3)
    assert peeked.columns == ["cell_id", "depth", "deep", "total"]
    assert peeked["total"].to_list() == peeked["depth"].to_list()
    assert peeked["deep"].to_list() == [True, True, True]


@pytest.mark.parametrize(
    ("step", "member"),
    [
        ("with_cells", "layer:counts"),
        ("filter_cells", "layer:log1p"),
        ("with_cells", "embedding:X_pca"),
        ("leiden", "graph:connectivities"),
        ("scrublet", "layer:counts"),
        ("with_cells", "layer:fresh"),
    ],
)
def test_peek_refuses_a_cell_table_that_needs_the_matrix(
    store: Path, step: str, member: str
) -> None:
    atlas = kyto.scan(store)
    plans = {
        ("with_cells", "layer:counts"): atlas.with_cells(
            total=total_counts(layer="counts")
        ),
        ("filter_cells", "layer:log1p"): atlas.filter(
            kyto.gene_expression("ENSG01", layer="log1p") > 0
        ).filter(cell_column("deep")),
        ("with_cells", "embedding:X_pca"): atlas.with_cells(
            first=kyto.cell_embedding("X_pca", 0)
        ),
        ("leiden", "graph:connectivities"): atlas.leiden(seed=0, into="cluster"),
        ("scrublet", "layer:counts"): atlas.scrublet(layer="counts", seed=0),
        ("with_cells", "layer:fresh"): atlas.log1p(
            layer="counts", into="fresh"
        ).with_cells(total=total_counts(layer="fresh")),
    }
    plan = plans[(step, member)]
    with pytest.raises(PlanError) as raised:
        plan.peek()
    error = raised.value
    assert error.code == "KY-E219"
    assert isinstance(error, ValueError)
    text = str(error)
    assert f"needs '{member}', which '{step}' reads" in text
    assert "cells.to_polars()" in text
    assert f"before '{step}'" in text
    # The same plan still inspects, and the plan before the step still peeks.
    assert plan.schema()["cells"]["columns"]
    assert atlas.peek(1).height == 1


@pytest.mark.parametrize("n", [-1, True, 1.5, "3", 2**64])
def test_peek_refuses_an_n_that_is_not_a_count(store: Path, n: object) -> None:
    with pytest.raises(PlanError) as raised:
        kyto.scan(store).peek(n)  # type: ignore[arg-type]
    assert raised.value.code == "KY-E215"
    assert "peek(..., n=" in str(raised.value)
    assert "an int of 0 or more" in str(raised.value)


def test_schema_of_a_scan_is_the_store_schema(store: Path) -> None:
    schema = kyto.scan(store).schema()
    assert schema["n_cells"] == N_CELLS
    assert schema["n_genes"] == N_GENES
    assert set(schema["tiling"]) == {"cell_block_size", "gene_block_size"}
    atlas = kyto.scan(store)
    assert schema["cells"] == atlas.cells.schema()
    assert schema["genes"] == atlas.genes.schema()
    assert [column["name"] for column in schema["cells"]["columns"]] == [
        "cell_id",
        "depth",
        "deep",
    ]
    assert schema["layers"] == [
        atlas.layer(name).schema() for name in ("counts", "log1p", "norm")
    ]
    assert schema["embeddings"] == [
        atlas.embedding(name).schema() for name in ("X_pca", "X_pca_loadings")
    ]
    assert schema["graphs"] == [
        atlas.graph(name).schema() for name in ("connectivities", "distances")
    ]
    assert schema["modalities"] == []
    assert list(schema) == [
        "n_cells",
        "n_genes",
        "tiling",
        "cells",
        "genes",
        "layers",
        "embeddings",
        "graphs",
        "modalities",
    ]


def test_schema_follows_the_plan(store: Path) -> None:
    atlas = kyto.scan(store)
    plan = (
        atlas.filter(cell_column("deep"))
        .with_cells(twice=cell_column("depth") * 2)
        .log1p(layer="counts", into="fresh")
        .pca(layer="fresh", into="X_fresh", n_components=2, seed=0)
    )
    schema = plan.schema()
    assert schema["n_cells"] is None
    assert schema["n_genes"] == N_GENES
    assert schema["cells"]["columns"][-1] == {"name": "twice", "dtype": "Float64"}
    assert [layer["name"] for layer in schema["layers"]][-1] == "fresh"
    assert "X_fresh" in [embedding["name"] for embedding in schema["embeddings"]]
    genes = atlas.filter(gene_column("symbol") != "SYM3").schema()
    assert (genes["n_cells"], genes["n_genes"]) == (N_CELLS, None)
    selected = atlas.select_genes(["ENSG01", "ENSG05"]).schema()
    assert selected["n_genes"] == 2
    # The earlier plan is unchanged (R2), and each call gives a new dict.
    assert atlas.schema()["n_cells"] == N_CELLS
    schema["layers"].clear()
    schema["cells"]["columns"].clear()
    assert plan.schema()["layers"] and plan.schema()["cells"]["columns"]
