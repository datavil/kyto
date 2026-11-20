"""Task 10.1: property tests of the Python API (hypothesis).

Two properties, each with a fixed seed (``derandomize``) and a small number of cases, so
``pytest`` draws the same cases on every run and every machine:

- **Counts round-trip, lazy equals eager.** A random count matrix of any shape (empty
  included) written as Matrix Market and converted reads back exactly; ``normalize_total``
  then ``log1p`` equal a NumPy reference computed eagerly in scanpy's ``float32`` steps;
  and the result has the same bits on 1 thread and on more (R11).
- **Random chains of valid methods never crash.** A chain of chain methods drawn with
  valid arguments (names that exist, numbers in range) on a small store, then every
  plan-only terminal and a ``collect`` on 1 thread and on more. Each step either works
  or raises a coded ``KytoError`` (R10): never another exception, never a Rust panic.
  Runs that succeed give the same bits on any number of threads (R11).

The Rust properties (storage round trips, kernels, threads and block sizes, optimized
against logical plans) are the crates' proptests, configured by
``tests/property_cases.rs``. Both suites read two environment variables for long runs:
``KYTO_PROPERTY_SCALE`` multiplies the number of cases, ``KYTO_PROPERTY_SEED`` draws
other cases (``KYTO_PROPERTY_SCALE=20 uv run pytest tests/test_properties.py``).
"""

# scipy ships without complete type information; strict pyright cannot see its types.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

import gzip
import itertools
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

import kyto
import kyto.qc
import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from kyto import _kyto, gene_column
from numpy.typing import NDArray
from polars.testing import assert_frame_equal

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., None])


def _positive(name: str, default: int) -> int:
    """A positive integer from the environment variable ``name``, else ``default``."""
    text = os.environ.get(name)
    if text is None:
        return default
    if not text.strip().isdigit() or int(text) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {text!r}")
    return int(text)


def _property(base_examples: int) -> Callable[[_F], _F]:
    """Settings of a property: ``base_examples`` times ``KYTO_PROPERTY_SCALE`` cases,
    derandomized unless ``KYTO_PROPERTY_SEED`` is set (then seeded with it), no example
    database (nothing written into the repo), no deadline (a run's time varies)."""

    def decorate(test: _F) -> _F:
        configured = settings(
            max_examples=base_examples * _positive("KYTO_PROPERTY_SCALE", 1),
            derandomize="KYTO_PROPERTY_SEED" not in os.environ,
            database=None,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        )(test)
        if "KYTO_PROPERTY_SEED" in os.environ:
            configured = seed(_positive("KYTO_PROPERTY_SEED", 1))(configured)
        return configured

    return decorate


_CODE = re.compile(r"KY-E\d{3}")


def _attempt(action: Callable[[], _T]) -> _T | None:
    """Run ``action``; return ``None`` if it raised a coded ``KytoError`` (R10).

    Any other exception (a ``TypeError`` from the bindings, a Rust panic, which PyO3
    raises as a ``BaseException``) fails the test.
    """
    try:
        return action()
    except (KeyboardInterrupt, SystemExit):
        raise
    # A Rust panic reaches Python as PyO3's PanicException, a BaseException.
    except BaseException as error:  # noqa: BLE001
        if not isinstance(error, kyto.KytoError):
            pytest.fail(
                f"uncoded error {type(error).__module__}.{type(error).__name__}: {error}"
            )
        assert _CODE.fullmatch(error.code), error.code
        assert error.code in str(error), str(error)
        assert error.hint, f"{error.code} has no hint"
        event(f"coded error {error.code}")
        return None


# Stores -------------------------------------------------------------------------------


_stores = itertools.count()


def _write_store(folder: Path, counts: NDArray[np.uint32]) -> Path:
    """Write ``counts`` (cells x genes) as 10x Matrix Market and convert it to a store.

    Cells are ``C0, C1, ...``; genes ``G0, G1, ...`` with symbols ``S0, S1, ...`` except
    the first two, ``MT-A`` and ``MT-B`` (a mitochondrial gene set). Returns the store.
    """
    number = next(_stores)
    source = folder / f"source{number}"
    source.mkdir()
    n_cells, n_genes = counts.shape
    cells, genes = np.nonzero(counts)
    lines = [
        f"{gene + 1} {cell + 1} {counts[cell, gene]}\n"
        for cell, gene in zip(cells, genes, strict=True)
    ]
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{n_genes} {n_cells} {len(lines)}\n" + "".join(lines)
    )
    (source / "barcodes.tsv").write_text(
        "".join(f"C{cell}\n" for cell in range(n_cells))
    )
    symbols = [
        ("MT-A", "MT-B")[gene] if gene < 2 else f"S{gene}" for gene in range(n_genes)
    ]
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            "".join(
                f"G{gene}\t{symbol}\tGene Expression\n"
                for gene, symbol in enumerate(symbols)
            ).encode()
        )
    )
    store = folder / f"store{number}.kyto"
    kyto.convert(source, store, layer="counts")
    return store


def _same_csr(first: Any, second: Any) -> None:
    """Two SciPy CSR matrices with the same shape, structure and value bits."""
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    np.testing.assert_array_equal(first.indptr, second.indptr)
    np.testing.assert_array_equal(first.indices, second.indices)
    assert first.data.tobytes() == second.data.tobytes()


def _same_bits(first: _kyto.Atlas, second: _kyto.Atlas) -> None:
    """Two materialized plans with the same schema and the same bits in every member."""
    schema = first.schema()
    assert schema == second.schema()
    assert_frame_equal(
        first.cells.to_polars(), second.cells.to_polars(), check_exact=True
    )
    assert_frame_equal(
        first.genes.to_polars(), second.genes.to_polars(), check_exact=True
    )
    for layer in schema["layers"]:
        name = layer["name"]
        _same_csr(first.layer(name).to_scipy(), second.layer(name).to_scipy())
    for embedding in schema["embeddings"]:
        name = embedding["name"]
        values = first.embedding(name).to_numpy()
        other = second.embedding(name).to_numpy()
        assert values.dtype == other.dtype and values.shape == other.shape, name
        assert values.tobytes() == other.tobytes(), name
    for graph in schema["graphs"]:
        name = graph["name"]
        _same_csr(first.graph(name).to_scipy(), second.graph(name).to_scipy())


# Property 1: round trip, lazy against eager -------------------------------------------


@st.composite
def _counts(draw: st.DrawFn) -> NDArray[np.uint32]:
    """A count matrix of 0 to 12 cells and 0 to 8 genes, mostly zeros."""
    shape = draw(st.tuples(st.integers(0, 12), st.integers(0, 8)))
    values = st.one_of(
        st.just(0), st.just(0), st.integers(1, 5), st.integers(1, 100_000)
    )
    return draw(hnp.arrays(np.uint32, shape, elements=values))


def _eager_log1p(
    counts: NDArray[np.uint32], target_sum: float | None
) -> NDArray[np.float32]:
    """scanpy's ``normalize_total`` then ``log1p``, eagerly, in its ``float32`` steps.

    The target is the median of the non-zero totals when ``target_sum`` is ``None``; a
    cell with no counts stays 0.
    """
    values = counts.astype(np.float32)
    totals = values.sum(axis=1, dtype=np.float64).astype(np.float32)
    if target_sum is None:
        positive = totals[totals > 0]
        target = np.float32(np.median(positive)) if positive.size else np.float32(1.0)
    else:
        target = np.float32(target_sum)
    divisor = np.where(totals > 0, totals / target, np.float32(1.0)).astype(np.float32)
    return np.log1p(values / divisor[:, None]).astype(np.float32)


@_property(base_examples=25)
@given(
    counts=_counts(),
    target_sum=st.sampled_from([None, 1.0, 1e4]),
    threads=st.integers(2, 4),
)
def test_counts_round_trip_and_lazy_log1p_equals_eager(
    tmp_path_factory: pytest.TempPathFactory,
    counts: NDArray[np.uint32],
    target_sum: float | None,
    threads: int,
) -> None:
    store = _write_store(tmp_path_factory.mktemp("round_trip"), counts)
    atlas = _kyto.scan(store)
    assert (atlas.count_cells(), atlas.count_genes()) == counts.shape
    stored = atlas.layer("counts").to_scipy()
    assert stored.dtype == np.uint32
    np.testing.assert_array_equal(stored.toarray(), counts)
    plan = (
        atlas.with_default_layer("counts")
        .normalize_total(target_sum, into="normalized")
        .log1p(layer="normalized", into="logged")
    )
    found = plan.layer("logged").to_scipy(threads=1)
    assert found.dtype == np.float32
    expected = _eager_log1p(counts, target_sum)
    np.testing.assert_allclose(found.toarray(), expected, rtol=2e-6, atol=1e-7)
    # Zeros are not stored (R9): exactly the input's non-zero entries come back.
    assert found.nnz == np.count_nonzero(counts)
    _same_csr(plan.layer("logged").to_scipy(threads=threads), found)


# Property 2: random chains never crash ------------------------------------------------


_N_CELLS = 30
_N_GENES = 14
_LAYERS = ("counts", "normalized", "logged", "scaled")


@pytest.fixture(scope="module")
def chain_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 30-cell, 14-gene store: two cell types, an all-zero cell and an all-zero gene."""
    generator = np.random.default_rng(101)
    counts = generator.integers(0, 4, size=(_N_CELLS, _N_GENES)).astype(np.uint32)
    counts[: _N_CELLS // 2, 2:6] += generator.integers(
        5, 30, size=(_N_CELLS // 2, 4)
    ).astype(np.uint32)
    counts[_N_CELLS // 2 :, 6:10] += generator.integers(
        5, 30, size=(_N_CELLS // 2, 4)
    ).astype(np.uint32)
    counts[3, :] = 0
    counts[:, 11] = 0
    return _write_store(tmp_path_factory.mktemp("chains"), counts)


@dataclass(frozen=True)
class _Chain:
    """A plan and what it holds so far, for drawing the next step's valid arguments."""

    atlas: _kyto.Atlas
    layers: tuple[str, ...] = ("counts",)
    cell_embeddings: tuple[tuple[str, int], ...] = ()
    graphs: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    cell_columns: tuple[str, ...] = ()
    has_hvg: bool = False
    steps: tuple[str, ...] = field(default=())

    def add(self, atlas: _kyto.Atlas, step: str, **changes: Any) -> "_Chain":
        return replace(self, atlas=atlas, steps=(*self.steps, step), **changes)


def _layer(data: st.DataObject, chain: _Chain) -> str | None:
    """An existing layer, or ``None`` (the default layer, ``counts``)."""
    return data.draw(st.sampled_from((None, *chain.layers)), label="layer")


def _into(data: st.DataObject, chain: _Chain) -> tuple[str, tuple[str, ...]]:
    """A layer to write (new or replacing one) and the layers after writing it."""
    into = data.draw(st.sampled_from(_LAYERS), label="into")
    return into, tuple(dict.fromkeys((*chain.layers, into)))


def _cell_metric(data: st.DataObject, chain: _Chain) -> _kyto.CellExpr[Any]:
    layer = _layer(data, chain)
    mitochondrial = gene_column("symbol").str.starts_with("MT-")
    choices = [
        lambda: kyto.qc.total_counts(layer=layer),
        lambda: kyto.qc.genes_detected(layer=layer),
        lambda: kyto.qc.percent_counts_in(mitochondrial, layer=layer),
        lambda: kyto.qc.total_counts_in(mitochondrial, layer=layer).log1p(),
    ]
    return data.draw(st.sampled_from(choices), label="cell metric")()


def _gene_metric(data: st.DataObject, chain: _Chain) -> _kyto.GeneExpr[Any]:
    layer = _layer(data, chain)
    choices = [
        lambda: kyto.qc.cells_detected(layer=layer),
        lambda: kyto.qc.gene_total_counts(layer=layer),
        lambda: kyto.qc.gene_mean_counts(layer=layer),
        lambda: kyto.qc.percent_dropout(layer=layer),
    ]
    return data.draw(st.sampled_from(choices), label="gene metric")()


def _with_cells(data: st.DataObject, chain: _Chain) -> _Chain | None:
    name = data.draw(st.sampled_from(("metric_a", "metric_b")), label="name")
    metric = _cell_metric(data, chain)
    atlas = _attempt(lambda: chain.atlas.with_cells(**{name: metric}))
    columns = tuple(dict.fromkeys((*chain.cell_columns, name)))
    return atlas and chain.add(atlas, f"with_cells({name})", cell_columns=columns)


def _with_genes(data: st.DataObject, chain: _Chain) -> _Chain | None:
    metric = _gene_metric(data, chain)
    atlas = _attempt(lambda: chain.atlas.with_genes(gene_metric=metric))
    return atlas and chain.add(atlas, "with_genes")


def _filter_cells(data: st.DataObject, chain: _Chain) -> _Chain | None:
    threshold = data.draw(st.sampled_from((0.0, 1.0, 20.0, 1e9)), label="threshold")
    metric = _cell_metric(data, chain)
    atlas = _attempt(lambda: chain.atlas.filter(metric >= threshold))
    return atlas and chain.add(atlas, f"filter(cells >= {threshold})")


def _filter_genes(data: st.DataObject, chain: _Chain) -> _Chain | None:
    if data.draw(st.booleans(), label="by symbol"):
        predicate = ~gene_column("symbol").str.starts_with("MT-")
    else:
        threshold = data.draw(st.sampled_from((0.0, 1.0, 3.0, 1e9)), label="threshold")
        predicate = _gene_metric(data, chain) >= threshold
    atlas = _attempt(lambda: chain.atlas.filter(predicate))
    return atlas and chain.add(atlas, "filter(genes)")


def _select_genes(data: st.DataObject, chain: _Chain) -> _Chain | None:
    ids = data.draw(
        st.lists(
            st.sampled_from([f"G{gene}" for gene in range(_N_GENES)]), unique=True
        ),
        label="gene ids",
    )
    atlas = _attempt(lambda: chain.atlas.select_genes(ids))
    return atlas and chain.add(atlas, f"select_genes({len(ids)})")


def _take_cells(data: st.DataObject, chain: _Chain) -> _Chain | None:
    ids = data.draw(
        st.lists(
            st.sampled_from([f"C{cell}" for cell in range(_N_CELLS)]), unique=True
        ),
        label="cell ids",
    )
    atlas = _attempt(lambda: chain.atlas.take_cells(ids))
    return atlas and chain.add(atlas, f"take_cells({len(ids)})")


def _normalize_total(data: st.DataObject, chain: _Chain) -> _Chain | None:
    target = data.draw(st.sampled_from((None, 1.0, 1e4)), label="target_sum")
    layer = _layer(data, chain)
    into, layers = _into(data, chain)
    atlas = _attempt(
        lambda: chain.atlas.normalize_total(target, layer=layer, into=into)
    )
    return atlas and chain.add(
        atlas, f"normalize_total({layer}->{into})", layers=layers
    )


def _log1p(data: st.DataObject, chain: _Chain) -> _Chain | None:
    layer = _layer(data, chain)
    into, layers = _into(data, chain)
    atlas = _attempt(lambda: chain.atlas.log1p(layer=layer, into=into))
    return atlas and chain.add(atlas, f"log1p({layer}->{into})", layers=layers)


def _scale(data: st.DataObject, chain: _Chain) -> _Chain | None:
    layer = _layer(data, chain)
    into, layers = _into(data, chain)
    zero_center = data.draw(st.booleans(), label="zero_center")
    max_value = data.draw(st.sampled_from((None, 0.5, 10.0)), label="max_value")
    atlas = _attempt(
        lambda: chain.atlas.scale(
            layer=layer, into=into, zero_center=zero_center, max_value=max_value
        )
    )
    return atlas and chain.add(atlas, f"scale({layer}->{into})", layers=layers)


def _highly_variable_genes(data: st.DataObject, chain: _Chain) -> _Chain | None:
    layer = _layer(data, chain)
    flavor = data.draw(
        st.sampled_from(("seurat", "cell_ranger", "seurat_v3")), label="flavor"
    )
    n_top_genes = data.draw(
        st.one_of(st.none(), st.integers(1, _N_GENES + 2)), label="n_top"
    )
    batch_key = data.draw(st.sampled_from((None, *chain.labels)), label="batch_key")
    atlas = _attempt(
        lambda: chain.atlas.highly_variable_genes(
            layer=layer, flavor=flavor, n_top_genes=n_top_genes, batch_key=batch_key
        )
    )
    return atlas and chain.add(atlas, f"highly_variable_genes({flavor})", has_hvg=True)


def _pca(data: st.DataObject, chain: _Chain) -> _Chain | None:
    layer = _layer(data, chain)
    into = data.draw(st.sampled_from(("pca", "pca_b")), label="into")
    n_components = data.draw(st.integers(1, 6), label="n_components")
    genes = (
        gene_column("highly_variable")
        if chain.has_hvg and data.draw(st.booleans())
        else None
    )
    atlas = _attempt(
        lambda: chain.atlas.pca(
            layer=layer, into=into, n_components=n_components, seed=7, genes=genes
        )
    )
    embeddings = tuple(
        (name, width) for name, width in chain.cell_embeddings if name != into
    ) + ((into, n_components),)
    return atlas and chain.add(
        atlas, f"pca({layer}->{into})", cell_embeddings=embeddings
    )


def _neighbors(data: st.DataObject, chain: _Chain) -> _Chain | None:
    if not chain.cell_embeddings:
        return None
    embedding, width = data.draw(
        st.sampled_from(chain.cell_embeddings), label="embedding"
    )
    n_neighbors = data.draw(st.integers(2, 8), label="n_neighbors")
    n_components = data.draw(
        st.one_of(st.none(), st.integers(1, width)), label="n_comps"
    )
    metric = data.draw(st.sampled_from(("euclidean", "cosine")), label="metric")
    search = data.draw(st.sampled_from(("exact", "approximate")), label="search")
    into = data.draw(st.sampled_from((None, "knn")), label="into")
    atlas = _attempt(
        lambda: chain.atlas.neighbors(
            embedding,
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            search=search,
            seed=3,
            into=into,
        )
    )
    graph = "connectivities" if into is None else f"{into}_connectivities"
    graphs = tuple(dict.fromkeys((*chain.graphs, graph)))
    return atlas and chain.add(atlas, f"neighbors({embedding}->{graph})", graphs=graphs)


def _cluster(data: st.DataObject, chain: _Chain) -> _Chain | None:
    if not chain.graphs:
        return None
    graph = data.draw(st.sampled_from(chain.graphs), label="graph")
    resolution = data.draw(st.sampled_from((0.3, 1.0, 3.0)), label="resolution")
    into = data.draw(st.sampled_from(("cluster_a", "cluster_b")), label="into")
    method = data.draw(st.sampled_from(("leiden", "louvain")), label="method")
    cluster = chain.atlas.leiden if method == "leiden" else chain.atlas.louvain
    atlas = _attempt(
        lambda: cluster(graph=graph, resolution=resolution, seed=5, into=into)
    )
    labels = tuple(dict.fromkeys((*chain.labels, into)))
    return atlas and chain.add(atlas, f"{method}({graph}->{into})", labels=labels)


def _umap(data: st.DataObject, chain: _Chain) -> _Chain | None:
    if not chain.graphs:
        return None
    graph = data.draw(st.sampled_from(chain.graphs), label="graph")
    n_epochs = data.draw(st.integers(1, 20), label="n_epochs")
    init = data.draw(st.sampled_from(("spectral", "random")), label="init")
    atlas = _attempt(
        lambda: chain.atlas.umap(
            graph=graph, into="umap", n_epochs=n_epochs, init=init, seed=11
        )
    )
    embeddings = tuple(
        (name, width) for name, width in chain.cell_embeddings if name != "umap"
    ) + (("umap", 2),)
    return atlas and chain.add(atlas, f"umap({graph})", cell_embeddings=embeddings)


def _scrublet(data: st.DataObject, chain: _Chain) -> _Chain | None:
    layer = _layer(data, chain)
    n_prin_comps = data.draw(st.integers(1, 6), label="n_prin_comps")
    n_neighbors = data.draw(
        st.one_of(st.none(), st.integers(2, 8)), label="n_neighbors"
    )
    atlas = _attempt(
        lambda: chain.atlas.scrublet(
            layer=layer, n_prin_comps=n_prin_comps, n_neighbors=n_neighbors, seed=13
        )
    )
    return atlas and chain.add(atlas, f"scrublet({layer})")


_STEPS: tuple[Callable[[st.DataObject, _Chain], _Chain | None], ...] = (
    _with_cells,
    _with_genes,
    _filter_cells,
    _filter_genes,
    _select_genes,
    _take_cells,
    _normalize_total,
    _log1p,
    _scale,
    _highly_variable_genes,
    _pca,
    _neighbors,
    _cluster,
    _umap,
    _scrublet,
)


def _plan_terminals(chain: _Chain) -> None:
    """Every terminal that reads no matrix: explain, estimate, schema, JSON round trip."""
    atlas = chain.atlas
    _attempt(lambda: atlas.explain())
    _attempt(lambda: atlas.explain(format="json"))
    _attempt(lambda: atlas.estimate())
    _attempt(lambda: atlas.schema())
    _attempt(lambda: atlas.peek(3))
    text = _attempt(atlas.to_json)
    if text is not None:
        rebuilt = _attempt(lambda: kyto.from_json(text))
        assert rebuilt is not None, "a plan's own JSON must load"
        assert rebuilt.to_json() == text


# The stages of a typical analysis, in order: each example takes each stage or not, so
# chains reach the graph steps often; then a few steps drawn from all of them, in any
# order.
_STAGES: tuple[tuple[Callable[[st.DataObject, _Chain], _Chain | None], ...], ...] = (
    (
        _with_cells,
        _with_genes,
        _filter_cells,
        _filter_genes,
        _select_genes,
        _take_cells,
    ),
    (_normalize_total,),
    (_log1p,),
    (_scale, _highly_variable_genes, _filter_genes),
    (_pca,),
    (_neighbors,),
    (_cluster, _umap),
    (_cluster, _umap, _scrublet, _with_cells),
)


@_property(base_examples=40)
@given(data=st.data(), threads=st.integers(2, 4))
def test_random_chains_never_crash(
    chain_store: Path,
    tmp_path_factory: pytest.TempPathFactory,
    data: st.DataObject,
    threads: int,
) -> None:
    chain = _Chain(atlas=_kyto.scan(chain_store).with_default_layer("counts"))
    stages = [
        data.draw(st.sampled_from(stage), label="stage step")
        for stage in _STAGES
        # Three in four: chains reach the graph steps (PCA, neighbours, clustering).
        if data.draw(st.sampled_from((True, True, True, False)), label="take stage")
    ]
    extra = data.draw(
        st.lists(st.sampled_from(_STEPS), max_size=3), label="extra steps"
    )
    for step in (*stages, *extra):
        following = step(data, chain)
        if following is None:
            # A coded error at plan build, or no valid arguments yet: go on without it.
            continue
        chain = following
        event(f"step {step.__name__.strip('_')}")
    _plan_terminals(chain)
    event(f"{len(chain.steps)} steps built")
    single = _attempt(lambda: chain.atlas.collect(threads=1))
    if single is None:
        return
    event("collected")
    memory_limit = data.draw(
        st.sampled_from((None, "1GB", 10_000)), label="memory_limit"
    )
    many = _attempt(
        lambda: chain.atlas.collect(memory_limit=memory_limit, threads=threads)
    )
    if many is not None:
        _same_bits(single, many)
    sunk = tmp_path_factory.mktemp("sink") / "sunk.kyto"
    chain.atlas.sink_zarr(sunk, threads=threads)
    _same_bits(single, _kyto.scan(sunk))
    if chain.labels:
        groupby = data.draw(st.sampled_from(chain.labels), label="groupby")
        method = data.draw(st.sampled_from(("t-test", "wilcoxon")), label="method")
        ranking = _attempt(
            lambda: chain.atlas.rank_genes_groups(groupby, method=method, threads=1)
        )
        if ranking is not None and _attempt(ranking.collect) is not None:
            event("ranked")
    if chain.cell_columns:
        column = data.draw(st.sampled_from(chain.cell_columns), label="pulled")
        pulled = _attempt(lambda: chain.atlas.pull(column, threads=1))
        if pulled is not None:
            other = chain.atlas.pull(column, threads=threads)
            assert pulled.equals(other, check_dtypes=True, check_names=True), column
            event("pulled")
    assert chain.atlas.count_cells(threads=threads) == single.count_cells()


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KYTO_PROPERTY_SCALE`` must be a positive integer: a typo fails loudly."""
    monkeypatch.setenv("KYTO_PROPERTY_SCALE", "3")
    assert _positive("KYTO_PROPERTY_SCALE", 1) == 3
    monkeypatch.setenv("KYTO_PROPERTY_SCALE", "0")
    with pytest.raises(ValueError, match="positive integer"):
        _positive("KYTO_PROPERTY_SCALE", 1)
    monkeypatch.delenv("KYTO_PROPERTY_SCALE")
    assert _positive("KYTO_PROPERTY_SCALE", 1) == 1


def test_t_test_ranking_without_genes_does_not_panic(chain_store: Path) -> None:
    """Found by the chain property (seed 5): a t-test over a plan with no genes left
    panicked (``unreachable!``) instead of ranking nothing."""
    plan = (
        _kyto.scan(chain_store)
        .with_default_layer("counts")
        .pca(into="pca", n_components=1, seed=7)
        .neighbors("pca", n_neighbors=2, seed=3)
        .leiden(resolution=0.3, seed=5, into="cluster")
        .select_genes([])
    )
    ranking = plan.rank_genes_groups("cluster", method="t-test", threads=1).collect()
    assert ranking.height == 0
