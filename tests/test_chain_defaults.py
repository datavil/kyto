"""Task 7.10: layer defaults that follow the chain.

A matrix step with no ``layer=`` reads the layer the step before it wrote (else the
default layer), a step with no ``into=`` names its output after itself, and both are
resolved when the plan is built and stored in full (R6), so the bare chain and the
explicit one are the same plan. A default name the plan already has is ``KY-E220``.
The pbmc3k Done-when (same plan JSON, same ``explain()``, same bytes) is in
``tests/test_reproducibility.py``.
"""

from pathlib import Path

import kyto
import pytest
from kyto import _kyto, cell_column
from test_chain import _store  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path)


def _explicit(atlas: _kyto.Atlas) -> _kyto.Atlas:
    return (
        atlas.with_default_layer("counts")
        .normalize_total(1e4, layer="counts", into="normalized")
        .log1p(layer="normalized", into="log1p")
        .scale(layer="log1p", into="scaled", max_value=10)
        .pca(layer="scaled", into="X_pca", n_components=1, seed=0)
        .neighbors("X_pca", n_neighbors=2)
        .leiden(into="leiden", seed=0)
        .louvain(into="louvain", seed=0)
        .umap(into="X_umap", seed=0)
    )


def _bare(atlas: _kyto.Atlas) -> _kyto.Atlas:
    return (
        atlas.with_default_layer("counts")
        .normalize_total(1e4)
        .log1p()
        .scale(max_value=10)
        .pca(n_components=1, seed=0)
        .neighbors(n_neighbors=2)
        .leiden(seed=0)
        .louvain(seed=0)
        .umap(seed=0)
    )


def test_the_bare_chain_is_the_explicit_plan(store: Path) -> None:
    atlas = _kyto.scan(store)
    explicit, bare = _explicit(atlas), _bare(atlas)
    assert bare.to_json() == explicit.to_json()
    assert str(bare.explain()) == str(explicit.explain())
    assert str(bare).splitlines()[-5:-2] == [
        'SCALE "log1p" -> "scaled" zero_center=true max_value=10',
        'LOG1P "normalized" -> "log1p"',
        'NORMALIZE_TOTAL "counts" -> "normalized" target_sum=10000',
    ]
    # The stored plan names everything, so it replays without the defaults.
    assert kyto.from_json(bare.to_json()).to_json() == bare.to_json()


def test_the_nearest_of_a_written_layer_and_the_default_layer_wins(
    store: Path,
) -> None:
    atlas = _kyto.scan(store).with_default_layer("counts").log1p()
    assert str(atlas.pca(n_components=1, seed=0)).startswith('PCA "log1p"')
    reset = atlas.with_default_layer("counts").pca(n_components=1, seed=0)
    assert str(reset).startswith('PCA "counts"')
    # Explicit arguments are unchanged.
    named = atlas.pca(layer="counts", into="pcs", n_components=1, seed=0)
    assert str(named).startswith('PCA "counts" -> "pcs"')
    # Expressions keep reading the default layer.
    assert repr(atlas.resolve(kyto.feature("ENSG01"))) == (
        "gene_expression('ENSG01', layer='counts')"
    )


def test_a_written_layer_needs_no_default_layer(store: Path) -> None:
    bare = _kyto.scan(store)
    with pytest.raises(kyto.KytoError) as raised:
        bare.log1p()
    assert raised.value.code == "KY-E211"
    atlas = bare.log1p(layer="counts")
    assert str(atlas.highly_variable_genes()).startswith(
        'HIGHLY_VARIABLE_GENES "log1p"'
    )
    # rank_genes_groups follows the chain too: it is checked against "log1p" now.
    frame = atlas.with_cells(group=cell_column("cell_id")).rank_genes_groups("group")
    assert frame.collect_schema().names()[:2] == ["group", "gene_id"]


@pytest.mark.parametrize(
    ("step", "kind", "name"),
    [
        ("normalize_total", "layer", "normalized"),
        ("log1p", "layer", "log1p"),
        ("scale", "layer", "scaled"),
        ("pca", "embedding", "X_pca"),
        ("umap", "embedding", "X_umap"),
        ("leiden", "cell column", "leiden"),
        ("louvain", "cell column", "louvain"),
    ],
)
def test_a_default_name_the_plan_has_is_ky_e220(
    store: Path, step: str, kind: str, name: str
) -> None:
    plan = _explicit(_kyto.scan(store))
    if step == "normalize_total":
        call = lambda: plan.normalize_total(1e4)
    elif step == "log1p":
        call = plan.log1p
    elif step == "scale":
        call = plan.scale
    elif step == "pca":
        call = lambda: plan.pca(n_components=1, seed=0)
    elif step == "umap":
        call = lambda: plan.umap(seed=0)
    elif step == "leiden":
        call = lambda: plan.leiden(seed=0)
    else:
        call = lambda: plan.louvain(seed=0)
    with pytest.raises(kyto.errors.PlanError) as raised:
        call()
    assert raised.value.code == "KY-E220"
    message = str(raised.value)
    assert f"{step}() without into= writes the {kind} '{name}'" in message
    assert "pass into=" in message
    # Naming the output, or replacing the member on purpose, still works.
    assert plan.log1p(into="log1p") is not None


def test_neighbors_without_an_embedding(store: Path) -> None:
    atlas = _kyto.scan(store).with_default_layer("counts")
    with pytest.raises(kyto.errors.PlanError) as raised:
        atlas.neighbors()
    assert raised.value.code == "KY-E215"
    assert "no pca step of the plan wrote a cell embedding" in str(raised.value)
    searched = atlas.pca(into="pcs", n_components=1, seed=0).neighbors(n_neighbors=2)
    assert str(searched).startswith('NEIGHBORS "pcs"')


def test_collect_starts_a_new_chain_at_the_default_layer(store: Path) -> None:
    """Documented on ``collect``: the collected plan is a scan, so a bare step reads
    its default layer, not the layer the collected plan wrote last."""
    base = _kyto.scan(store).with_default_layer("counts").normalize_total(1e4)
    assert str(base.log1p()).startswith('LOG1P "normalized"')
    assert str(base.collect().log1p(into="x")).startswith('LOG1P "counts"')
