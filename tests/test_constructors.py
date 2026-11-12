"""Task 3.4: the expression constructors.

``cell_column``, ``gene_column``, ``gene_expression``, ``cell_profile``,
``cell_embedding``, ``gene_embedding`` and ``feature``: the class, axis and dtype each
returns, explicit arguments (R7), the coded errors ``KY-E208`` and ``KY-E209``, and that
``repr`` is code that rebuilds the expression. ``feature`` only parses its name; the
resolver is task 3.5. Doctests: ``tests/test_docstrings.py``. Rust tests:
``crates/kyto-core/tests/it/feature_names.rs``.
"""

from collections.abc import Callable
from typing import Any

import kyto
import pytest
from kyto import (
    _kyto,
    cell_column,
    cell_embedding,
    cell_profile,
    feature,
    gene_column,
    gene_embedding,
    gene_expression,
)
from kyto.dtypes import Numeric, Unknown

_CELL: list[tuple[Callable[[], Any], str, type]] = [
    (lambda: cell_column("n_genes"), "cell_column('n_genes')", Unknown),
    (lambda: gene_expression("CD3E"), "gene_expression('CD3E')", Numeric),
    (
        lambda: gene_expression("CD3E", layer="log1p"),
        "gene_expression('CD3E', layer='log1p')",
        Numeric,
    ),
    (lambda: cell_embedding("pca", 0), "cell_embedding('pca', 0)", Numeric),
    (lambda: feature("CD3E"), "feature('CD3E')", Unknown),
    (lambda: feature("log1p:CD3E"), "feature('log1p:CD3E')", Unknown),
    (lambda: feature("meta:cluster"), "feature('meta:cluster')", Unknown),
]

_GENE: list[tuple[Callable[[], Any], str, type]] = [
    (lambda: gene_column("symbol"), "gene_column('symbol')", Unknown),
    (lambda: cell_profile("AAAC-1"), "cell_profile('AAAC-1')", Numeric),
    (
        lambda: cell_profile("AAAC-1", layer="counts"),
        "cell_profile('AAAC-1', layer='counts')",
        Numeric,
    ),
    (lambda: gene_embedding("loadings", 3), "gene_embedding('loadings', 3)", Numeric),
]


@pytest.mark.parametrize(("build", "code", "dtype"), _CELL)
def test_cell_constructors(build: Callable[[], Any], code: str, dtype: type) -> None:
    expression = build()
    assert repr(expression) == code
    assert expression.dtype is dtype
    assert isinstance(expression, _kyto.CellExpr)


@pytest.mark.parametrize(("build", "code", "dtype"), _GENE)
def test_gene_constructors(build: Callable[[], Any], code: str, dtype: type) -> None:
    expression = build()
    assert repr(expression) == code
    assert expression.dtype is dtype
    assert isinstance(expression, _kyto.GeneExpr)


@pytest.mark.parametrize(("build", "code", "dtype"), _CELL + _GENE)
def test_repr_rebuilds_the_expression(
    build: Callable[[], Any], code: str, dtype: type
) -> None:
    namespace = {name: getattr(kyto, name) for name in kyto.__all__}
    rebuilt = eval(code, namespace)
    assert repr(rebuilt) == repr(build())
    assert type(rebuilt) is type(build())


def test_constructors_are_exported() -> None:
    for name in (
        "cell_column",
        "gene_column",
        "gene_expression",
        "cell_profile",
        "cell_embedding",
        "gene_embedding",
        "feature",
    ):
        assert name in kyto.__all__
        assert getattr(kyto, name) is getattr(_kyto, name)


def test_layer_is_keyword_only_and_names_positional_only() -> None:
    call: Any = gene_expression
    with pytest.raises(TypeError):
        call("CD3E", "log1p")
    with pytest.raises(TypeError):
        call(gene_id="CD3E")
    call = cell_profile
    with pytest.raises(TypeError):
        call("AAAC-1", "counts")
    call = feature
    with pytest.raises(TypeError):
        call(name="CD3E")


def test_constructors_mix_axes_like_columns() -> None:
    mixed: Any = gene_expression("CD3E")
    with pytest.raises(kyto.AxisError) as raised:
        _ = mixed & cell_profile("AAAC-1")
    assert raised.value.code == "KY-E202"
    mixed_feature: Any = feature("CD3E")
    with pytest.raises(kyto.AxisError):
        _ = mixed_feature + gene_embedding("loadings", 0)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("", "it is empty"),
        (":CD3E", "nothing comes before ':'"),
        ("log1p:", "nothing follows 'log1p:'"),
        (
            "log 1p:CD3E",
            (
                "the prefix 'log 1p' cannot be a layer or embedding name or 'meta': "
                "names match [A-Za-z0-9_.-]+"
            ),
        ),
    ],
)
def test_feature_rejects_names_that_cannot_resolve(name: str, problem: str) -> None:
    with pytest.raises(kyto.ResolutionError) as raised:
        feature(name)
    assert raised.value.code == "KY-E208"
    assert isinstance(raised.value, ValueError)
    assert raised.value.fields == {"name": name, "problem": problem}
    assert "'log1p:CD3E'" in str(raised.value)


def test_feature_splits_at_the_first_colon() -> None:
    assert repr(feature("counts:HLA:DRA")) == "feature('counts:HLA:DRA')"


@pytest.mark.parametrize(
    ("component", "got"),
    [
        (-1, "a negative int"),
        (2**64, "an int above 2**64 - 1"),
        (True, "a bool"),
        (1.0, "a float"),
        ("0", "a str"),
        (None, "a NoneType"),
        (object(), "an object"),
    ],
)
@pytest.mark.parametrize("constructor", [cell_embedding, gene_embedding])
def test_embedding_component_is_ky_e209(
    constructor: Callable[..., Any], component: object, got: str
) -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        constructor("pca", component)
    assert raised.value.code == "KY-E209"
    assert raised.value.fields == {
        "constructor": constructor.__name__,
        "name": "pca",
        "got": got,
        "component": repr(component),
    }
    assert f"{constructor.__name__}('pca', 0)" in str(raised.value)


def test_largest_component_is_accepted() -> None:
    assert repr(cell_embedding("pca", 2**64 - 1)) == (
        f"cell_embedding('pca', {2**64 - 1})"
    )
