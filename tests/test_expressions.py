"""Task 3.3: typed expressions, ``CellExpr`` and ``GeneExpr`` (R5).

Done when: ``cell_column(...) & gene_column(...)`` raises ``KY-E202`` when the plan is
built (here: when the two are combined, before any data is read), and ``.str`` on a
numeric expression fails in pyright (``tests/typing/expressions.py``, checked by
``uv run pyright``). Rust tests: ``crates/kyto-core/tests/expressions.rs``.
"""

import ast
import doctest
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kyto
import kyto.dtypes
import pytest
from kyto import _kyto, cell_column, gene_column
from kyto.dtypes import Boolean, Numeric, String, Unknown

# Operators are called through this table so pyright does not reject the deliberately
# mixed or ill-typed operands (the static checks live in tests/typing/expressions.py).
_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "&": lambda left, right: left & right,
    "|": lambda left, right: left | right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "+": lambda left, right: left + right,
    "-": lambda left, right: left - right,
    "*": lambda left, right: left * right,
    "/": lambda left, right: left / right,
}


@pytest.mark.parametrize("operation", list(_OPERATORS))
def test_cell_and_gene_expressions_never_mix(operation: str) -> None:
    apply = _OPERATORS[operation]
    cell: Any = cell_column("n_genes") > 200 if operation in "&|" else cell_column("n")
    gene: Any = gene_column("highly_variable")
    for left, right in ((cell, gene), (gene, cell)):
        with pytest.raises(kyto.AxisError) as raised:
            apply(left, right)
        error = raised.value
        assert error.code == "KY-E202"
        assert isinstance(error, TypeError)
        assert error.fields["operation"] == operation
        assert error.fields["cell_expression"] == repr(cell)
        assert error.fields["gene_expression"] == repr(gene)


def test_done_when_message() -> None:
    with pytest.raises(kyto.AxisError, match="KY-E202") as raised:
        _OPERATORS["&"](cell_column("n_genes"), gene_column("highly_variable"))
    assert str(raised.value).startswith(
        "KY-E202: '&' mixes a cell expression (cell_column('n_genes')) with a gene "
        "expression (gene_column('highly_variable'))"
    )


def test_dtype_follows_the_operations() -> None:
    column = cell_column("n_genes")
    assert column.dtype is Unknown
    assert (column + 1).dtype is Numeric
    assert (1 - column).dtype is Numeric
    assert (column > 1).dtype is Boolean
    assert (200 < column).dtype is Boolean
    assert ((column > 1) & (column < 5)).dtype is Boolean
    assert (~(column > 1)).dtype is Boolean
    assert gene_column("symbol").str.starts_with("MT-").dtype is Boolean
    assert issubclass(String, kyto.dtypes.ExprDtype)


def test_repr_is_the_code_that_builds_the_expression() -> None:
    column = cell_column("n_genes")
    assert repr(200 < column) == "cell_column('n_genes') > 200"
    assert repr(2 / column) == "2 / cell_column('n_genes')"
    assert repr(True & (column >= 1.5)) == "True & (cell_column('n_genes') >= 1.5)"
    assert repr(~((column == "a") | (column != "b"))) == (
        "~((cell_column('n_genes') == 'a') | (cell_column('n_genes') != 'b'))"
    )
    assert repr(gene_column("it's").str.ends_with("\n")) == (
        "gene_column('it\\'s').str.ends_with('\\n')"
    )
    assert str(column) == repr(column)


def test_str_on_a_numeric_expression_is_ky_e205() -> None:
    numeric: Any = cell_column("n_genes") + 1
    with pytest.raises(kyto.ExpressionError) as raised:
        numeric.str.starts_with("MT-")
    error = raised.value
    assert error.code == "KY-E205"
    assert isinstance(error, TypeError)
    assert error.fields == {
        "operation": "str.starts_with",
        "accepted": "a string expression",
        "got": "a numeric expression",
        "operand": "cell_column('n_genes') + 1",
    }


_WRONG_OPERANDS: dict[str, tuple[Callable[[Any], object], str, str]] = {
    "boolean plus": (lambda c: (c > 1) + 1, "+", "a boolean expression"),
    "times string": (lambda c: c * "x", "*", "a string value"),
    "plus bool": (lambda c: c + True, "+", "a boolean value"),
    "numeric and": (lambda c: (c + 1) & (c > 1), "&", "a numeric expression"),
    "not numeric": (lambda c: ~(c + 1), "~", "a numeric expression"),
    "numeric equals string": (lambda c: (c + 1) == "x", "==", "a string value"),
    "plus None": (lambda c: c + None, "+", "a NoneType"),
    "greater than list": (lambda c: c > [1, 2], ">", "a list"),
    "plus huge int": (lambda c: c + 2**63, "+", "an int outside that range"),
}


@pytest.mark.parametrize("case", list(_WRONG_OPERANDS))
def test_wrong_operands_are_ky_e205(case: str) -> None:
    build, operation, got = _WRONG_OPERANDS[case]
    with pytest.raises(kyto.ExpressionError) as raised:
        build(cell_column("n_genes"))
    assert raised.value.code == "KY-E205"
    assert raised.value.fields["operation"] == operation
    assert raised.value.fields["got"] == got


def test_long_foreign_operands_are_shortened() -> None:
    column: Any = cell_column("n")
    with pytest.raises(kyto.ExpressionError) as raised:
        _ = column + list(range(1000))
    operand = raised.value.fields["operand"]
    assert isinstance(operand, str)
    assert operand.endswith("...")
    assert len(operand) == 83


def test_boundary_integers_are_accepted() -> None:
    column = cell_column("n")
    assert repr(column + (2**63 - 1)) == f"cell_column('n') + {2**63 - 1}"
    assert repr(column + -(2**63)) == f"cell_column('n') + {-(2**63)}"


def test_unknown_columns_are_left_to_plan_build() -> None:
    column = cell_column("anything")
    for built in (
        column + 1,
        column & True,
        column.str.starts_with("x"),
        column == "x",
    ):
        assert built.dtype in (Numeric, Boolean)


_TRUTH_USES: dict[str, Callable[[Any], object]] = {
    "bool": lambda e: bool(e),
    "and": lambda e: e and True,
    "not": lambda e: not e,
    "chained comparison": lambda e: 1 < e < 3,
    "if": lambda e: e if e else None,
}


@pytest.mark.parametrize("case", list(_TRUTH_USES))
def test_truth_value_is_ky_e206(case: str) -> None:
    with pytest.raises(kyto.ExpressionError) as raised:
        _TRUTH_USES[case](cell_column("n_genes"))
    assert raised.value.code == "KY-E206"
    assert "&, | and ~" in str(raised.value)


def test_expressions_are_immutable_and_unhashable() -> None:
    column = cell_column("n_genes")
    before = repr(column)
    _ = (column + 1) > 2
    assert repr(column) == before
    with pytest.raises(AttributeError):
        column.dtype = Numeric  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(column)
    with pytest.raises(TypeError):
        pickle.dumps(column)


def test_classes_are_generic_at_run_time() -> None:
    alias = _kyto.CellExpr[Numeric]
    assert alias.__origin__ is _kyto.CellExpr  # type: ignore[attr-defined]
    assert _kyto.GeneExpr[Boolean].__args__ == (Boolean,)  # type: ignore[attr-defined]
    assert isinstance(cell_column("a"), _kyto.CellExpr)
    assert not isinstance(cell_column("a"), _kyto.GeneExpr)


def test_no_constructor_from_python() -> None:
    with pytest.raises(TypeError):
        _kyto.CellExpr()  # type: ignore[call-arg]


def test_docstring_examples_run() -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs = {"cell_column": cell_column, "gene_column": gene_column}
    sources = {
        "CellExpr": _kyto.CellExpr.__doc__,
        "GeneExpr": _kyto.GeneExpr.__doc__,
        "CellStringNamespace": _kyto.CellStringNamespace.__doc__,
        "GeneStringNamespace": _kyto.GeneStringNamespace.__doc__,
        "cell_column": cell_column.__doc__,
        "gene_column": gene_column.__doc__,
        "kyto.dtypes": kyto.dtypes.__doc__,
    }
    for name, docstring in sources.items():
        assert docstring is not None, name
        runner.run(parser.get_doctest(docstring, dict(globs), name, None, 0))
    assert runner.failures == 0
    assert runner.tries == 10


def test_stub_docstring_examples_run() -> None:
    stub = Path(_kyto.__file__).with_name("_kyto.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    names = {
        "CellExpr",
        "GeneExpr",
        "CellStringNamespace",
        "GeneStringNamespace",
        "cell_column",
        "gene_column",
    }
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs = {"cell_column": cell_column, "gene_column": gene_column}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef):
            continue
        if node.name not in names:
            continue
        nested = list(ast.walk(node))
        for item in nested:
            if isinstance(item, ast.ClassDef | ast.FunctionDef):
                docstring = ast.get_docstring(item)
                if docstring:
                    test = parser.get_doctest(
                        docstring, dict(globs), item.name, None, 0
                    )
                    runner.run(test)
    assert runner.failures == 0
    assert runner.tries == 10


def test_deep_trees_are_ky_e207_not_a_crash() -> None:
    expression = cell_column("n")
    for _ in range(999):
        expression = expression + 1
    assert repr(expression).count("+ 1") == 999
    with pytest.raises(kyto.ExpressionError) as raised:
        _ = expression + 1
    assert raised.value.code == "KY-E207"
    assert raised.value.fields == {"operation": "+", "depth": 1001, "limit": 1000}
    # A deep tree is also refused when it is the right operand.
    with pytest.raises(kyto.ExpressionError, match="KY-E207"):
        _ = cell_column("m") > expression - 1


def test_repr_escapes_non_printable_characters() -> None:
    name = "a\0b\x1b \xa0\u200b\x7fé z"
    text = repr(cell_column(name))
    assert text.isprintable()
    assert text == "cell_column(" + repr(name) + ")"
