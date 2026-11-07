"""Static checks of the expression stubs (task 3.3), run by ``uv run pyright``.

Never executed. Each line marked ``# pyright: ignore[rule]`` must be a type error:
``reportUnnecessaryTypeIgnoreComment`` (on for this file below) fails the local gate if
one of them stops erroring, for example if ``.str`` becomes available on a numeric
expression. Every other line must type-check. The result of a rejected line has an
unknown type; the two ``reportUnknown*`` rules that would flag that knock-on effect are
off here, so each ignore names only the error under test.
"""
# pyright: reportUnnecessaryTypeIgnoreComment=error
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from kyto import cell_column, gene_column, gene_expression
from kyto._kyto import Atlas, CellExpr, GeneExpr
from kyto.dtypes import Boolean, ExprDtype, Numeric, String, Unknown


def accepted(
    numeric: CellExpr[Numeric], text: GeneExpr[String], flag: CellExpr[Boolean]
) -> list[CellExpr[ExprDtype] | GeneExpr[ExprDtype]]:
    """Valid expressions: none of these may error."""
    column: CellExpr[Unknown] = cell_column("n_genes")
    symbol: GeneExpr[Unknown] = gene_column("symbol")
    keep: CellExpr[Boolean] = (column > 200) & ~flag | (numeric / 2 <= 1.5)
    mito: GeneExpr[Boolean] = symbol.str.starts_with("MT-") | text.str.ends_with("-1")
    shifted: CellExpr[Numeric] = 1 - column * numeric + 0.5
    equal: CellExpr[Boolean] = (column == "B cell") | (numeric != 2) | (flag == True)
    return [keep, mito, shifted, equal, True & flag, text < "Z"]


def rejected(
    numeric: CellExpr[Numeric],
    boolean: GeneExpr[Boolean],
    other_boolean: GeneExpr[Boolean],
    text: CellExpr[String],
) -> None:
    """Invalid expressions: each line must be a pyright error."""
    # Done-when of 3.3: `.str` on a numeric expression fails in pyright.
    _ = numeric.str  # pyright: ignore[reportAttributeAccessIssue]
    _ = (cell_column("n_genes") + 1).str  # pyright: ignore[reportAttributeAccessIssue]
    _ = boolean.str  # pyright: ignore[reportAttributeAccessIssue]
    # R5: cell and gene expressions never mix (KY-E202 at run time as well).
    _ = cell_column("a") & gene_column("b")  # pyright: ignore[reportOperatorIssue]
    _ = (cell_column("a") > 1) | (gene_column("b") > 1)  # pyright: ignore[reportOperatorIssue]
    # Logic only on booleans, arithmetic only on numbers.
    _ = numeric & (numeric > 1)  # pyright: ignore[reportOperatorIssue]
    _ = ~numeric  # pyright: ignore[reportOperatorIssue]
    _ = text + 1  # pyright: ignore[reportOperatorIssue]
    _ = boolean < other_boolean  # pyright: ignore[reportOperatorIssue]
    # Ordering needs both sides of one kind (`==` cannot be checked statically).
    _ = text < 1.5  # pyright: ignore[reportOperatorIssue]
    _ = numeric >= "a"  # pyright: ignore[reportOperatorIssue]
    # An expression has no truth value (KY-E206 at run time).
    if numeric > 1:  # pyright: ignore[reportGeneralTypeIssues]
        pass


def resolved(atlas: Atlas, numeric: CellExpr[Numeric], text: GeneExpr[String]) -> None:
    """``Atlas.resolve`` keeps the axis and the kind (task 3.5)."""
    cell: CellExpr[Numeric] = atlas.resolve(numeric)
    gene: GeneExpr[String] = atlas.resolve(text)
    _ = (cell, gene)
    _ = atlas.resolve(numeric).str  # pyright: ignore[reportAttributeAccessIssue]
    _ = atlas.resolve("CD3E")  # pyright: ignore[reportCallIssue, reportArgumentType]


def null_tests(numeric: CellExpr[Numeric], text: GeneExpr[String]) -> None:
    """Null tests take any kind and give a boolean of the same axis (task 3.7)."""
    missing: CellExpr[Boolean] = numeric.is_null()
    present: GeneExpr[Boolean] = text.is_not_null()
    keep: CellExpr[Boolean] = ~cell_column("cell_type").is_null() & missing
    _ = (present, keep)
    _ = numeric.is_null().str  # pyright: ignore[reportAttributeAccessIssue]
    _ = numeric.is_null() & text.is_null()  # pyright: ignore[reportOperatorIssue]


def lineage(atlas: Atlas, numeric: CellExpr[Numeric], flag: GeneExpr[Boolean]) -> None:
    """``recompute`` keeps axis and kind; ``filter_genes`` takes a boolean gene
    expression only (task 3.8)."""
    again: CellExpr[Numeric] = numeric.recompute()
    filtered: Atlas = atlas.filter_genes(flag)
    _ = (again, filtered)
    _ = atlas.filter_genes(numeric > 1)  # pyright: ignore[reportArgumentType]
    _ = atlas.filter_genes(gene_column("highly_variable"))  # Unknown until resolved
    _ = atlas.filter_genes(gene_column("n_cells") + 1)  # pyright: ignore[reportArgumentType]


def gene_symbols(atlas: Atlas) -> None:
    """``by`` takes ``'id'`` or ``'symbol'`` only; a symbol read is numeric and resolves
    to the same axis and kind (task 3.9)."""
    by_symbol: CellExpr[Numeric] = gene_expression("CD3E", layer="log1p", by="symbol")
    resolved: CellExpr[Numeric] = atlas.resolve(gene_expression("CD3E", by="id"))
    _ = (by_symbol, resolved)
    _ = gene_expression("CD3E", by="name")  # pyright: ignore[reportArgumentType]
