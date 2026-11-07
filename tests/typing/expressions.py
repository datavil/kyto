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

from kyto import cell_column, gene_column
from kyto._kyto import CellExpr, GeneExpr
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
