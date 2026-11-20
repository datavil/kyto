"""Dtype kinds of kyto expressions (task 3.3).

``CellExpr`` and ``GeneExpr`` are generic over one of these classes, so a type checker
knows what an expression holds: ``CellExpr[Numeric]`` has no ``.str``, and ``&`` takes
only ``Boolean`` expressions. They are markers, never instantiated; ``expr.dtype``
returns the class itself.

``Unknown`` is the kind of a column before the plan resolves it against the schema
(task 3.5). Type checkers accept it wherever another kind is expected, and kyto checks
it when the plan is built.

Examples
--------
>>> from kyto import cell_column
>>> from kyto.dtypes import Boolean, Unknown
>>> cell_column("n_genes").dtype is Unknown
True
>>> (cell_column("n_genes") > 200).dtype is Boolean
True
"""

from typing import final

__all__ = ["Boolean", "ExprDtype", "Numeric", "String", "Unknown"]


class ExprDtype:
    """Base class of the dtype kinds; ``CellExpr[ExprDtype]`` is any cell expression.

    Examples
    --------
    >>> from kyto.dtypes import ExprDtype, Numeric
    >>> issubclass(Numeric, ExprDtype)
    True
    """


@final
class Boolean(ExprDtype):
    """``True`` / ``False``: comparisons, ``&``, ``|``, ``~`` and ``str`` predicates.

    Examples
    --------
    >>> from kyto import cell_column
    >>> from kyto.dtypes import Boolean
    >>> (cell_column("n_genes") > 200).dtype is Boolean
    True
    """


@final
class Numeric(ExprDtype):
    """Integers and floats: matrix values, embeddings and arithmetic.

    Examples
    --------
    >>> from kyto import gene_expression
    >>> from kyto.dtypes import Numeric
    >>> gene_expression("CD3E").dtype is Numeric
    True
    """


@final
class String(ExprDtype):
    """Text: the ID column and categorical columns.

    Examples
    --------
    >>> from kyto import cell_column, scan
    >>> scan(store).resolve(cell_column("cell_id")).dtype
    <class 'kyto.dtypes.String'>
    """


@final
class Unknown(ExprDtype):
    """Not known until the plan is built: a column not yet resolved (task 3.5).

    Examples
    --------
    >>> from kyto import cell_column
    >>> from kyto.dtypes import Unknown
    >>> cell_column("n_genes").dtype is Unknown
    True
    """
