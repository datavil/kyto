"""Type stub for the compiled extension module `kyto._kyto`."""

import builtins
import os
from typing import ClassVar, Generic, NoReturn, TypeAlias, TypeVar, final, overload

from kyto.dtypes import Boolean, ExprDtype, Numeric, String, Unknown

__version__: str

@final
class Atlas:
    """A lazy, immutable plan over a kyto store.

    An ``Atlas`` records what to compute; it reads no matrix data. Every method returns
    a new ``Atlas`` and leaves the one it was called on unchanged (R2), so an ``Atlas``
    can be shared and reused freely. Plans built from one another share their common
    part; nothing is copied.

    Examples
    --------
    >>> atlas = scan(store)
    >>> counts = atlas.with_default_layer("counts")
    >>> print(atlas)
    SCAN "..." (2 cells x 3 genes)
    """

    def with_default_layer(self, layer: str) -> Atlas:
        """Set the layer that short gene names refer to.

        Parameters
        ----------
        layer : str
            Name of a layer of the store, as stored (case-sensitive).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no layer named ``layer``.

        Examples
        --------
        >>> atlas = scan(store)
        >>> print(atlas.with_default_layer("counts"))
        WITH_DEFAULT_LAYER "counts"
        SCAN "..." (2 cells x 3 genes)
        """

_D_co = TypeVar("_D_co", bound=ExprDtype, covariant=True)

@final
class CellExpr(Generic[_D_co]):
    """An expression with one value per cell.

    Built with ``cell_column`` and combined with operators; it records what to compute and
    reads nothing. The type parameter is the dtype kind (``kyto.dtypes``): ``.str``
    exists only on ``String`` (and ``Unknown``) expressions, ``&``, ``|`` and ``~``
    only on ``Boolean`` ones, arithmetic only on ``Numeric`` ones. Combining it with a
    ``GeneExpr`` raises ``AxisError`` (``KY-E202``, R5).

    Examples
    --------
    >>> cell_column("n_genes") > 200
    cell_column('n_genes') > 200
    """

    __hash__: ClassVar[None]  # type: ignore[assignment]

    @property
    def dtype(self) -> type[_D_co]:
        """The dtype kind: a class from ``kyto.dtypes``."""

    def __class_getitem__(cls, item: object, /) -> object: ...
    def __eq__(  # type: ignore[override]
        self, other: CellExpr[ExprDtype] | builtins.bool | float | builtins.str, /
    ) -> CellExpr[Boolean]:
        """Compare with ``==``: a boolean expression.

        Both sides must be of one kind (``KY-E205`` otherwise); type checkers cannot
        check that for ``==`` and ``!=``, they fall back to ``object.__eq__``.
        """

    def __ne__(  # type: ignore[override]
        self, other: CellExpr[ExprDtype] | builtins.bool | float | builtins.str, /
    ) -> CellExpr[Boolean]:
        """Compare with ``!=``: a boolean expression.

        Both sides must be of one kind (``KY-E205`` otherwise); type checkers cannot
        check that for ``==`` and ``!=``, they fall back to ``object.__eq__``.
        """

    @overload
    def __lt__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Boolean]:
        """Compare with ``<``: a boolean expression; both sides of one kind."""

    @overload
    def __lt__(
        self: _CellExprString, other: _CellExprString | builtins.str, /
    ) -> CellExpr[Boolean]: ...
    @overload
    def __le__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Boolean]:
        """Compare with ``<=``: a boolean expression; both sides of one kind."""

    @overload
    def __le__(
        self: _CellExprString, other: _CellExprString | builtins.str, /
    ) -> CellExpr[Boolean]: ...
    @overload
    def __gt__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Boolean]:
        """Compare with ``>``: a boolean expression; both sides of one kind."""

    @overload
    def __gt__(
        self: _CellExprString, other: _CellExprString | builtins.str, /
    ) -> CellExpr[Boolean]: ...
    @overload
    def __ge__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Boolean]:
        """Compare with ``>=``: a boolean expression; both sides of one kind."""

    @overload
    def __ge__(
        self: _CellExprString, other: _CellExprString | builtins.str, /
    ) -> CellExpr[Boolean]: ...
    def __add__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Numeric]:
        """``self + other``."""

    def __radd__(self: _CellExprNumeric, other: float, /) -> CellExpr[Numeric]:
        """``other + self``."""

    def __sub__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Numeric]:
        """``self - other``."""

    def __rsub__(self: _CellExprNumeric, other: float, /) -> CellExpr[Numeric]:
        """``other - self``."""

    def __mul__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Numeric]:
        """``self * other``."""

    def __rmul__(self: _CellExprNumeric, other: float, /) -> CellExpr[Numeric]:
        """``other * self``."""

    def __truediv__(
        self: _CellExprNumeric, other: _CellExprNumeric | float, /
    ) -> CellExpr[Numeric]:
        """``self / other`` (true division)."""

    def __rtruediv__(self: _CellExprNumeric, other: float, /) -> CellExpr[Numeric]:
        """``other / self`` (true division)."""

    def __and__(
        self: _CellExprBoolean, other: _CellExprBoolean | builtins.bool, /
    ) -> CellExpr[Boolean]:
        """``self & other`` (logical and)."""

    def __rand__(self: _CellExprBoolean, other: builtins.bool, /) -> CellExpr[Boolean]:
        """``other & self`` (logical and)."""

    def __or__(
        self: _CellExprBoolean, other: _CellExprBoolean | builtins.bool, /
    ) -> CellExpr[Boolean]:
        """``self | other`` (logical or)."""

    def __ror__(self: _CellExprBoolean, other: builtins.bool, /) -> CellExpr[Boolean]:
        """``other | self`` (logical or)."""

    def __invert__(self: _CellExprBoolean) -> CellExpr[Boolean]:
        """``~self`` (logical not)."""

    def __bool__(self) -> NoReturn:
        """Always raises ``ExpressionError`` (``KY-E206``): use ``&``, ``|``, ``~``."""

    @property
    def str(self: _CellExprString) -> CellStringNamespace:
        """String functions; only on ``String`` (or ``Unknown``) expressions."""

@final
class CellStringNamespace:
    """String functions of a ``CellExpr`` (``CellExpr.str``).

    Examples
    --------
    >>> cell_column("cell_id").str.ends_with("-1")
    cell_column('cell_id').str.ends_with('-1')
    """

    def starts_with(self, prefix: builtins.str, /) -> CellExpr[Boolean]:
        """Whether each value starts with ``prefix``.

        Parameters
        ----------
        prefix : str
            The prefix, compared byte for byte (case-sensitive).

        Returns
        -------
        CellExpr[Boolean]
            A new expression. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not a string expression.

        Examples
        --------
        >>> cell_column("symbol").str.starts_with("MT-")
        cell_column('symbol').str.starts_with('MT-')
        """

    def ends_with(self, suffix: builtins.str, /) -> CellExpr[Boolean]:
        """Whether each value ends with ``suffix``.

        Parameters
        ----------
        suffix : str
            The suffix, compared byte for byte (case-sensitive).

        Returns
        -------
        CellExpr[Boolean]
            A new expression. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not a string expression.

        Examples
        --------
        >>> cell_column("name").str.ends_with("-1")
        cell_column('name').str.ends_with('-1')
        """

_CellExprBoolean: TypeAlias = CellExpr[Boolean] | CellExpr[Unknown]
_CellExprNumeric: TypeAlias = CellExpr[Numeric] | CellExpr[Unknown]
_CellExprString: TypeAlias = CellExpr[String] | CellExpr[Unknown]

def cell_column(name: builtins.str, /) -> CellExpr[Unknown]:
    """A column of the cell table, as a ``CellExpr``.

    Parameters
    ----------
    name : str
        The column name. It is resolved against the schema when the plan is built
        (task 3.5); until then its dtype is ``Unknown``.

    Returns
    -------
    CellExpr[Unknown]
        An expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> cell_column("n_genes") > 200
    cell_column('n_genes') > 200
    """

@final
class GeneExpr(Generic[_D_co]):
    """An expression with one value per gene.

    Built with ``gene_column`` and combined with operators; it records what to compute and
    reads nothing. The type parameter is the dtype kind (``kyto.dtypes``): ``.str``
    exists only on ``String`` (and ``Unknown``) expressions, ``&``, ``|`` and ``~``
    only on ``Boolean`` ones, arithmetic only on ``Numeric`` ones. Combining it with a
    ``CellExpr`` raises ``AxisError`` (``KY-E202``, R5).

    Examples
    --------
    >>> gene_column("symbol").str.starts_with("MT-")
    gene_column('symbol').str.starts_with('MT-')
    """

    __hash__: ClassVar[None]  # type: ignore[assignment]

    @property
    def dtype(self) -> type[_D_co]:
        """The dtype kind: a class from ``kyto.dtypes``."""

    def __class_getitem__(cls, item: object, /) -> object: ...
    def __eq__(  # type: ignore[override]
        self, other: GeneExpr[ExprDtype] | builtins.bool | float | builtins.str, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``==``: a boolean expression.

        Both sides must be of one kind (``KY-E205`` otherwise); type checkers cannot
        check that for ``==`` and ``!=``, they fall back to ``object.__eq__``.
        """

    def __ne__(  # type: ignore[override]
        self, other: GeneExpr[ExprDtype] | builtins.bool | float | builtins.str, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``!=``: a boolean expression.

        Both sides must be of one kind (``KY-E205`` otherwise); type checkers cannot
        check that for ``==`` and ``!=``, they fall back to ``object.__eq__``.
        """

    @overload
    def __lt__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``<``: a boolean expression; both sides of one kind."""

    @overload
    def __lt__(
        self: _GeneExprString, other: _GeneExprString | builtins.str, /
    ) -> GeneExpr[Boolean]: ...
    @overload
    def __le__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``<=``: a boolean expression; both sides of one kind."""

    @overload
    def __le__(
        self: _GeneExprString, other: _GeneExprString | builtins.str, /
    ) -> GeneExpr[Boolean]: ...
    @overload
    def __gt__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``>``: a boolean expression; both sides of one kind."""

    @overload
    def __gt__(
        self: _GeneExprString, other: _GeneExprString | builtins.str, /
    ) -> GeneExpr[Boolean]: ...
    @overload
    def __ge__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Boolean]:
        """Compare with ``>=``: a boolean expression; both sides of one kind."""

    @overload
    def __ge__(
        self: _GeneExprString, other: _GeneExprString | builtins.str, /
    ) -> GeneExpr[Boolean]: ...
    def __add__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Numeric]:
        """``self + other``."""

    def __radd__(self: _GeneExprNumeric, other: float, /) -> GeneExpr[Numeric]:
        """``other + self``."""

    def __sub__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Numeric]:
        """``self - other``."""

    def __rsub__(self: _GeneExprNumeric, other: float, /) -> GeneExpr[Numeric]:
        """``other - self``."""

    def __mul__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Numeric]:
        """``self * other``."""

    def __rmul__(self: _GeneExprNumeric, other: float, /) -> GeneExpr[Numeric]:
        """``other * self``."""

    def __truediv__(
        self: _GeneExprNumeric, other: _GeneExprNumeric | float, /
    ) -> GeneExpr[Numeric]:
        """``self / other`` (true division)."""

    def __rtruediv__(self: _GeneExprNumeric, other: float, /) -> GeneExpr[Numeric]:
        """``other / self`` (true division)."""

    def __and__(
        self: _GeneExprBoolean, other: _GeneExprBoolean | builtins.bool, /
    ) -> GeneExpr[Boolean]:
        """``self & other`` (logical and)."""

    def __rand__(self: _GeneExprBoolean, other: builtins.bool, /) -> GeneExpr[Boolean]:
        """``other & self`` (logical and)."""

    def __or__(
        self: _GeneExprBoolean, other: _GeneExprBoolean | builtins.bool, /
    ) -> GeneExpr[Boolean]:
        """``self | other`` (logical or)."""

    def __ror__(self: _GeneExprBoolean, other: builtins.bool, /) -> GeneExpr[Boolean]:
        """``other | self`` (logical or)."""

    def __invert__(self: _GeneExprBoolean) -> GeneExpr[Boolean]:
        """``~self`` (logical not)."""

    def __bool__(self) -> NoReturn:
        """Always raises ``ExpressionError`` (``KY-E206``): use ``&``, ``|``, ``~``."""

    @property
    def str(self: _GeneExprString) -> GeneStringNamespace:
        """String functions; only on ``String`` (or ``Unknown``) expressions."""

@final
class GeneStringNamespace:
    """String functions of a ``GeneExpr`` (``GeneExpr.str``).

    Examples
    --------
    >>> gene_column("symbol").str.starts_with("MT-")
    gene_column('symbol').str.starts_with('MT-')
    """

    def starts_with(self, prefix: builtins.str, /) -> GeneExpr[Boolean]:
        """Whether each value starts with ``prefix``.

        Parameters
        ----------
        prefix : str
            The prefix, compared byte for byte (case-sensitive).

        Returns
        -------
        GeneExpr[Boolean]
            A new expression. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not a string expression.

        Examples
        --------
        >>> gene_column("symbol").str.starts_with("MT-")
        gene_column('symbol').str.starts_with('MT-')
        """

    def ends_with(self, suffix: builtins.str, /) -> GeneExpr[Boolean]:
        """Whether each value ends with ``suffix``.

        Parameters
        ----------
        suffix : str
            The suffix, compared byte for byte (case-sensitive).

        Returns
        -------
        GeneExpr[Boolean]
            A new expression. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not a string expression.

        Examples
        --------
        >>> gene_column("name").str.ends_with("-1")
        gene_column('name').str.ends_with('-1')
        """

_GeneExprBoolean: TypeAlias = GeneExpr[Boolean] | GeneExpr[Unknown]
_GeneExprNumeric: TypeAlias = GeneExpr[Numeric] | GeneExpr[Unknown]
_GeneExprString: TypeAlias = GeneExpr[String] | GeneExpr[Unknown]

def gene_column(name: builtins.str, /) -> GeneExpr[Unknown]:
    """A column of the gene table, as a ``GeneExpr``.

    Parameters
    ----------
    name : str
        The column name. It is resolved against the schema when the plan is built
        (task 3.5); until then its dtype is ``Unknown``.

    Returns
    -------
    GeneExpr[Unknown]
        An expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> gene_column("symbol").str.starts_with("MT-")
    gene_column('symbol').str.starts_with('MT-')
    """

def scan(store: str | os.PathLike[str], /) -> Atlas:
    """Build the root `Atlas` of the local store at `store` (private until 7.1)."""

def raise_error_example(code: str, /) -> bool:
    """Raise the registered example error for `code`; return False if unknown."""

def error_example_text(code: str, /) -> tuple[str, str] | None:
    """Return the Rust message and hint of the example error for `code`."""

def convert(
    source: str,
    target: str,
    /,
    *,
    layer: str,
    on_duplicate_ids: str,
    overwrite: bool,
) -> None:
    """Import `source` into a new store at `target` (see `kyto.convert`)."""

def sink_h5ad(
    store: str,
    target: str,
    /,
    *,
    x_layer: str | None,
    overwrite: bool,
) -> None:
    """Export the store at `store` as an `.h5ad` file at `target` (private, task 2.7)."""
