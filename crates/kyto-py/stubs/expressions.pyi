"""Hand-written part of the `kyto._kyto` stub (task 7.6).

`scripts/stubs.py` merges it into the stub that `pyo3-stub-gen` generates from the
bindings, because `pyo3-stub-gen` cannot express it: the generic expression classes
(`CellExpr[Boolean]`, methods typed on `self`, overloads per dtype kind), their
`TypeVar`s and aliases, and the `TypedDict`s that type the dicts `schema()`,
`explain()` and `estimate()` return. `tests/test_stubs.py` checks it against the
extension at run time (names, signatures and docstrings).
"""

import builtins
from typing import (
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    final,
    overload,
)

from kyto.dtypes import Boolean, ExprDtype, Numeric, String, Unknown

class ColumnLineage(TypedDict):
    """The ``lineage`` of a stored derived column (format.md section 8)."""

    layer: str
    node: str

class ColumnSchema(TypedDict):
    """One ``columns`` entry of ``schema.json`` (format.md section 8)."""

    name: str
    dtype: Literal[
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "String",
        "Enum",
    ]
    n_categories: NotRequired[int]
    ordered: NotRequired[bool]
    lineage: NotRequired[ColumnLineage]

class TableSchema(TypedDict):
    """The ``cells`` or ``genes`` entry of ``schema.json``."""

    columns: list[ColumnSchema]

class LayerSchema(TypedDict):
    """One ``layers`` entry of ``schema.json``."""

    name: str
    dtype: Literal["float32", "uint32"]
    nnz: int

class EmbeddingSchema(TypedDict):
    """One ``embeddings`` entry of ``schema.json``."""

    name: str
    axis: Literal["cells", "genes"]
    width: int
    dtype: Literal["float32", "float64"]

class GraphSchema(TypedDict):
    """One ``graphs`` entry of ``schema.json``."""

    name: str
    axis: Literal["cells", "genes"]
    dtype: Literal["float32", "float64"]
    nnz: int

class ModalitySchema(TypedDict):
    """A modality's schema (provisional: format v0 stores none)."""

    name: str
    n_features: int
    features: TableSchema
    layers: list[LayerSchema]

class TilingSchema(TypedDict):
    """The ``tiling`` entry of ``schema.json``: the tile sizes of every layer."""

    cell_block_size: int
    gene_block_size: int

class AtlasSchema(TypedDict):
    """What ``Atlas.schema()`` returns: a plan's output schema, shaped like
    ``schema.json``; a size is ``None`` if not known without running the plan."""

    n_cells: int | None
    n_genes: int | None
    tiling: TilingSchema
    cells: TableSchema
    genes: TableSchema
    layers: list[LayerSchema]
    embeddings: list[EmbeddingSchema]
    graphs: list[GraphSchema]
    modalities: list[ModalitySchema]

class CostEstimate(TypedDict):
    """What running a plan would cost (``Atlas.estimate()``); sizes in bytes."""

    passes: int
    bytes_read: int
    peak_memory: int
    held_memory: int
    passes_saved: int
    device: Literal["cpu", "auto", "cuda", "wgpu", "metal"]

class ExplainedStep(TypedDict):
    """One logical step of ``Atlas.explain(format="json")``."""

    id: int
    name: str
    parameters: str
    input: int | None
    note: str | None

class ExplainedOutput(TypedDict):
    """What a terminal reads from the optimized plan (``Atlas.explain``)."""

    node: int
    reads: list[str]

class ExplainedNode(TypedDict):
    """One optimized node of ``Atlas.explain(format="json")`` and how it runs."""

    index: int
    name: str
    parameters: str
    reads: list[str] | None
    input: int | None
    origin: list[int]
    device: Literal["cpu", "cuda", "metal", "vulkan", "dx12", "wgpu"]
    device_note: str | None
    moves: str | None
    passes: int
    statistics_passes: int
    apply_pass: bool
    blocks: Literal["cells", "in_memory", "none"]
    block_cells: int | None
    holds_input: bool

class ExplainedDevice(TypedDict):
    """The device a plan asks for (``Atlas.on``) and the devices found."""

    requested: Literal["cpu", "auto", "cuda", "wgpu", "metal"]
    found: str

class ExplainedPlan(TypedDict):
    """The optimized plan of ``Atlas.explain(format="json")``."""

    device: ExplainedDevice
    outputs: list[ExplainedOutput]
    nodes: list[ExplainedNode]

class Explanation(TypedDict):
    """What ``Atlas.explain(format="json")`` returns: the logical and optimized trees."""

    logical: list[ExplainedStep]
    optimized: ExplainedPlan

class PlanDiagnostic(TypedDict):
    """One problem ``validate_plan`` found in a plan JSON (type checking only)."""

    code: str
    path: str
    message: str
    hint: str

class Estimate(TypedDict):
    """What a handle's data looks like, from metadata alone; ``None`` if not known."""

    n_rows: int | None
    n_columns: int | None
    nnz: int | None
    dense_bytes: int | None

class _SupportsDLPack(Protocol):  # noqa: PYI046 (used by the generated Atlas.with_embedding)
    """An object that exports its memory through DLPack (a torch tensor on the CPU).

    What ``Atlas.with_embedding`` reads besides a NumPy array (task 6.11), through
    ``numpy.from_dlpack``.
    """

    def __dlpack__(self) -> object: ...

_D_co = TypeVar("_D_co", bound=ExprDtype, covariant=True)
_D = TypeVar("_D", bound=ExprDtype)  # noqa: PYI018 (used by the generated Atlas.resolve)

@final
class CellExpr(Generic[_D_co]):
    """An expression with one value per cell.

    Built with ``cell_column``, ``gene_expression``, ``cell_embedding`` or ``feature``
    and combined with operators; it records what to compute and reads nothing. The type
    parameter is the dtype kind (``kyto.dtypes``): ``.str`` exists only on ``String``
    (and ``Unknown``) expressions, ``&``, ``|`` and ``~`` only on ``Boolean`` ones,
    arithmetic only on ``Numeric`` ones. Combining it with a ``GeneExpr`` raises
    ``AxisError`` (``KY-E202``, R5).

    Examples
    --------
    >>> cell_column("n_genes") > 200
    cell_column('n_genes') > 200
    """

    __hash__: ClassVar[None]  # type: ignore[assignment]

    @property
    def dtype(self) -> type[_D_co]:
        """The dtype kind: a class from ``kyto.dtypes``.

        Examples
        --------
         >>> cell_column("total_counts").dtype
         <class 'kyto.dtypes.Unknown'>
         >>> (cell_column("total_counts") > 0).dtype
         <class 'kyto.dtypes.Boolean'>
        """

    def __class_getitem__(cls, item: object) -> object:
        """Subscripting the class, as in ``CellExpr[Numeric]``, gives a
        ``types.GenericAlias`` for annotations evaluated at run time.
        """

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

    def is_null(self) -> CellExpr[Boolean]:
        """Whether each value is missing (null): a boolean expression.

        Missing metadata is null (R9): a categorical code ``-1``, a masked
        nullable entry or a float ``NaN`` of an imported ``.h5ad``. A ``NaN``
        that kyto itself writes stays a value, not null. Matrix values and embeddings are never null (an absent
        sparse value is 0), so on them the result is always ``False``. Comparing
        with ``None`` (``x == None``) raises ``KY-E212``; use this instead.

        Returns
        -------
        The same expression class, of dtype ``Boolean``, never null itself.
        Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E207``: the expression would be too deep.

        Examples
        --------
         >>> cell_column("label").is_null()
         cell_column('label').is_null()
        """

    def is_not_null(self) -> CellExpr[Boolean]:
        """Whether each value is present (not null): a boolean expression.

        The negation of ``is_null()``; see there for what is null (R9).

        Returns
        -------
        The same expression class, of dtype ``Boolean``, never null itself.
        Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E207``: the expression would be too deep.

        Examples
        --------
         >>> cell_column("label").is_not_null()
         cell_column('label').is_not_null()
        """

    def log1p(self: _CellExprNumeric) -> CellExpr[Numeric]:
        """The natural logarithm of one plus each value: a float expression.

        ``numpy.log1p``: ``-1`` gives ``-inf``, a value below ``-1`` gives
        ``NaN``, a missing value stays missing. scanpy's ``log1p_`` QC columns
        are ``qc.total_counts().log1p()`` and the like. A ``Float32`` input
        gives ``Float32``, every other number ``Float64``.

        Returns
        -------
        The same expression class, numeric. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not numeric. ``KY-E207``: the
            expression would be too deep.

        Examples
        --------
         >>> cell_column("total_counts").log1p()
         cell_column('total_counts').log1p()
        """

    def recompute(self) -> CellExpr[_D_co]:
        """This column, computed again from its lineage instead of read as stored.

        A derived column the store computed (one whose ``schema.json`` entry
        records ``lineage``, such as ``n_genes``) goes stale once a step such as
        a gene ``filter`` changes its inputs: reading it then raises ``KY-E301``.
        A recomputed read is valid there. Nothing is computed now: the executor
        (task 5.x) computes it when the plan runs.

        Returns
        -------
        The same expression class and dtype. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E213``: the expression is not a plain column read.

        Examples
        --------
        >>> cell_column("n_genes").recompute()
        cell_column('n_genes').recompute()
        """

    def __bool__(self) -> NoReturn:
        """Always raises ``ExpressionError`` (``KY-E206``): use ``&``, ``|``, ``~``."""

    @property
    def str(self: _CellExprString) -> CellStringNamespace:
        """String functions; only on ``String`` (or ``Unknown``) expressions.

        Examples
        --------
         >>> cell_column("label").str.starts_with("MT-")
         cell_column('label').str.starts_with('MT-')
        """

@final
class CellStringNamespace:
    """String functions of a ``CellExpr`` (``CellExpr.str``).

    Examples
    --------
    >>> cell_column("cell_id").str.ends_with("-1")
    cell_column('cell_id').str.ends_with('-1')
    """

    def starts_with(self, prefix: builtins.str) -> CellExpr[Boolean]:
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

    def ends_with(self, suffix: builtins.str) -> CellExpr[Boolean]:
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

@final
class GeneExpr(Generic[_D_co]):
    """An expression with one value per gene.

    Built with ``gene_column``, ``cell_profile`` or ``gene_embedding`` and combined with
    operators; it records what to compute and reads nothing. The type parameter is the
    dtype kind (``kyto.dtypes``): ``.str`` exists only on ``String`` (and ``Unknown``)
    expressions, ``&``, ``|`` and ``~`` only on ``Boolean`` ones, arithmetic only on
    ``Numeric`` ones. Combining it with a ``CellExpr`` raises ``AxisError``
    (``KY-E202``, R5).

    Examples
    --------
    >>> gene_column("symbol").str.starts_with("MT-")
    gene_column('symbol').str.starts_with('MT-')
    """

    __hash__: ClassVar[None]  # type: ignore[assignment]

    @property
    def dtype(self) -> type[_D_co]:
        """The dtype kind: a class from ``kyto.dtypes``.

        Examples
        --------
         >>> gene_column("total_counts").dtype
         <class 'kyto.dtypes.Unknown'>
         >>> (gene_column("total_counts") > 0).dtype
         <class 'kyto.dtypes.Boolean'>
        """

    def __class_getitem__(cls, item: object) -> object:
        """Subscripting the class, as in ``CellExpr[Numeric]``, gives a
        ``types.GenericAlias`` for annotations evaluated at run time.
        """

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

    def is_null(self) -> GeneExpr[Boolean]:
        """Whether each value is missing (null): a boolean expression.

        Missing metadata is null (R9): a categorical code ``-1``, a masked
        nullable entry or a float ``NaN`` of an imported ``.h5ad``. A ``NaN``
        that kyto itself writes stays a value, not null. Matrix values and embeddings are never null (an absent
        sparse value is 0), so on them the result is always ``False``. Comparing
        with ``None`` (``x == None``) raises ``KY-E212``; use this instead.

        Returns
        -------
        The same expression class, of dtype ``Boolean``, never null itself.
        Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E207``: the expression would be too deep.

        Examples
        --------
         >>> gene_column("label").is_null()
         gene_column('label').is_null()
        """

    def is_not_null(self) -> GeneExpr[Boolean]:
        """Whether each value is present (not null): a boolean expression.

        The negation of ``is_null()``; see there for what is null (R9).

        Returns
        -------
        The same expression class, of dtype ``Boolean``, never null itself.
        Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E207``: the expression would be too deep.

        Examples
        --------
         >>> gene_column("label").is_not_null()
         gene_column('label').is_not_null()
        """

    def log1p(self: _GeneExprNumeric) -> GeneExpr[Numeric]:
        """The natural logarithm of one plus each value: a float expression.

        ``numpy.log1p``: ``-1`` gives ``-inf``, a value below ``-1`` gives
        ``NaN``, a missing value stays missing. scanpy's ``log1p_`` QC columns
        are ``qc.total_counts().log1p()`` and the like. A ``Float32`` input
        gives ``Float32``, every other number ``Float64``.

        Returns
        -------
        The same expression class, numeric. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E205``: the expression is not numeric. ``KY-E207``: the
            expression would be too deep.

        Examples
        --------
         >>> gene_column("total_counts").log1p()
         gene_column('total_counts').log1p()
        """

    def recompute(self) -> GeneExpr[_D_co]:
        """This column, computed again from its lineage instead of read as stored.

        A derived column the store computed (one whose ``schema.json`` entry
        records ``lineage``, such as ``n_genes``) goes stale once a step such as
        a gene ``filter`` changes its inputs: reading it then raises ``KY-E301``.
        A recomputed read is valid there. Nothing is computed now: the executor
        (task 5.x) computes it when the plan runs.

        Returns
        -------
        The same expression class and dtype. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E213``: the expression is not a plain column read.

        Examples
        --------
        >>> cell_column("n_genes").recompute()
        cell_column('n_genes').recompute()
        """

    def __bool__(self) -> NoReturn:
        """Always raises ``ExpressionError`` (``KY-E206``): use ``&``, ``|``, ``~``."""

    @property
    def str(self: _GeneExprString) -> GeneStringNamespace:
        """String functions; only on ``String`` (or ``Unknown``) expressions.

        Examples
        --------
         >>> gene_column("label").str.starts_with("MT-")
         gene_column('label').str.starts_with('MT-')
        """

@final
class GeneStringNamespace:
    """String functions of a ``GeneExpr`` (``GeneExpr.str``).

    Examples
    --------
    >>> gene_column("symbol").str.starts_with("MT-")
    gene_column('symbol').str.starts_with('MT-')
    """

    def starts_with(self, prefix: builtins.str) -> GeneExpr[Boolean]:
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

    def ends_with(self, suffix: builtins.str) -> GeneExpr[Boolean]:
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
