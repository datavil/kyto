"""Type stub for the compiled extension module `kyto._kyto`."""

import builtins
import os
from typing import (
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    NotRequired,
    TypeAlias,
    TypedDict,
    TypeVar,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.sparse  # pyright: ignore[reportMissingTypeStubs]

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
    @overload
    def resolve(self, expression: CellExpr[_D], /) -> CellExpr[_D]:
        """Resolve the names in an expression against this plan (R6).

        Every ``feature(...)`` name becomes the one column, gene or embedding component
        it names; every gene or cell read without a ``layer`` gets the plan's default
        layer; every column, gene, layer and embedding name is checked against the
        store's schema and gene IDs as read when the plan was built; every column gets
        its dtype from the schema, and the operations on it are checked again. The
        result names everything in full, so it keeps its meaning whatever is added to
        the store or the plan later. Full names are ``meta:<column>``,
        ``<layer>:<gene_id>`` and ``<embedding>:<component>``; a name that matches more
        than one feature is an error, never a guess.

        Parameters
        ----------
        expression : CellExpr or GeneExpr
            The expression to resolve.

        Returns
        -------
        CellExpr or GeneExpr
            A new expression of the same axis; ``expression`` is unchanged. Nothing is
            read or copied.

        Raises
        ------
        ResolutionError
            ``KY-E201``: a ``feature(...)`` name matches more than one feature.
            ``KY-E210``: a name matches nothing, or a cell ID is empty.
            ``KY-E211``: a gene or cell is read from the default layer, and the plan has
            none.
        MissingMemberError
            ``KY-E109``: a layer or embedding is not in the store.
        ExpressionError
            ``KY-E205``: a column's stored dtype does not fit an operation on it.

        Examples
        --------
        >>> atlas = scan(store).with_default_layer("counts")
        >>> atlas.resolve(feature("ENSG01") > 1)
        gene_expression('ENSG01', layer='counts') > 1
        >>> atlas.resolve(feature("meta:cell_id").str.starts_with("AAA"))
        cell_column('cell_id').str.starts_with('AAA')
        """
    @overload
    def resolve(self, expression: GeneExpr[_D], /) -> GeneExpr[_D]: ...
    @property
    def cells(self) -> Table:
        """The cell table: a read-only, lazy handle (nothing is read).

        Returns
        -------
        Table
            A new handle on this plan's cell table.

        Examples
        --------
        >>> scan(store).cells.names()
        ['cell_id']
        """
    @property
    def genes(self) -> Table:
        """The gene table: a read-only, lazy handle (nothing is read).

        Returns
        -------
        Table
            A new handle on this plan's gene table.

        Examples
        --------
        >>> scan(store).genes.names()
        ['gene_id', 'symbol', 'feature_type']
        """
    def layer(self, name: builtins.str, /) -> Matrix:
        """A layer, as a read-only, lazy matrix handle (nothing is read).

        Parameters
        ----------
        name : str
            Name of a layer of the store, as stored (case-sensitive).

        Returns
        -------
        Matrix
            A new handle on the layer.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no layer named ``name``.

        Examples
        --------
        >>> scan(store).layer("counts").schema()
        {'name': 'counts', 'dtype': 'uint32', 'nnz': 3}
        """
    def embedding(self, name: builtins.str, /) -> Embedding:
        """An embedding, as a read-only, lazy handle (nothing is read).

        Parameters
        ----------
        name : str
            Name of an embedding of the store, as stored (case-sensitive).

        Returns
        -------
        Embedding
            A new handle on the embedding.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no embedding named ``name``.

        Examples
        --------
        >>> scan(store).embedding("X_pca")
        Traceback (most recent call last):
        ...
        kyto.errors.MissingMemberError: KY-E109: no embedding named 'X_pca' in the store at '...'...
        """
    def graph(self, name: builtins.str, /) -> Graph:
        """A graph, as a read-only, lazy handle (nothing is read).

        Parameters
        ----------
        name : str
            Name of a graph of the store, as stored (case-sensitive).

        Returns
        -------
        Graph
            A new handle on the graph.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no graph named ``name``.

        Examples
        --------
        >>> scan(store).graph("connectivities")
        Traceback (most recent call last):
        ...
        kyto.errors.MissingMemberError: KY-E109: no graph named 'connectivities' in the store at '...'...
        """
    def modality(self, name: builtins.str, /) -> Modality:
        """A modality, as a read-only, lazy handle (nothing is read).

        Store format v0 holds no modalities, so for now this always raises.

        Parameters
        ----------
        name : str
            Name of a modality of the store, as stored (case-sensitive).

        Returns
        -------
        Modality
            A new handle on the modality.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no modality named ``name``.

        Examples
        --------
        >>> scan(store).modality("adt")
        Traceback (most recent call last):
        ...
        kyto.errors.MissingMemberError: KY-E109: no modality named 'adt' in the store at '...'...
        """

# The dicts that handle `schema()` and `estimate()` return (task 3.6). Type-checking
# only: they do not exist at run time.

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

class Estimate(TypedDict):
    """What a handle's data looks like, from metadata alone; ``None`` if not known."""

    n_rows: int | None
    n_columns: int | None
    nnz: int | None
    dense_bytes: int | None

@final
class Table:
    """The cell or gene table of a plan: read-only and lazy.

    Taken from ``Atlas.cells`` or ``Atlas.genes``. It reads nothing: ``names()``,
    ``schema()`` and ``estimate()`` answer from the schema read when the plan was built,
    and the terminals ``to_polars()`` and ``lazy()`` run the plan (task 7.3). It has no
    method that changes it or the plan (R2).

    Examples
    --------
    >>> atlas = scan(store)
    >>> atlas.cells
    <Table cells: 2 rows x 1 columns>
    """

    def names(self) -> list[str]:
        """The column names, ID column first, in stored order.

        Returns
        -------
        list of str
            A new list; reads no data.

        Examples
        --------
        >>> scan(store).genes.names()
        ['gene_id', 'symbol', 'feature_type']
        """
    def schema(self) -> TableSchema:
        """The table's entry in ``schema.json`` (format.md section 8).

        Returns
        -------
        dict
            A new ``{"columns": [...]}`` dict, one dict per column with ``name`` and
            ``dtype``, ``n_categories`` and ``ordered`` for an ``Enum`` column, and
            ``lineage`` when the store records one. Changing it changes nothing else.

        Examples
        --------
        >>> scan(store).cells.schema()
        {'columns': [{'name': 'cell_id', 'dtype': 'String'}]}
        """
    def estimate(self) -> Estimate:
        """What the table looks like, from metadata alone (no data is read).

        Returns
        -------
        dict
            A new dict: ``n_rows`` (cells or genes), ``n_columns``, ``nnz`` (``None``)
            and ``dense_bytes`` (``None``: string sizes are not in the schema). A value
            not known without running the plan is ``None``. The cost estimate (bytes
            read, peak memory) is task 4.6.

        Examples
        --------
        >>> scan(store).genes.estimate()
        {'n_rows': 3, 'n_columns': 3, 'nnz': None, 'dense_bytes': None}
        """
    def to_polars(self) -> pl.DataFrame:
        """Run the plan and return the table as a Polars ``DataFrame``.

        Returns
        -------
        polars.DataFrame
            A copy the caller owns (R2): changing it never changes the store or a
            later read.

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store).cells.to_polars()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Table.to_polars' is not implemented yet...
        """
    def lazy(self) -> pl.LazyFrame:
        """The table as a Polars ``LazyFrame`` over the plan.

        Returns
        -------
        polars.LazyFrame
            A new lazy frame; collecting it returns a copy the caller owns (R2).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store).cells.lazy()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Table.lazy' is not implemented yet...
        """

@final
class Matrix:
    """A layer of a plan, a sparse cells x genes matrix: read-only and lazy.

    Taken from ``Atlas.layer(name)``. It reads nothing: ``names()``, ``schema()`` and
    ``estimate()`` answer from metadata, and the terminals ``to_scipy()`` and
    ``to_dense(max_bytes=...)`` run the plan (task 7.3). There is no implicit dense
    array (R8): no ``__array__``, and dense output only through ``to_dense``. It has
    no method that changes it or the plan (R2).

    Examples
    --------
    >>> scan(store).layer("counts")
    <Matrix 'counts': 2 cells x 3 genes, nnz 3, uint32>
    """

    def names(self) -> list[str]:
        """The column names: the gene IDs, in stored order.

        Returns
        -------
        list of str
            A new list, read from the gene table when the plan was built.

        Examples
        --------
        >>> scan(store).layer("counts").names()
        ['ENSG01', 'ENSG02', 'ENSG03']
        """
    def schema(self) -> LayerSchema:
        """The layer's entry in ``schema.json`` (format.md section 8).

        Returns
        -------
        dict
            A new dict with ``name``, ``dtype`` (``"float32"`` or ``"uint32"``) and
            ``nnz``. Changing it changes nothing else.

        Examples
        --------
        >>> scan(store).layer("counts").schema()
        {'name': 'counts', 'dtype': 'uint32', 'nnz': 3}
        """
    def estimate(self) -> Estimate:
        """What the matrix looks like, from metadata alone (no data is read).

        Returns
        -------
        dict
            A new dict: ``n_rows`` (cells), ``n_columns`` (genes), ``nnz`` (stored
            entries) and ``dense_bytes`` (what ``to_dense`` would allocate). A value not
            known without running the plan is ``None``. The cost estimate (bytes read,
            peak memory) is task 4.6.

        Examples
        --------
        >>> scan(store).layer("counts").estimate()
        {'n_rows': 2, 'n_columns': 3, 'nnz': 3, 'dense_bytes': 24}
        """
    def to_scipy(self) -> scipy.sparse.csr_matrix:
        """Run the plan and return the matrix as a SciPy CSR matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cells x genes in the layer's stored dtype; a copy the caller owns (R2).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store).layer("counts").to_scipy()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Matrix.to_scipy' is not implemented yet...
        """
    def to_dense(self, *, max_bytes: int) -> npt.NDArray[np.float32 | np.uint32]:
        """Run the plan and return the matrix as a dense NumPy array, if it fits.

        The only way to get a dense matrix (R8): absent entries become 0 (R9).

        Parameters
        ----------
        max_bytes : int
            The largest array, in bytes, the caller accepts. Required: there is no
            default, so densifying is always a stated choice. ``estimate()["dense_bytes"]``
            gives the size without reading the matrix.

        Returns
        -------
        numpy.ndarray
            Cells x genes in the layer's stored dtype; a copy the caller owns (R2).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store).layer("counts").to_dense(max_bytes=1024)
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Matrix.to_dense' is not implemented yet...
        """

@final
class Embedding:
    """An embedding of a plan, a dense array with one row per cell or gene: read-only and
    lazy.

    Taken from ``Atlas.embedding(name)``. It reads nothing: ``names()``, ``schema()``
    and ``estimate()`` answer from metadata, and the terminal ``to_numpy()`` runs the
    plan (task 7.3). An embedding is stored dense, so ``to_numpy()`` densifies nothing
    (R8). It has no method that changes it or the plan (R2).

    Examples
    --------
    >>> scan(store_with_members).embedding("X_pca")
    <Embedding 'X_pca': 2 cells x 2 components, float32>
    """

    def names(self) -> list[str]:
        """The column names: ``<embedding>:<component>`` for each component, from 0.

        Returns
        -------
        list of str
            A new list; reads no data. For a cell embedding these are the full names
            ``feature(...)`` accepts (docs/feature_names.md).

        Examples
        --------
        >>> scan(store_with_members).embedding("X_pca").names()
        ['X_pca:0', 'X_pca:1']
        """
    def schema(self) -> EmbeddingSchema:
        """The embedding's entry in ``schema.json`` (format.md section 8).

        Returns
        -------
        dict
            A new dict with ``name``, ``axis`` (``"cells"`` or ``"genes"``), ``width``
            and ``dtype`` (``"float32"`` or ``"float64"``). Changing it changes nothing
            else.

        Examples
        --------
        >>> scan(store_with_members).embedding("X_pca").schema()
        {'name': 'X_pca', 'axis': 'cells', 'width': 2, 'dtype': 'float32'}
        """
    def estimate(self) -> Estimate:
        """What the embedding looks like, from metadata alone (no data is read).

        Returns
        -------
        dict
            A new dict: ``n_rows`` (cells or genes), ``n_columns`` (components), ``nnz``
            (``None``: it is dense) and ``dense_bytes`` (what ``to_numpy`` would
            allocate). A value not known without running the plan is ``None``.

        Examples
        --------
        >>> scan(store_with_members).embedding("X_pca").estimate()
        {'n_rows': 2, 'n_columns': 2, 'nnz': None, 'dense_bytes': 16}
        """
    def to_numpy(self) -> npt.NDArray[np.float32 | np.float64]:
        """Run the plan and return the embedding as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Rows x components in the stored dtype; a copy the caller owns (R2).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store_with_members).embedding("X_pca").to_numpy()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Embedding.to_numpy' is not implemented yet...
        """

@final
class Graph:
    """A graph of a plan, a sparse cells x cells matrix of edge weights: read-only and
    lazy.

    Taken from ``Atlas.graph(name)``. It reads nothing: ``names()``, ``schema()`` and
    ``estimate()`` answer from metadata, and the terminals ``to_scipy()`` and
    ``to_edges()`` run the plan (task 7.3). It has no method that changes it or the
    plan (R2).

    Examples
    --------
    >>> scan(store_with_members).graph("connectivities")
    <Graph 'connectivities': 2 x 2 cells, nnz 3, float32>
    """

    def names(self) -> list[str]:
        """The column names of its edge list (``to_edges()``).

        Returns
        -------
        list of str
            A new list: ``['source', 'target', 'weight']``.

        Examples
        --------
        >>> scan(store_with_members).graph("connectivities").names()
        ['source', 'target', 'weight']
        """
    def schema(self) -> GraphSchema:
        """The graph's entry in ``schema.json`` (format.md section 8).

        Returns
        -------
        dict
            A new dict with ``name``, ``axis`` (``"cells"``), ``dtype`` (``"float32"``
            or ``"float64"``) and ``nnz``. Changing it changes nothing else.

        Examples
        --------
        >>> scan(store_with_members).graph("connectivities").schema()
        {'name': 'connectivities', 'axis': 'cells', 'dtype': 'float32', 'nnz': 3}
        """
    def estimate(self) -> Estimate:
        """What the graph looks like, from metadata alone (no data is read).

        Returns
        -------
        dict
            A new dict: ``n_rows`` and ``n_columns`` (cells), ``nnz`` (stored edges)
            and ``dense_bytes`` (a dense cells x cells array). A value not known
            without running the plan is ``None``.

        Examples
        --------
        >>> scan(store_with_members).graph("connectivities").estimate()
        {'n_rows': 2, 'n_columns': 2, 'nnz': 3, 'dense_bytes': 16}
        """
    def to_scipy(self) -> scipy.sparse.csr_matrix:
        """Run the plan and return the graph as a SciPy CSR matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cells x cells edge weights in the stored dtype; a copy the caller owns
            (R2). A stored edge may weigh 0 (format.md section 7).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store_with_members).graph("connectivities").to_scipy()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Graph.to_scipy' is not implemented yet...
        """
    def to_edges(self) -> pl.DataFrame:
        """Run the plan and return the edges as a Polars ``DataFrame``.

        Returns
        -------
        polars.DataFrame
            One row per stored edge, with the columns ``names()``; a copy the caller
            owns (R2).

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: terminals run with task 7.3.

        Examples
        --------
        >>> scan(store_with_members).graph("connectivities").to_edges()
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Graph.to_edges' is not implemented yet...
        """

@final
class Modality:
    """A modality of a plan, a second feature space on the same cells: read-only and lazy.

    Taken from ``Atlas.modality(name)``. Provisional and minimal: store format v0 holds
    no modalities, so no store has one yet, and the format version that adds them fixes
    the schema and the terminals. It has no method that changes it or the plan (R2).

    Examples
    --------
    >>> scan(store).modality("adt")
    Traceback (most recent call last):
    ...
    kyto.errors.MissingMemberError: KY-E109: no modality named 'adt' in the store at '...'...
    """

    def names(self) -> list[str]:
        """The names of its layers.

        Returns
        -------
        list of str
            A new list; reads no data.
        """
    def schema(self) -> ModalitySchema:
        """The modality's schema (provisional, as format v0 stores none).

        Returns
        -------
        dict
            A new dict with ``name``, ``n_features``, ``features`` (a table entry) and
            ``layers`` (layer entries). Changing it changes nothing else.
        """
    def estimate(self) -> Estimate:
        """What the modality looks like, from metadata alone (no data is read).

        Returns
        -------
        dict
            A new dict: ``n_rows`` (cells), ``n_columns`` (features), ``nnz`` and
            ``dense_bytes`` (``None``).
        """

_D_co = TypeVar("_D_co", bound=ExprDtype, covariant=True)
_D = TypeVar("_D", bound=ExprDtype)

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

    def is_null(self) -> CellExpr[Boolean]:
        """Whether each value is missing (null); never null itself (R9).

        Missing metadata is null; ``NaN`` is a value, and matrix values and embeddings
        are never null (an absent sparse value is 0). Use this, not ``x == None``
        (``KY-E212``). Nothing is read.
        """

    def is_not_null(self) -> CellExpr[Boolean]:
        """Whether each value is present (not null); the negation of ``is_null()``."""

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

    def is_null(self) -> GeneExpr[Boolean]:
        """Whether each value is missing (null); never null itself (R9).

        Missing metadata is null; ``NaN`` is a value, and matrix values and embeddings
        are never null (an absent sparse value is 0). Use this, not ``x == None``
        (``KY-E212``). Nothing is read.
        """

    def is_not_null(self) -> GeneExpr[Boolean]:
        """Whether each value is present (not null); the negation of ``is_null()``."""

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

def gene_expression(
    gene_id: builtins.str, /, *, layer: builtins.str | None = None
) -> CellExpr[Numeric]:
    """The values of one gene across cells, as a ``CellExpr``.

    Parameters
    ----------
    gene_id : str
        The gene ID. It is checked against the schema when the plan is built (task 3.5).
    layer : str or None, default=None
        The layer to read. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> gene_expression("CD3E", layer="log1p") > 1.5
    gene_expression('CD3E', layer='log1p') > 1.5
    >>> gene_expression("CD3E").dtype
    <class 'kyto.dtypes.Numeric'>
    """

def cell_profile(
    cell_id: builtins.str, /, *, layer: builtins.str | None = None
) -> GeneExpr[Numeric]:
    """The values of one cell across genes, as a ``GeneExpr``.

    Parameters
    ----------
    cell_id : str
        The cell ID. It is checked against the schema when the plan is built (task 3.5).
    layer : str or None, default=None
        The layer to read. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    GeneExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> cell_profile("AAACATACAACCAC-1", layer="counts") > 0
    cell_profile('AAACATACAACCAC-1', layer='counts') > 0
    """

def cell_embedding(name: builtins.str, component: int, /) -> CellExpr[Numeric]:
    """One component of a cell embedding, as a ``CellExpr``.

    Parameters
    ----------
    name : str
        The embedding name. It is checked against the schema when the plan is built
        (task 3.5).
    component : int
        The component (column of the embedding), 0-based.

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Raises
    ------
    ExpressionError
        ``KY-E209`` if ``component`` is not a non-negative ``int``.

    Examples
    --------
    >>> cell_embedding("pca", 0) * 2
    cell_embedding('pca', 0) * 2
    """

def gene_embedding(name: builtins.str, component: int, /) -> GeneExpr[Numeric]:
    """One component of a gene embedding, as a ``GeneExpr``.

    Parameters
    ----------
    name : str
        The embedding name. It is checked against the schema when the plan is built
        (task 3.5).
    component : int
        The component (column of the embedding), 0-based.

    Returns
    -------
    GeneExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Raises
    ------
    ExpressionError
        ``KY-E209`` if ``component`` is not a non-negative ``int``.

    Examples
    --------
    >>> gene_embedding("loadings", 1) > 0
    gene_embedding('loadings', 1) > 0
    """

def feature(name: builtins.str, /) -> CellExpr[Unknown]:
    """A cell feature by name, as a ``CellExpr``.

    The name is short (``"CD3E"``, ``"cluster"``) or full (``"<prefix>:<name>"``, such
    as ``"log1p:CD3E"`` or ``"meta:cluster"``, split at the first ``:``). It is only
    parsed here: the plan resolves it against the schema when it is built and stores
    the full name (R6, task 3.5), so the expression's dtype is ``Unknown`` until then.

    Parameters
    ----------
    name : str
        The short or full feature name.

    Returns
    -------
    CellExpr[Unknown]
        An expression that reads nothing until a plan using it runs.

    Raises
    ------
    ResolutionError
        ``KY-E208`` if ``name`` cannot name any feature: it is empty, a part around its
        first ``:`` is empty, or the prefix is not a valid layer or embedding name.

    Examples
    --------
    >>> feature("log1p:CD3E") > 1.5
    feature('log1p:CD3E') > 1.5
    >>> feature("CD3E").dtype
    <class 'kyto.dtypes.Unknown'>
    >>> feature("log1p:")
    Traceback (most recent call last):
        ...
    kyto.errors.ResolutionError: KY-E208: invalid feature name 'log1p:'...
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
