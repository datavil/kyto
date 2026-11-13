"""Type stub for the compiled extension module `kyto._kyto`."""

import builtins
import os
from collections.abc import Callable
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
        RemovedFeatureError
            ``KY-E203``: a gene read names a gene removed earlier in the plan.
        StaleColumnError
            ``KY-E301``: a stored derived cell column (its ``schema.json`` entry records
            ``lineage``) is read after ``filter_genes``; read
            ``cell_column(name).recompute()`` instead.
        LineageError
            ``KY-E302``: a ``.recompute()`` names a column that records no lineage.

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
    def filter_genes(self, predicate: _GeneExprBoolean, /) -> Atlas:
        """Keep the genes where a boolean gene expression is true.

        The predicate is resolved against this plan now (R6): its names are checked and
        stored in full. Which genes it keeps is known only when the plan runs. From
        here on, a derived cell column that the store computed over all genes (one whose
        ``schema.json`` entry records ``lineage``, such as a stored ``n_genes``) is stale:
        reading it raises ``KY-E301``; read ``cell_column(name).recompute()`` instead.

        Parameters
        ----------
        predicate : GeneExpr[Boolean] or GeneExpr[Unknown]
            One value per gene; genes where it is ``True`` are kept, genes where it is
            ``False`` or null are dropped. A column read (``Unknown`` until resolved)
            must be stored as ``Boolean``.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ExpressionError
            ``KY-E205``: ``predicate`` is not a ``GeneExpr``, or not boolean once
            resolved.
        ResolutionError
            ``KY-E210``: a name in ``predicate`` matches nothing, and the other
            errors of ``resolve``.

        Examples
        --------
        >>> atlas = scan(store)
        >>> print(atlas.filter_genes(gene_column("feature_type") == "Gene Expression"))
        FILTER_GENES gene_column('feature_type') == 'Gene Expression'
        SCAN "..." (2 cells x 3 genes)
        """
    @overload
    def explain(self, *, format: Literal["text"] = "text") -> str:
        """How the plan would run: its logical and optimized trees (task 4.7).

        The logical tree lists the plan's steps from the last down to the scan, each with
        its node ID and its parameters, every name resolved in full as the plan stored it
        (R6: a ``feature`` or a gene read by symbol shows the gene ID and layer it
        names), and, in brackets, a note when the step has a cheaper alternative (a
        centred ``scale`` stores every gene of every cell; PCA centres sparse data
        implicitly). The optimized tree is the plan optimized with every rule (for a terminal
        reading all of its output): one line for what the output reads, then one line per
        node from the top down to the scan, each with the logical nodes it came from
        (``(node 1, 2)`` for a fused pass) and, in brackets, how it runs: ``device``
        (``cpu``), ``passes`` over its input's blocks (with the ``statistics`` passes over
        all cells and the ``apply`` pass when there are statistics passes), and
        ``blocks=N cells`` (streamed in blocks of the store's cell block size), ``in
        memory`` or ``no data``. ``estimate()`` gives the cost of the whole plan. Nothing
        is read.

        Parameters
        ----------
        format : {"text", "json"}, default="text"
            ``"text"``: the two trees as text, under ``logical plan`` and ``optimized
            plan``. ``"json"``: the same as a dict with ``logical`` (one dict per step:
            ``id``, ``name``, ``parameters``, ``input``, ``note``) and ``optimized`` (``outputs``:
            ``node``, ``reads``; ``nodes``: ``index``, ``name``, ``parameters``,
            ``reads`` (a scan's, else ``None``), ``input``, ``origin``, ``device``,
            ``passes``, ``statistics_passes``, ``apply_pass``, ``blocks`` (``"cells"``,
            ``"in_memory"`` or ``"none"``) and ``block_cells``), in the text's order.

        Returns
        -------
        str or dict
            A ``str`` for ``format="text"``, a ``dict`` for ``format="json"``; new
            objects each call (the plan is unchanged).

        Raises
        ------
        PlanError
            ``KY-E215``: ``format`` is not ``"text"`` or ``"json"``.

        Examples
        --------
        >>> genes = gene_column("feature_type") == "Gene Expression"
        >>> print(scan(store).filter_genes(genes).explain())
        logical plan
          node 1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- node 0
          node 0 SCAN "..." (2 cells x 3 genes)
        optimized plan
          output #1 reads [...]
          #1 FILTER_GENES ... <- #0 (node 1) [device=cpu passes=1 blocks=4096 cells]
          #0 SCAN ... (node 0) [device=cpu passes=1 blocks=4096 cells]
        >>> explained = scan(store).filter_genes(genes).explain(format="json")
        >>> [step["name"] for step in explained["logical"]]
        ['FILTER_GENES', 'SCAN']
        >>> explained["optimized"]["nodes"][0]["passes"]
        1
        """
    @overload
    def explain(self, *, format: Literal["json"]) -> Explanation: ...
    def estimate(self) -> CostEstimate:
        """What running the plan would cost, from its schema alone (task 4.6).

        Estimated for the plan as ``explain()`` shows it (every optimizer rule, a
        terminal reading all of its output, the CPU, 2 cell blocks in flight).
        ``passes`` counts the passes over the store's matrix: each statistics pass
        streams the data again from the store, and one more pass delivers what is read
        from the stream. ``bytes_read`` is the decoded size of the matrix tiles those
        passes read (fewer bytes are on disk, compressed; the cell and gene tables are
        not counted). ``peak_memory`` is the most memory the plan holds at once: the
        blocks in flight in every streamed step (sized for blocks of mean density) plus
        statistics, in-memory results and the cell and gene tables the run reads (a
        filter, ``take_cells``, ``select_genes``, a computed column or a sink reads a
        whole table; string lengths are assumed, task 5.10); within 25% of what the
        executor allocates on the test fixtures. Nothing is read.

        Returns
        -------
        CostEstimate
            A new dict with ``passes``, ``bytes_read`` and ``peak_memory`` (bytes, as
            ``int``) and ``device`` (``"cpu"``).

        Examples
        --------
        >>> scan(store).estimate()
        {'passes': 1, 'bytes_read': 16428, 'peak_memory': 1048592, 'device': 'cpu'}
        """
    def _execute(
        self,
        *,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> int:
        """Run the plan and stream its output to nowhere (private, task 5.7).

        Task 5.7's hook until the terminals of 7.2 and 7.3 return the data. Runs the
        plan as ``estimate()`` costs it on the CPU executor, drops every block it
        streams and returns how many cell rows it streamed. The GIL is released while
        the plan computes, so other Python threads run. Ctrl-C (or any exception a
        signal handler raises) cancels the run: it stops at its next block, its threads
        end, and the exception is raised from here; the process stays usable.

        Parameters
        ----------
        threads : int or None, default=None
            Threads that read and compute blocks; ``None`` for one per CPU core. The
            result does not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread whenever the run
            moved: cell rows computed and in the whole run, both summed over its passes
            (each statistics pass, then the output's stream, computes every cell). The
            last call has ``done == total``; a store without cells gets none. An
            exception it raises cancels the run and is raised from here.

        Returns
        -------
        int
            The cell rows the output streamed (the plan's cells).

        Raises
        ------
        PlanError
            ``KY-E215``: ``threads`` is below 1, or ``progress`` is not callable.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.
        KeyboardInterrupt
            Ctrl-C while the plan computes; the run has stopped.

        Examples
        --------
        >>> calls = []
        >>> scan(store)._execute(threads=1, progress=lambda done, total: calls.append((done, total)))
        2
        >>> calls[-1]
        (2, 2)
        """
    def to_json(self) -> str:
        """Write the plan as JSON text, to store it or send it elsewhere.

        The text is versioned (``"version": 1``) and lists the plan's nodes from the
        scan up, each with its stable ID (its position from the scan, 0), the ID of its
        input, and its step: the plan method that built it and its arguments with every
        name resolved in full (R6). The scan holds the store's location, its schema and
        its gene IDs and symbols as read when the plan was built, so ``from_json`` needs
        no store. Equal plans write equal text, and ``from_json(atlas.to_json())``
        gives back an equal plan.

        Returns
        -------
        str
            The plan JSON, on one line (a new string; the plan is unchanged). Nothing
            is read.

        Examples
        --------
        >>> genes = gene_column("feature_type") == "Gene Expression"
        >>> atlas = scan(store).filter_genes(genes)
        >>> text = atlas.to_json()
        >>> text[:39]
        '{"format":"kyto-plan","version":1,"node'
        >>> print(from_json(text))
        FILTER_GENES gene_column('feature_type') == 'Gene Expression'
        SCAN "..." (2 cells x 3 genes)
        """
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

# The dicts that handle `schema()` and `estimate()` return (task 3.6), and
# `Atlas.explain(format="json")` (task 4.7). Type-checking only: they do not exist
# at run time.

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

class CostEstimate(TypedDict):
    """What running a plan would cost (``Atlas.estimate()``); sizes in bytes."""

    passes: int
    bytes_read: int
    peak_memory: int
    device: Literal["cpu"]

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
    device: Literal["cpu"]
    passes: int
    statistics_passes: int
    apply_pass: bool
    blocks: Literal["cells", "in_memory", "none"]
    block_cells: int | None

class ExplainedPlan(TypedDict):
    """The optimized plan of ``Atlas.explain(format="json")``."""

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
            A new list, read from the gene table when the plan was built. After
            ``filter_genes``, the genes that may remain: which ones is known only
            when the plan runs, and ``estimate()`` gives ``n_columns`` as ``None``.

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

        Missing metadata is null (an imported ``.h5ad``'s float ``NaN`` too); matrix
        values and embeddings are never null (an absent sparse value is 0). Use this, not ``x == None``
        (``KY-E212``). Nothing is read.
        """

    def is_not_null(self) -> CellExpr[Boolean]:
        """Whether each value is present (not null); the negation of ``is_null()``."""

    def recompute(self) -> CellExpr[_D_co]:
        """This column, computed again from its lineage instead of read as stored.

        A derived column the store computed (one whose ``schema.json`` entry records
        ``lineage``, such as ``n_genes``) goes stale once a step such as
        ``filter_genes`` changes its inputs: reading it then raises ``KY-E301``. A
        recomputed read is valid there. Nothing is computed now: the executor (task
        5.x) computes it when the plan runs.

        Returns
        -------
        CellExpr
            A new expression of the same dtype. Nothing is read.

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

        Missing metadata is null (an imported ``.h5ad``'s float ``NaN`` too); matrix
        values and embeddings are never null (an absent sparse value is 0). Use this, not ``x == None``
        (``KY-E212``). Nothing is read.
        """

    def is_not_null(self) -> GeneExpr[Boolean]:
        """Whether each value is present (not null); the negation of ``is_null()``."""

    def recompute(self) -> GeneExpr[_D_co]:
        """This column, computed again from its lineage instead of read as stored.

        A derived column the store computed (one whose ``schema.json`` entry records
        ``lineage``, such as ``n_genes``) goes stale once a step such as
        ``filter_genes`` changes its inputs: reading it then raises ``KY-E301``. A
        recomputed read is valid there. Nothing is computed now: the executor (task
        5.x) computes it when the plan runs.

        Returns
        -------
        GeneExpr
            A new expression of the same dtype. Nothing is read.

        Raises
        ------
        ExpressionError
            ``KY-E213``: the expression is not a plain column read.

        Examples
        --------
        >>> gene_column("n_cells").recompute()
        gene_column('n_cells').recompute()
        """

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
    gene_id: builtins.str,
    /,
    *,
    layer: builtins.str | None = None,
    by: Literal["id", "symbol"] = "id",
) -> CellExpr[Numeric]:
    """The values of one gene across cells, as a ``CellExpr``.

    Parameters
    ----------
    gene_id : str
        The gene ID, or its symbol with ``by='symbol'``. It is checked against the
        schema when the plan is built (task 3.5).
    layer : str or None, default=None
        The layer to read. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).
    by : {'id', 'symbol'}, default='id'
        How ``gene_id`` names the gene. ``'id'``: it is the gene ID (the ``gene_id``
        column). ``'symbol'``: it is a gene symbol (the ``symbol`` column of the gene
        table), resolved once to the ID of the one gene that carries it when the plan
        is built (R6). A symbol several genes carry raises ``ResolutionError``
        (``KY-E204``) listing their IDs; one no gene carries, or a store without
        symbols, raises ``KY-E210``. Symbols are matched exactly, case included.

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Raises
    ------
    ResolutionError
        ``KY-E214`` if ``by`` is not ``'id'`` or ``'symbol'``.

    Examples
    --------
    >>> gene_expression("CD3E", layer="log1p") > 1.5
    gene_expression('CD3E', layer='log1p') > 1.5
    >>> gene_expression("CD3E").dtype
    <class 'kyto.dtypes.Numeric'>
    >>> gene_expression("CD3E", by="symbol")
    gene_expression('CD3E', by='symbol')
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

def total_counts(*, layer: builtins.str | None = None) -> CellExpr[Numeric]:
    """Each cell's total count: the sum of its values over every gene, as a ``CellExpr``.

    scanpy's ``total_counts`` (``scanpy.pp.calculate_qc_metrics``), computed when a
    plan reads it, over the genes present then, so it never goes stale. Summed in
    ``float64`` (scanpy: ``float32``).

    Parameters
    ----------
    layer : str or None, default=None
        The layer to sum. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> total_counts(layer="counts") > 500
    total_counts(layer='counts') > 500
    """

def genes_detected(*, layer: builtins.str | None = None) -> CellExpr[Numeric]:
    """Each cell's number of genes with a non-zero value, as a ``CellExpr``.

    scanpy's ``n_genes_by_counts`` (``scanpy.pp.calculate_qc_metrics``), computed when
    a plan reads it, over the genes present then, so it never goes stale.

    Parameters
    ----------
    layer : str or None, default=None
        The layer to count in. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    CellExpr[Numeric]
        A numeric (``UInt32``) expression that reads nothing until a plan using it runs.

    Examples
    --------
    >>> genes_detected(layer="counts") >= 200
    genes_detected(layer='counts') >= 200
    """

def total_counts_in(
    genes: _GeneExprBoolean, /, *, layer: builtins.str | None = None
) -> CellExpr[Numeric]:
    """Each cell's total count over a gene set, as a ``CellExpr``.

    scanpy's ``total_counts_<qc_var>`` (``scanpy.pp.calculate_qc_metrics``), computed
    when a plan reads it, over the genes present then, so it never goes stale. Summed
    in ``float64`` (scanpy: ``float32``).

    Parameters
    ----------
    genes : GeneExpr
        The gene set: a boolean expression on gene-table columns, such as
        ``gene_column('symbol').str.starts_with('MT-')``. A gene where it is null is
        not in the set. Checked when the plan is built.
    layer : str or None, default=None
        The layer to sum. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Raises
    ------
    ExpressionError
        ``KY-E205`` if ``genes`` is not a ``GeneExpr`` or not boolean (a column of
        unknown dtype is checked when the plan is built, as is an expression that
        reads matrix values).

    Examples
    --------
    >>> total_counts_in(gene_column("symbol").str.starts_with("MT-"))
    total_counts_in(gene_column('symbol').str.starts_with('MT-'))
    """

def percent_counts_in(
    genes: _GeneExprBoolean, /, *, layer: builtins.str | None = None
) -> CellExpr[Numeric]:
    """Each cell's percentage of counts in a gene set, as a ``CellExpr``.

    scanpy's ``pct_counts_<qc_var>`` (``scanpy.pp.calculate_qc_metrics``):
    ``total_counts_in(genes) / total_counts() * 100``, ``NaN`` for a cell without
    counts. Computed when a plan reads it, over the genes present then, so it never
    goes stale; in ``float64`` (scanpy: ``float32``).

    Parameters
    ----------
    genes : GeneExpr
        The gene set: a boolean expression on gene-table columns, such as
        ``gene_column('symbol').str.starts_with('MT-')``. A gene where it is null is
        not in the set. Checked when the plan is built.
    layer : str or None, default=None
        The layer to sum. ``None`` reads the plan's default layer
        (``Atlas.with_default_layer``).

    Returns
    -------
    CellExpr[Numeric]
        A numeric expression that reads nothing until a plan using it runs.

    Raises
    ------
    ExpressionError
        ``KY-E205`` if ``genes`` is not a ``GeneExpr`` or not boolean (a column of
        unknown dtype is checked when the plan is built, as is an expression that
        reads matrix values).

    Examples
    --------
    >>> mitochondrial = gene_column("symbol").str.starts_with("MT-")
    >>> percent_counts_in(mitochondrial, layer="counts") < 5
    percent_counts_in(gene_column('symbol').str.starts_with('MT-'), layer='counts') < 5
    """

def from_json(text: builtins.str, /) -> Atlas:
    """Read a plan from the JSON text that ``Atlas.to_json`` wrote.

    The plan is rebuilt step by step with the same checks as when it was first built,
    against the store snapshot the text holds (schema, gene IDs and symbols). No store
    is opened or read; the store is checked when the plan runs.

    Parameters
    ----------
    text : str
        A plan JSON, as ``Atlas.to_json`` writes it.

    Returns
    -------
    Atlas
        A new plan, equal to the one that wrote ``text``.

    Raises
    ------
    PlanError
        ``KY-E217``: ``text`` is not JSON, not a plan JSON, or breaks its format (the
        message names the line and column, or the node).
        ``KY-E218``: the text has a ``version`` this kyto does not read.
    KytoError
        Any error a plan method raises for its arguments, when the text was edited
        into a plan that does not build (such as ``KY-E109`` for a missing layer).

    Examples
    --------
    >>> text = (
    ...     '{"format":"kyto-plan","version":1,"nodes":[{"id":0,"input":null,"step":'
    ...     '{"method":"scan","source":"pbmc.kyto","schema":{"n_cells":2,"n_genes":1,'
    ...     '"tiling":{"cell_block_size":4096,"gene_block_size":1024},'
    ...     '"cells":{"columns":[{"name":"cell_id","dtype":"String"}]},'
    ...     '"genes":{"columns":[{"name":"gene_id","dtype":"String"}]},'
    ...     '"layers":[{"name":"counts","dtype":"uint32","nnz":2}],'
    ...     '"embeddings":[],"graphs":[]},"gene_ids":["CD3E"],"symbols":null}}]}'
    ... )
    >>> print(from_json(text))
    SCAN "pbmc.kyto" (2 cells x 1 genes)
    >>> from_json('{"format":"kyto-plan","version":2,"nodes":[]}')
    Traceback (most recent call last):
        ...
    kyto.errors.PlanError: KY-E218: the plan JSON has version 2, but this kyto reads only version 1...
    """

def validate_plan(text: builtins.str, /) -> list[PlanDiagnostic]:
    """Check a plan JSON and list what is wrong with it, without raising.

    The checks are the ones ``from_json`` makes, in its order: JSON syntax, the
    ``version`` (another version is one ``KY-E218`` and nothing else), the shape
    against the published JSON Schema (``plan_json_schema()``; every mismatch is
    listed), then, when the shape is right, each step replayed through its plan method,
    which stops at the first step that does not build (later steps build on it). No
    store is opened or read.

    Parameters
    ----------
    text : str
        A plan JSON, as ``Atlas.to_json`` writes it (any formatting).

    Returns
    -------
    list[PlanDiagnostic]
        A new list, empty when ``from_json(text)`` reads the text; otherwise one dict
        per problem with ``code`` (``"KY-E217"``), ``path`` (a JSON Pointer into the
        text, such as ``"/nodes/2/step/layer"``; ``""`` for the whole text), ``message``
        (what is wrong there) and ``hint`` (how to fix it). The first has the code
        ``from_json`` raises; past the shape check, it is that error's message and hint
        (a shape problem is worded by the schema check, more precisely than
        ``from_json`` words it).

    Examples
    --------
    >>> diagnostics = validate_plan('{"format": "kyto-plan", "version": 1, "nodes": []}')
    >>> [(diagnostic["code"], diagnostic["path"]) for diagnostic in diagnostics]
    [('KY-E217', '/nodes')]
    >>> validate_plan('{"format": "kyto-plan", "version": 2, "nodes": []}')[0]["code"]
    'KY-E218'
    >>> validate_plan("not json")[0]["message"]
    'the plan JSON is invalid at line 1 column 2: expected ident'
    """

def plan_json_schema() -> builtins.str:
    """The JSON Schema (draft 2020-12) of the plan JSON this kyto writes and reads.

    It describes the shape of the text ``Atlas.to_json`` writes (version 1); a text can
    match it and still not build (an unknown layer, a gene filter reading a cell column).
    ``validate_plan`` checks both.

    Returns
    -------
    str
        The schema as JSON text, the same on every call (a new ``str``).

    Examples
    --------
    >>> import json
    >>> schema = json.loads(plan_json_schema())
    >>> schema["$schema"]
    'https://json-schema.org/draft/2020-12/schema'
    >>> schema["title"]
    'kyto plan JSON, version 1'
    """

def scan(store: str | os.PathLike[str], /) -> Atlas:
    """Open the kyto store at ``store`` as the root of a lazy plan.

    Reads the store's ``schema.json`` and its gene IDs (and gene symbols, when the gene
    table has a ``symbol`` column), so names can be resolved and checked when the plan
    is built (R6). Reads no matrix tile and no other column (R1). Every method of the
    returned ``Atlas`` builds a new plan on top of this one (R2).

    Parameters
    ----------
    store : str or os.PathLike[str]
        Path of a local kyto store, as ``convert`` writes it.

    Returns
    -------
    Atlas
        A new plan with one ``SCAN`` step. Nothing of the matrix is read or copied.

    Raises
    ------
    CorruptStoreError
        ``KY-E116``: ``store`` does not exist or is not a kyto store.
        ``KY-E113``, ``KY-E122``: its gene table or ``schema.json`` breaks the format.
    FormatVersionError
        ``KY-E115``: the store has a ``format_version`` this kyto does not read.
    StoreIOError
        ``KY-E110``: a file of the store cannot be read.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> folder = Path(tempfile.mkdtemp())
    >>> _ = (folder / "matrix.mtx").write_text(
    ...     "%%MatrixMarket matrix coordinate integer general\\n3 2 3\\n"
    ...     "1 1 5\\n3 1 1\\n2 2 7\\n"
    ... )
    >>> _ = (folder / "barcodes.tsv").write_text("AAAC-1\\nAAAG-1\\n")
    >>> _ = (folder / "features.tsv").write_text(
    ...     "ENSG01\\tCD3E\\tGene Expression\\nENSG02\\tCD4\\tGene Expression\\n"
    ...     "ENSG03\\tMS4A1\\tGene Expression\\n"
    ... )
    >>> convert(folder, folder / "pbmc.kyto")
    >>> atlas = scan(folder / "pbmc.kyto")
    >>> print(atlas)
    SCAN "..." (2 cells x 3 genes)
    >>> scan(folder / "missing.kyto")
    Traceback (most recent call last):
        ...
    kyto.errors.CorruptStoreError: KY-E116: ...
    """

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
