"""Type stub for the compiled extension module `kyto._kyto`."""

import builtins
import os
from collections.abc import Callable, Sequence
from typing import (
    Any,
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
            ``lineage``) is read after a gene ``filter``; read
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
    def filter(self, predicate: _CellExprBoolean | _GeneExprBoolean, /) -> Atlas:
        """Keep the cells or the genes where a boolean expression is true.

        A ``CellExpr`` filters cells, a ``GeneExpr`` filters genes (R5). The predicate is
        resolved against this plan now (R6): its names are checked and stored in full.
        Which cells or genes it keeps is known only when the plan runs. After a gene
        filter, a derived cell column that the store computed over all genes (one whose
        ``schema.json`` entry records ``lineage``, such as a stored ``n_genes``) is stale:
        reading it raises ``KY-E301``; read ``cell_column(name).recompute()`` instead.

        Parameters
        ----------
        predicate : CellExpr[Boolean] or GeneExpr[Boolean]
            One value per cell or gene; where it is ``True`` the cell or gene is kept,
            where it is ``False`` or null it is dropped. A column read (``Unknown`` until
            resolved) must be stored as ``Boolean``.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ExpressionError
            ``KY-E205``: ``predicate`` is not a ``CellExpr`` or a ``GeneExpr``, or not
            boolean once resolved.
        ResolutionError
            ``KY-E210``: a name in ``predicate`` matches nothing, and the other
            errors of ``resolve``.

        Examples
        --------
        >>> atlas = scan(store)
        >>> print(atlas.filter(gene_column("feature_type") == "Gene Expression"))
        FILTER_GENES gene_column('feature_type') == 'Gene Expression'
        SCAN "..." (2 cells x 3 genes)
        >>> print(atlas.filter(total_counts(layer="counts") >= 6))
        FILTER_CELLS total_counts(layer='counts') >= 6
        SCAN "..." (2 cells x 3 genes)
        """
    def select_genes(self, gene_ids: Sequence[str], /) -> Atlas:
        """Keep the given genes, in the given order.

        Parameters
        ----------
        gene_ids : list of str
            Gene IDs (the store's ``gene_id`` column, not symbols), each once.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ResolutionError
            ``KY-E210``: a gene ID is not in the store, with the closest IDs.
        RemovedFeatureError
            ``KY-E203``: a gene was removed earlier in the plan.
        PlanError
            ``KY-E215``: a gene ID is listed twice.

        Examples
        --------
        >>> print(scan(store).select_genes(["ENSG03", "ENSG01"]))
        SELECT_GENES ['ENSG03', 'ENSG01']
        SCAN "..." (2 cells x 3 genes)
        """
    def take_cells(self, cell_ids: Sequence[str], /) -> Atlas:
        """Keep the given cells.

        The cells stream in the store's order, not in the order given (deviations.md).
        Whether every ID names a cell is checked when the plan runs (the cell IDs are
        not read when the plan is built).

        Parameters
        ----------
        cell_ids : list of str
            Cell IDs (the store's ``cell_id`` column), each once.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ResolutionError
            ``KY-E210``: a cell ID is empty (here), or names no cell of the plan (when
            it runs).
        PlanError
            ``KY-E215``: a cell ID is listed twice.

        Examples
        --------
        >>> print(scan(store).take_cells(["AAAG-1"]))
        TAKE_CELLS ['AAAG-1']
        SCAN "..." (2 cells x 3 genes)
        """
    def with_cells(self, **columns: CellExpr[ExprDtype]) -> Atlas:
        """Add or replace cell columns, each computed from a cell expression.

        The expressions are resolved against this plan now (R6) and computed when the
        plan runs. A name equal to a stored column replaces it.

        Parameters
        ----------
        **columns : CellExpr
            The new columns, by name.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ExpressionError
            ``KY-E205``: a value is not a ``CellExpr``.
        ResolutionError
            ``KY-E210``: a name in an expression matches nothing, and the other errors
            of ``resolve``.
        PlanError
            ``KY-E215``: no column is given.

        Examples
        --------
        >>> print(scan(store).with_cells(n_counts=total_counts(layer="counts")))
        WITH_CELLS "n_counts" = total_counts(layer='counts')
        SCAN "..." (2 cells x 3 genes)
        """
    def with_genes(self, **columns: GeneExpr[ExprDtype]) -> Atlas:
        """Add or replace gene columns, each computed from a gene expression.

        The expressions are resolved against this plan now (R6) and computed when the
        plan runs. A name equal to a stored column replaces it.

        Parameters
        ----------
        **columns : GeneExpr
            The new columns, by name.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ExpressionError
            ``KY-E205``: a value is not a ``GeneExpr``.
        ResolutionError
            ``KY-E210``: a name in an expression matches nothing, and the other errors
            of ``resolve``.
        PlanError
            ``KY-E215``: no column is given.

        Examples
        --------
        >>> print(scan(store).with_genes(is_cd3e=gene_column("symbol") == "CD3E"))
        WITH_GENES "is_cd3e" = gene_column('symbol') == 'CD3E'
        SCAN "..." (2 cells x 3 genes)
        """
    def with_embedding(
        self, name: str, values: npt.NDArray[np.float32 | np.float64], /
    ) -> Atlas:
        """Add a cell embedding computed outside kyto (not implemented yet).

        Arrives with task 6.11 (interop hooks: scvi-tools and Harmony outputs).

        Parameters
        ----------
        name : str
            Name of the new embedding.
        values : numpy.ndarray
            One row per cell of the plan, in ``cell_ids()`` order.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged.

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: always, until task 6.11.

        Examples
        --------
        >>> scan(store).with_embedding("X_scvi", None)
        Traceback (most recent call last):
        ...
        kyto.errors.NotImplementedYetError: KY-E601: 'Atlas.with_embedding' is not implemented yet...
        """
    def normalize_total(
        self,
        target_sum: float | None = None,
        *,
        layer: str | None = None,
        into: str,
    ) -> Atlas:
        """Scale every cell to the same total count (scanpy's ``normalize_total``).

        Each cell's values are multiplied by ``target_sum / total``, its total over the
        genes of the plan; a cell with total 0 stays 0. Writes ``float32``.

        Parameters
        ----------
        target_sum : float or None, default=None
            The total of every cell afterwards; ``None`` for the median of the cells'
            totals (one more pass over the cells).
        layer : str or None, default=None
            The layer read; ``None`` for the default layer.
        into : str
            The layer written (a new name, or an existing layer to replace).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``target_sum`` is not a positive finite ``float32``.
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer is not in the plan.
        StoreInputError
            ``KY-E102``: ``into`` is not a valid layer name.

        Examples
        --------
        >>> atlas = scan(store).normalize_total(1e4, layer="counts", into="normalized")
        >>> print(atlas)
        NORMALIZE_TOTAL "counts" -> "normalized" target_sum=10000
        SCAN "..." (2 cells x 3 genes)
        """
    def log1p(self, *, layer: str | None = None, into: str) -> Atlas:
        """Take ``log(1 + x)`` of every value (natural log, scanpy's ``log1p``).

        Writes ``float32``. Absent values stay absent (``log1p(0) == 0``).

        Parameters
        ----------
        layer : str or None, default=None
            The layer read; ``None`` for the default layer.
        into : str
            The layer written (a new name, or an existing layer to replace).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer is not in the plan.

        Examples
        --------
        >>> print(scan(store).with_default_layer("counts").log1p(into="log1p"))
        LOG1P "counts" -> "log1p"
        WITH_DEFAULT_LAYER "counts"
        SCAN "..." (2 cells x 3 genes)
        """
    def scale(
        self,
        *,
        layer: str | None = None,
        into: str,
        zero_center: bool = True,
        max_value: float | None = None,
    ) -> Atlas:
        """Scale every gene to mean 0 and variance 1 (scanpy's ``scale``).

        Centring stores every gene of every cell (``explain()`` notes it; PCA centres
        sparse data implicitly). Writes ``float32``.

        Parameters
        ----------
        layer : str or None, default=None
            The layer read; ``None`` for the default layer.
        into : str
            The layer written (a new name, or an existing layer to replace).
        zero_center : bool, default=True
            Subtract each gene's mean; ``False`` only divides by the standard deviation
            (and keeps the data sparse).
        max_value : float or None, default=None
            Clip the scaled values to ``[-max_value, max_value]`` (``[.., max_value]``
            without centring); ``None`` does not clip.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``max_value`` is not positive.
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer is not in the plan.

        Examples
        --------
        >>> print(scan(store).scale(layer="counts", into="scaled", max_value=10))
        SCALE "counts" -> "scaled" zero_center=true max_value=10
        SCAN "..." (2 cells x 3 genes)
        """
    def highly_variable_genes(
        self,
        *,
        layer: str | None = None,
        flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "seurat",
        n_top_genes: int | None = None,
        batch_key: str | None = None,
    ) -> Atlas:
        """Mark the highly variable genes (scanpy's ``highly_variable_genes``).

        Adds scanpy's gene columns: ``highly_variable`` (``Boolean``), ``means``,
        ``dispersions`` and ``dispersions_norm`` (``seurat``, ``cell_ranger``) or
        ``means``, ``variances``, ``variances_norm`` and ``highly_variable_rank``
        (``seurat_v3``), all ``Float64``, and with ``batch_key``
        ``highly_variable_nbatches`` (``Int64``) and ``highly_variable_intersection``.
        Keep the genes with ``filter(gene_column("highly_variable"))``.

        Parameters
        ----------
        layer : str or None, default=None
            The layer read (log data for ``seurat`` and ``cell_ranger``, counts for
            ``seurat_v3``); ``None`` for the default layer.
        flavor : {"seurat", "cell_ranger", "seurat_v3"}, default="seurat"
            The method, as scanpy's.
        n_top_genes : int or None, default=None
            Mark this many genes; ``None`` uses scanpy's dispersion cutoffs (not with
            ``seurat_v3``, which needs it).
        batch_key : str or None, default=None
            A cell column: genes are ranked in each batch and combined as scanpy does.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: an unknown ``flavor``, ``n_top_genes`` below 1, or
            ``seurat_v3`` without ``n_top_genes``.
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer or ``batch_key`` is not in the plan.

        Examples
        --------
        >>> atlas = scan(store).highly_variable_genes(layer="counts", n_top_genes=2)
        >>> print(atlas)
        HIGHLY_VARIABLE_GENES "counts" flavor=seurat n_top_genes=2
        SCAN "..." (2 cells x 3 genes)
        """
    def pca(
        self,
        *,
        layer: str | None = None,
        into: str,
        n_components: int = 50,
        seed: int,
    ) -> Atlas:
        """Principal components of a layer (scanpy's ``pca``, implicit centring).

        Streams the layer in ``2 + PCA_POWER_ITERATIONS`` passes (a randomized block
        Krylov method) and never densifies it. Adds the cell embedding ``into`` (one row
        per cell, ``n_components`` wide) and the gene embedding ``into + "_loadings"``.

        Parameters
        ----------
        layer : str or None, default=None
            The layer read; ``None`` for the default layer.
        into : str
            Name of the cell embedding written.
        n_components : int, default=50
            Components computed.
        seed : int
            Seed of the random start (R7, R11: the same seed gives the same result).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``n_components`` is below 1 or above the cells or genes, or
            ``seed`` is negative.
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer is not in the plan.

        Examples
        --------
        >>> print(scan(store).pca(layer="counts", into="pca", n_components=1, seed=0))
        PCA "counts" -> "pca" n_components=1 seed=0
        SCAN "..." (2 cells x 3 genes)
        """
    def neighbors(
        self,
        embedding: str,
        /,
        *,
        n_neighbors: int = 15,
        n_components: int | None = None,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        search: Literal["exact", "approximate"] = "exact",
        seed: int = 0,
        into: str | None = None,
    ) -> Atlas:
        """The k nearest neighbours of every cell in a cell embedding (scanpy's
        ``neighbors``).

        Adds the cell graph of distances, ``"distances"`` (or ``into + "_distances"``),
        and declares the connectivities graph, ``"connectivities"`` (or
        ``into + "_connectivities"``).

        Parameters
        ----------
        embedding : str
            The cell embedding searched, such as the ``into`` of ``pca``.
        n_neighbors : int, default=15
            Neighbours per cell, the cell itself included (scanpy's meaning).
        n_components : int or None, default=None
            Use the first this many columns of the embedding; ``None`` uses all.
        metric : {"euclidean", "cosine"}, default="euclidean"
            The distance.
        search : {"exact", "approximate"}, default="exact"
            Exact blocked search, or HNSW (approximate, faster on large atlases).
        seed : int, default=0
            Seed of the approximate search (R7, R11).
        into : str or None, default=None
            Prefix of the graph names; ``None`` for scanpy's names.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``n_neighbors`` below 2 or above the cells, ``n_components``
            out of range, a negative ``seed``, or an unknown ``metric`` or ``search``.
        MissingMemberError
            ``KY-E109``: ``embedding`` is not a cell embedding of the plan.

        Examples
        --------
        >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
        >>> print(atlas.neighbors("pca", n_neighbors=2))
        NEIGHBORS "pca" -> "distances", "connectivities" n_neighbors=2 metric=Euclidean search=Exact seed=0
        PCA "counts" -> "pca" n_components=1 seed=0
        SCAN "..." (2 cells x 3 genes)
        """
    def leiden(
        self,
        *,
        graph: str = "connectivities",
        resolution: float = 1.0,
        seed: int,
        into: str,
    ) -> Atlas:
        """Cluster the cells of a cell graph with Leiden (scanpy's ``leiden``).

        Adds the cell column ``into``: one cluster label per cell.

        Parameters
        ----------
        graph : str, default="connectivities"
            The cell graph clustered, such as the connectivities ``neighbors`` declares.
        resolution : float, default=1.0
            Higher values give more, smaller clusters.
        seed : int
            Seed of the random steps (R7, R11: the same seed gives the same clusters).
        into : str
            The cell column written.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``resolution`` is not positive, ``seed`` is not an int of 0 or
            more, or ``into`` is not a valid column name.
        MissingMemberError
            ``KY-E109``: ``graph`` is not a cell graph of the plan.

        Examples
        --------
        >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
        >>> atlas = atlas.neighbors("pca", n_neighbors=2).leiden(into="leiden", seed=0)
        >>> print(str(atlas).splitlines()[0])
        LEIDEN "connectivities" -> "leiden" resolution=1 seed=0
        """
    def louvain(
        self,
        *,
        graph: str = "connectivities",
        resolution: float = 1.0,
        seed: int,
        into: str,
    ) -> Atlas:
        """Cluster the cells of a cell graph with Louvain (scanpy's ``louvain``).

        Adds the cell column ``into``: one cluster label per cell.

        Parameters
        ----------
        graph : str, default="connectivities"
            The cell graph clustered, such as the connectivities ``neighbors`` declares.
        resolution : float, default=1.0
            Higher values give more, smaller clusters.
        seed : int
            Seed of the random steps (R7, R11: the same seed gives the same clusters).
        into : str
            The cell column written.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: ``resolution`` is not positive, ``seed`` is not an int of 0 or
            more, or ``into`` is not a valid column name.
        MissingMemberError
            ``KY-E109``: ``graph`` is not a cell graph of the plan.

        Examples
        --------
        >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
        >>> atlas = atlas.neighbors("pca", n_neighbors=2).louvain(into="louvain", seed=0)
        >>> print(str(atlas).splitlines()[0])
        LOUVAIN "connectivities" -> "louvain" resolution=1 seed=0
        """
    def umap(
        self,
        *,
        graph: str = "connectivities",
        into: str,
        n_components: int = 2,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_epochs: int | None = None,
        alpha: float = 1.0,
        gamma: float = 1.0,
        negative_sample_rate: int = 5,
        init: Literal["spectral", "random"] = "spectral",
        seed: int,
    ) -> Atlas:
        """Lay the cells of a cell graph out in a few dimensions (scanpy's ``umap``).

        Adds the cell embedding ``into`` (``float32``, ``n_components`` wide).

        Parameters
        ----------
        graph : str, default="connectivities"
            The cell graph laid out, such as the connectivities ``neighbors`` declares.
        into : str
            The cell embedding written.
        n_components : int, default=2
            Dimensions of the layout (1 to 100).
        min_dist : float, default=0.5
            The smallest distance between points of the layout (at most ``spread``).
        spread : float, default=1.0
            The scale of the layout's distances.
        n_epochs : int or None, default=None
            Epochs of the optimisation; ``None`` for scanpy's default.
        alpha : float, default=1.0
            The initial learning rate.
        gamma : float, default=1.0
            The weight of the negative samples (repulsion).
        negative_sample_rate : int, default=5
            Negative samples per positive sample.
        init : {"spectral", "random"}, default="spectral"
            Where the layout starts.
        seed : int
            Seed of the random steps (R7, R11).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: an argument out of range (``n_components`` from 1 to 100,
            positive ``spread`` and ``alpha``, ``min_dist`` from 0 to ``spread``), an
            unknown ``init``, or an ``into`` that is not a valid embedding name.
        MissingMemberError
            ``KY-E109``: ``graph`` is not a cell graph of the plan.

        Examples
        --------
        >>> atlas = scan(store).pca(layer="counts", into="pca", n_components=1, seed=0)
        >>> atlas = atlas.neighbors("pca", n_neighbors=2).umap(into="X_umap", seed=0)
        >>> print(str(atlas).splitlines()[0])
        UMAP "connectivities" -> "X_umap" n_components=2 min_dist=0.5 spread=1 n_epochs=auto alpha=1 gamma=1 negative_sample_rate=5 init=Spectral seed=0
        """
    def scrublet(
        self,
        *,
        layer: str | None = None,
        batch_key: str | None = None,
        sim_doublet_ratio: float = 2.0,
        expected_doublet_rate: float = 0.05,
        stdev_doublet_rate: float = 0.02,
        n_prin_comps: int = 30,
        n_neighbors: int | None = None,
        threshold: float | None = None,
        seed: int,
        into: tuple[str, str] | None = None,
    ) -> Atlas:
        """Score every cell as a possible doublet (scanpy's ``pp.scrublet``).

        Adds two cell columns: the doublet score (``Float64``) and the call
        (``Boolean``), ``"doublet_score"`` and ``"predicted_doublet"`` unless ``into``
        names them. Each batch (all cells without ``batch_key``) runs scanpy's steps on
        its own: genes in fewer than 3 cells and cells with fewer than 3 genes are left
        out (their score and call are null), then the counts are normalised, the highly
        variable genes kept, doublets simulated from seeded random pairs of cells, the
        cells and doublets embedded by PCA, and each cell scored by the doublets among
        its nearest neighbours. The call compares the score to ``threshold``, or to
        scikit-image's ``threshold_minimum`` of the simulated doublets' scores (no
        doublet is called where it finds no threshold). The layer must hold raw counts.
        The run holds the layer in memory, as scanpy does.

        Parameters
        ----------
        layer : str or None, default=None
            The layer scored (raw counts); ``None`` for the plan's default layer.
        batch_key : str or None, default=None
            A cell column: each of its values is scored on its own.
        sim_doublet_ratio : float, default=2.0
            Doublets simulated per cell.
        expected_doublet_rate : float, default=0.05
            The expected share of doublets (above 0, below 1).
        stdev_doublet_rate : float, default=0.02
            The uncertainty of ``expected_doublet_rate``.
        n_prin_comps : int, default=30
            Principal components of the embedding (below the cells and the genes kept).
        n_neighbors : int or None, default=None
            Neighbours per cell; ``None`` for ``round(0.5 * sqrt(cells))``.
        threshold : float or None, default=None
            The score above which a cell is called a doublet; ``None`` finds it.
        seed : int
            Seed of the simulated doublets (R7, R11).
        into : tuple of (str, str) or None, default=None
            The names of the score and call columns; ``None`` for scanpy's names.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        PlanError
            ``KY-E215``: an argument out of range (positive ``sim_doublet_ratio``,
            ``expected_doublet_rate`` between 0 and 1, ``n_prin_comps`` and
            ``n_neighbors`` of 1 or more, a finite ``threshold``), or invalid or equal
            column names in ``into``. When the plan runs: ``n_prin_comps`` not below a
            batch's cells and genes kept, a batch of fewer than 2 scored cells or no
            variable gene, a ``sim_doublet_ratio`` giving no doublet or more than the
            pairs of cells, or a null batch.
        MissingMemberError
            ``KY-E109``: ``layer`` is not a layer of the plan.
        ResolutionError
            ``KY-E210``: ``batch_key`` is not a cell column of the plan.

        Examples
        --------
        >>> print(str(scan(store).scrublet(layer="counts", seed=0)).splitlines()[0])
        SCRUBLET "counts" -> ["doublet_score", "predicted_doublet"] sim_doublet_ratio=2 expected_doublet_rate=0.05 stdev_doublet_rate=0.02 n_prin_comps=30 n_neighbors=auto threshold=auto seed=0
        """
    def rank_genes_groups(
        self,
        groupby: str,
        /,
        *,
        method: Literal["t-test", "wilcoxon"] = "t-test",
        layer: str | None = None,
        groups: Sequence[str] | None = None,
        reference: str | None = None,
        n_genes: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        correction: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
        threads: int | None = None,
    ) -> pl.LazyFrame:
        """Rank the genes of each group by a differential expression test (scanpy's
        ``rank_genes_groups`` then ``rank_genes_groups_df``).

        The arguments and the plan's layer and ``groupby`` column are checked now; the
        plan runs only when the returned frame is collected (R1), with the GIL released.
        One row per (group, gene): groups in order (an ``Enum`` column's categories,
        numbers by value, else text sorted; or as ``groups`` lists them), each group's
        genes by statistic, descending (ties in gene order). Cells with a null label are
        in no group and in every rest.

        Parameters
        ----------
        groupby : str
            The cell column whose values are the groups (stored, or added by
            ``with_cells``).
        method : {"t-test", "wilcoxon"}, default="t-test"
            Welch's t-test, or Wilcoxon's rank-sum test (normal approximation).
        layer : str or None, default=None
            The layer tested (log data, as scanpy expects); ``None`` for the default
            layer.
        groups : list of str or None, default=None
            The groups tested, in this order; ``None`` tests every group. The rest is
            still every other cell.
        reference : str or None, default=None
            The group every other is compared with; ``None`` compares with the rest.
        n_genes : int or None, default=None
            Keep the first this many genes of each group; ``None`` keeps every gene.
        rankby_abs : bool, default=False
            Order genes by the absolute statistic.
        tie_correct : bool, default=False
            Wilcoxon only: correct the variance for ties.
        correction : {"benjamini-hochberg", "bonferroni"}, default="benjamini-hochberg"
            How the p-values are adjusted.
        threads : int or None, default=None
            Threads that read and compute blocks; ``None`` for one per CPU core. The
            result does not depend on it (R11).

        Returns
        -------
        polars.LazyFrame
            Columns ``group``, ``gene_id`` (``String``), ``statistic``,
            ``log_fold_change``, ``p_value``, ``adjusted_p_value`` (``Float64``).
            Collecting it runs the plan and returns a new frame the caller owns
            (R2), handed to Polars without a copy (the Arrow C stream interface).

        Raises
        ------
        PlanError
            ``KY-E215``: an unknown ``method`` or ``correction``, ``n_genes`` below 1,
            ``threads`` below 1; when collected, an empty or repeated ``groups``, no
            labelled cell, or a tested group of one cell.
        ResolutionError
            ``KY-E211``: ``layer`` is ``None`` and the plan has no default layer.
        MissingMemberError
            ``KY-E109``: the layer or ``groupby`` is not in the plan.
        ResolutionError
            ``KY-E214`` (when collected): a ``groups`` or ``reference`` label no cell
            has.

        Examples
        --------
        >>> frame = scan(store).rank_genes_groups("cell_id", layer="counts")
        >>> frame.collect_schema().names()
        ['group', 'gene_id', 'statistic', 'log_fold_change', 'p_value', 'adjusted_p_value']
        """
    def on(self, *, device: Literal["cpu", "auto", "cuda", "wgpu", "metal"]) -> Atlas:
        """Run the plan on a device (R7: the device is an argument, not a setting).

        Today every plan runs on the CPU; the GPU devices arrive with task 9.1.

        Parameters
        ----------
        device : {"cpu"}
            The device the plan's nodes run on.

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        NotImplementedYetError
            ``KY-E601``: ``"auto"``, ``"cuda"``, ``"wgpu"`` or ``"metal"`` (task 9.1).
        PlanError
            ``KY-E215``: any other device name.

        Examples
        --------
        >>> scan(store).on(device="cpu").estimate()["device"]
        'cpu'
        """
    def collect(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> Atlas:
        """Run the plan and keep its result as a new, materialized ``Atlas``.

        The plan's output (its layers, and its cell and gene tables with the computed
        columns) is written to a temporary kyto store in the system's temporary
        directory, and the returned ``Atlas`` scans that store (with this plan's default
        layer, if the output has it). The store is deleted when the last ``Atlas`` built
        on it is garbage-collected, so ``to_json()`` of such a plan names a store that
        lives only as long as the plan. To keep the result, use ``sink_zarr``. The GIL is
        released while the plan computes; Ctrl-C cancels the run cleanly.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
            of bytes: blocks shrink to fit, and a plan that cannot fit raises before
            anything is read. ``None`` runs with the default blocks.
        threads : int or None, default=None
            Threads that read and compute blocks; ``None`` for one per CPU core. The
            result does not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread whenever the run
            moved (cell rows, summed over the passes).

        Returns
        -------
        Atlas
            A new plan with one ``SCAN`` step (and ``WITH_DEFAULT_LAYER``) over the
            result; this one is unchanged. The data are copied into the temporary store.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        DeviceError
            ``KY-E502``: the system cannot start ``threads`` threads.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.
        LineageError
            ``KY-E303``: the plan recomputes a column (``.recompute()``) whose store
            does not hold the plan that computed it.
        KeyboardInterrupt
            Ctrl-C while the plan computes; the run has stopped.

        Examples
        --------
        >>> atlas = scan(store).filter(total_counts(layer="counts") >= 7)
        >>> print(atlas.collect(threads=1))
        SCAN "...atlas.kyto" (1 cells x 3 genes)
        """
    def sink_zarr(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        overwrite: bool = False,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> None:
        """Run the plan and write its output as a new kyto store at ``path``.

        Writes the plan's layers, and its cell and gene tables with the computed columns
        (derived columns keep their lineage; stale ones are dropped), streaming one cell
        block at a time. The store is written next to ``path`` and renamed into place
        when complete, so a failed or cancelled run leaves nothing (and an overwritten
        store is as it was). The GIL is released while the plan computes; Ctrl-C
        cancels the run cleanly.

        Parameters
        ----------
        path : str or os.PathLike[str]
            The new store.
        overwrite : bool, default=False
            Replace a kyto store already at ``path``, once the new one is complete.
            Anything else at ``path`` is never replaced, nor the store the plan reads.
        memory_limit : str or int or None, default=None
            The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
            of bytes: blocks shrink to fit, and a plan that cannot fit raises before
            anything is read. ``None`` runs with the default blocks.
        threads : int or None, default=None
            Threads that read and compute blocks; ``None`` for one per CPU core. The
            result does not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread whenever the run
            moved (cell rows, summed over the passes).

        Returns
        -------
        None
            The data are written to ``path``; this plan is unchanged.

        Raises
        ------
        StoreInputError
            ``KY-E118``: ``path`` exists and ``overwrite`` is false, it is not a kyto
            store, or it is (or lies inside) the store the plan reads.
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted. ``KY-E216``: the plan already ends in a sink.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        DeviceError
            ``KY-E502``: the system cannot start ``threads`` threads.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.
        LineageError
            ``KY-E303``: the plan recomputes a column (``.recompute()``) whose store
            does not hold the plan that computed it.
        KeyboardInterrupt
            Ctrl-C while the plan computes; the run has stopped and nothing is left.

        Examples
        --------
        >>> target = store.parent / "normalized.kyto"
        >>> scan(store).normalize_total(1e4, layer="counts", into="normalized").sink_zarr(target)
        >>> scan(target).layer("normalized").schema()["dtype"]
        'float32'
        >>> scan(store).log1p(layer="counts", into="log1p").sink_zarr(target, overwrite=True)
        >>> scan(target).layer("log1p").schema()["dtype"]
        'float32'
        """
    def sink_h5ad(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        overwrite: bool = False,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> None:
        """Run the plan and write its output as an ``.h5ad`` file at ``path``, for
        scanpy and anndata.

        The plan's default layer becomes ``X`` (no ``X`` without one) and every other
        layer goes under ``layers``, all as CSR; the cell and gene tables with the
        computed columns become ``obs`` and ``var``. The plan is streamed into a
        temporary kyto store next to ``path``, exported, and the store deleted; the
        file is renamed into place when complete, so a failed or cancelled run leaves
        nothing. The GIL is released while the plan computes; Ctrl-C cancels the run
        cleanly.

        Parameters
        ----------
        path : str or os.PathLike[str]
            The new file.
        overwrite : bool, default=False
            Replace a file already at ``path``, once the new one is complete. A
            directory is never replaced.
        memory_limit : str or int or None, default=None
            The most memory the run may use, such as ``"4GB"``, ``"512MiB"`` or a number
            of bytes: blocks shrink to fit, and a plan that cannot fit raises before
            anything is read. ``None`` runs with the default blocks.
        threads : int or None, default=None
            Threads that read and compute blocks; ``None`` for one per CPU core. The
            result does not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread whenever the run
            moved (cell rows, summed over the passes).

        Returns
        -------
        None
            The data are written to ``path``; this plan is unchanged.

        Raises
        ------
        StoreInputError
            ``KY-E123``: ``path`` exists and ``overwrite`` is false, or is a directory.
            ``KY-E124``: a name ``.h5ad`` cannot hold.
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted. ``KY-E216``: the plan already ends in a sink.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        DeviceError
            ``KY-E502``: the system cannot start ``threads`` threads.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.
        KeyboardInterrupt
            Ctrl-C while the plan computes; the run has stopped and nothing is left.

        Examples
        --------
        >>> target = store.parent / "pbmc.h5ad"
        >>> scan(store).with_default_layer("counts").sink_h5ad(target)
        >>> target.exists()
        True
        >>> scan(store).sink_h5ad(target)
        Traceback (most recent call last):
        ...
        kyto.errors.StoreInputError: KY-E123: cannot write '...pbmc.h5ad': ...
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
        >>> print(scan(store).filter(genes).explain())
        logical plan
          node 1 FILTER_GENES gene_column('feature_type') == 'Gene Expression' <- node 0
          node 0 SCAN "..." (2 cells x 3 genes)
        optimized plan
          output #1 reads [...]
          #1 FILTER_GENES ... <- #0 (node 1) [device=cpu passes=1 blocks=4096 cells]
          #0 SCAN ... (node 0) [device=cpu passes=1 blocks=4096 cells]
        >>> explained = scan(store).filter(genes).explain(format="json")
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
        >>> atlas = scan(store).filter(genes)
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

    def pull(
        self,
        feature: str | CellExpr[ExprDtype] | GeneExpr[ExprDtype],
        /,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.Series:
        """One feature of every cell (or gene) of the plan's output, as a Polars ``Series``.

        Runs the plan now, computing only what the feature reads (R1): no layer is read
        that the feature does not read, and nothing is written. The feature is resolved
        against this plan when called (R6), so a typo fails before anything runs.

        Parameters
        ----------
        feature : str or CellExpr or GeneExpr
            A feature name, as ``feature`` takes it (a cell column such as
            ``"total_counts"``, a gene ID of the default layer such as ``"ENSG01"``, or a
            full name such as ``"meta:cell_type"`` or ``"log1p:ENSG01"``), or an
            expression. To read a gene by its symbol, pass
            ``gene_expression("CD3E", by="symbol")`` or use ``pull_genes(..., by="symbol")``.
            A ``GeneExpr`` gives one value per gene instead of per cell.
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it; the
            result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does not
            depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan runs.

        Returns
        -------
        polars.Series
            A new series the caller owns (R2): one row per cell (``cell_ids()`` order) or,
            for a ``GeneExpr``, per gene (``gene_ids()`` order). It is named ``feature``
            for a name, and the expression's text for an expression. A sparse value that
            is absent is 0 (R9).
            Handed to Polars without a copy (the Arrow C stream interface).

        Raises
        ------
        ResolutionError
            ``KY-E210`` (with close matches), ``KY-E201``, ``KY-E204``, ``KY-E211`` and the
            other errors of ``resolve``.
        ExpressionError
            ``KY-E205``: ``feature`` is not a ``str``, ``CellExpr`` or ``GeneExpr``.
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store).with_default_layer("counts")
        >>> atlas.pull("ENSG01").to_list()
        [5, 0]
        >>> atlas.pull(gene_expression("CD3E", by="symbol") * 2).to_list()
        [10.0, 0.0]
        >>> atlas.pull(gene_column("symbol")).to_list()
        ['CD3E', 'CD4', 'MS4A1']
        """
    def select(
        self,
        *features: str,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
        **named: str | CellExpr[ExprDtype],
    ) -> pl.DataFrame:
        """Features of every cell of the plan's output, as a Polars ``DataFrame`` with
        ``cell_id`` first.

        Runs the plan now, computing only what the features read (R1); nothing is
        written. Each feature is resolved against this plan when called (R6).

        Parameters
        ----------
        *features : str
            Feature names, as ``pull`` takes them; each column is named as written.
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it; the
            result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does not
            depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan runs.
        **named : str or CellExpr
            More features, each a name or a cell expression, in a column named by its
            keyword (after the positional ones). ``memory_limit``, ``threads`` and
            ``progress`` are taken by the arguments above.

        Returns
        -------
        polars.DataFrame
            A new frame the caller owns (R2): ``cell_id`` (``String``), then one column per
            feature in the order given, one row per cell in ``cell_ids()`` order. Even one
            feature gives a frame (R3). A sparse value that is absent is 0 (R9).
            Handed to Polars without a copy (the Arrow C stream interface).

        Raises
        ------
        ResolutionError
            ``KY-E210`` (with close matches), ``KY-E201``, ``KY-E204``, ``KY-E211`` and the
            other errors of ``resolve``.
        ExpressionError
            ``KY-E205``: a positional feature that is not a ``str``, or a keyword one that
            is not a ``str`` or ``CellExpr``.
        PlanError
            ``KY-E215``: no feature, a name given twice or ``cell_id``, or a
            ``memory_limit``, ``threads`` or ``progress`` that is not accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store).with_default_layer("counts")
        >>> frame = atlas.select("ENSG01", n_counts=total_counts())
        >>> frame.columns
        ['cell_id', 'ENSG01', 'n_counts']
        >>> frame.rows()
        [('AAAC-1', 5, 6.0), ('AAAG-1', 0, 7.0)]
        """
    def pull_genes(
        self,
        genes: Sequence[str],
        /,
        *,
        layer: str | None = None,
        by: Literal["id", "symbol"] = "id",
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.DataFrame:
        """Genes' values in every cell of the plan's output, as a Polars ``DataFrame`` with
        ``cell_id`` first: one column per gene, named as given.

        The way to read genes by symbol: ``pull_genes(["CD3E"], by="symbol")``. Each
        symbol is resolved to its gene ID when called (R6). Runs the plan now, reading
        only ``layer`` (R1); nothing is written.

        Parameters
        ----------
        genes : list of str
            Gene IDs, or symbols with ``by="symbol"``.
        layer : str or None, default=None
            The layer to read; ``None`` for the plan's default layer.
        by : {"id", "symbol"}, default="id"
            Whether ``genes`` holds gene IDs or symbols. A symbol must name exactly one
            gene (``KY-E204`` lists the genes otherwise).
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it; the
            result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does not
            depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan runs.

        Returns
        -------
        polars.DataFrame
            A new frame the caller owns (R2): ``cell_id``, then one column per gene, one
            row per cell in ``cell_ids()`` order, in the layer's dtype. One gene still
            gives a frame (R3). An absent sparse value is 0 (R9).
            Handed to Polars without a copy (the Arrow C stream interface).

        Raises
        ------
        ResolutionError
            ``KY-E210`` (an unknown gene or symbol, with close matches), ``KY-E204`` (a
            symbol of several genes), ``KY-E203`` (a gene removed by an earlier step),
            ``KY-E211`` (no ``layer`` and no default layer), ``KY-E214`` (``by`` not
            ``"id"`` or ``"symbol"``).
        MissingMemberError
            ``KY-E109``: no layer named ``layer``.
        PlanError
            ``KY-E215``: ``genes`` is not a list of str, is empty or names a gene twice, or
            a ``memory_limit``, ``threads`` or ``progress`` that is not accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store)
        >>> atlas.pull_genes(["CD3E", "CD4"], layer="counts", by="symbol").rows()
        [('AAAC-1', 5, 0), ('AAAG-1', 0, 7)]
        """
    def features(self) -> pl.DataFrame:
        """The feature names this plan resolves, as a Polars ``DataFrame``; reads no data.

        One row per cell column (``meta:<column>``), per gene of each layer
        (``<layer>:<gene_id>``, with the gene's symbol) and per component of each cell
        embedding (``<embedding>:<component>``): the full names ``feature``, ``pull`` and
        ``select`` take (docs/feature_names.md). Genes removed by ``select_genes`` are
        left out; after a filter on data (``filter`` with a ``GeneExpr``) the genes are
        those that may remain, known only when the plan runs.

        Returns
        -------
        polars.DataFrame
            A new frame: ``feature`` (``String``), ``kind`` (``"cell column"``,
            ``"gene"`` or ``"embedding component"``) and ``symbol`` (``String``, null
            except for genes with a symbol). Nothing is read from the store.
            Handed to Polars without a copy (the Arrow C stream interface).

        Examples
        --------
        >>> frame = scan(store).features()
        >>> frame["feature"].to_list()
        ['meta:cell_id', 'counts:ENSG01', 'counts:ENSG02', 'counts:ENSG03']
        >>> frame.row(1)
        ('counts:ENSG01', 'gene', 'CD3E')
        """
    def cell_ids(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.Series:
        """The cell IDs of the plan's output, in order, as a Polars ``Series``.

        Reads the store's cell IDs in place when no step changes the cells; runs the plan
        (reading only what its filters read) otherwise.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core.
        progress : callable or None, default=None
            Called as ``progress(done, total)`` while the plan runs; never when nothing
            runs.

        Returns
        -------
        polars.Series
            A new ``String`` series named ``cell_id`` the caller owns (R2): the row order
            of every cell result (``select``, ``cells.to_polars()``).

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        CorruptStoreError
            ``KY-E127``: the store changed since it was scanned.

        Examples
        --------
        >>> scan(store).cell_ids().to_list()
        ['AAAC-1', 'AAAG-1']
        >>> scan(store).filter(total_counts(layer="counts") > 6).cell_ids().to_list()
        ['AAAG-1']
        """
    def gene_ids(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.Series:
        """The gene IDs of the plan's output, in order, as a Polars ``Series``.

        Given by the plan when it knows them (the store's genes, or a ``select_genes``
        selection; nothing is read); runs the plan when a filter on data removes genes.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core.
        progress : callable or None, default=None
            Called as ``progress(done, total)`` while the plan runs; never when nothing
            runs.

        Returns
        -------
        polars.Series
            A new ``String`` series named ``gene_id`` the caller owns (R2): the gene order
            of every gene result (``genes.to_polars()``, a layer's columns).

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.

        Examples
        --------
        >>> scan(store).gene_ids().to_list()
        ['ENSG01', 'ENSG02', 'ENSG03']
        >>> scan(store).select_genes(["ENSG03", "ENSG01"]).gene_ids().to_list()
        ['ENSG03', 'ENSG01']
        """
    def count_cells(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> int:
        """The number of cells of the plan's output.

        Given by the plan when it knows it (no cell filter or ``take_cells``; nothing is
        read); runs the plan, reading only what its filters read, otherwise.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core.
        progress : callable or None, default=None
            Called as ``progress(done, total)`` while the plan runs; never when nothing
            runs.

        Returns
        -------
        int
            The cell count.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.

        Examples
        --------
        >>> scan(store).count_cells()
        2
        >>> scan(store).filter(total_counts(layer="counts") > 6).count_cells()
        1
        """
    def count_genes(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> int:
        """The number of genes of the plan's output.

        Given by the plan when it knows it (no filter on gene data; nothing is read);
        runs the plan otherwise.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core.
        progress : callable or None, default=None
            Called as ``progress(done, total)`` while the plan runs; never when nothing
            runs.

        Returns
        -------
        int
            The gene count.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.

        Examples
        --------
        >>> scan(store).count_genes()
        3
        >>> scan(store).filter(gene_column("symbol") != "CD4").count_genes()
        2
        """
    def metadata(self) -> dict[str, Any]:
        """How the scanned store was made: its ``provenance.json`` (docs/format.md section
        9), as a ``dict``.

        Keys: ``kyto_version``, ``dependencies``, ``plan`` (the plan JSON that wrote the
        store, or ``None`` for an import), ``source``, ``seeds``, ``device`` and ``run``.
        It describes the store this plan scans, not the plan's own steps
        (``to_json()``). kyto stores no free-form metadata such as anndata's ``uns``.

        Returns
        -------
        dict
            A new dict the caller owns, read from the store (no matrix data).

        Raises
        ------
        StoreIOError
            ``KY-E110``: the store has no readable ``provenance.json``.

        Examples
        --------
        >>> scan(store).metadata()["source"]["kind"]
        'mtx'
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
    def to_polars(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.DataFrame:
        """Run the plan and return the table as a Polars ``DataFrame``.

        The table after every step of the plan: its rows are the cells (or genes) the
        plan keeps, in store order, and its columns those of ``names()``, computed
        columns included. A plan with steps is run into a temporary store first (as
        ``Atlas.collect``); a plain ``scan`` is read in place. The GIL is released
        while the plan runs; Ctrl-C cancels it cleanly.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        polars.DataFrame
            A copy the caller owns (R2): read from the store into new memory that is
            handed to Polars without a further copy (the Arrow C stream interface), so
            changing it never changes the store or a later read. String columns are
            ``Enum``, as stored.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> scan(store).genes.to_polars()["gene_id"].to_list()
        ['ENSG01', 'ENSG02', 'ENSG03']
        """
    def lazy(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.LazyFrame:
        """Run the plan and return the table as a Polars ``LazyFrame``.

        Runs the plan now, as ``to_polars`` (the table's schema, such as the
        categories of an ``Enum`` column, is known only then), and returns a lazy
        query over the result, to go on with in Polars.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        polars.LazyFrame
            A new lazy frame over a copy the caller owns (R2), handed over as
            ``to_polars``'s.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> scan(store).cells.lazy().select("cell_id").collect()["cell_id"].to_list()
        ['AAAC-1', 'AAAG-1']
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
            a gene ``filter``, the genes that may remain: which ones is known only
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
    def to_scipy(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> scipy.sparse.csr_matrix:
        """Run the plan and return the matrix as a SciPy CSR matrix.

        The layer after every step of the plan: the cells and genes the plan keeps, in
        store order. A plan with steps is run into a temporary store first (as
        ``Atlas.collect``); a plain ``scan`` is read in place. The GIL is released
        while the plan runs; Ctrl-C cancels it cleanly.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cells x genes in the layer's stored dtype (``float32`` or ``uint32``), with
            sorted indices and no stored zeros, ``int32`` index arrays (``int64`` when
            ``nnz`` or the genes need them); a copy the caller owns (R2): the arrays
            hold memory read from the store, handed to NumPy without a further copy
            (one index array is converted to SciPy's dtype).

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> matrix = scan(store).layer("counts").to_scipy()
        >>> matrix.shape, matrix.dtype, matrix.nnz
        ((2, 3), dtype('uint32'), 3)
        """
    def to_dense(
        self,
        *,
        max_bytes: int,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> npt.NDArray[np.float32 | np.uint32]:
        """Run the plan and return the matrix as a dense NumPy array, if it fits.

        The only way to get a dense matrix (R8): absent entries become 0 (R9). When the
        size is known from metadata (``estimate()["dense_bytes"]``) it is checked
        before the plan runs; otherwise (after a filter) once the plan has run, before
        the layer is read.

        Parameters
        ----------
        max_bytes : int
            The largest array, in bytes, the caller accepts. Required: there is no
            default, so densifying is always a stated choice. ``estimate()["dense_bytes"]``
            gives the size without reading the matrix.
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        numpy.ndarray
            Cells x genes, C order, in the layer's stored dtype (``float32`` or
            ``uint32``); a copy the caller owns (R2): densified once into new memory,
            handed to NumPy without a further copy.

        Raises
        ------
        MemoryLimitError
            ``KY-E402``: the array is larger than ``max_bytes``, or cannot be
            allocated. ``KY-E401``: the plan cannot run within ``memory_limit``.
        PlanError
            ``KY-E215``: a ``max_bytes`` that is not an int of 0 or more, or a
            ``memory_limit``, ``threads`` or ``progress`` that is not accepted.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> scan(store).layer("counts").to_dense(max_bytes=1024)
        array([[5, 0, 1],
               [0, 7, 0]], dtype=uint32)
        >>> scan(store).layer("counts").to_dense(max_bytes=8)
        Traceback (most recent call last):
        ...
        kyto.errors.MemoryLimitError: KY-E402: 'Matrix.to_dense' cannot allocate a dense 2 x 3 array of 24 bytes: max_bytes is 8...
        """

@final
class Embedding:
    """An embedding of a plan, a dense array with one row per cell or gene: read-only and
    lazy.

    Taken from ``Atlas.embedding(name)``. It reads nothing: ``names()``, ``schema()``
    and ``estimate()`` answer from metadata, and the terminals ``to_numpy()`` and
    ``to_torch(device=...)`` run the plan (task 7.3). An embedding is stored dense, so ``to_numpy()`` densifies nothing (R8). It has no method that changes it or the plan (R2).

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
    def to_numpy(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Run the plan and return the embedding as a NumPy array.

        The embedding after every step of the plan: one row per cell (or gene) the
        plan keeps, in store order. It is stored dense, so this densifies nothing (R8).

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        numpy.ndarray
            Rows x components, C order, in the stored dtype; a copy the caller owns
            (R2): read from the store into new memory, handed to NumPy without a
            further copy.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0)
        >>> array = atlas.embedding("X_pca").to_numpy()
        >>> array.shape, array.dtype
        ((2, 1), dtype('float32'))
        """
    def to_torch(
        self,
        *,
        device: builtins.str,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> Any:
        """Run the plan and return the embedding as a PyTorch tensor on ``device``.

        As ``to_numpy``, then ``torch.from_numpy(array).to(device)``: on the CPU the
        tensor shares the new array's memory (no further copy); any other device gets
        one more copy, from the host. A tensor made on the device without the host
        copy (DLPack) is task 9.5. Needs ``torch`` installed.

        Parameters
        ----------
        device : str
            A device PyTorch accepts, such as ``"cpu"``, ``"cuda"``, ``"cuda:1"`` or
            ``"mps"``. Required (R7: the device is always an argument).
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        torch.Tensor
            Rows x components in the stored dtype, on ``device``; a copy the caller
            owns (R2).

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.
        ModuleNotFoundError
            ``torch`` is not installed.

        Examples
        --------
        >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0)
        >>> atlas.embedding("X_pca").to_torch(device="cpu").shape  # doctest: +SKIP
        torch.Size([2, 1])
        """

@final
class Graph:
    """A graph of a plan, a sparse cells x cells matrix of edge weights: read-only and
    lazy.

    Taken from ``Atlas.graph(name)``. It reads nothing: ``names()``, ``schema()`` and
    ``estimate()`` answer from metadata, and the terminals ``to_scipy()`` and
    ``to_edges()`` run the plan (task 7.3). It has no
    method that changes it or the plan (R2).

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
    def to_scipy(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> scipy.sparse.csr_matrix:
        """Run the plan and return the graph as a SciPy CSR matrix.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        scipy.sparse.csr_matrix
            Cells x cells edge weights in the stored dtype; a copy the caller owns
            (R2), handed over as ``Matrix.to_scipy``'s. A stored edge may weigh 0
            (format.md section 7).

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0).neighbors("X_pca", n_neighbors=2)
        >>> graph = atlas.graph("distances").to_scipy()
        >>> graph.shape, graph.dtype
        ((2, 2), dtype('float32'))
        """
    def to_edges(
        self,
        *,
        memory_limit: str | int | None = None,
        threads: int | None = None,
        progress: Callable[[int, int], object] | None = None,
    ) -> pl.DataFrame:
        """Run the plan and return the edges as a Polars ``DataFrame``.

        Parameters
        ----------
        memory_limit : str or int or None, default=None
            The most memory running the plan may use, as ``Atlas.collect`` takes it;
            the result is held in memory besides.
        threads : int or None, default=None
            Threads that run the plan; ``None`` for one per CPU core. The result does
            not depend on it (R11).
        progress : callable or None, default=None
            Called as ``progress(done, total)`` on the calling thread while the plan
            runs; never for a plain ``scan``, which runs nothing.

        Returns
        -------
        polars.DataFrame
            One row per stored edge, row by row: ``source`` and ``target`` are cell
            positions (``UInt32``, rows of ``cells.to_polars()`` and of ``to_scipy()``)
            and ``weight`` the stored weight (``Float32`` or ``Float64``); a copy the
            caller owns (R2), handed over as ``Table.to_polars``'s.

        Raises
        ------
        PlanError
            ``KY-E215``: a ``memory_limit``, ``threads`` or ``progress`` that is not
            accepted.
        MemoryLimitError
            ``KY-E401``: the plan cannot run within ``memory_limit``.
        NotImplementedYetError
            ``KY-E601``: the plan has a step the executor does not run yet.

        Examples
        --------
        >>> atlas = scan(store).log1p(layer="counts", into="log1p").pca(layer="log1p", into="X_pca", n_components=1, seed=0).neighbors("X_pca", n_neighbors=2)
        >>> atlas.graph("distances").to_edges().columns
        ['source', 'target', 'weight']
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
        a gene ``filter`` changes its inputs: reading it then raises ``KY-E301``. A
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
        a gene ``filter`` changes its inputs: reading it then raises ``KY-E301``. A
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

def handoff_example() -> tuple[pl.DataFrame, int, int]:
    """Return a Rust-built frame handed to Polars and its `x` and `n` buffer addresses."""

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

def concat(
    sources: list[str],
    target: str,
    /,
    *,
    label: str,
    keys: list[str] | None,
    join: str,
    layer: str,
    on_duplicate_ids: str,
    overwrite: bool,
) -> None:
    """Combine `sources` into a new store at `target` (see `kyto.concat`)."""

def sink_h5ad(
    store: str,
    target: str,
    /,
    *,
    x_layer: str | None,
    overwrite: bool,
) -> None:
    """Export the store at `store` as an `.h5ad` file at `target` (private, task 2.7)."""
