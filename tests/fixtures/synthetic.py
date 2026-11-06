"""Seeded synthetic count data with the edge cases later tasks test.

What it does: :func:`make_synthetic` builds a small cells x genes count matrix from a
:class:`SyntheticSpec`, deterministically from its seed, with:

- cell types with marker genes (structure for HVG, PCA, clustering, DE),
- batches with different sequencing depth (``batch_key`` paths),
- the 13 human mitochondrial genes (``MT-`` prefix, task 6.1) and a few cells with a
  high mitochondrial fraction,
- repeated gene symbols with unique gene IDs (task 3.9),
- all-zero cells and all-zero genes, and one gene seen in a single cell.

Determinism across platforms: the generator owns its ``numpy.random.Generator`` (never
a shared one, memory/test_seed_isolation.md) and uses only uniform draws, integer
draws and basic float arithmetic (no ``exp``/``log``-based samplers such as
``poisson``, whose libm can round differently per OS). The resulting digest is checked
on every CI runner.

What it does not: model real biology faithfully. It is a test input, not a simulator.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fixtures.dataset import Dataset

MITOCHONDRIAL_SYMBOLS: tuple[str, ...] = (
    "MT-ND1",
    "MT-ND2",
    "MT-CO1",
    "MT-CO2",
    "MT-ATP8",
    "MT-ATP6",
    "MT-CO3",
    "MT-ND3",
    "MT-ND4L",
    "MT-ND4",
    "MT-ND5",
    "MT-ND6",
    "MT-CYB",
)

# Detection probability levels of ordinary genes (low to high expression).
_BASE_LEVELS: NDArray[np.float64] = np.array([1 / 64, 1 / 16, 1 / 4, 1 / 2])
_MAX_PROBABILITY = 0.95
_MAX_CONTINUE = 0.9
_MAX_EXTRA_COUNTS = 40


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of a synthetic dataset. The defaults are the ``synthetic`` fixture.

    Parameters
    ----------
    seed : int, default=0
        Seed of the generator's own ``numpy.random.Generator``.
    n_cells : int, default=500
        Number of cells.
    n_genes : int, default=400
        Number of genes, including the 13 mitochondrial genes.
    n_batches : int, default=3
        Number of batches; batch ``b`` has depth factor ``1 - b / (2 * n_batches)``.
    n_cell_types : int, default=4
        Number of cell types.
    n_markers : int, default=12
        Marker genes per cell type (detection boosted 4x in that type).
    n_empty_cells : int, default=4
        Cells with no counts at all.
    n_empty_genes : int, default=8
        Genes with no counts at all.
    n_duplicate_symbols : int, default=6
        Pairs of genes that share a symbol (distinct gene IDs).
    n_high_mito_cells : int, default=10
        Cells dominated by mitochondrial counts (as in dying cells).

    Raises
    ------
    ValueError
        If the counts cannot fit (for example more special genes than genes).
    """

    seed: int = 0
    n_cells: int = 500
    n_genes: int = 400
    n_batches: int = 3
    n_cell_types: int = 4
    n_markers: int = 12
    n_empty_cells: int = 4
    n_empty_genes: int = 8
    n_duplicate_symbols: int = 6
    n_high_mito_cells: int = 10

    def __post_init__(self) -> None:
        for field_name, value in (
            ("n_empty_cells", self.n_empty_cells),
            ("n_empty_genes", self.n_empty_genes),
            ("n_duplicate_symbols", self.n_duplicate_symbols),
            ("n_high_mito_cells", self.n_high_mito_cells),
            ("n_markers", self.n_markers),
        ):
            if value < 0:
                raise ValueError(
                    f"SyntheticSpec: {field_name}={value} must be zero or positive."
                )
        n_special_genes = (
            len(MITOCHONDRIAL_SYMBOLS)
            + self.n_empty_genes
            + 1  # the single-cell gene
            + 2 * self.n_duplicate_symbols
        )
        if self.n_genes < n_special_genes + self.n_markers:
            raise ValueError(
                f"SyntheticSpec: n_genes={self.n_genes} is too small; it needs at "
                f"least {n_special_genes + self.n_markers} for the mitochondrial, "
                "empty, single-cell, duplicate-symbol and marker genes. Raise n_genes "
                "or lower the other counts."
            )
        if self.n_cells < self.n_empty_cells + self.n_high_mito_cells + 1:
            raise ValueError(
                f"SyntheticSpec: n_cells={self.n_cells} is too small for "
                f"n_empty_cells={self.n_empty_cells} and "
                f"n_high_mito_cells={self.n_high_mito_cells}. Raise n_cells."
            )
        if self.n_batches < 1 or self.n_cell_types < 1:
            raise ValueError(
                "SyntheticSpec: n_batches and n_cell_types must be at least 1, got "
                f"{self.n_batches} and {self.n_cell_types}."
            )


def _string_array(values: list[str]) -> NDArray[np.str_]:
    return np.array(values, dtype=np.str_)


def make_synthetic(spec: SyntheticSpec | None = None) -> Dataset:
    """Build a synthetic dataset, identical for the same spec on every platform.

    Parameters
    ----------
    spec : SyntheticSpec or None, default=None
        Sizes and seed. ``None`` means ``SyntheticSpec()``, the ``synthetic`` fixture.

    Returns
    -------
    Dataset
        A new dataset named ``"synthetic"``. Cell columns: ``cell_id``, ``batch``
        (``batch_0`` ...), ``cell_type`` (``type_0`` ...). Gene columns: ``gene_id``
        (``SYNG00000001`` ...), ``symbol``.

    Examples
    --------
    >>> from fixtures.synthetic import SyntheticSpec, make_synthetic
    >>> data = make_synthetic(SyntheticSpec(seed=1, n_cells=50, n_genes=60))
    >>> data.n_cells, data.n_genes
    (50, 60)
    >>> data.digest() == make_synthetic(SyntheticSpec(seed=1, n_cells=50, n_genes=60)).digest()
    True
    """
    spec = SyntheticSpec() if spec is None else spec
    rng = np.random.default_rng(spec.seed)
    n_cells, n_genes = spec.n_cells, spec.n_genes

    cell_type = rng.integers(0, spec.n_cell_types, n_cells)
    batch = rng.integers(0, spec.n_batches, n_cells)

    # Gene roles, all disjoint: mitochondrial, empty, single-cell, duplicate pairs.
    order = rng.permutation(n_genes)
    n_mito = len(MITOCHONDRIAL_SYMBOLS)
    mito_genes = np.sort(order[:n_mito])
    empty_genes = order[n_mito : n_mito + spec.n_empty_genes]
    single_cell_gene = int(order[n_mito + spec.n_empty_genes])
    start = n_mito + spec.n_empty_genes + 1
    duplicate_genes = order[start : start + 2 * spec.n_duplicate_symbols]
    # Markers: any gene that is neither mitochondrial, empty nor the single-cell gene.
    marker_candidates = order[start:]

    # Detection probability: gene level x marker boost x batch depth x cell size.
    base = _BASE_LEVELS[rng.integers(0, _BASE_LEVELS.shape[0], n_genes)]
    base = base * rng.integers(4, 9, n_genes) / 8
    base[mito_genes] = 1 / 2
    boost = np.ones((spec.n_cell_types, n_genes))
    for type_index in range(spec.n_cell_types):
        markers = rng.choice(marker_candidates, spec.n_markers, replace=False)
        boost[type_index, markers] = 4
    batch_depth = 1 - np.arange(spec.n_batches) / (2 * spec.n_batches)
    cell_size = rng.integers(4, 13, n_cells) / 8
    probability = base[None, :] * boost[cell_type]
    probability *= (batch_depth[batch] * cell_size)[:, None]

    # Dying cells: mitochondrial genes everywhere, everything else suppressed.
    special_cells = rng.permutation(n_cells)
    high_mito_cells = special_cells[: spec.n_high_mito_cells]
    empty_cells = special_cells[
        spec.n_high_mito_cells : spec.n_high_mito_cells + spec.n_empty_cells
    ]
    probability[high_mito_cells] *= 1 / 4
    probability[np.ix_(high_mito_cells, mito_genes)] = _MAX_PROBABILITY
    probability = np.minimum(probability, _MAX_PROBABILITY)

    # Counts: detected with `probability`, then a geometric number of extra counts
    # (continue with probability min(p, 0.9), at most 40 extra).
    alive = rng.random((n_cells, n_genes)) < probability
    counts = alive.astype(np.int64)
    continue_probability = np.minimum(probability, _MAX_CONTINUE)
    for _ in range(_MAX_EXTRA_COUNTS):
        alive &= rng.random((n_cells, n_genes)) < continue_probability
        counts += alive

    counts[empty_cells] = 0
    counts[:, empty_genes] = 0
    counts[:, single_cell_gene] = 0
    non_empty = np.setdiff1d(np.arange(n_cells), empty_cells)
    counts[int(rng.choice(non_empty)), single_cell_gene] = 3

    symbols = [f"GENE{index:04d}" for index in range(n_genes)]
    for symbol, gene in zip(MITOCHONDRIAL_SYMBOLS, mito_genes.tolist(), strict=True):
        symbols[gene] = symbol
    for first, second in duplicate_genes.reshape(-1, 2).tolist():
        symbols[second] = symbols[first]

    rows, columns = np.nonzero(counts)
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n_cells), out=indptr[1:])
    return Dataset(
        name="synthetic",
        indptr=indptr,
        indices=columns.astype(np.int32),
        data=counts[rows, columns].astype(np.float32),
        cells={
            "cell_id": _string_array([f"cell_{index:04d}" for index in range(n_cells)]),
            "batch": _string_array([f"batch_{value}" for value in batch.tolist()]),
            "cell_type": _string_array(
                [f"type_{value}" for value in cell_type.tolist()]
            ),
        },
        genes={
            "gene_id": _string_array([f"SYNG{index:08d}" for index in range(n_genes)]),
            "symbol": _string_array(symbols),
        },
    )
