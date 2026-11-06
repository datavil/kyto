"""The in-memory form of a fixture dataset: a CSR count matrix plus cell and gene columns.

What it does: holds one dataset as plain numpy arrays (cells x genes CSR, string
columns per axis), checks its invariants and computes its content digest.

What it does not: read or write files, know how a dataset was made, or import
anndata (see :mod:`fixtures.interop`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from fixtures.hashing import array_digest


# eq=False: comparing datasets field by field would compare arrays; compare digests.
@dataclass(frozen=True, eq=False)
class Dataset:
    """A cells x genes count matrix in CSR form, with named columns per axis.

    Parameters
    ----------
    name : str
        Fixture name, for example ``"synthetic"`` or ``"pbmc3k"``.
    indptr : NDArray[np.int64]
        CSR row pointers, length ``n_cells + 1``.
    indices : NDArray[np.int32]
        CSR gene index of each stored value, sorted within each cell.
    data : NDArray[np.float32]
        Stored counts (integer-valued, all non-zero), aligned with ``indices``.
    cells : dict[str, NDArray[np.str_]]
        Cell columns, each of length ``n_cells``. Always holds ``"cell_id"`` (unique).
    genes : dict[str, NDArray[np.str_]]
        Gene columns, each of length ``n_genes``. Always holds ``"gene_id"`` (unique)
        and ``"symbol"`` (may repeat).

    Raises
    ------
    ValueError
        If the arrays break a CSR or column invariant (see :meth:`validate`).

    Notes
    -----
    Construction makes the given arrays read-only (they are not copied).
    """

    name: str
    indptr: NDArray[np.int64]
    indices: NDArray[np.int32]
    data: NDArray[np.float32]
    cells: Mapping[str, NDArray[np.str_]]
    genes: Mapping[str, NDArray[np.str_]]

    def __post_init__(self) -> None:
        self.validate()
        # Fixtures are cached and shared between tests: freeze every array so a test
        # cannot change what the next one sees.
        for array in (self.indptr, self.indices, self.data):
            array.setflags(write=False)
        object.__setattr__(self, "cells", MappingProxyType(dict(self.cells)))
        object.__setattr__(self, "genes", MappingProxyType(dict(self.genes)))
        for columns in (self.cells, self.genes):
            for values in columns.values():
                values.setflags(write=False)

    @property
    def n_cells(self) -> int:
        """Number of cells (rows)."""
        return int(self.indptr.shape[0] - 1)

    @property
    def n_genes(self) -> int:
        """Number of genes (columns)."""
        return int(self.genes["gene_id"].shape[0])

    def validate(self) -> None:
        """Check the CSR and column invariants.

        Raises
        ------
        ValueError
            If a dtype is wrong, ``indptr`` is not a valid row pointer, indices are
            out of range or unsorted within a cell, a stored value is zero, a column
            has the wrong length, or ``cell_id`` / ``gene_id`` repeat.
        """
        where = f"Dataset {self.name!r}"
        expected = {
            "indptr": (self.indptr, np.int64),
            "indices": (self.indices, np.int32),
            "data": (self.data, np.float32),
        }
        for label, (array, dtype) in expected.items():
            if array.dtype != dtype or array.ndim != 1:
                raise ValueError(
                    f"{where}: {label} must be a 1-d {np.dtype(dtype)} array, "
                    f"got {array.ndim}-d {array.dtype}."
                )
        for axis, columns, key in (
            ("cells", self.cells, "cell_id"),
            ("genes", self.genes, "gene_id"),
        ):
            if key not in columns:
                raise ValueError(f"{where}: {axis} needs a {key!r} column.")
        if "symbol" not in self.genes:
            raise ValueError(f"{where}: genes needs a 'symbol' column.")
        n_cells = int(self.cells["cell_id"].shape[0])
        n_genes = self.n_genes
        if self.indptr.shape[0] != n_cells + 1 or self.indptr[0] != 0:
            raise ValueError(
                f"{where}: indptr must have n_cells + 1 = {n_cells + 1} entries "
                f"starting at 0, got {self.indptr.shape[0]}."
            )
        if np.any(np.diff(self.indptr) < 0) or self.indptr[-1] != self.data.shape[0]:
            raise ValueError(
                f"{where}: indptr must be non-decreasing and end at nnz = "
                f"{self.data.shape[0]}."
            )
        if self.indices.shape != self.data.shape:
            raise ValueError(f"{where}: indices and data must have the same length.")
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= n_genes
        ):
            raise ValueError(f"{where}: gene indices must lie in [0, {n_genes}).")
        # Within a cell, indices must increase strictly; across a cell boundary they
        # may drop. Mark the steps that start a new cell and ignore those.
        steps = np.diff(self.indices.astype(np.int64))
        row_start = np.zeros(self.indices.shape[0], dtype=bool)
        row_start[self.indptr[1:-1][self.indptr[1:-1] < self.indices.shape[0]]] = True
        if np.any((steps <= 0) & ~row_start[1:]):
            raise ValueError(
                f"{where}: gene indices must be strictly increasing within each cell."
            )
        if not np.all(np.isfinite(self.data)) or np.any(self.data <= 0):
            raise ValueError(
                f"{where}: stored values must be finite and positive (non-zero counts)."
            )
        if np.any(self.data != np.round(self.data)):
            raise ValueError(f"{where}: stored values must be integer counts.")
        for axis, columns, length in (
            ("cells", self.cells, n_cells),
            ("genes", self.genes, n_genes),
        ):
            for column, values in columns.items():
                if values.dtype.kind != "U" or values.shape != (length,):
                    raise ValueError(
                        f"{where}: {axis} column {column!r} must be a 1-d string "
                        f"array of length {length}, got {values.dtype} {values.shape}."
                    )
        for axis, values in (
            ("cell_id", self.cells["cell_id"]),
            ("gene_id", self.genes["gene_id"]),
        ):
            if np.unique(values).shape[0] != values.shape[0]:
                raise ValueError(f"{where}: {axis} values must be unique.")

    def digest(self) -> str:
        """Return the platform-independent SHA-256 digest of the whole dataset.

        Covers the matrix (``indptr``, ``indices``, ``data`` values), the shape and
        every cell and gene column.

        Returns
        -------
        str
            64 lowercase hex characters.
        """
        arrays: dict[str, NDArray[np.generic]] = {
            "matrix/indptr": self.indptr,
            "matrix/indices": self.indices,
            "matrix/data": self.data,
            "matrix/shape": np.array([self.n_cells, self.n_genes], dtype=np.int64),
        }
        arrays.update({f"cells/{key}": value for key, value in self.cells.items()})
        arrays.update({f"genes/{key}": value for key, value in self.genes.items()})
        return array_digest(arrays)

    def to_dense(self) -> NDArray[np.float32]:
        """Return the counts as a new dense ``(n_cells, n_genes)`` float32 array.

        Returns
        -------
        NDArray[np.float32]
            A copy; only for small fixtures.
        """
        dense = np.zeros((self.n_cells, self.n_genes), dtype=np.float32)
        rows = np.repeat(np.arange(self.n_cells), np.diff(self.indptr))
        dense[rows, self.indices] = self.data
        return dense
