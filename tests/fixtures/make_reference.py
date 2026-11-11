"""Generate the scanpy reference outputs that parity tests compare kyto against.

What it does: runs the standard scanpy steps planned in tasks 6.1 to 6.10 on each
fixture and writes one compressed ``.npz`` per (fixture, step) to
``fixtures/reference/<fixture>/<step>.npz``, plus ``manifest.json`` with the scanpy,
anndata and numpy versions, every step's parameters, the fixture digest the outputs
came from, and a platform-independent digest per array (:mod:`fixtures.hashing`).

Large outputs are summarised on ``pbmc3k`` to keep the repository small: the
normalised matrices are stored as per-cell and per-gene sums, and the scaled matrix
as its gene means and standard deviations. ``synthetic`` stores them in full.

Run from the repository root after changing a fixture or upgrading scanpy::

    uv run --directory tests python -m fixtures.make_reference

What it does not: run kyto, or decide tolerances (task 10.2 owns the parity table).
"""

# scanpy, pandas and scipy ship without complete type information; strict pyright
# cannot see their types, so the unknown-type checks are relaxed in this file only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
from collections.abc import Callable, Mapping
from importlib.metadata import version
from typing import TypedDict

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray

from fixtures.interop import to_anndata
from fixtures.reference import MANIFEST_PATH, REFERENCE_DIR, array_entry
from fixtures.registry import EXPECTED_DIGESTS, FIXTURE_NAMES, FixtureName, load_fixture

VERSIONED_PACKAGES: tuple[str, ...] = (
    "anndata",
    "numpy",
    "scanpy",
    "scikit-learn",
    "scikit-misc",
    "scipy",
)

Arrays = dict[str, NDArray[np.generic]]
Parameters = dict[str, str | int | float | bool | None | list[int]]


class ReferenceEntry(TypedDict):
    """Manifest entry of one (fixture, step) reference file."""

    file: str
    fixture_digest: str
    parameters: Parameters
    arrays: dict[str, dict[str, object]]


class Manifest(TypedDict):
    """Top level of ``manifest.json``."""

    generator: str
    versions: dict[str, str]
    references: dict[str, ReferenceEntry]


# Per-fixture sizes: the small synthetic data needs smaller n_top_genes and n_comps.
_N_TOP_GENES: dict[FixtureName, int] = {"synthetic": 100, "pbmc3k": 2000}
_N_COMPS: dict[FixtureName, int] = {"synthetic": 20, "pbmc3k": 50}
_PERCENT_TOP = [50, 100, 200]
_TARGET_SUM = 1e4
_N_NEIGHBORS = 15
_SCALE_MAX_VALUE = 10.0
_SEED = 0
# Genes seen in fewer cells are dropped before HVG and later steps on pbmc3k, as in
# the scanpy tutorial. Without it scanpy's cell_ranger flavour raises ("Bin edges must
# be unique") on the ~19k all-zero genes. synthetic keeps its empty genes (edge case).
_MIN_CELLS_PER_GENE: dict[FixtureName, int] = {"synthetic": 0, "pbmc3k": 3}


def _frame_arrays(frame: object, prefix: str) -> Arrays:
    """Return every column of a pandas DataFrame as ``<prefix>/<column>`` arrays."""
    import pandas as pd

    assert isinstance(frame, pd.DataFrame)
    arrays: Arrays = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype.kind == "O":
            values = values.astype(np.str_)
        arrays[f"{prefix}/{column}"] = values
    return arrays


def _csr_arrays(matrix: object, prefix: str) -> Arrays:
    """Return a scipy CSR matrix as ``<prefix>/{data,indices,indptr,shape}`` arrays."""
    from scipy.sparse import csr_matrix

    csr = csr_matrix(matrix)
    csr.sort_indices()
    return {
        f"{prefix}/data": np.asarray(csr.data),
        f"{prefix}/indices": np.asarray(csr.indices, dtype=np.int64),
        f"{prefix}/indptr": np.asarray(csr.indptr, dtype=np.int64),
        f"{prefix}/shape": np.asarray(csr.shape, dtype=np.int64),
    }


def _sum_arrays(matrix: object, prefix: str) -> Arrays:
    """Return float64 per-cell sums, per-gene sums and the number of stored values."""
    from scipy.sparse import csr_matrix

    csr = csr_matrix(matrix, dtype=np.float64)
    return {
        f"{prefix}/cell_sum": np.asarray(csr.sum(axis=1)).ravel(),
        f"{prefix}/gene_sum": np.asarray(csr.sum(axis=0)).ravel(),
        f"{prefix}/nnz": np.asarray([csr.nnz], dtype=np.int64),
    }


def _counts(name: FixtureName) -> AnnData:
    adata = to_anndata(load_fixture(name))
    adata.var["mt"] = adata.var["symbol"].str.startswith("MT-").to_numpy()
    return adata


def _filtered_counts(name: FixtureName) -> AnnData:
    """Counts without the genes seen in fewer than ``_MIN_CELLS_PER_GENE`` cells."""
    import scanpy as sc

    adata = _counts(name)
    if _MIN_CELLS_PER_GENE[name] > 0:
        sc.pp.filter_genes(adata, min_cells=_MIN_CELLS_PER_GENE[name])
    return adata


def _log_normalized(name: FixtureName) -> AnnData:
    """Filtered counts normalised to ``_TARGET_SUM`` per cell, then log1p."""
    import scanpy as sc

    adata = _filtered_counts(name)
    sc.pp.normalize_total(adata, target_sum=_TARGET_SUM)
    sc.pp.log1p(adata)
    return adata


def _hvg_genes(name: FixtureName) -> AnnData:
    """The log-normalised data restricted to the ``seurat`` HVGs (PCA input)."""
    import scanpy as sc

    adata = _log_normalized(name)
    frame = sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=_N_TOP_GENES[name], inplace=False
    )
    assert frame is not None
    return adata[:, frame["highly_variable"].to_numpy()].copy()


def _pca(name: FixtureName) -> AnnData:
    import scanpy as sc

    adata = _hvg_genes(name)
    sc.pp.pca(
        adata,
        n_comps=_N_COMPS[name],
        zero_center=True,
        svd_solver="arpack",
        random_state=_SEED,
    )
    return adata


def qc(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``calculate_qc_metrics`` with the ``MT-`` gene set (task 6.1)."""
    import scanpy as sc

    parameters: Parameters = {
        "function": "scanpy.pp.calculate_qc_metrics",
        "qc_vars": "mt = symbol starts with 'MT-'",
        "percent_top": _PERCENT_TOP,
        "log1p": True,
    }
    result = sc.pp.calculate_qc_metrics(
        _counts(name),
        qc_vars=["mt"],
        percent_top=_PERCENT_TOP,
        log1p=True,
        inplace=False,
    )
    assert result is not None
    cells, genes = result
    return parameters, _frame_arrays(cells, "cells") | _frame_arrays(genes, "genes")


def normalize_log1p(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``normalize_total`` (fixed target and median target), then ``log1p`` (6.2)."""
    import scanpy as sc
    from scipy.sparse import csr_matrix

    full = name == "synthetic"
    parameters: Parameters = {
        "function": "scanpy.pp.normalize_total then scanpy.pp.log1p",
        "target_sum": _TARGET_SUM,
        "median": "target_sum=None",
        "stored": "full CSR" if full else "sums only",
    }
    arrays: Arrays = {}
    for label, target_sum in (("target", _TARGET_SUM), ("median", None)):
        adata = _counts(name)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # A copy: `log1p` below changes `adata.X` in place.
        normalized = csr_matrix(adata.X, copy=True)
        sc.pp.log1p(adata)
        for step, matrix in (("normalized", normalized), ("log1p", adata.X)):
            prefix = f"{label}/{step}"
            arrays |= _csr_arrays(matrix, prefix) if full else {}
            arrays |= _sum_arrays(matrix, prefix)
    return parameters, arrays


def hvg(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``highly_variable_genes`` for each flavour, with ``batch_key`` on batches (6.3)."""
    import scanpy as sc

    n_top_genes = _N_TOP_GENES[name]
    parameters: Parameters = {
        "function": "scanpy.pp.highly_variable_genes",
        "n_top_genes": n_top_genes,
        "input": "genes in >= min_cells_per_gene cells; seurat and cell_ranger: log1p "
        "of target_sum normalisation; seurat_v3: counts",
        "min_cells_per_gene": _MIN_CELLS_PER_GENE[name],
    }
    log_normalized = _log_normalized(name)
    counts = _filtered_counts(name)
    batch_keys: list[str | None] = [None]
    if "batch" in counts.obs.columns:
        batch_keys.append("batch")
    arrays: Arrays = {"genes/gene_id": counts.var_names.to_numpy().astype(np.str_)}
    for flavor in ("seurat", "cell_ranger", "seurat_v3"):
        for batch_key in batch_keys:
            source = counts if flavor == "seurat_v3" else log_normalized
            frame = sc.pp.highly_variable_genes(
                source,
                flavor=flavor,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                inplace=False,
            )
            label = flavor if batch_key is None else f"{flavor}_by_{batch_key}"
            arrays |= _frame_arrays(frame, label)
    return parameters, arrays


def scale(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``scale`` with clipping on the HVG subset (task 6.4)."""
    import scanpy as sc

    full = name == "synthetic"
    parameters: Parameters = {
        "function": "scanpy.pp.scale",
        "input": "log1p of target_sum normalisation, seurat HVGs",
        "min_cells_per_gene": _MIN_CELLS_PER_GENE[name],
        "max_value": _SCALE_MAX_VALUE,
        "zero_center": True,
        "stored": "full dense" if full else "gene mean and std only",
    }
    adata = _hvg_genes(name)
    sc.pp.scale(adata, max_value=_SCALE_MAX_VALUE, zero_center=True)
    arrays: Arrays = {
        "genes/gene_id": adata.var_names.to_numpy().astype(np.str_),
        "genes/mean": adata.var["mean"].to_numpy(),
        "genes/std": adata.var["std"].to_numpy(),
    }
    if full:
        arrays["scaled"] = np.asarray(adata.X)
    return parameters, arrays


def pca(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``pca`` with implicit centering on sparse data (task 6.5)."""
    parameters: Parameters = {
        "function": "scanpy.pp.pca",
        "input": "log1p of target_sum normalisation, seurat HVGs, not scaled",
        "min_cells_per_gene": _MIN_CELLS_PER_GENE[name],
        "n_comps": _N_COMPS[name],
        "zero_center": True,
        "svd_solver": "arpack",
        "random_state": _SEED,
    }
    adata = _pca(name)
    return parameters, {
        "genes/gene_id": adata.var_names.to_numpy().astype(np.str_),
        "embedding": np.asarray(adata.obsm["X_pca"]),
        "components": np.asarray(adata.varm["PCs"]),
        "variance": np.asarray(adata.uns["pca"]["variance"]),
        "variance_ratio": np.asarray(adata.uns["pca"]["variance_ratio"]),
    }


def neighbors(name: FixtureName) -> tuple[Parameters, Arrays]:
    """Exact kNN distances and UMAP connectivities on the PCA (tasks 6.6, 6.7)."""
    import scanpy as sc

    parameters: Parameters = {
        "function": "scanpy.pp.neighbors",
        "input": "pca reference embedding",
        "min_cells_per_gene": _MIN_CELLS_PER_GENE[name],
        "n_neighbors": _N_NEIGHBORS,
        "n_pcs": _N_COMPS[name],
        "method": "umap",
        "metric": "euclidean",
        "transformer": "exact (n_obs < 4096)",
        "random_state": _SEED,
    }
    adata = _pca(name)
    sc.pp.neighbors(
        adata,
        n_neighbors=_N_NEIGHBORS,
        n_pcs=_N_COMPS[name],
        method="umap",
        metric="euclidean",
        random_state=_SEED,
    )
    return parameters, _csr_arrays(adata.obsp["distances"], "distances") | _csr_arrays(
        adata.obsp["connectivities"], "connectivities"
    )


def rank_genes(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``rank_genes_groups`` by cell type, t-test and Wilcoxon (task 6.10)."""
    import scanpy as sc

    parameters: Parameters = {
        "function": "scanpy.tl.rank_genes_groups",
        "input": "log1p of target_sum normalisation, all genes",
        "groupby": "cell_type",
        "reference": "rest",
        "n_genes": "all",
        "corr_method": "benjamini-hochberg",
    }
    arrays: Arrays = {}
    for method in ("t-test", "wilcoxon"):
        adata = _log_normalized(name)
        sc.tl.rank_genes_groups(
            adata, groupby="cell_type", method=method, n_genes=adata.n_vars
        )
        frame = sc.get.rank_genes_groups_df(adata, group=None)
        arrays |= _frame_arrays(frame, method)
    return parameters, arrays


Step = Callable[[FixtureName], tuple[Parameters, Arrays]]

# Steps per fixture. pbmc3k has no cell groups, so it has no rank_genes reference.
STEPS: dict[FixtureName, dict[str, Step]] = {
    "synthetic": {
        "qc": qc,
        "normalize_log1p": normalize_log1p,
        "hvg": hvg,
        "scale": scale,
        "pca": pca,
        "neighbors": neighbors,
        "rank_genes": rank_genes,
    },
    "pbmc3k": {
        "qc": qc,
        "normalize_log1p": normalize_log1p,
        "hvg": hvg,
        "scale": scale,
        "pca": pca,
        "neighbors": neighbors,
    },
}


def array_entries(
    arrays: Mapping[str, NDArray[np.generic]],
) -> dict[str, dict[str, object]]:
    """Return the manifest record (dtype, shape, digest) of every array."""
    return {key: array_entry(key, value) for key, value in sorted(arrays.items())}


def installed_versions() -> dict[str, str]:
    """Return the installed version of every package that shapes the references."""
    return {package: version(package) for package in VERSIONED_PACKAGES}


def main() -> None:
    """Regenerate every reference file and the manifest."""
    references: dict[str, ReferenceEntry] = {}
    for name in FIXTURE_NAMES:
        digest = load_fixture(name).digest()
        if digest != EXPECTED_DIGESTS[name]:
            raise SystemExit(
                f"make_reference: fixture {name!r} has digest {digest}, registry "
                f"expects {EXPECTED_DIGESTS[name]}. Fix the fixture or update "
                "EXPECTED_DIGESTS on purpose first."
            )
        for step, function in STEPS[name].items():
            parameters, arrays = function(name)
            relative = f"{name}/{step}.npz"
            path = REFERENCE_DIR / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Array names contain "/", so none can collide with `allow_pickle`.
            np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
            references[f"{name}/{step}"] = {
                "file": relative,
                "fixture_digest": digest,
                "parameters": parameters,
                "arrays": array_entries(arrays),
            }
            print(f"{relative}: {path.stat().st_size / 1024:.0f} KiB")
    manifest: Manifest = {
        "generator": "uv run --directory tests python -m fixtures.make_reference",
        "versions": installed_versions(),
        "references": references,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
