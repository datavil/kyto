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
    "igraph",
    "leidenalg",
    "numpy",
    "scanpy",
    "scikit-image",
    "scikit-learn",
    "scikit-misc",
    "scipy",
    "umap-learn",
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
# Leiden and Louvain resolutions: scanpy's default 1 and the clustering tutorial's.
_RESOLUTIONS: tuple[float, ...] = (0.02, 0.5, 1.0, 2.0)
# UMAP seeds whose trustworthiness gives umap-learn's range (task 6.9), and the
# neighbourhood sizes it is measured at (scikit-learn's default 5, and n_neighbors).
_UMAP_SEEDS = [0, 1, 2, 3, 4]
_TRUSTWORTHINESS_NEIGHBORS = [5, 15]
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


def _neighbors(name: FixtureName) -> AnnData:
    """The PCA reference with its exact kNN graph and UMAP connectivities."""
    import scanpy as sc

    adata = _pca(name)
    sc.pp.neighbors(
        adata,
        n_neighbors=_N_NEIGHBORS,
        n_pcs=_N_COMPS[name],
        method="umap",
        metric="euclidean",
        random_state=_SEED,
    )
    return adata


def neighbors(name: FixtureName) -> tuple[Parameters, Arrays]:
    """Exact kNN distances and UMAP connectivities on the PCA (tasks 6.6, 6.7)."""
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
    adata = _neighbors(name)
    return parameters, _csr_arrays(adata.obsp["distances"], "distances") | _csr_arrays(
        adata.obsp["connectivities"], "connectivities"
    )


def _rb_quality(
    adjacency: object, membership: NDArray[np.int64], resolution: float
) -> float:
    """Modularity with a resolution (leidenalg's ``RBConfigurationVertexPartition``
    on the directed graph scanpy builds) of ``membership``, normalised by the total
    edge weight: ``sum_c (W_c - resolution * K_out_c * K_in_c / m) / m``."""
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(adjacency, dtype=np.float64)
    rows, columns = matrix.nonzero()
    weights = np.asarray(matrix[rows, columns]).ravel()
    total = float(weights.sum())
    inside = float(weights[membership[rows] == membership[columns]].sum())
    n_clusters = int(membership.max()) + 1
    out_strength = np.bincount(
        membership, np.asarray(matrix.sum(axis=1)).ravel(), n_clusters
    )
    in_strength = np.bincount(
        membership, np.asarray(matrix.sum(axis=0)).ravel(), n_clusters
    )
    expected = float((out_strength * in_strength).sum()) / total
    return (inside - resolution * expected) / total


def leiden(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``tl.leiden`` with leidenalg on the connectivities, several resolutions (6.8)."""
    import scanpy as sc

    parameters: Parameters = {
        "function": "scanpy.tl.leiden",
        "input": "neighbors connectivities (as the neighbors reference)",
        "flavor": "leidenalg",
        "directed": True,
        "use_weights": True,
        "n_iterations": -1,
        "partition_type": "RBConfigurationVertexPartition",
        "random_state": _SEED,
        "resolutions": ", ".join(str(resolution) for resolution in _RESOLUTIONS),
    }
    adata = _neighbors(name)
    arrays: Arrays = {}
    for resolution in _RESOLUTIONS:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=_SEED,
            flavor="leidenalg",
            key_added="leiden",
        )
        labels = adata.obs["leiden"].astype(np.int64).to_numpy()
        quality = _rb_quality(adata.obsp["connectivities"], labels, resolution)
        arrays[f"{resolution}/labels"] = labels.astype(np.int32)
        arrays[f"{resolution}/quality"] = np.array([quality])
    return parameters, arrays


def louvain(name: FixtureName) -> tuple[Parameters, Arrays]:
    """Louvain as ``tl.louvain`` (flavor ``vtraag``) runs it, several resolutions (6.8).

    The ``louvain`` package has no Python 3.13 build; its successor leidenalg (same
    author, same moves and aggregation) runs Louvain with its documented loop. As
    scanpy's default: the directed graph, no weights,
    ``RBConfigurationVertexPartition``, levels until one moves nothing."""
    import leidenalg
    from scanpy._utils import get_igraph_from_adjacency
    from scipy.sparse import csr_matrix

    parameters: Parameters = {
        "function": "leidenalg move_nodes/aggregate_partition loop, as scanpy.tl.louvain",
        "input": "neighbors connectivities (as the neighbors reference)",
        "flavor": "vtraag",
        "directed": True,
        "use_weights": False,
        "partition_type": "RBConfigurationVertexPartition",
        "random_state": _SEED,
        "resolutions": ", ".join(str(resolution) for resolution in _RESOLUTIONS),
    }
    adata = _neighbors(name)
    adjacency = csr_matrix(adata.obsp["connectivities"])
    graph = get_igraph_from_adjacency(adjacency, directed=True)
    unweighted = csr_matrix(
        (np.ones(adjacency.nnz), adjacency.indices, adjacency.indptr),
        shape=adjacency.shape,
    )
    arrays: Arrays = {}
    for resolution in _RESOLUTIONS:
        partition = leidenalg.RBConfigurationVertexPartition(
            graph, resolution_parameter=resolution
        )
        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(_SEED)
        # leidenalg's documented Louvain loop (`refine_partition = False` crashes in
        # leidenalg 0.10.2): move nodes, aggregate, until a level moves nothing.
        aggregate = partition.aggregate_partition()
        while optimiser.move_nodes(aggregate) > 0:
            partition.from_coarse_partition(aggregate)
            aggregate = aggregate.aggregate_partition()
        partition.renumber_communities()
        labels = np.asarray(partition.membership, dtype=np.int64)
        quality = _rb_quality(unweighted, labels, resolution)
        arrays[f"{resolution}/labels"] = labels.astype(np.int32)
        arrays[f"{resolution}/quality"] = np.array([quality])
    return parameters, arrays


def umap(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``tl.umap`` on the neighbours above, and its trustworthiness (task 6.9).

    ``embedding`` is scanpy's ``X_umap`` at seed 0. ``trustworthiness/<k>`` holds
    scikit-learn's trustworthiness of the layout against ``X_pca`` (the space the
    neighbours were searched in) at ``k`` neighbours, one value per seed in
    ``_UMAP_SEEDS``: umap-learn's range, which kyto's layout must fall in.
    """
    import scanpy as sc
    from sklearn.manifold import trustworthiness

    parameters: Parameters = {
        "function": "scanpy.tl.umap",
        "input": "neighbors reference graph",
        "min_cells_per_gene": _MIN_CELLS_PER_GENE[name],
        "min_dist": 0.5,
        "spread": 1.0,
        "n_components": 2,
        "init_pos": "spectral",
        "random_state": _UMAP_SEEDS,
        "trustworthiness_n_neighbors": _TRUSTWORTHINESS_NEIGHBORS,
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
    space = np.asarray(adata.obsm["X_pca"])[:, : _N_COMPS[name]]
    arrays: Arrays = {}
    scores: dict[int, list[float]] = {k: [] for k in _TRUSTWORTHINESS_NEIGHBORS}
    for seed in _UMAP_SEEDS:
        sc.tl.umap(adata, random_state=seed)
        layout = np.asarray(adata.obsm["X_umap"])
        if seed == _UMAP_SEEDS[0]:
            arrays["embedding"] = layout
            arrays["a"] = np.asarray([adata.uns["umap"]["params"]["a"]])
            arrays["b"] = np.asarray([adata.uns["umap"]["params"]["b"]])
        for k in _TRUSTWORTHINESS_NEIGHBORS:
            scores[k].append(float(trustworthiness(space, layout, n_neighbors=k)))
    for k, values in scores.items():
        arrays[f"trustworthiness/{k}"] = np.asarray(values)
    return parameters, arrays


def _scrublet_arrays(uns: Mapping[str, object], prefix: str) -> Arrays:
    """One run's simulated doublets (parents, scores) and threshold (NaN for none)."""
    threshold = uns.get("threshold")
    return {
        f"{prefix}/doublet_parents": np.asarray(uns["doublet_parents"], dtype=np.int64),
        f"{prefix}/doublet_scores_sim": np.asarray(
            uns["doublet_scores_sim"], dtype=np.float64
        ),
        f"{prefix}/threshold": np.asarray(
            [np.nan if threshold is None else threshold], dtype=np.float64
        ),
    }


def scrublet(name: FixtureName) -> tuple[Parameters, Arrays]:
    """``pp.scrublet`` on the counts, alone and per batch where there are (6.12).

    Per run ``<label>`` (``all``, ``by_batch``): the cells' scores (NaN where Scrublet
    filtered the cell out), calls (False there) and whether each was scored, and per
    batch (``all`` for one) the simulated doublets' parents (positions among the
    batch's scored cells), their scores and the threshold.
    """
    import scanpy as sc

    parameters: Parameters = {
        "function": "scanpy.pp.scrublet",
        "input": "counts (no gene filter)",
        "random_state": _SEED,
        "other": "scanpy's defaults",
    }
    arrays: Arrays = {}
    batch_keys: list[str | None] = [None]
    if "batch" in _counts(name).obs.columns:
        batch_keys.append("batch")
    for batch_key in batch_keys:
        adata = _counts(name)
        sc.pp.scrublet(adata, batch_key=batch_key, random_state=_SEED, verbose=False)
        label = "all" if batch_key is None else f"by_{batch_key}"
        scores = adata.obs["doublet_score"].to_numpy(dtype=np.float64)
        scored = ~np.isnan(scores)
        predicted = adata.obs["predicted_doublet"].to_numpy()
        arrays[f"{label}/cells/doublet_score"] = scores
        arrays[f"{label}/cells/scored"] = scored
        arrays[f"{label}/cells/predicted_doublet"] = np.asarray(
            [
                bool(value) if ok else False
                for value, ok in zip(predicted, scored, strict=True)
            ]
        )
        uns = adata.uns["scrublet"]
        if batch_key is None:
            arrays |= _scrublet_arrays(uns, f"{label}/batches/all")
        else:
            for batch, batch_uns in uns["batches"].items():
                arrays |= _scrublet_arrays(batch_uns, f"{label}/batches/{batch}")
    return parameters, arrays


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
        "leiden": leiden,
        "louvain": louvain,
        "umap": umap,
        "scrublet": scrublet,
        "rank_genes": rank_genes,
    },
    "pbmc3k": {
        "qc": qc,
        "normalize_log1p": normalize_log1p,
        "hvg": hvg,
        "scale": scale,
        "pca": pca,
        "neighbors": neighbors,
        "leiden": leiden,
        "louvain": louvain,
        "umap": umap,
        "scrublet": scrublet,
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
