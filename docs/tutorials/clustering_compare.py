"""Compare kyto's run of the clustering tutorial with scanpy's (task 10.7).

``compare`` reads what ``clustering.py`` wrote (``kyto.h5ad``, ``run.json``) and
scanpy's processed ``AnnData`` from ``clustering_scanpy.py``, and measures what the
plan's Done-when asks: equal QC columns and highly variable genes, PCA variances, the
Leiden adjusted Rand index per clustering and UMAP trustworthiness, plus the doublet
calls. As a script it prints the table:

```bash
uv run python docs/tutorials/clustering_compare.py <kyto output folder> <scanpy.h5ad> \
    <s1d1.h5>
```
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from numpy.typing import NDArray
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

# anndata, scanpy and scikit-learn ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

# Cell QC columns both tools write; counts compare exactly, percentages to float32.
QC_COUNTS = (
    "n_genes_by_counts",
    "total_counts",
    "total_counts_mt",
    "total_counts_ribo",
    "total_counts_hb",
)
QC_PERCENTS = ("pct_counts_mt", "pct_counts_ribo", "pct_counts_hb")
LEIDEN = ("leiden", "leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00")
# Trustworthiness needs all pairwise distances: score a seeded sample of cells.
TRUSTWORTHINESS_CELLS = 3000
TRUSTWORTHINESS_NEIGHBORS = (5, 15)
# The clustering whose Wilcoxon ranking the tutorial reads, and the genes compared.
MARKER_GROUPS = "leiden_res_0.50"
MARKER_TOP = 10


class Comparison(NamedTuple):
    """What ``compare`` measured.

    Attributes
    ----------
    same_cells : bool
        Both kept the same cells, in the same order, with the same sample labels.
    qc_counts_equal : dict of str to bool
        Per count column: equal on every cell.
    qc_percent_error : dict of str to float
        Per percentage column: the largest difference relative to scanpy's value
        (kyto computes in ``float64``, scanpy in ``float32``).
    same_genes : bool
        Both kept the same genes after the filters (by gene ID).
    hvg_kyto, hvg_scanpy, hvg_common : int
        Highly variable genes of each, and of both.
    pca_variance_error : float
        Largest relative difference of the 50 components' variances (kyto's from the
        variance of its ``X_pca`` columns).
    leiden : dict of str to tuple of (int, int, float)
        Per clustering: kyto's clusters, scanpy's clusters, adjusted Rand index.
    trustworthiness : dict of int to tuple of (float, float)
        Per neighbour count: kyto's and scanpy's UMAP against their own ``X_pca``.
    doublets : tuple of (int, int, int, float)
        Doublets called by kyto, by scanpy and by both, and the correlation of the
        two doublet scores over the cells. (The share of cells called alike says
        little: calling no doublet at all would score 98.8%.)
    marker_overlap : dict of str to int
        Per kyto ``leiden_res_0.50`` cluster: how many of its first 10 Wilcoxon genes
        are among the first 10 of the scanpy cluster sharing most cells with it
        (Jaccard), scanpy's ranking restricted to kyto's highly variable genes.
    """

    same_cells: bool
    qc_counts_equal: dict[str, bool]
    qc_percent_error: dict[str, float]
    same_genes: bool
    hvg_kyto: int
    hvg_scanpy: int
    hvg_common: int
    pca_variance_error: float
    leiden: dict[str, tuple[int, int, float]]
    trustworthiness: dict[int, tuple[float, float]]
    doublets: tuple[int, int, int, float]
    marker_overlap: dict[str, int]


def _values(data: ad.AnnData, column: str) -> NDArray[np.float64]:
    return np.asarray(data.obs[column], dtype=np.float64)


def compare(
    kyto: ad.AnnData,
    scanpy: ad.AnnData,
    gene_ids: dict[str, str],
    kept_genes: list[str],
    ranked: pl.DataFrame,
) -> Comparison:
    """Measure kyto's tutorial result against scanpy's.

    Parameters
    ----------
    kyto : anndata.AnnData
        ``kyto.h5ad`` from ``clustering.py`` (highly variable genes, ``var["gene_id"]``).
    scanpy : anndata.AnnData
        scanpy's processed data from ``clustering_scanpy.py`` (every kept gene).
    gene_ids : dict of str to str
        scanpy's gene names (symbols made unique by ``read_10x_h5``) to gene IDs.
    kept_genes : list of str
        The gene IDs kyto kept after the filters (``run.json``'s ``kept_genes``).
    ranked : polars.DataFrame
        kyto's Wilcoxon ranking (``ranked.parquet``).

    Returns
    -------
    Comparison
        The measurements; see :class:`Comparison`.
    """
    same_cells = bool(
        kyto.obs_names.equals(scanpy.obs_names)
        and (kyto.obs["sample"].astype(str) == scanpy.obs["sample"].astype(str)).all()
    )
    counts_equal = {
        column: bool(np.array_equal(_values(kyto, column), _values(scanpy, column)))
        for column in QC_COUNTS
    }
    percent_error: dict[str, float] = {}
    for column in QC_PERCENTS:
        reference = _values(scanpy, column)
        difference = np.abs(_values(kyto, column) - reference)
        percent_error[column] = float(
            np.max(difference / np.maximum(np.abs(reference), 1e-30), initial=0.0)
        )

    scanpy_genes = {gene_ids[name] for name in scanpy.var_names}
    kyto_ids = set(np.asarray(kyto.var["gene_id"], dtype=str).tolist())
    kyto_flags = np.asarray(kyto.var["highly_variable"], dtype=bool)
    hvg_kyto = set(np.asarray(kyto.var["gene_id"], dtype=str)[kyto_flags].tolist())
    scanpy_flags = np.asarray(scanpy.var["highly_variable"], dtype=bool)
    hvg_scanpy = {
        gene_ids[name]
        for name in np.asarray(scanpy.var_names, dtype=str)[scanpy_flags].tolist()
    }
    same_genes = set(kept_genes) == scanpy_genes and len(kept_genes) == len(
        scanpy_genes
    )

    kyto_variance = np.var(np.asarray(kyto.obsm["X_pca"], np.float64), axis=0, ddof=1)
    scanpy_variance = np.asarray(scanpy.uns["pca"]["variance"], np.float64)
    pca_error = float(np.max(np.abs(kyto_variance - scanpy_variance) / scanpy_variance))

    leiden = {
        column: (
            len(set(kyto.obs[column].tolist())),
            len(set(scanpy.obs[column].tolist())),
            float(adjusted_rand_score(scanpy.obs[column], kyto.obs[column])),
        )
        for column in LEIDEN
    }

    sample = np.random.default_rng(0).choice(
        kyto.n_obs, min(TRUSTWORTHINESS_CELLS, kyto.n_obs), replace=False
    )
    trust: dict[int, tuple[float, float]] = {}
    for neighbors in TRUSTWORTHINESS_NEIGHBORS:
        pair = tuple(
            float(
                trustworthiness(
                    np.asarray(data.obsm["X_pca"])[sample],
                    np.asarray(data.obsm["X_umap"])[sample],
                    n_neighbors=neighbors,
                )
            )
            for data in (kyto, scanpy)
        )
        trust[neighbors] = (pair[0], pair[1])

    kyto_calls = np.asarray(kyto.obs["predicted_doublet"], dtype=bool)
    scanpy_calls = np.asarray(scanpy.obs["predicted_doublet"], dtype=bool)
    doublets = (
        int(kyto_calls.sum()),
        int(scanpy_calls.sum()),
        int((kyto_calls & scanpy_calls).sum()),
        float(
            np.corrcoef(
                _values(kyto, "doublet_score"), _values(scanpy, "doublet_score")
            )[0, 1]
        ),
    )

    kyto_labels = np.asarray(kyto.obs[MARKER_GROUPS], dtype=str)
    scanpy_labels = np.asarray(scanpy.obs[MARKER_GROUPS], dtype=str)
    marker_overlap: dict[str, int] = {}
    for group in ranked["group"].unique(maintain_order=True).to_list():
        members = kyto_labels == group
        jaccard = {
            label: float(
                np.sum(members & (scanpy_labels == label))
                / np.sum(members | (scanpy_labels == label))
            )
            for label in sorted(set(scanpy_labels.tolist()))
        }
        match = max(jaccard, key=lambda label: jaccard[label])
        scanpy_names = sc.get.rank_genes_groups_df(scanpy, group=match)["names"]
        scanpy_top = [
            gene_ids[name] for name in scanpy_names if gene_ids[name] in kyto_ids
        ][:MARKER_TOP]
        kyto_top = ranked.filter(pl.col("group") == group)["gene_id"].head(MARKER_TOP)
        marker_overlap[group] = len(set(scanpy_top) & set(kyto_top.to_list()))
    return Comparison(
        same_cells,
        counts_equal,
        percent_error,
        same_genes,
        len(hvg_kyto),
        len(hvg_scanpy),
        len(hvg_kyto & hvg_scanpy),
        pca_error,
        leiden,
        trust,
        doublets,
        marker_overlap,
    )


def scanpy_gene_ids(sample: Path) -> dict[str, str]:
    """scanpy's gene names to gene IDs, as ``read_10x_h5`` and the tutorial name them.

    Parameters
    ----------
    sample : Path
        One of the tutorial's 10x ``.h5`` files (both hold the same genes).

    Returns
    -------
    dict of str to str
        ``var_names`` after ``var_names_make_unique()`` to ``var["gene_ids"]``.
    """
    data = sc.read_10x_h5(sample)
    data.var_names_make_unique()
    return dict(zip(data.var_names, data.var["gene_ids"], strict=True))


def table(result: Comparison) -> str:
    """The comparison as a Markdown table."""
    rows = [
        "| Check | kyto | scanpy | Agreement |",
        "|---|---|---|---|",
        f"| Cells kept (same IDs and samples) | | | {result.same_cells} |",
    ]
    rows += [
        f"| `{column}` | | | equal: {equal} |"
        for column, equal in result.qc_counts_equal.items()
    ]
    rows += [
        f"| `{column}` | `Float64` | `float32` | largest relative difference "
        f"{error:.1e} |"
        for column, error in result.qc_percent_error.items()
    ]
    rows.append(f"| Genes kept (same IDs) | | | {result.same_genes} |")
    rows.append(
        f"| Highly variable genes | {result.hvg_kyto} | {result.hvg_scanpy} | "
        f"{result.hvg_common} in common |"
    )
    rows.append(
        f"| PCA variances (50) | | | largest relative difference "
        f"{result.pca_variance_error:.1e} |"
    )
    rows += [
        f"| `{column}` clusters | {kyto} | {scanpy} | ARI {ari:.3f} |"
        for column, (kyto, scanpy, ari) in result.leiden.items()
    ]
    rows += [
        f"| UMAP trustworthiness, {neighbors} neighbours | {kyto:.4f} | {scanpy:.4f} | |"
        for neighbors, (kyto, scanpy) in result.trustworthiness.items()
    ]
    kyto_calls, scanpy_calls, both, correlation = result.doublets
    rows.append(
        f"| Doublets called | {kyto_calls} | {scanpy_calls} | {both} by both; "
        f"scores correlate {correlation:.3f} |"
    )
    overlaps = list(result.marker_overlap.values())
    rows.append(
        f"| Top {MARKER_TOP} Wilcoxon genes per `{MARKER_GROUPS}` cluster | | | "
        f"in common: {overlaps} (mean {np.mean(overlaps):.1f}) |"
    )
    return "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: clustering_compare.py <kyto output folder> <scanpy.h5ad> <s1d1.h5>"
        )
    folder = Path(sys.argv[1])
    run = json.loads((folder / "run.json").read_text())
    measured = compare(
        ad.read_h5ad(folder / "kyto.h5ad"),
        ad.read_h5ad(sys.argv[2]),
        scanpy_gene_ids(Path(sys.argv[3])),
        run["kept_genes"],
        pl.read_parquet(folder / "ranked.parquet"),
    )
    print(table(measured))
    print({key: value for key, value in run.items() if key != "kept_genes"})
