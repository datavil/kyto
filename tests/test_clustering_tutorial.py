"""Task 10.7: scanpy's clustering tutorial, end to end, in kyto.

Done when: the notebook runs end to end in the gate, and its results are compared with
scanpy's run of the tutorial (QC columns and HVG sets equal; PCA variance to 1e-3;
Leiden adjusted Rand index and UMAP trustworthiness reported).

``docs/tutorials/clustering.py`` runs as a script, in its own process, on the
tutorial's two samples (fetched into the fixture cache, SHA-256 checked) and must draw
every plot. scanpy's run of the same tutorial (``clustering_scanpy.py``) is the
reference: it is computed once per set of reference package versions and cached next
to the samples, since it does not change between kyto commits. The comparison table is
printed (``pytest -s``) and the measured values are held to the bounds below.
Skips the comparison without scanpy.
"""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from fixtures.pbmc import CACHE_DIR
from fixtures.tutorial_samples import TUTORIAL_SAMPLES, fetch_tutorial_samples

anndata = pytest.importorskip("anndata")
pytest.importorskip("cellestial")

TUTORIALS = Path(__file__).parents[1] / "docs" / "tutorials"
PLOTS = (
    "qc_violins",
    "qc_scatter",
    "highly_variable_genes",
    "pca",
    "pca_3_4",
    "umap_sample",
    "umap_leiden",
    "umap_doublets",
    "umap_qc",
    "umap_resolutions",
    "markers_res_0.02",
    "markers_res_0.50",
    "ranked_genes",
    "umap_cluster_7_genes",
)
# The packages whose versions decide scanpy's result; a change recomputes it.
REFERENCE_PACKAGES = (
    "scanpy",
    "anndata",
    "numpy",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "umap-learn",
    "pynndescent",
    "numba",
    "igraph",
    "leidenalg",
)
# Measured on macOS arm64 (Nov 14, 2026): 0.819, 0.979, 0.920, 0.878. kyto's Leiden
# runs to convergence while the tutorial's first call stops after 2 iterations, and the
# neighbour graphs differ (exact search against pynndescent), so the bound has room.
LEIDEN_ARI_FLOOR = 0.75
# Measured 0.9634 (kyto) against 0.9627 (scanpy) at 5 neighbours.
TRUSTWORTHINESS_MARGIN = 0.01
# Doublets: kyto's exact search calls 249 to 255 in s1d1 and 181 to 207 in s1d3 over
# seeds 0 to 5, as scanpy's with `use_approx_neighbors=False`; the tutorial's scanpy run
# (pynndescent) calls 213, of which kyto calls 211. Scores correlate 0.987.
DOUBLET_CALLS = (400, 480)
DOUBLET_SHARED_FLOOR = 0.9
DOUBLET_SCORE_CORRELATION_FLOOR = 0.95
# Measured mean 6.8 of 10 (the 11 large clusters 7 to 10).
MARKER_OVERLAP_FLOOR = 5.5


def _run(script: Path, *arguments: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]


@pytest.fixture(scope="module")
def samples() -> dict[str, Path]:
    return fetch_tutorial_samples()


@pytest.fixture(scope="module")
def kyto_run(
    samples: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The tutorial's output folder, written by the notebook run as a script."""
    output = tmp_path_factory.mktemp("clustering") / "output"
    _run(TUTORIALS / "clustering.py", samples["s1d1"], samples["s1d3"], output)
    return output


def _reference_key() -> str:
    digest = hashlib.sha256()
    digest.update((TUTORIALS / "clustering_scanpy.py").read_bytes())
    for sample in TUTORIAL_SAMPLES:
        digest.update(sample.sha256.encode())
    for package in REFERENCE_PACKAGES:
        digest.update(f"{package}=={importlib.metadata.version(package)}".encode())
    return digest.hexdigest()[:16]


@pytest.fixture(scope="module")
def scanpy_reference(samples: dict[str, Path]) -> Path:
    """scanpy's processed ``AnnData`` of the tutorial, computed once and cached."""
    pytest.importorskip("scanpy")
    path = CACHE_DIR / "clustering_tutorial" / f"scanpy-{_reference_key()}.h5ad"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part.h5ad")
        _run(
            TUTORIALS / "clustering_scanpy.py",
            samples["s1d1"],
            samples["s1d3"],
            partial,
        )
        partial.replace(path)
    return path


def test_notebook_runs_and_draws_every_plot(kyto_run: Path) -> None:
    for name in PLOTS:
        assert (kyto_run / f"{name}.png").stat().st_size > 0, name
    run = json.loads((kyto_run / "run.json").read_text())
    assert run["total"] > 0
    result = anndata.read_h5ad(kyto_run / "kyto.h5ad")
    assert result.n_vars == 2000
    assert {"X_pca", "X_umap"} <= set(result.obsm)


def test_matches_scanpys_run(
    kyto_run: Path, scanpy_reference: Path, samples: dict[str, Path]
) -> None:
    from clustering_compare import compare, scanpy_gene_ids, table

    run = json.loads((kyto_run / "run.json").read_text())
    result = compare(
        anndata.read_h5ad(kyto_run / "kyto.h5ad"),
        anndata.read_h5ad(scanpy_reference),
        scanpy_gene_ids(samples["s1d1"]),
        run["kept_genes"],
        pl.read_parquet(kyto_run / "ranked.parquet"),
    )
    print(table(result))

    assert result.same_cells
    assert all(result.qc_counts_equal.values()), result.qc_counts_equal
    # kyto's Float64 percentages against scanpy's float32: one float32 rounding.
    assert max(result.qc_percent_error.values()) <= 2.0**-22, result.qc_percent_error
    assert result.same_genes
    assert result.hvg_kyto == result.hvg_scanpy == result.hvg_common == 2000
    assert result.pca_variance_error <= 1e-3
    for column, (_, _, ari) in result.leiden.items():
        assert ari >= LEIDEN_ARI_FLOOR, (column, ari)
    for neighbors, (kyto, scanpy) in result.trustworthiness.items():
        assert kyto >= scanpy - TRUSTWORTHINESS_MARGIN, (neighbors, kyto, scanpy)
    kyto_calls, scanpy_calls, both, correlation = result.doublets
    assert DOUBLET_CALLS[0] <= kyto_calls <= DOUBLET_CALLS[1], result.doublets
    assert both >= DOUBLET_SHARED_FLOOR * scanpy_calls, result.doublets
    assert correlation >= DOUBLET_SCORE_CORRELATION_FLOOR, result.doublets
    overlaps = list(result.marker_overlap.values())
    assert sum(overlaps) / len(overlaps) >= MARKER_OVERLAP_FLOOR, overlaps
