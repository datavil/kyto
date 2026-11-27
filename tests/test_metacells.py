"""``metacells`` (task 6.16, opt-in): e-means metacells on pbmc3k.

The Done-when, on pbmc3k (the 50k half is ``benchmarks/scanpy_workflow/metacells.py``,
run by hand; numbers in performance.md):

- e-means at 3 epochs reaches a training MSE at most that of Lloyd's k-means at 25
  iterations (scikit-learn ``KMeans``, same k, random init), at k = n/50 and n/200;
  both are random-init methods whose error swings by up to 7% with the seed at
  k = 13, so each side is the mean over 50 seeds (``metacells.lloyd_mse``; ten seeds
  can rank them either way: over 100 seeds e-means is 1.1% +- 0.3% lower at k = 13);
- labels and centroids are bit-identical at 1, 4 and all threads;
- a plan without ``metacells`` gives the same ``explain()``, ``estimate()`` and
  output as before the task (``metacells_baseline.json``, written by the build of
  commit 1318b27);
- ``explain()`` shows the node when it is called.

Parity: scanpy has no metacells. kyto replays its own reference, the paper's
algorithm in NumPy (``tests/parity/emeans.py``), bit for bit.
"""

import importlib
import json
import os
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from kyto import _kyto, cell_column, gene_column, qc
from metacells_baseline import (
    BASELINE,
    STORE,
    THREADS,
    checksums,
    pbmc3k_store,
    pipeline,
    snapshot,
)
from numpy.typing import NDArray
from parity import tolerance
from parity.emeans import emeans, paper_batch_size, paper_first_step

ALL_THREADS = os.cpu_count() or 1
SEEDS = range(50)


@pytest.fixture(scope="module")
def pbmc3k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """pbmc3k as a store, converted from its 10x matrix directory."""
    return pbmc3k_store(tmp_path_factory.mktemp("pbmc3k"))


def _preprocessed(store: Path) -> _kyto.Atlas:
    """scanpy's tutorial steps up to a 50-component PCA."""
    mito = gene_column("symbol").str.starts_with("MT-")
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(
            n_genes_by_counts=qc.genes_detected(),
            pct_counts_mt=qc.percent_counts_in(mito),
        )
        .filter(cell_column("n_genes_by_counts") >= 200)
        .filter(cell_column("n_genes_by_counts") < 2500)
        .filter(cell_column("pct_counts_mt") < 5)
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .highly_variable_genes(layer="log1p", n_top_genes=2000)
        .filter(gene_column("highly_variable"))
        .pca(layer="log1p", into="X_pca", n_components=50, seed=0)
    )


@pytest.fixture(scope="module")
def with_pca(pbmc3k: Path) -> _kyto.Atlas:
    """The preprocessed pbmc3k, run once: a store holding its ``X_pca``."""
    return _preprocessed(pbmc3k).collect()


def _labels(atlas: _kyto.Atlas, column: str = "metacells") -> NDArray[np.int64]:
    values = atlas.pull(cell_column(column)).to_numpy()
    return np.array([int(value) for value in values], dtype=np.int64)


def _mse(points: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Training MSE: mean squared distance to the mean of each cell's group."""
    centroids = np.zeros((labels.max() + 1, points.shape[1]))
    np.add.at(centroids, labels, points)
    centroids /= np.maximum(np.bincount(labels), 1)[:, None]
    return float(((points - centroids[labels]) ** 2).sum() / len(points))


def _lloyd_mse(points: NDArray[np.float64], k: int, seed: int) -> float:
    """Training MSE of scikit-learn's Lloyd at 25 iterations from random cells."""
    cluster: Any = importlib.import_module("sklearn.cluster")
    model: Any = cluster.KMeans(
        k,
        init="random",
        n_init=1,
        max_iter=25,
        tol=0.0,
        algorithm="lloyd",
        random_state=seed,
    ).fit(points)
    return float(model.inertia_) / len(points)


@pytest.mark.parametrize("divisor", [50, 200])
def test_three_epochs_reach_lloyds_error_at_25_iterations(
    with_pca: _kyto.Atlas, divisor: int
) -> None:
    points = with_pca.embedding("X_pca").to_numpy().astype(np.float64)
    k = len(points) // divisor
    ours = [
        _mse(points, _labels(with_pca.metacells("X_pca", n_metacells=k, seed=seed)))
        for seed in SEEDS
    ]
    lloyd = [_lloyd_mse(points, k, seed) for seed in SEEDS]
    bound = np.mean(lloyd) * (1.0 + tolerance("metacells.lloyd_mse"))
    assert np.mean(ours) <= bound, (k, np.mean(ours), np.mean(lloyd))


@pytest.mark.parametrize(("divisor", "seed"), [(50, 0), (200, 0), (50, 7)])
def test_bit_for_bit_with_the_numpy_reference(
    with_pca: _kyto.Atlas, divisor: int, seed: int
) -> None:
    points = with_pca.embedding("X_pca").to_numpy()
    k = len(points) // divisor
    atlas = with_pca.metacells("X_pca", n_metacells=k, seed=seed, into="groups")
    expected = emeans(points, k, 3, seed)
    assert np.array_equal(_labels(atlas, "groups"), expected.labels)
    centroids = atlas.embedding("groups_centroids").to_numpy()
    assert centroids.dtype == np.float32
    assert centroids.shape == (len(points), 50)
    want = expected.centroids[expected.labels].astype(np.float32)
    assert centroids.tobytes() == want.tobytes()


def test_a_component_subset_and_more_epochs_match_the_reference(
    with_pca: _kyto.Atlas,
) -> None:
    points = with_pca.embedding("X_pca").to_numpy()
    atlas = with_pca.metacells(
        "X_pca", n_metacells=40, n_components=10, epochs=5, seed=3
    )
    expected = emeans(points[:, :10], 40, 5, 3)
    assert np.array_equal(_labels(atlas), expected.labels)
    assert atlas.embedding("metacells_centroids").to_numpy().shape == (len(points), 10)


def test_a_first_step_clamped_to_one_matches_the_reference(
    with_pca: _kyto.Atlas,
) -> None:
    # 25 epochs at k = 100: Eq. 6 is above 1, so the first update forgets every
    # cluster's count (gamma = 0) and clusters no cell joined keep their centroid.
    points = with_pca.embedding("X_pca").to_numpy()
    batch = paper_batch_size(len(points), 100, 25)
    assert paper_first_step(len(points), 100, 25, batch) == 1.0
    atlas = with_pca.metacells("X_pca", n_metacells=100, epochs=25, seed=0)
    assert np.array_equal(_labels(atlas), emeans(points, 100, 25, 0).labels)
    assert np.isfinite(atlas.embedding("metacells_centroids").to_numpy()).all()


def test_identical_on_one_four_and_all_threads(pbmc3k: Path, tmp_path: Path) -> None:
    plan = _preprocessed(pbmc3k).metacells(n_metacells=52, seed=0)
    found: list[dict[str, str]] = []
    for threads in sorted({1, 4, ALL_THREADS}):
        output = tmp_path / f"{threads}.kyto"
        plan.sink_zarr(output, threads=threads)
        found.append(checksums(output))
    assert all(sums == found[0] for sums in found)
    assert any(name.startswith("embeddings/metacells_centroids") for name in found[0])


def test_a_plan_without_metacells_is_unchanged(pbmc3k: Path, tmp_path: Path) -> None:
    expected = json.loads(BASELINE.read_text())
    found = snapshot(pbmc3k, tmp_path / "out.kyto")
    assert found["explain"] == expected["explain"]
    assert found["estimate"] == expected["estimate"]
    assert found["checksums"] == expected["checksums"]
    assert "METACELLS" not in found["explain"]


def test_explain_shows_the_node_and_it_reads_its_embedding(pbmc3k: Path) -> None:
    plan = pipeline(pbmc3k).metacells(n_metacells=50, seed=4)
    text = plan.explain(threads=THREADS)
    assert (
        'METACELLS "X_pca" -> "metacells", "metacells_centroids" n_metacells=50 '
        "epochs=3 seed=4"
    ) in text
    assert STORE not in text
    schema = plan.schema()
    assert any(column["name"] == "metacells" for column in schema["cells"]["columns"])
    assert any(
        embedding["name"] == "metacells_centroids" and embedding["width"] == 30
        for embedding in schema["embeddings"]
    )
    # Round trip through the plan JSON.
    assert str(kyto.from_json(plan.to_json())) == str(plan)


def test_after_a_cell_filter_only_the_kept_cells_are_grouped(
    with_pca: _kyto.Atlas,
) -> None:
    kept = with_pca.filter(cell_column("n_genes_by_counts") >= 1000)
    points = kept.embedding("X_pca").to_numpy()
    atlas = kept.metacells("X_pca", n_metacells=20, seed=1)
    assert np.array_equal(_labels(atlas), emeans(points, 20, 3, 1).labels)


def test_arguments_are_coded(pbmc3k: Path, with_pca: _kyto.Atlas) -> None:
    for arguments, message in [
        ({"n_metacells": 0}, "n_metacells"),
        ({"n_metacells": 2639}, "the plan has 2638 cells"),
        ({"n_metacells": 5, "epochs": 0}, "epochs"),
        ({"n_metacells": 5, "epochs": 26}, "epochs"),
        ({"n_metacells": 5, "n_components": 51}, "n_components"),
        ({"n_metacells": 5, "into": "cell_id"}, "reserved"),
        ({"n_metacells": 5, "into": "X"}, "replace the embedding grouped"),
    ]:
        embedding = "X_centroids" if arguments.get("into") == "X" else "X_pca"
        plan = (
            with_pca
            if embedding == "X_pca"
            else with_pca.metacells("X_pca", n_metacells=2, seed=0, into="X")
        )
        with pytest.raises(kyto.errors.PlanError, match=message) as raised:
            plan.metacells(embedding, seed=0, **arguments)  # pyright: ignore[reportArgumentType]
        assert raised.value.code == "KY-E215"
    with pytest.raises(kyto.errors.MissingMemberError) as missing:
        with_pca.metacells("X_umap", n_metacells=5, seed=0)
    assert missing.value.code == "KY-E109"
    taken = with_pca.metacells(n_metacells=5, seed=0)
    with pytest.raises(kyto.errors.PlanError) as default_taken:
        taken.metacells(n_metacells=5, seed=1)
    assert default_taken.value.code == "KY-E220"
    # More metacells than the cells a filter leaves: known only when the plan runs.
    filtered = _preprocessed(pbmc3k).filter(cell_column("n_genes_by_counts") >= 2400)
    with pytest.raises(kyto.errors.PlanError, match="n_metacells") as late:
        filtered.metacells(n_metacells=2000, seed=0).embedding(
            "metacells_centroids"
        ).to_numpy()
    assert late.value.code == "KY-E215"
