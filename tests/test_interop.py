"""Task 6.11: interop hooks, out through NumPy and DLPack, back with ``with_embedding``.

Done when: a notebook example runs scvi on a kyto atlas and stores the latent space
back (``docs/tutorials/interop.py``, run here as a script).

Round trips, always run: an embedding or a dense matrix exported through NumPy or
DLPack and handed back with ``with_embedding`` reads back bit for bit (``float32`` and
``float64``, any memory order), through every later step, sink and terminal; the
values are copied once; after a cell filter the rows are those of the kept cells.

Parity, with the ``interop`` dependency group (scvi-tools, harmonypy) and scanpy:
scvi-tools trained on kyto's export equals scvi-tools trained on scanpy's AnnData of
the same cells and genes, bit for bit; Harmony on kyto's PCA is compared with Harmony
on scanpy's PCA (tolerances ``interop.*``: kyto's PCA differs from scanpy's in the
trailing components, and Harmony carries that through). Those tests skip, visibly,
without the group (``uv sync --group interop``).
"""

# scanpy, anndata, scvi-tools and harmonypy ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, cast

import kyto
import numpy as np
import pytest
from fixtures.pbmc import fetch_pbmc3k
from kyto import _kyto, cell_column, gene_column, qc
from kyto.errors import ExpressionError, PlanError
from numpy.typing import NDArray
from parity import tolerance

N_CELLS = 40
N_GENES = 12
NOTEBOOK = Path(__file__).parents[1] / "docs" / "tutorials" / "interop.py"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 40-cell, 12-gene store, layer ``counts``, in two groups of cells."""
    generator = np.random.default_rng(611)
    counts = generator.poisson(3.0, size=(N_CELLS, N_GENES))
    counts[: N_CELLS // 2, : N_GENES // 2] += 6
    rows, columns = np.nonzero(counts)
    source = tmp_path / "source"
    source.mkdir()
    entries = "".join(
        f"{gene + 1} {cell + 1} {counts[cell, gene]}\n"
        for cell, gene in zip(rows, columns, strict=True)
    )
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        f"{N_GENES} {N_CELLS} {len(rows)}\n{entries}"
    )
    (source / "barcodes.tsv").write_text(
        "".join(f"C{cell:02d}-1\n" for cell in range(N_CELLS))
    )
    (source / "genes.tsv").write_text(
        "".join(f"ENSG{gene:02d}\tG{gene}\n" for gene in range(N_GENES))
    )
    path = tmp_path / "atlas.kyto"
    kyto.convert(source, path, layer="counts")
    return path


def _pca(store: Path) -> _kyto.Atlas:
    return (
        kyto.scan(store)
        .normalize_total(1e4, layer="counts")
        .log1p()
        .pca(n_components=4, seed=0)
    )


def _same_bits(left: NDArray[Any], right: NDArray[Any]) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _awkward_float64(n_rows: int, width: int) -> NDArray[np.float64]:
    """Signed zeros, subnormals, extremes and values with no short decimal form."""
    specials = np.array(
        [
            0.0,
            -0.0,
            5e-324,
            -2.2250738585072014e-308,
            1.7976931348623157e308,
            0.1,
            1 / 3,
        ]
    )
    return np.resize(specials, n_rows * width).reshape(n_rows, width) * np.where(
        np.arange(n_rows * width).reshape(n_rows, width) % 5 == 0, 1.0, 0.5
    )


# --- Round trips (no optional dependency) ------------------------------------------


def test_numpy_round_trip_is_bit_for_bit(store: Path) -> None:
    atlas = _pca(store)
    exported = atlas.embedding("X_pca").to_numpy()
    back = atlas.with_embedding("X_copy", exported)
    assert _same_bits(back.embedding("X_copy").to_numpy(), exported)
    # float64, in Fortran order and as a strided view: rows and columns as indexed.
    awkward = _awkward_float64(N_CELLS, 6)
    for values in (awkward, np.asfortranarray(awkward), awkward[:, ::2]):
        read = kyto.scan(store).with_embedding("X_given", values)
        assert _same_bits(
            read.embedding("X_given").to_numpy(), np.ascontiguousarray(values)
        )
    schema = kyto.scan(store).with_embedding("X_given", awkward).embedding("X_given")
    assert schema.schema()["dtype"] == "float64"
    assert schema.schema()["width"] == 6


def test_dlpack_round_trip_is_bit_for_bit(store: Path) -> None:
    atlas = _pca(store)
    exported = cast("NDArray[np.float32]", np.from_dlpack(atlas.embedding("X_pca")))

    class OnlyDLPack:
        """A producer that is not a NumPy array (as a torch tensor is not)."""

        def __init__(self, array: NDArray[Any]) -> None:
            self.array = array

        def __dlpack__(self, **kwargs: Any) -> object:
            return self.array.__dlpack__(**kwargs)

        def __dlpack_device__(self) -> tuple[int, int]:
            return self.array.__dlpack_device__()

    given: list[NDArray[np.float32] | OnlyDLPack] = [exported, OnlyDLPack(exported)]
    for values in given:
        back = atlas.with_embedding("X_copy", values)
        assert _same_bits(back.embedding("X_copy").to_numpy(), exported)


def test_torch_round_trip_is_bit_for_bit(store: Path) -> None:
    torch = pytest.importorskip(
        "torch", reason="torch is not installed (the interop dependency group)"
    )
    atlas = _pca(store)
    tensor = torch.from_dlpack(atlas.embedding("X_pca"))
    back = atlas.with_embedding("X_torch", tensor)
    assert _same_bits(back.embedding("X_torch").to_numpy(), tensor.numpy())
    # A transposed (non-contiguous) tensor and a float64 one.
    double = torch.from_numpy(_awkward_float64(6, N_CELLS)).T
    read = kyto.scan(store).with_embedding("X_double", double)
    assert _same_bits(
        read.embedding("X_double").to_numpy(), double.contiguous().numpy()
    )
    with pytest.raises(PlanError, match="KY-E215.*not float32 or float64"):
        kyto.scan(store).with_embedding("X_half", tensor.to(torch.float16))


def test_a_dense_matrix_export_comes_back_as_an_embedding(store: Path) -> None:
    log1p = kyto.scan(store).normalize_total(1e4, layer="counts").log1p()
    dense = log1p.layer("log1p").to_dense(max_bytes=1 << 20)
    back = log1p.with_embedding("X_dense", dense)
    assert _same_bits(back.embedding("X_dense").to_numpy(), dense)


def test_values_are_copied_once(store: Path) -> None:
    values = np.ones((N_CELLS, 3), np.float32)
    atlas = kyto.scan(store).with_embedding("X_given", values)
    values[:] = 7.0
    assert (atlas.embedding("X_given").to_numpy() == 1.0).all()
    exported = atlas.embedding("X_given").to_numpy()
    exported[:] = 9.0
    assert (atlas.embedding("X_given").to_numpy() == 1.0).all()


def test_later_steps_sinks_and_plan_json_carry_it(store: Path, tmp_path: Path) -> None:
    atlas = _pca(store)
    pcs = atlas.embedding("X_pca").to_numpy()
    handed_back = atlas.with_embedding("X_back", pcs).neighbors("X_back", into="back")
    computed = atlas.neighbors("X_pca")
    for name in ("distances", "connectivities"):
        left = handed_back.graph(f"back_{name}").to_scipy()
        right = computed.graph(name).to_scipy()
        assert (left - right).count_nonzero() == 0
        assert (left.indices == right.indices).all()
    given = kyto.scan(store).with_embedding("X_given", _awkward_float64(N_CELLS, 3))
    expected = given.embedding("X_given").to_numpy()
    given.sink_zarr(tmp_path / "out.kyto")
    stored = kyto.scan(tmp_path / "out.kyto").embedding("X_given").to_numpy()
    assert _same_bits(stored, expected)
    replayed = kyto.from_json(given.to_json())
    assert replayed.to_json() == given.to_json()
    assert _same_bits(replayed.embedding("X_given").to_numpy(), expected)
    pytest.importorskip("anndata", reason="anndata is not installed on this platform")
    adata = given.to_anndata()
    assert _same_bits(np.asarray(adata.obsm["X_given"]), expected)


def test_after_a_cell_filter_rows_are_the_kept_cells(store: Path) -> None:
    first_half = cell_column("cell_id") < "C20"
    kept = kyto.scan(store).filter(first_half)
    values = np.arange(20 * 2, dtype=np.float32).reshape(20, 2)
    atlas = kept.with_embedding("X_given", values)
    assert _same_bits(atlas.embedding("X_given").to_numpy(), values)
    # A filter after it keeps the rows of the cells it keeps.
    later = kyto.scan(store).with_embedding(
        "X_given", np.arange(N_CELLS, dtype=np.float32).reshape(N_CELLS, 1)
    )
    rows = later.filter(first_half).embedding("X_given").to_numpy()
    assert rows.ravel().tolist() == list(range(20))
    # The row count is checked when the plan runs.
    wrong = kept.with_embedding("X_given", np.zeros((N_CELLS, 2), np.float32))
    with pytest.raises(PlanError) as raised:
        wrong.embedding("X_given").to_numpy()
    assert raised.value.code == "KY-E215"
    assert "20 cells reach the step when the plan runs" in str(raised.value)


@pytest.mark.parametrize(
    ("values", "error", "part"),
    [
        (np.zeros((N_CELLS + 1, 2), np.float32), PlanError, "the plan has 40 cells"),
        (np.zeros(N_CELLS, np.float32), PlanError, "it has 1 dimensions"),
        (np.zeros((N_CELLS, 2, 1), np.float32), PlanError, "it has 3 dimensions"),
        (np.zeros((N_CELLS, 0), np.float32), PlanError, "it has no column"),
        (np.zeros((N_CELLS, 2), np.int64), PlanError, "not float32 or float64"),
        (np.zeros((N_CELLS, 2), np.float16), PlanError, "not float32 or float64"),
        (np.full((N_CELLS, 2), np.nan), PlanError, "row 0, column 0"),
        (
            np.where(np.arange(2 * N_CELLS).reshape(N_CELLS, 2) == 7, np.inf, 0.0),
            PlanError,
            "row 3, column 1",
        ),
        ([[0.0, 1.0]] * N_CELLS, ExpressionError, "a NumPy array or a DLPack tensor"),
        (None, ExpressionError, "got a NoneType"),
    ],
)
def test_refuses_what_it_cannot_hold(
    store: Path, values: Any, error: type[Exception], part: str
) -> None:
    with pytest.raises(error) as raised:
        kyto.scan(store).with_embedding("X_given", values)
    assert getattr(raised.value, "code", None) in {"KY-E215", "KY-E205"}
    assert part in str(raised.value)


def test_refuses_bad_names(store: Path) -> None:
    values = np.zeros((N_CELLS, 2), np.float32)
    for name in ("a/b", ""):
        with pytest.raises(PlanError, match="KY-E215"):
            kyto.scan(store).with_embedding(name, values)
    with pytest.raises(PlanError, match="KY-E215.*gene embedding"):
        _pca(store).with_embedding("X_pca_loadings", values)


def test_masked_swapped_and_unaligned_arrays(store: Path) -> None:
    values = np.arange(2 * N_CELLS, dtype=np.float32).reshape(N_CELLS, 2)
    with pytest.raises(PlanError, match="KY-E215.*mask would be ignored"):
        kyto.scan(store).with_embedding("X", np.ma.masked_array(values))
    with pytest.raises(PlanError, match="KY-E215.*other byte order"):
        kyto.scan(store).with_embedding("X", values.astype(">f4"))
    # An unaligned view is read through an aligned copy, values unchanged.
    buffer = np.zeros(values.nbytes + 1, np.uint8)
    unaligned = buffer[1:].view(np.float32).reshape(N_CELLS, 2)
    unaligned[:] = values
    assert not unaligned.flags.aligned
    read = kyto.scan(store).with_embedding("X", unaligned)
    assert _same_bits(read.embedding("X").to_numpy(), values)


# --- The notebook and parity with scanpy (the interop dependency group) ------------


def _interop() -> tuple[Any, Any]:
    scvi = pytest.importorskip(
        "scvi", reason="scvi-tools is not installed (uv sync --group interop)"
    )
    harmonypy = pytest.importorskip(
        "harmonypy", reason="harmonypy is not installed (uv sync --group interop)"
    )
    return scvi, harmonypy


@pytest.fixture(scope="module")
def pbmc3k_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """pbmc3k's 10x folder, extracted from the verified fixture archive."""
    target = tmp_path_factory.mktemp("pbmc3k")
    with tarfile.open(fetch_pbmc3k(), "r:gz") as archive:
        archive.extractall(target, filter="data")
    return target / "filtered_gene_bc_matrices" / "hg19"


def test_notebook_runs_scvi_and_stores_the_latent_space_back(
    pbmc3k_folder: Path, tmp_path: Path
) -> None:
    _interop()
    output = tmp_path / "output"
    completed = subprocess.run(
        [sys.executable, str(NOTEBOOK), str(pbmc3k_folder), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    stored = kyto.scan(output / "pbmc3k_integrated.kyto")
    latent = stored.embedding("X_scVI").to_numpy()
    assert latent.shape == (2700, 10)
    assert latent.dtype == np.float32
    assert np.isfinite(latent).all()
    assert stored.embedding("X_pca_harmony").to_numpy().shape == (2700, 30)
    cells = stored.cells.to_polars()
    assert cells["leiden_scvi"].n_unique() > 1
    assert cells["leiden_harmony"].n_unique() > 1
    assert stored.embedding("X_umap_scvi").to_numpy().shape == (2700, 2)


@pytest.fixture(scope="module")
def pbmc3k_pair(pbmc3k_folder: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """kyto's plan of pbmc3k's 1200 ``seurat_v3`` genes, and scanpy's AnnData of them."""
    _interop()
    sc = pytest.importorskip("scanpy", reason="scanpy is not installed")
    from fixtures.interop import to_anndata
    from fixtures.pbmc import load_pbmc3k

    path = tmp_path_factory.mktemp("pbmc3k_store") / "pbmc3k.kyto"
    kyto.convert(pbmc3k_folder, path, layer="counts")
    variable = (
        kyto.scan(path)
        .with_default_layer("counts")
        .filter(qc.genes_detected() >= 200)
        .filter(qc.cells_detected() >= 3)
        .highly_variable_genes(n_top_genes=1200, flavor="seurat_v3")
        .filter(gene_column("highly_variable"))
    )
    reference = to_anndata(load_pbmc3k())
    sc.pp.filter_cells(reference, min_genes=200)
    sc.pp.filter_genes(reference, min_cells=3)
    sc.pp.highly_variable_genes(reference, flavor="seurat_v3", n_top_genes=1200)
    reference = reference[:, reference.var["highly_variable"]].copy()
    return variable, reference, sc


def _scvi_latent(scvi: Any, adata: Any) -> NDArray[np.float32]:
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=10)
    model.train(max_epochs=10, accelerator="cpu", enable_progress_bar=False)
    return model.get_latent_representation()


def test_scvi_on_kytos_export_equals_scvi_on_scanpys_anndata(pbmc3k_pair: Any) -> None:
    scvi, _ = _interop()
    variable, reference, _ = pbmc3k_pair
    exported = variable.to_anndata()
    assert list(exported.obs_names) == list(reference.obs_names)
    assert list(exported.var_names) == list(reference.var_names)
    counts = exported.X.astype(np.float32)
    assert (counts - reference.X).count_nonzero() == 0
    latent = _scvi_latent(scvi, exported)
    expected = _scvi_latent(scvi, reference.copy())
    # Same input, same seed: the same latent space, bit for bit (measured).
    assert _same_bits(latent, expected)
    stored = variable.with_embedding("X_scVI", latent).embedding("X_scVI").to_numpy()
    assert _same_bits(stored, expected)


def _harmony(harmonypy: Any, components: NDArray[Any], batch: NDArray[Any]) -> Any:
    run = harmonypy.run_harmony(
        components, {"batch": batch}, "batch", random_state=0, verbose=False
    )
    corrected = np.asarray(run.Z_corr)
    return corrected if corrected.shape == components.shape else corrected.T


def test_harmony_on_kytos_pca_against_harmony_on_scanpys(pbmc3k_pair: Any) -> None:
    _, harmonypy = _interop()
    variable, reference, sc = pbmc3k_pair
    from sklearn.neighbors import NearestNeighbors

    pca = (
        variable.normalize_total(1e4, layer="counts")
        .log1p()
        .pca(n_components=30, seed=0)
    )
    components = pca.embedding("X_pca").to_numpy()
    reference = reference.copy()
    sc.pp.normalize_total(reference, target_sum=1e4)
    sc.pp.log1p(reference)
    sc.pp.pca(reference, n_comps=30, random_state=0)
    # Two stand-in batches (pbmc3k is one sample). scanpy's `harmony_integrate` is
    # `run_harmony` on `obsm["X_pca"]` plus a transpose that harmonypy 2 no longer
    # needs (scanpy 1.12.5 fails on it), so the reference calls harmonypy itself.
    batch = np.array(["a", "b"])[np.arange(components.shape[0]) % 2]
    corrected = _harmony(harmonypy, components, batch)
    expected = _harmony(harmonypy, reference.obsm["X_pca"], batch)
    stored = pca.with_embedding("X_pca_harmony", corrected.astype(np.float32))
    assert _same_bits(
        stored.embedding("X_pca_harmony").to_numpy(), corrected.astype(np.float32)
    )
    rows = np.linalg.norm(corrected - expected, axis=1) / np.linalg.norm(
        expected, axis=1
    )
    median = float(np.median(rows))
    assert median <= tolerance("interop.harmony_rows"), median

    def knn(values: NDArray[Any]) -> NDArray[np.intp]:
        found = NearestNeighbors(n_neighbors=16).fit(values)
        return found.kneighbors(values, return_distance=False)[:, 1:]

    left, right = knn(corrected), knn(expected)
    overlap = float(
        np.mean([len(set(a) & set(b)) / 15 for a, b in zip(left, right, strict=True)])
    )
    assert overlap >= tolerance("interop.harmony_knn"), overlap
    print(
        f"Harmony vs scanpy: median row error {median:.1e}, kNN overlap {overlap:.3f}"
    )
