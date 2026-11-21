"""Reproducibility (task 8.7, R11): the same input, plan and seed give a byte-identical
store, on one thread and on every core, in both sinks.

The Done-when: a reference pipeline on pbmc3k that reaches every kind of sink member
(layers, cell and gene table columns, cell and gene embeddings, graphs:
``normalize_total -> log1p -> highly_variable_genes -> pca -> neighbors -> leiden ->
umap``) is written twice, at ``threads=1`` and at one thread per core, and the two
writes' checksums are compared. The checksum covers the bytes format.md section 10.3
names: every file of a ``.kyto`` store, with ``provenance.json`` taken without its
``run`` object (timestamps and timings) and its ``source.path``; every byte of an
``.h5ad`` file (kyto's HDF5 writer records no object times, so nothing else varies).
"""

import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Protocol, cast

import h5py  # pyright: ignore[reportMissingTypeStubs]
import kyto
import pytest
from fixtures.pbmc import _MEMBER_PREFIX, fetch_pbmc3k  # pyright: ignore[reportPrivateUsage]
from kyto import _kyto


@pytest.fixture(scope="module")
def pbmc3k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """pbmc3k as a store, converted from its 10x matrix directory."""
    directory = tmp_path_factory.mktemp("pbmc3k")
    matrix = directory / "hg19"
    matrix.mkdir()
    with tarfile.open(fetch_pbmc3k()) as archive:
        for name in ["matrix.mtx", "genes.tsv", "barcodes.tsv"]:
            member = archive.extractfile(_MEMBER_PREFIX + name)
            assert member is not None
            (matrix / name).write_bytes(member.read())
    store = directory / "pbmc3k.kyto"
    kyto.convert(matrix, store)
    return store


def _pipeline(store: Path) -> _kyto.Atlas:
    """The reference pipeline: two layers, gene and cell columns, cell and gene
    embeddings, two graphs, every seeded step seeded."""
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(total=kyto.qc.total_counts())
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .highly_variable_genes(layer="log1p", n_top_genes=1000)
        .pca(
            layer="log1p",
            into="X_pca",
            n_components=30,
            seed=0,
            genes=kyto.gene_column("highly_variable"),
        )
        .neighbors("X_pca", n_neighbors=15, search="approximate", seed=1)
        .leiden(into="leiden", seed=2)
        .umap(into="X_umap", n_epochs=100, seed=3)
    )


def _kyto_checksums(store: Path) -> dict[str, str]:
    """SHA-256 of every file of ``store`` by relative path (format.md section 10.3):
    ``provenance.json`` is hashed without ``run`` and ``source.path``, the rest as is."""
    checksums: dict[str, str] = {}
    for path in sorted(store.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.relative_to(store).as_posix() == "provenance.json":
            provenance = json.loads(data)
            del provenance["run"]
            if provenance["source"] is not None:
                del provenance["source"]["path"]
            data = json.dumps(provenance).encode()
        checksums[path.relative_to(store).as_posix()] = hashlib.sha256(data).hexdigest()
    return checksums


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ObjectInfo(Protocol):
    """The part of h5py's ``ObjInfo`` read here (h5py ships no stubs)."""

    atime: int
    btime: int
    ctime: int
    mtime: int


def _object_times(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Access, birth, change and modification time of every object of an HDF5 file
    (0 when the object records none), by object path."""
    times: dict[str, tuple[int, int, int, int]] = {}

    def record(name: str, item: Any) -> None:
        info = cast(_ObjectInfo, h5py.h5o.get_info(item.id))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        times[name] = (info.atime, info.btime, info.ctime, info.mtime)

    with h5py.File(path, "r") as file:
        record("/", file)
        file.visititems(lambda name, item: record("/" + name, item))  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType, reportUnknownArgumentType]
    return times


ALL_THREADS = os.cpu_count() or 1


def test_kyto_store_is_byte_identical_on_any_thread_count(
    pbmc3k: Path, tmp_path: Path
) -> None:
    pipeline = _pipeline(pbmc3k)
    one, every = tmp_path / "one.kyto", tmp_path / "every.kyto"
    pipeline.sink_zarr(one, threads=1)
    pipeline.sink_zarr(every, threads=ALL_THREADS)

    first, second = _kyto_checksums(one), _kyto_checksums(every)
    assert first.keys() == second.keys()
    assert [name for name in first if first[name] != second[name]] == []
    # The pipeline reached every kind of member, so all of them were compared.
    members = {
        name.split("/")[0] + "/" + name.split("/")[1] for name in first if "/" in name
    }
    assert {
        "layers/normalized",
        "layers/log1p",
        "embeddings/X_pca",
        "embeddings/X_umap",
        "embeddings/X_pca_loadings",
        "graphs/distances",
        "graphs/connectivities",
    } <= members
    assert {"cells.parquet", "genes.parquet", "provenance.json"} <= first.keys()


def test_h5ad_is_byte_identical_on_any_thread_count(
    pbmc3k: Path, tmp_path: Path
) -> None:
    pipeline = _pipeline(pbmc3k)
    one, every = tmp_path / "one.h5ad", tmp_path / "every.h5ad"
    pipeline.sink_h5ad(one, threads=1)
    pipeline.sink_h5ad(every, threads=ALL_THREADS)
    assert _file_checksum(one) == _file_checksum(every)
    # HDF5 stamps times in whole seconds, so two quick writes could match by chance:
    # check that no object (root, groups, datasets) carries a time at all.
    times = _object_times(one)
    assert times.keys() > {"/", "/obs", "/X/data"}
    assert set(times.values()) == {(0, 0, 0, 0)}
