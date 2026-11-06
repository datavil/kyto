"""The 10x Genomics pbmc3k dataset: download once, verify, parse into a Dataset.

What it does: fetches the filtered gene-barcode matrices of 3k PBMCs (the 10x source
that ``scanpy.datasets.pbmc3k``'s ``pbmc3k_raw.h5ad`` was built from) into a gitignored cache, checks the archive's SHA-256,
and parses the Matrix Market file straight from the archive (nothing is extracted to
disk) into a cells x genes :class:`~fixtures.dataset.Dataset`.

Gene symbols are kept as published, repeats included (91 symbols repeat; task 3.9),
with the Ensembl IDs as the unique gene key. scanpy's ``pbmc3k()`` instead indexes by
symbol made unique (``var_names_make_unique``); the counts are the same.

What it does not: commit or modify raw data. The archive lives only in the cache.
"""

import hashlib
import io
import os
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fixtures.dataset import Dataset

PBMC3K_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
    "pbmc3k_filtered_gene_bc_matrices.tar.gz"
)
PBMC3K_SHA256 = "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5"
PBMC3K_ARCHIVE = "pbmc3k_filtered_gene_bc_matrices.tar.gz"
_MEMBER_PREFIX = "filtered_gene_bc_matrices/hg19/"
_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_ATTEMPTS = 3
_CHUNK_BYTES = 1 << 20

# Gitignored; CI keeps it between runs with actions/cache (.github/workflows/ci.yml).
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "fixtures"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _request() -> urllib.request.Request:
    # The 10x CDN answers 403 to urllib's default User-Agent.
    return urllib.request.Request(
        PBMC3K_URL, headers={"User-Agent": "kyto-test-fixtures"}
    )


def fetch_pbmc3k(cache_dir: Path = CACHE_DIR) -> Path:
    """Return the path of the verified pbmc3k archive, downloading it if needed.

    The download goes to a temporary file in ``cache_dir`` and is renamed into place
    only after its SHA-256 matches, so an interrupted download never poisons the cache.

    Parameters
    ----------
    cache_dir : Path, default=CACHE_DIR
        Directory that holds the archive (``<repo>/.cache/fixtures``).

    Returns
    -------
    Path
        ``cache_dir / PBMC3K_ARCHIVE``, with SHA-256 ``PBMC3K_SHA256``.

    Raises
    ------
    ValueError
        If the downloaded file has another SHA-256 (a cached file that fails the check
        is deleted and downloaded again).
    urllib.error.URLError
        If all download attempts fail (no network, server error).
    """
    path = cache_dir / PBMC3K_ARCHIVE
    if path.exists() and _file_sha256(path) != PBMC3K_SHA256:
        # A truncated or corrupted cache (for example a bad CI cache restore).
        path.unlink()
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _download(path)
    actual = _file_sha256(path)
    if actual != PBMC3K_SHA256:
        raise ValueError(
            f"fetch_pbmc3k: {path} has SHA-256 {actual}, expected {PBMC3K_SHA256}."
        )
    return path


def _download(path: Path) -> None:
    """Download the archive to ``path``, retrying transient network failures."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_once(path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def _download_once(path: Path) -> None:
    cache_dir = path.parent
    handle, temporary_name = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(handle, "wb") as output,
            urllib.request.urlopen(
                _request(), timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                output.write(chunk)
        actual = _file_sha256(temporary)
        if actual != PBMC3K_SHA256:
            raise ValueError(
                f"fetch_pbmc3k: download from {PBMC3K_URL} has SHA-256 {actual}, "
                f"expected {PBMC3K_SHA256}. The source changed or the transfer "
                "was corrupted; retry, and if it persists, check the URL."
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.extractfile(_MEMBER_PREFIX + name)
    if member is None:
        raise ValueError(f"pbmc3k archive: {_MEMBER_PREFIX + name} is not a file.")
    with member:
        return member.read()


def _tsv_column(text: bytes, column: int, n_columns: int) -> NDArray[np.str_]:
    values: list[str] = []
    for line in text.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != n_columns:
            raise ValueError(
                f"pbmc3k archive: expected {n_columns} tab-separated fields, got "
                f"{len(fields)} in line {line!r}."
            )
        values.append(fields[column])
    return np.array(values, dtype=np.str_)


def load_pbmc3k(cache_dir: Path = CACHE_DIR) -> Dataset:
    """Return pbmc3k as a cells x genes Dataset (2700 x 32738), fetching it if needed.

    Parameters
    ----------
    cache_dir : Path, default=CACHE_DIR
        Directory that holds the archive; see :func:`fetch_pbmc3k`.

    Returns
    -------
    Dataset
        A new dataset named ``"pbmc3k"``. Cell columns: ``cell_id`` (barcode). Gene
        columns: ``gene_id`` (Ensembl ID), ``symbol`` (as published, may repeat).

    Raises
    ------
    ValueError
        If the archive fails its SHA-256 check or its contents are malformed.
    """
    with tarfile.open(fetch_pbmc3k(cache_dir), "r:gz") as archive:
        genes_text = _read_member(archive, "genes.tsv")
        barcodes_text = _read_member(archive, "barcodes.tsv")
        matrix_text = _read_member(archive, "matrix.mtx")

    stream = io.TextIOWrapper(io.BytesIO(matrix_text), encoding="ascii")
    header = stream.readline()
    if not header.startswith("%%MatrixMarket matrix coordinate"):
        raise ValueError(f"pbmc3k archive: unexpected matrix.mtx header {header!r}.")
    line = stream.readline()
    while line.startswith("%"):
        line = stream.readline()
    n_rows, n_columns, nnz = (int(field) for field in line.split())
    entries = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if entries.shape != (nnz, 3):
        raise ValueError(
            f"pbmc3k archive: matrix.mtx declares {nnz} entries, found "
            f"{entries.shape[0]} rows of {entries.shape[1]} fields."
        )
    # Matrix Market is 1-based and, for 10x v1, genes x cells.
    gene = entries[:, 0].astype(np.int64) - 1
    cell = entries[:, 1].astype(np.int64) - 1
    values = entries[:, 2]
    if np.any((gene < 0) | (gene >= n_rows) | (cell < 0) | (cell >= n_columns)):
        raise ValueError(
            f"pbmc3k archive: matrix.mtx has an entry outside {n_rows} x {n_columns}."
        )
    if np.any(values != np.round(values)) or np.any(values <= 0):
        raise ValueError("pbmc3k archive: counts must be positive integers.")

    gene_id = _tsv_column(genes_text, 0, 2)
    symbol = _tsv_column(genes_text, 1, 2)
    cell_id = _tsv_column(barcodes_text, 0, 1)
    if (gene_id.shape[0], cell_id.shape[0]) != (n_rows, n_columns):
        raise ValueError(
            f"pbmc3k archive: matrix is {n_rows} x {n_columns} but there are "
            f"{gene_id.shape[0]} genes and {cell_id.shape[0]} barcodes."
        )

    order = np.lexsort((gene, cell))
    gene, cell, values = gene[order], cell[order], values[order]
    duplicate = (np.diff(cell) == 0) & (np.diff(gene) == 0)
    if np.any(duplicate):
        raise ValueError("pbmc3k archive: matrix.mtx repeats a (gene, cell) entry.")
    indptr = np.zeros(n_columns + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=n_columns), out=indptr[1:])
    return Dataset(
        name="pbmc3k",
        indptr=indptr,
        indices=gene.astype(np.int32),
        data=values.astype(np.float32),
        cells={"cell_id": cell_id},
        genes={"gene_id": gene_id, "symbol": symbol},
    )
