"""The two samples of scanpy's clustering tutorial: download once, verify, cache.

What it does: fetches ``s1d1`` and ``s1d3``, the 10x Cell Ranger ``.h5`` files scanpy's
basic clustering tutorial reads (figshare DOI 10.6084/m9.figshare.22716739.v1, which the
tutorial's ``pooch`` registry points at; 22.8 MB and 20.4 MB), into the gitignored
fixture cache, and checks each file's SHA-256 (the MD5 figshare publishes was checked
when the digests were recorded). Task 10.7 runs the tutorial on them.

What it does not: parse the files (``kyto.convert``/``kyto.concat`` and
``scanpy.read_10x_h5`` read them) or commit raw data.
"""

import hashlib
import http.client
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from fixtures.pbmc import CACHE_DIR


class TutorialSample(NamedTuple):
    """One downloadable sample: its file name, URL and SHA-256."""

    name: str
    url: str
    sha256: str

    @property
    def file_name(self) -> str:
        """The cached file's name (the tutorial's)."""
        return f"{self.name}_filtered_feature_bc_matrix.h5"


# In the tutorial's order; the URLs are figshare's downloads of the DOI's files.
TUTORIAL_SAMPLES = (
    TutorialSample(
        "s1d1",
        "https://ndownloader.figshare.com/files/40347877",
        "322c30a7a4905f7f113472442d4aa2c81a1ad736c86651f6c0b81e5b2ff94ac8",
    ),
    TutorialSample(
        "s1d3",
        "https://ndownloader.figshare.com/files/40347880",
        "ca40b287fac57ac2048f11f56f1b630c7c188718908fc217c6baf2e49bdb1982",
    ),
)
_DOWNLOAD_TIMEOUT_SECONDS = 300
_DOWNLOAD_ATTEMPTS = 3
_CHUNK_BYTES = 1 << 20


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_tutorial_samples(cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    """Return the verified tutorial samples, downloading any that are missing.

    Each download goes to a temporary file in ``cache_dir`` and is renamed into place
    only after its SHA-256 matches, so an interrupted download never poisons the cache;
    a cached file that fails the check is deleted and downloaded again.

    Parameters
    ----------
    cache_dir : Path, default=CACHE_DIR
        Directory that holds the files (``<repo>/.cache/fixtures``).

    Returns
    -------
    dict of str to Path
        Sample name (``"s1d1"``, ``"s1d3"``, in that order) to its ``.h5`` file.

    Raises
    ------
    ValueError
        If a downloaded file has another SHA-256.
    urllib.error.URLError
        If all download attempts of a file fail (no network, server error).
    """
    paths: dict[str, Path] = {}
    for sample in TUTORIAL_SAMPLES:
        path = cache_dir / sample.file_name
        if path.exists() and _file_sha256(path) != sample.sha256:
            path.unlink()
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            _download(sample, path)
        actual = _file_sha256(path)
        if actual != sample.sha256:
            raise ValueError(
                f"fetch_tutorial_samples: {path} has SHA-256 {actual}, "
                f"expected {sample.sha256}."
            )
        paths[sample.name] = path
    return paths


def _download(sample: TutorialSample, path: Path) -> None:
    """Download one sample to ``path``, retrying transient network failures."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_once(sample, path)
            return
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ):
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def _download_once(sample: TutorialSample, path: Path) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    temporary = Path(temporary_name)
    request = urllib.request.Request(
        sample.url, headers={"User-Agent": "kyto-test-fixtures"}
    )
    try:
        with (
            os.fdopen(handle, "wb") as output,
            urllib.request.urlopen(
                request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                output.write(chunk)
        actual = _file_sha256(temporary)
        if actual != sample.sha256:
            raise ValueError(
                f"fetch_tutorial_samples: download of {sample.name} from {sample.url} "
                f"has SHA-256 {actual}, expected {sample.sha256}. The source changed "
                "or the transfer was corrupted; retry, and if it persists, check the "
                "URL."
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
