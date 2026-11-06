"""Load stored scanpy reference outputs, checked against their manifest.

What it does: reads ``fixtures/reference/manifest.json`` and the ``.npz`` file of one
(fixture, step), and checks that every array has the dtype, shape and digest recorded
when it was generated and that it was generated from the current fixture.

What it does not: generate references (see :mod:`fixtures.make_reference`) or need
scanpy; loading uses numpy only.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fixtures.hashing import array_digest
from fixtures.registry import EXPECTED_DIGESTS, FixtureName

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
MANIFEST_PATH = REFERENCE_DIR / "manifest.json"

JsonObject = dict[str, object]
_REGENERATE = (
    "Regenerate with `uv run --directory tests python -m fixtures.make_reference`."
)


def array_entry(name: str, array: NDArray[np.generic]) -> dict[str, object]:
    """Return the manifest record of one array: little-endian dtype, shape, digest.

    Parameters
    ----------
    name : str
        Array name inside its reference file (part of the digest).
    array : NDArray[np.generic]
        The array.

    Returns
    -------
    dict[str, object]
        Keys ``dtype`` (for example ``"<f4"``), ``shape`` (list of int), ``digest``.
    """
    return {
        "dtype": array.dtype.newbyteorder("<").str,
        "shape": list(array.shape),
        "digest": array_digest({name: array}),
    }


def load_manifest() -> JsonObject:
    """Return the parsed ``manifest.json``.

    Returns
    -------
    dict[str, object]
        Keys ``generator``, ``versions`` and ``references`` (see
        :class:`fixtures.make_reference.Manifest`). A new object on every call.
    """
    return cast(JsonObject, json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def reference_keys() -> list[str]:
    """Return every stored reference as ``"<fixture>/<step>"``, sorted.

    Returns
    -------
    list[str]
        For example ``["pbmc3k/hvg", ..., "synthetic/rank_genes"]``.
    """
    return sorted(cast(JsonObject, load_manifest()["references"]))


def load_reference(fixture: FixtureName, step: str) -> dict[str, NDArray[np.generic]]:
    """Return the stored scanpy outputs of one step on one fixture, verified.

    Parameters
    ----------
    fixture : {"synthetic", "pbmc3k"}
        Fixture the reference was computed on.
    step : str
        Step name, for example ``"qc"``, ``"hvg"`` or ``"pca"`` (see
        ``manifest.json`` for the parameters of each).

    Returns
    -------
    dict[str, NDArray[np.generic]]
        Arrays by name, newly loaded (the caller owns them).

    Raises
    ------
    KeyError
        If there is no reference for ``fixture/step``.
    ValueError
        If the reference was generated from another version of the fixture, or an
        array's names, dtype, shape or digest differ from the manifest. Regenerate
        with ``uv run --directory tests python -m fixtures.make_reference``.

    Examples
    --------
    >>> qc = load_reference("synthetic", "qc")
    >>> qc["cells/total_counts"].shape
    (500,)
    """
    key = f"{fixture}/{step}"
    references = cast(JsonObject, load_manifest()["references"])
    if key not in references:
        raise KeyError(
            f"load_reference: no reference {key!r}; available: {sorted(references)}."
        )
    entry = cast(JsonObject, references[key])
    if entry["fixture_digest"] != EXPECTED_DIGESTS[fixture]:
        raise ValueError(
            f"load_reference: {key} was generated from fixture digest "
            f"{entry['fixture_digest']}, but {fixture!r} now has "
            f"{EXPECTED_DIGESTS[fixture]}. {_REGENERATE}"
        )
    expected = cast(dict[str, dict[str, object]], entry["arrays"])
    with np.load(
        REFERENCE_DIR / cast(str, entry["file"]), allow_pickle=False
    ) as archive:
        arrays: dict[str, NDArray[np.generic]] = {
            name: archive[name] for name in archive.files
        }
    if sorted(arrays) != sorted(expected):
        raise ValueError(
            f"load_reference: {key} holds arrays {sorted(arrays)}, manifest lists "
            f"{sorted(expected)}. {_REGENERATE}"
        )
    for name, array in arrays.items():
        recorded = expected[name]
        actual = array_entry(name, array)
        if actual != recorded:
            raise ValueError(
                f"load_reference: {key} array {name!r} is {actual}, manifest records "
                f"{recorded}. {_REGENERATE}"
            )
    return arrays
