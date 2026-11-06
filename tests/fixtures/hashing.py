"""Content digests of arrays that are identical on every OS and architecture.

What it does: hashes the canonical bytes of named arrays (fixed little-endian dtypes,
UTF-8 for strings, shapes included) with SHA-256.

What it does not: hash pickles, ``.npz`` or ``.h5ad`` file bytes. Those depend on the
writer (zip timestamps, HDF5 layout), so the same data could hash differently.
"""

import hashlib
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

# Every numeric array is converted to one of these before hashing, so a platform with a
# different native byte order or default integer width hashes the same values the same.
_CANONICAL_DTYPES: dict[str, str] = {
    "b": "|b1",
    "i": "<i8",
    "u": "<u8",
    "f": "<f8",
}


def _string_bytes(array: NDArray[np.str_]) -> bytes:
    parts: list[bytes] = []
    for value in array.ravel().tolist():
        encoded = str(value).encode("utf-8")
        parts.append(len(encoded).to_bytes(8, "little"))
        parts.append(encoded)
    return b"".join(parts)


def array_digest(arrays: Mapping[str, NDArray[np.generic]]) -> str:
    """Return the SHA-256 hex digest of named arrays, independent of platform.

    Arrays are visited in sorted name order. Each contributes its name, kind, shape and
    values: booleans as one byte, integers widened to little-endian 64-bit, floats
    widened to little-endian float64 (exact for float32), strings as length-prefixed
    UTF-8, every NaN as one bit pattern. Widening means the digest checks values, not
    the storage dtype; check dtypes separately where they matter.

    Parameters
    ----------
    arrays : Mapping[str, NDArray[np.generic]]
        Named arrays. Supported kinds: bool, signed and unsigned integer, float,
        unicode string.

    Returns
    -------
    str
        64 lowercase hex characters.

    Raises
    ------
    TypeError
        If an array has an unsupported dtype kind (object, complex, bytes, ...).

    Examples
    --------
    >>> import numpy as np
    >>> a = array_digest({"x": np.array([1, 2], dtype=np.int32)})
    >>> a == array_digest({"x": np.array([1, 2], dtype=np.int64)})
    True
    >>> a == array_digest({"x": np.array([1, 3], dtype=np.int32)})
    False
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        kind = array.dtype.kind
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(kind.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        if kind == "U":
            digest.update(_string_bytes(array.astype(np.str_)))
        elif kind in _CANONICAL_DTYPES:
            canonical = np.ascontiguousarray(array, dtype=_CANONICAL_DTYPES[kind])
            if kind == "f":
                # NaN payload and sign bits differ between platforms (x86 makes 0/0 a
                # negative NaN, ARM a positive one); hash every NaN as one pattern.
                canonical = np.where(np.isnan(canonical), np.nan, canonical).astype(
                    "<f8"
                )
            digest.update(canonical.tobytes())
        else:
            raise TypeError(
                f"array_digest: array {name!r} has dtype {array.dtype} (kind {kind!r}); "
                "accepted kinds are bool, int, uint, float and unicode string. "
                "Convert it to one of those first."
            )
    return digest.hexdigest()
