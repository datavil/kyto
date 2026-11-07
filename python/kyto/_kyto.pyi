"""Type stub for the compiled extension module `kyto._kyto`."""

import os
from typing import final

__version__: str

@final
class Atlas:
    """A lazy, immutable plan over a kyto store.

    An ``Atlas`` records what to compute; it reads no matrix data. Every method returns
    a new ``Atlas`` and leaves the one it was called on unchanged (R2), so an ``Atlas``
    can be shared and reused freely. Plans built from one another share their common
    part; nothing is copied.

    Examples
    --------
    >>> atlas = scan(store)
    >>> counts = atlas.with_default_layer("counts")
    >>> print(atlas)
    SCAN "..." (2 cells x 3 genes)
    """

    def with_default_layer(self, layer: str) -> Atlas:
        """Set the layer that short gene names refer to.

        Parameters
        ----------
        layer : str
            Name of a layer of the store, as stored (case-sensitive).

        Returns
        -------
        Atlas
            A new plan; this one is unchanged. Nothing is read or copied.

        Raises
        ------
        MissingMemberError
            ``KY-E109``: the store has no layer named ``layer``.

        Examples
        --------
        >>> atlas = scan(store)
        >>> print(atlas.with_default_layer("counts"))
        WITH_DEFAULT_LAYER "counts"
        SCAN "..." (2 cells x 3 genes)
        """

def scan(store: str | os.PathLike[str], /) -> Atlas:
    """Build the root `Atlas` of the local store at `store` (private until 7.1)."""

def raise_error_example(code: str, /) -> bool:
    """Raise the registered example error for `code`; return False if unknown."""

def error_example_text(code: str, /) -> tuple[str, str] | None:
    """Return the Rust message and hint of the example error for `code`."""

def convert(
    source: str,
    target: str,
    /,
    *,
    layer: str,
    on_duplicate_ids: str,
    overwrite: bool,
) -> None:
    """Import `source` into a new store at `target` (see `kyto.convert`)."""

def sink_h5ad(
    store: str,
    target: str,
    /,
    *,
    x_layer: str | None,
    overwrite: bool,
) -> None:
    """Export the store at `store` as an `.h5ad` file at `target` (private, task 2.7)."""
