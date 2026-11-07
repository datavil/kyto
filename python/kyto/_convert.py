"""``kyto.convert``: import an ``.h5ad``, 10x ``.h5`` or Matrix Market input (task 2.6).

What it does: checks nothing itself; passes the arguments to the Rust importer
(``kyto_store::convert``), which streams the input into a new store.

What it does not: read the store back (that is ``kyto.scan``, task 7.x).
"""

import os
from typing import Literal

from kyto import _kyto


def convert(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    layer: str = "counts",
    on_duplicate_ids: Literal["error", "suffix"] = "error",
    overwrite: bool = False,
) -> None:
    """Import a single-cell matrix file into a new kyto store.

    The matrix is streamed one cell block at a time; nothing exists at ``target``
    unless the whole import succeeds. Values equal what ``anndata.read_h5ad``,
    ``scanpy.read_10x_h5(gex_only=False)`` and
    ``scanpy.read_10x_mtx(var_names="gene_ids", gex_only=False)`` read, with the
    differences listed in ``deviations.md``: every feature is kept, genes are keyed by
    their ID (symbols go to the ``symbol`` column), string columns become categoricals,
    integer matrices become ``uint32`` layers and float matrices ``float32``. Missing
    ``obs``/``var`` values become null: categorical code ``-1``, masked nullable
    entries, and ``NaN`` in a float column (anndata has no nullable float). A ``NaN``
    in a matrix stays a value.

    Parameters
    ----------
    source : str or os.PathLike[str]
        One of: an ``.h5ad`` file written by anndata >= 0.8 (``X`` dense, CSR or CSC;
        ``layers`` and ``obs``/``var`` columns are imported too); a 10x ``.h5`` file from
        Cell Ranger (v2 or v3); a 10x matrix directory holding ``matrix.mtx``,
        ``barcodes.tsv`` and ``features.tsv`` or ``genes.tsv`` (each optionally
        gzipped), or the ``.mtx`` file in it.
    target : str or os.PathLike[str]
        Path of the new store (a directory).
    layer : str, default="counts"
        Layer name of the main matrix (``X``, or the 10x counts). ``.h5ad`` layers keep
        their own names.
    on_duplicate_ids : {"error", "suffix"}, default="error"
        Repeated cell or gene IDs: ``"error"`` refuses the input before any matrix is
        read; ``"suffix"`` appends ``-1``, ``-2``, ... to later occurrences, as
        ``AnnData.obs_names_make_unique`` does.
    overwrite : bool, default=False
        Replace an existing kyto store at ``target`` once the new one is complete.
        Anything other than a kyto store is never replaced.

    Returns
    -------
    None
        The store is written to ``target``; nothing is returned or kept in memory.

    Raises
    ------
    DuplicateIdError
        ``KY-E101``: repeated IDs with ``on_duplicate_ids="error"``.
    InputFileError
        ``KY-E120``: an unknown file kind, encoding or dtype; ``KY-E121``: invalid
        input values or structure, such as a negative count in an integer matrix.
    StoreInputError
        ``KY-E102``/``KY-E103`` for a bad layer name (an ``.h5ad`` that already has a
        ``counts`` layer needs another ``layer``, such as ``layer="X"``), ``KY-E104`` for a bad
        ``on_duplicate_ids``, ``KY-E111``/``KY-E112`` for null IDs or bad columns,
        ``KY-E118`` if ``target`` exists and may not be replaced.
    StoreIOError
        ``KY-E110``: ``source`` is missing or unreadable, or the store cannot be written.

    Notes
    -----
    Memory: the cell and gene tables, one cell block per layer and, for matrices not
    stored by cell (CSC ``X``, Matrix Market), up to 256 MiB of entries per layer;
    such inputs are read ``1 + entries / 256 MiB`` times instead of being loaded whole.
    Dense ``X`` is read in slabs of at most 256 MiB.

    Examples
    --------
    >>> import json, tempfile
    >>> from pathlib import Path
    >>> import kyto as ky
    >>> folder = Path(tempfile.mkdtemp())
    >>> _ = (folder / "matrix.mtx").write_text(
    ...     "%%MatrixMarket matrix coordinate integer general\\n3 2 3\\n"
    ...     "1 1 5\\n3 1 1\\n2 2 7\\n"
    ... )
    >>> _ = (folder / "barcodes.tsv").write_text("AAAC-1\\nAAAG-1\\n")
    >>> _ = (folder / "features.tsv").write_text(
    ...     "ENSG01\\tCD3E\\tGene Expression\\nENSG02\\tCD4\\tGene Expression\\n"
    ...     "ENSG03\\tMS4A1\\tGene Expression\\n"
    ... )
    >>> ky.convert(folder, folder / "pbmc.kyto")
    >>> schema = json.loads((folder / "pbmc.kyto" / "schema.json").read_text())
    >>> schema["n_cells"], schema["n_genes"], schema["layers"]
    (2, 3, [{'name': 'counts', 'dtype': 'uint32', 'nnz': 3}])
    """
    _kyto.convert(
        os.fspath(source),
        os.fspath(target),
        layer=layer,
        on_duplicate_ids=on_duplicate_ids,
        overwrite=overwrite,
    )
