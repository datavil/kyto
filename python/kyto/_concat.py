"""``kyto.concat``: combine several samples into one store (task 2.9).

What it does: checks nothing itself; passes the arguments to the Rust writer
(``kyto_store::concat``), which streams every source into a new store.

What it does not: build a lazy plan over several stores; the combined store is
written once.
"""

import os
from collections.abc import Sequence
from typing import Literal

from kyto import _kyto


def concat(
    sources: Sequence[str | os.PathLike[str]],
    target: str | os.PathLike[str],
    *,
    label: str = "sample",
    keys: Sequence[str] | None = None,
    join: Literal["inner", "outer"] = "inner",
    layer: str = "counts",
    on_duplicate_ids: Literal["error", "suffix"] = "error",
    overwrite: bool = False,
) -> None:
    """Combine several samples into a new kyto store, as ``anndata.concat``.

    The cells of every source are written one after the other, in source order, one
    cell block at a time; nothing exists at ``target`` unless the whole write succeeds.
    The result equals ``anndata.concat(adatas, label=label, keys=keys, join=join,
    merge="same")`` on the sources read by anndata, with the differences listed in
    ``deviations.md``: gene columns equal in every source are kept (anndata's default
    ``merge=None`` keeps none), integer cell columns missing in a source hold nulls
    (anndata casts them to float with ``NaN``), and the importer's own differences
    (see ``kyto.convert``).

    Parameters
    ----------
    sources : Sequence of str or os.PathLike[str]
        The samples: kyto stores, or any input ``kyto.convert`` reads (``.h5ad``,
        10x ``.h5``, a 10x matrix directory or its ``.mtx`` file), mixed freely.
    target : str or os.PathLike[str]
        Path of the new store (a directory). It may not be a source.
    label : str, default="sample"
        Name of the cell column naming each cell's source, an Enum whose categories
        are ``keys`` in order. A column of that name in the sources is replaced.
    keys : Sequence of str or None, default=None
        The label of each source, one per source, all different. ``None`` takes the
        source names: the file or directory name without ``.h5ad``, ``.h5``,
        ``.mtx``, ``.mtx.gz``, ``.kyto`` or ``.zarr``.
    join : {"inner", "outer"}, default="inner"
        Genes are matched by ``gene_id``. ``"inner"`` keeps the genes of every source,
        in the first source's order; ``"outer"`` the genes of any source, sorted by ID
        unless all sources have the same genes in the same order (as pandas). A gene
        a source lacks has no values in its cells (zeros, R9). Layers and cell columns
        are joined the same way: a source without a layer gives no values, a source
        without a cell column nulls.
    layer : str, default="counts"
        Layer name of an input's main matrix (``X``, or the 10x counts), as in
        ``kyto.convert``. Stores keep their layer names.
    on_duplicate_ids : {"error", "suffix"}, default="error"
        A cell ID in more than one source: ``"error"`` refuses before any matrix is
        read; ``"suffix"`` appends ``-1``, ``-2``, ... to later occurrences, as
        ``AnnData.obs_names_make_unique`` after ``anndata.concat``. Inputs apply it to
        their own repeated IDs too, as ``kyto.convert`` does.
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
        ``KY-E101``: a cell ID in more than one source with
        ``on_duplicate_ids="error"``.
    StoreInputError
        ``KY-E104`` for no sources, sources with layers but no layer name in common
        (``join="inner"``), a ``keys`` of another length, repeated keys (or
        sources with the same name and no ``keys``), a ``label`` of ``"cell_id"``, or
        a bad ``join``/``on_duplicate_ids``; ``KY-E112`` for a cell column whose
        dtypes cannot be combined (text or booleans in one source, numbers in
        another);
        ``KY-E118`` if ``target`` exists and may not be replaced, or is a source.
    InputFileError
        ``KY-E120``/``KY-E121`` for an input ``kyto.convert`` refuses.
    StoreIOError
        ``KY-E110``: a source is missing or unreadable, or the store cannot be
        written.

    Notes
    -----
    Memory: the cell and gene tables of every source and of the result, and one
    cell block per layer of the source being read (plus an input's own buffers, as
    in ``kyto.convert``). ``uint32`` and ``float32`` layers of the same name give a
    ``float32`` layer. Embeddings and graphs of store sources are not carried over.

    Examples
    --------
    >>> import json, tempfile
    >>> from pathlib import Path
    >>> import polars as pl
    >>> import kyto as ky
    >>> folder = Path(tempfile.mkdtemp())
    >>> for name, genes in [("a", "ENSG01\\tCD3E\\nENSG02\\tCD4\\n"), ("b", "ENSG02\\tCD4\\n")]:
    ...     sample = folder / name
    ...     sample.mkdir()
    ...     n_genes = genes.count("\\n")
    ...     _ = (sample / "matrix.mtx").write_text(
    ...         "%%MatrixMarket matrix coordinate integer general\\n"
    ...         f"{n_genes} 1 1\\n{n_genes} 1 7\\n"
    ...     )
    ...     _ = (sample / "barcodes.tsv").write_text(f"{name}-AAAC-1\\n")
    ...     _ = (sample / "genes.tsv").write_text(genes)
    >>> ky.concat([folder / "a", folder / "b"], folder / "both.kyto", join="outer")
    >>> schema = json.loads((folder / "both.kyto" / "schema.json").read_text())
    >>> schema["n_cells"], schema["n_genes"]
    (2, 2)
    >>> pl.read_parquet(folder / "both.kyto" / "cells.parquet")["sample"].to_list()
    ['a', 'b']
    """
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]  # one path, not one per character
    _kyto.concat(
        [os.fspath(source) for source in sources],
        os.fspath(target),
        label=label,
        # A bare string is one key, not one per character.
        keys=None if keys is None else [keys] if isinstance(keys, str) else list(keys),
        join=join,
        layer=layer,
        on_duplicate_ids=on_duplicate_ids,
        overwrite=overwrite,
    )
