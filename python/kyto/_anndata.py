"""``Atlas.to_anndata``'s reading side (task 7.8): an ``.h5ad`` written by kyto, read
with anndata under scanpy's names.

What it does: reads the temporary ``.h5ad`` the terminal wrote (2.7's export of the
plan's output), renames the ``obsm`` keys as the Rust side decided (``pca`` to
``X_pca``) and, if asked, names the genes by a text gene column, made unique as
``AnnData.var_names_make_unique`` does.

What it does not: run plans, check arguments (the Rust side does, with coded errors),
or import anndata at kyto's import (anndata is an optional dependency, ``kyto[plot]``).
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata

# anndata and pandas ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false


def read(
    path: str, obsm_keys: dict[str, str], var_names: str | None
) -> "anndata.AnnData":
    """Read the ``.h5ad`` at ``path`` into memory under scanpy's names.

    Parameters
    ----------
    path : str
        The ``.h5ad`` file kyto wrote.
    obsm_keys : dict of str to str
        The new ``obsm`` key of each cell embedding (``{"pca": "X_pca"}``).
    var_names : str or None
        A text gene column to name the genes by; ``None`` keeps the gene IDs.

    Returns
    -------
    anndata.AnnData
        A new object read from the file (the file can be deleted afterwards).

    Examples
    --------
    >>> import anndata, numpy as np, pandas as pd, tempfile
    >>> from pathlib import Path
    >>> var = pd.DataFrame(
    ...     {"symbol": ["A", "A", None]},
    ...     index=pd.Index(["G1", "G2", "G3"], name="gene_id"),
    ... )
    >>> source = anndata.AnnData(np.zeros((1, 3), np.float32), var=var)
    >>> source.obsm["pca"] = np.zeros((1, 2), np.float32)
    >>> path = Path(tempfile.mkdtemp()) / "x.h5ad"
    >>> source.write_h5ad(path)
    >>> adata = read(str(path), {"pca": "X_pca"}, "symbol")
    >>> list(adata.var_names), list(adata.obsm)
    (['A', 'A-1', 'G3'], ['X_pca'])
    """
    import anndata
    import pandas as pd

    adata = anndata.read_h5ad(path)
    adata.obsm = {obsm_keys.get(key, key): adata.obsm[key] for key in adata.obsm}
    if var_names is not None and var_names != adata.var.index.name:
        gene_ids = adata.var.index.astype(str)
        values = adata.var[var_names]
        labels = [
            gene_id if pd.isna(value) else str(value)
            for gene_id, value in zip(gene_ids, values, strict=True)
        ]
        adata.var.insert(0, str(adata.var.index.name or "gene_id"), list(gene_ids))
        # anndata warns on repeated names until they are made unique, just below,
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not unique", category=UserWarning
            )
            # And on suffixes it picks: kyto's docstring states the rule instead.
            warnings.filterwarnings(
                "ignore", message="Suffix used", category=UserWarning
            )
            adata.var_names = labels
            adata.var_names_make_unique()
        adata.var.index.name = None
    return adata
