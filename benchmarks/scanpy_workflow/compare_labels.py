"""Compare kyto's and scanpy's workflow outputs: cells kept, Leiden clusters, ARI (10.8).

Usage: ``compare_labels.py SCANPY.h5ad KYTO_STORE``. Run in an environment with both
kyto and scanpy (the dev environment). Prints one JSON object.
"""

# anndata and pandas ship without complete type information (as in tests/).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import sys

import anndata
import kyto as ky
import numpy as np
from sklearn.metrics import adjusted_rand_score


def compare(scanpy_path: str, kyto_path: str) -> dict[str, object]:
    """Align both outputs by cell ID and compare their Leiden labels.

    Parameters
    ----------
    scanpy_path : str
        scanpy's output ``.h5ad``.
    kyto_path : str
        kyto's output store.

    Returns
    -------
    dict[str, object]
        Cells and clusters on each side, shared cells, ARI and cluster sizes.
    """
    reference = anndata.read_h5ad(scanpy_path, backed="r")
    scanpy_labels = dict(
        zip(reference.obs_names, reference.obs["leiden"].astype(str), strict=True)
    )
    cells = ky.scan(kyto_path).cells.to_polars()
    kyto_labels = dict(
        zip(
            cells["cell_id"].to_list(), cells["leiden"].cast(str).to_list(), strict=True
        )
    )
    shared = [cell for cell in kyto_labels if cell in scanpy_labels]
    ari = adjusted_rand_score(
        [scanpy_labels[cell] for cell in shared], [kyto_labels[cell] for cell in shared]
    )
    sizes = {
        "scanpy": sorted(
            np.unique(list(scanpy_labels.values()), return_counts=True)[1].tolist(),
            reverse=True,
        ),
        "kyto": sorted(
            np.unique(list(kyto_labels.values()), return_counts=True)[1].tolist(),
            reverse=True,
        ),
    }
    return {
        "cells": {
            "scanpy": len(scanpy_labels),
            "kyto": len(kyto_labels),
            "shared": len(shared),
        },
        "clusters": {"scanpy": len(sizes["scanpy"]), "kyto": len(sizes["kyto"])},
        "ari": round(float(ari), 4),
        "cluster_sizes": sizes,
    }


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2])))
