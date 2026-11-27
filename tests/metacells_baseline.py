"""What a plan without ``metacells`` gave before task 6.16: its snapshot (opt-in check).

Task 6.16's Done-when: a plan without ``metacells`` gives the same ``explain()``,
``estimate()`` and output as before the task. This module builds that plan on pbmc3k
(QC filter, normalize, log1p, HVG, PCA, neighbours, Leiden, UMAP: every kind of sink
member) and records its ``explain()`` text, its ``estimate()`` and the checksums of
its ``sink_zarr`` store (``test_reproducibility``'s, at 4 threads).
``tests/metacells_baseline.json`` was written by this module under the build of
commit 1318b27, the last before 6.16; ``tests/test_metacells.py`` compares the current
build with it. Regenerate only on purpose, when a later task changes these outputs:

    uv run --directory tests python metacells_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import kyto
from fixtures.pbmc import _MEMBER_PREFIX, fetch_pbmc3k  # pyright: ignore[reportPrivateUsage]
from kyto import _kyto

#: The snapshot, next to this module.
BASELINE = Path(__file__).with_name("metacells_baseline.json")
#: Threads of the snapshot's explain, estimate and run.
THREADS = 4
#: Stands for the store's path in the snapshot's text.
STORE = "<store>"


def pbmc3k_store(directory: Path) -> Path:
    """pbmc3k converted from its 10x matrix directory into ``directory``."""
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


def pipeline(store: Path) -> _kyto.Atlas:
    """The plan without ``metacells``."""
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(total=kyto.qc.total_counts())
        .filter(kyto.cell_column("total") >= 1000)
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .highly_variable_genes(layer="log1p", n_top_genes=1000)
        .filter(kyto.gene_column("highly_variable"))
        .pca(layer="log1p", into="X_pca", n_components=30, seed=0)
        .neighbors("X_pca", n_neighbors=15, seed=1)
        .leiden(into="leiden", seed=2)
        .umap(into="X_umap", n_epochs=50, seed=3)
    )


def checksums(store: Path, scanned: Path | None = None) -> dict[str, str]:
    """SHA-256 of every file of ``store`` by relative path, ``provenance.json``
    without ``run`` and ``source.path`` (as ``test_reproducibility``), without
    ``kyto_version`` (a release changes it, not the plan: 10.6 moved it from 0.0.0)
    and with the
    ``scanned`` store's path in its plan as ``<store>``."""
    found: dict[str, str] = {}
    for path in sorted(store.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        name = path.relative_to(store).as_posix()
        if name == "provenance.json":
            provenance = json.loads(data)
            del provenance["run"]
            del provenance["kyto_version"]
            if provenance["source"] is not None:
                del provenance["source"]["path"]
            text = json.dumps(provenance)
            if scanned is not None:
                text = text.replace(json.dumps(str(scanned))[1:-1], STORE)
            data = text.encode()
        found[name] = hashlib.sha256(data).hexdigest()
    return found


def snapshot(store: Path, output: Path) -> dict[str, Any]:
    """The plan's explain text and estimate (the store path as ``<store>``) and its
    sunk store's checksums."""
    plan = pipeline(store)
    plan.sink_zarr(output, threads=THREADS)
    estimate = json.loads(json.dumps(plan.estimate(threads=THREADS)))
    return {
        "explain": plan.explain(threads=THREADS).replace(str(store), STORE),
        "estimate": json.loads(json.dumps(estimate).replace(str(store), STORE)),
        "checksums": checksums(output, store),
    }


def main() -> None:
    """Write the snapshot of the current build to ``BASELINE`` (or ``argv[1]``)."""
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else BASELINE
    with tempfile.TemporaryDirectory() as directory:
        store = pbmc3k_store(Path(directory))
        found = snapshot(store, Path(directory) / "out.kyto")
    target.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
