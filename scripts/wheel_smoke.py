"""Smoke-test an installed kyto wheel: import it and run a small pipeline (task 10.6).

Does: import ``kyto`` from the environment that runs this script (never from the
source tree), convert a 10x Matrix Market directory into a store, and run QC, filters,
``normalize_total``, ``log1p``, highly variable genes, PCA, neighbours and Leiden on
it, checking the shapes of what comes back. Without ``--matrix-dir`` it writes a small
synthetic directory first (no download, no dependency beyond the wheel's own).

Does not: compare results with scanpy (the parity tests do), or build anything.

Usage (from any directory, with the interpreter of the environment under test; ``-I``
keeps the source tree's ``python/`` off the import path)::

    python -I scripts/wheel_smoke.py
    python -I scripts/wheel_smoke.py --matrix-dir filtered_gene_bc_matrices/hg19 \\
        --expect-cells 2638
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

N_CELLS = 300
N_GENES = 120


def _write_synthetic(directory: Path) -> None:
    """Write a 10x (Cell Ranger v2) Matrix Market directory with fixed counts.

    Parameters
    ----------
    directory : Path
        The directory to create; it receives ``matrix.mtx``, ``genes.tsv`` and
        ``barcodes.tsv``.
    """
    directory.mkdir(parents=True)
    state = 12345
    entries: list[tuple[int, int, int]] = []
    for cell in range(N_CELLS):
        for gene in range(N_GENES):
            # A linear congruential generator: the same counts on every platform.
            state = (1103515245 * state + 12345) % 2**31
            if state % 3 == 0:
                # Two cell groups that differ in their first 20 genes.
                boost = 6 if (cell < N_CELLS // 2) == (gene < 20) else 1
                entries.append((gene + 1, cell + 1, 1 + state % 5 * boost))
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        f"{N_GENES} {N_CELLS} {len(entries)}",
        *(f"{gene} {cell} {count}" for gene, cell, count in entries),
    ]
    (directory / "matrix.mtx").write_text("\n".join(lines) + "\n", encoding="utf-8")
    symbols = [f"MT-G{gene}" if gene < 3 else f"GENE{gene}" for gene in range(N_GENES)]
    (directory / "genes.tsv").write_text(
        "".join(f"ENSG{gene:011d}\t{symbols[gene]}\n" for gene in range(N_GENES)),
        encoding="utf-8",
    )
    (directory / "barcodes.tsv").write_text(
        "".join(f"CELL{cell:05d}-1\n" for cell in range(N_CELLS)), encoding="utf-8"
    )


def run(matrix_dir: Path, workdir: Path, synthetic: bool) -> int:
    """Convert ``matrix_dir`` and run the pipeline; return the cells kept.

    Parameters
    ----------
    matrix_dir : Path
        A 10x Matrix Market directory.
    workdir : Path
        Where the store is written.
    synthetic : bool
        Whether ``matrix_dir`` is the small synthetic one (smaller thresholds).

    Returns
    -------
    int
        The number of cells after the QC filter.
    """
    import kyto as ky
    from kyto import cell_column, gene_column, qc

    store = workdir / "smoke.kyto"
    ky.convert(str(matrix_dir), str(store), layer="counts")
    atlas = ky.scan(str(store)).with_default_layer("counts")
    min_genes, max_genes, n_top, n_components = (
        (10, N_GENES + 1, 40, 5) if synthetic else (200, 2500, 2000, 50)
    )
    mito = gene_column("symbol").str.starts_with("MT-")
    plan = (
        atlas.with_cells(
            n_genes_by_counts=qc.genes_detected(),
            pct_counts_mt=qc.percent_counts_in(mito),
        )
        .filter(
            (cell_column("n_genes_by_counts") >= min_genes)
            & (cell_column("n_genes_by_counts") < max_genes)
            & (cell_column("pct_counts_mt") < (100 if synthetic else 5))
        )
        .normalize_total(1e4)
        .log1p()
        .with_default_layer("log1p")
        .highly_variable_genes(n_top_genes=n_top)
        .filter(gene_column("highly_variable"))
        .pca(n_components=n_components, seed=0)
        .neighbors()
        .leiden(seed=0)
    )
    result = plan.collect()
    n_cells = result.count_cells()
    components = result.embedding("X_pca").to_numpy()
    if components.shape != (n_cells, n_components):
        raise SystemExit(
            f"X_pca has shape {components.shape}, expected ({n_cells}, {n_components})"
        )
    clusters = result.cells.to_polars()["leiden"]
    if clusters.len() != n_cells or clusters.null_count() != 0:
        raise SystemExit("leiden does not label every kept cell")
    print(
        f"kyto {ky.__version__} from {Path(ky.__file__).parent}: {n_cells} cells, "
        f"{result.count_genes()} genes, {clusters.n_unique()} Leiden clusters"
    )
    return n_cells


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, run the smoke test and return the exit status.

    Parameters
    ----------
    argv : list of str or None, default=None
        The arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 on success, 1 if ``--expect-cells`` does not match.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "kyto").splitlines()[0])
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        help="a 10x Matrix Market directory (default: a small synthetic one)",
    )
    parser.add_argument(
        "--expect-cells", type=int, help="fail unless the QC filter keeps this many"
    )
    arguments = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(
        prefix="kyto-smoke-", ignore_cleanup_errors=True
    ) as scratch:
        workdir = Path(scratch)
        matrix_dir: Path | None = arguments.matrix_dir
        synthetic = matrix_dir is None
        if matrix_dir is None:
            matrix_dir = workdir / "matrix"
            _write_synthetic(matrix_dir)
        n_cells = run(matrix_dir, workdir, synthetic)
    expected: int | None = arguments.expect_cells
    if expected is not None and n_cells != expected:
        print(f"expected {expected} cells after QC, got {n_cells}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
