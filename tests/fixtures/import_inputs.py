"""Importer inputs and the values anndata and scanpy read from them (task 2.6).

What it does: writes every fixture in every input format ``kyto.convert`` reads
(``.h5ad`` with CSR, CSC and dense ``X``, 10x ``.h5`` v2 and v3, 10x Matrix Market
directories v2 and v3), reads each back with anndata or scanpy, and writes what kyto
must store: per layer the canonical CSR (``<layer>.indptr`` u64, ``.indices`` u32,
``.data`` f32 or u32, little-endian), the tables (``cells.parquet``,
``genes.parquet``, Enum columns as strings) and the ``schema.json`` column entries
(plus each Enum's category list).
A ``manifest.json`` lists the cases. The Rust test
``crates/kyto-store/tests/import_parity.rs`` converts every case and compares.

Run as ``python -m fixtures.import_inputs <out_dir>`` from ``tests/``. Exits with code
77 when anndata or scanpy is not installed (platforms without wheels).

What it does not: run kyto. The reference is anndata's reader, not this writer:
expected values always come from reading the written file back.
"""

# pandas, scipy, h5py and anndata ship without complete type information; strict
# pyright cannot see their types, so the unknown-type checks are relaxed in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportAttributeAccessIssue=false

import gzip
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fixtures.dataset import Dataset
from fixtures.registry import FIXTURE_NAMES, load_fixture

SKIP_EXIT_CODE = 77


def _canonical_csr(matrix: Any) -> Any:
    from scipy.sparse import csr_matrix

    csr = csr_matrix(matrix, copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    return csr


def _write_layer(directory: Path, name: str, matrix: Any) -> dict[str, str]:
    """Write the canonical CSR kyto must store for ``matrix``; return its dtype entry."""
    csr = _canonical_csr(matrix)
    integer = np.issubdtype(csr.dtype, np.integer)
    data = csr.data.astype("<u4" if integer else "<f4")
    (directory / f"{name}.indptr").write_bytes(csr.indptr.astype("<u8").tobytes())
    (directory / f"{name}.indices").write_bytes(csr.indices.astype("<u4").tobytes())
    (directory / f"{name}.data").write_bytes(data.tobytes())
    return {"name": name, "dtype": "uint32" if integer else "float32"}


def _objects(series: Any) -> list[Any]:
    """Values of a pandas series as Python objects, missing values as ``None``."""
    import pandas as pd

    return [None if pd.isna(value) else value for value in series.astype(object)]


def _column_entry(name: str, series: Any) -> tuple[pl.Series, dict[str, Any]]:
    """The expected kyto column and schema entry of one pandas column."""
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        if categories.dtype.kind in "OUS" or isinstance(
            categories.dtype, pd.StringDtype
        ):
            values = pl.Series(name, _objects(series), dtype=pl.String)
            entry = {
                "name": name,
                "dtype": "Enum",
                "n_categories": len(categories),
                "ordered": bool(dtype.ordered),
                "categories": [str(category) for category in categories],
            }
            return values.cast(pl.String), entry
        numeric = pl.Series(name, np.asarray(categories))
        codes = np.asarray(series.cat.codes)
        values = numeric.gather(pl.Series(np.where(codes < 0, 0, codes)))
        values = pl.select(
            pl.when(pl.Series(codes >= 0)).then(values).otherwise(None).alias(name)
        ).to_series()
        return values, {"name": name, "dtype": str(values.dtype)}
    if dtype.kind in "OUS" or isinstance(dtype, pd.StringDtype):
        values = pl.Series(name, _objects(series), dtype=pl.String)
        # kyto stores plain strings as an Enum of the sorted distinct values.
        categories = sorted(set(values.drop_nulls().to_list()))
        entry = {
            "name": name,
            "dtype": "Enum",
            "n_categories": len(categories),
            "ordered": False,
            "categories": categories,
        }
        return values.cast(pl.String), entry
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        # Nullable Int64 / boolean: no pyarrow, so go through Python objects.
        polars_dtype = pl.Series(np.array([], dtype=dtype.numpy_dtype)).dtype
        values = pl.Series(name, _objects(series), dtype=polars_dtype)
    else:
        values = pl.Series(name, series.to_numpy())
    return values, {"name": name, "dtype": str(values.dtype)}


def _write_table(
    directory: Path, file_name: str, id_name: str, ids: Any, frame: Any
) -> list[dict[str, Any]]:
    columns = [pl.Series(id_name, [str(value) for value in ids], dtype=pl.String)]
    entries: list[dict[str, Any]] = [{"name": id_name, "dtype": "String"}]
    for name in frame.columns:
        values, entry = _column_entry(str(name), frame[name])
        columns.append(values)
        entries.append(entry)
    pl.DataFrame(columns).write_parquet(directory / file_name)
    return entries


def _expected(
    directory: Path,
    layers: dict[str, Any],
    cell_ids: Any,
    cells: Any,
    gene_ids: Any,
    genes: Any,
) -> None:
    directory.mkdir(parents=True)
    schema = {
        "cells": _write_table(directory, "cells.parquet", "cell_id", cell_ids, cells),
        "genes": _write_table(directory, "genes.parquet", "gene_id", gene_ids, genes),
        "layers": [
            _write_layer(directory, name, matrix) for name, matrix in layers.items()
        ],
    }
    (directory / "schema.json").write_text(json.dumps(schema, indent=1))


def _with_extra_columns(adata: Any) -> Any:
    """Add one obs and var column of every encoding anndata writes (synthetic only)."""
    import pandas as pd

    n_cells, n_genes = adata.shape
    rows = np.arange(n_cells)
    obs = adata.obs
    obs["int16"] = (rows % 7).astype(np.int16)
    obs["uint8"] = (rows % 200).astype(np.uint8)
    obs["float64"] = np.where(rows % 5 == 0, np.nan, rows / 3)
    obs["float32"] = (rows / 7).astype(np.float32)
    obs["flag"] = rows % 2 == 0
    obs["nullable_int"] = pd.array(
        [None if row % 4 == 0 else row for row in rows], dtype="Int64"
    )
    obs["nullable_bool"] = pd.array(
        [None if row % 3 == 0 else row % 2 == 0 for row in rows], dtype="boolean"
    )
    obs["stage"] = pd.Categorical(
        [["late", None, "early"][row % 3] for row in rows],
        categories=["early", "mid", "late"],
        ordered=True,
    )
    obs["cluster"] = pd.Categorical(rows % 3 * 10)
    adata.var["highly_variable"] = np.arange(n_genes) % 3 == 0
    adata.var["dispersion"] = np.arange(n_genes, dtype=np.float32) / 2
    return adata


def _h5ad_cases(dataset: Dataset, out: Path) -> list[dict[str, Any]]:
    import anndata as ad
    from scipy.sparse import csc_matrix

    from fixtures.interop import to_anndata

    cases: list[dict[str, Any]] = []
    base: Any = to_anndata(dataset)
    if dataset.name == "synthetic":
        base = _with_extra_columns(base)
        base.layers["counts_int"] = base.X.astype(np.int32)
        base.layers["dense_float64"] = base.X.toarray().astype(np.float64)
    variants: dict[str, tuple[Any, Literal["gzip", "lzf"] | None]] = {
        "csr": (base.X, None),
        "csc": (csc_matrix(base.X), "gzip"),
        "dense": (base.X.toarray(), "lzf"),
    }
    for variant, (matrix, compression) in variants.items():
        # Layers only in the CSR file; the others hold X alone.
        adata = base if variant == "csr" else ad.AnnData(matrix, base.obs, base.var)
        path = out / "inputs" / f"{dataset.name}_{variant}.h5ad"
        adata.write_h5ad(path, compression=compression)
        read: Any = ad.read_h5ad(path)
        # anndata 0.13 lists X as layers[None]; kyto stores it as `counts`.
        names = sorted(name for name in read.layers if name is not None)
        layers = {"counts": read.X} | {name: read.layers[name] for name in names}
        case = f"{dataset.name}_h5ad_{variant}"
        _expected(
            out / "expected" / case,
            layers,
            read.obs_names,
            read.obs,
            read.var_names,
            read.var,
        )
        cases.append({"case": case, "input": str(path), "kind": "h5ad"})
    return cases


def _fixed_strings(values: NDArray[np.str_]) -> NDArray[np.bytes_]:
    return np.array([value.encode() for value in values], dtype=np.bytes_)


def _tenx_cases(dataset: Dataset, out: Path) -> list[dict[str, Any]]:
    import h5py
    import scanpy as sc

    cases: list[dict[str, Any]] = []
    n_cells, n_genes = dataset.n_cells, dataset.n_genes
    counts = dataset.data.astype(np.int32)
    barcodes = _fixed_strings(dataset.cells["cell_id"])
    gene_ids = _fixed_strings(dataset.genes["gene_id"])
    symbols = _fixed_strings(dataset.genes["symbol"])
    gzip_options: dict[str, Any] = {"compression": "gzip", "compression_opts": 4}
    for version in ("v2", "v3"):
        path = out / "inputs" / f"{dataset.name}_10x_{version}.h5"
        with h5py.File(path, "w") as file:
            group = file.create_group("matrix" if version == "v3" else "GRCh38")
            group.create_dataset("barcodes", data=barcodes, **gzip_options)
            group.create_dataset("data", data=counts, **gzip_options)
            group.create_dataset("indices", data=dataset.indices.astype(np.int64))
            group.create_dataset("indptr", data=dataset.indptr.astype(np.int64))
            group.create_dataset("shape", data=np.array([n_genes, n_cells], np.int32))
            if version == "v3":
                features = group.create_group("features")
                features.create_dataset("id", data=gene_ids, **gzip_options)
                features.create_dataset("name", data=symbols, **gzip_options)
                kinds = np.full(n_genes, b"Gene Expression", dtype="S16")
                kinds[-2:] = b"Antibody Capture"
                features.create_dataset("feature_type", data=kinds)
                features.create_dataset(
                    "genome", data=np.full(n_genes, b"GRCh38", dtype="S6")
                )
                features.create_dataset(
                    "_all_tag_keys", data=np.array([b"genome"], dtype="S6")
                )
            else:
                group.create_dataset("genes", data=gene_ids)
                group.create_dataset("gene_names", data=symbols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read: Any = sc.read_10x_h5(path, gex_only=False)
        var = read.var.drop(columns=["gene_ids"])
        var.insert(0, "symbol", np.asarray(read.var_names, dtype=object))
        var = var.rename(columns={"feature_types": "feature_type"})
        case = f"{dataset.name}_10x_{version}"
        _expected(
            out / "expected" / case,
            {"counts": read.X.astype(np.int64)},
            read.obs_names,
            read.obs,
            read.var["gene_ids"],
            var,
        )
        cases.append({"case": case, "input": str(path), "kind": "10x_h5"})
    return cases


def _mtx_cases(dataset: Dataset, out: Path) -> list[dict[str, Any]]:
    import scanpy as sc
    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix

    cases: list[dict[str, Any]] = []
    matrix = csr_matrix(
        (dataset.data.astype(np.int64), dataset.indices, dataset.indptr),
        shape=(dataset.n_cells, dataset.n_genes),
    )
    for version in ("v2", "v3"):
        directory = out / "inputs" / f"{dataset.name}_mtx_{version}"
        directory.mkdir(parents=True)
        suffix = ".gz" if version == "v3" else ""
        mmwrite(directory / "matrix.mtx", matrix.T.tocoo(), field="integer")
        rows = [
            f"{gene_id}\t{symbol}" + ("\tGene Expression" if version == "v3" else "")
            for gene_id, symbol in zip(
                dataset.genes["gene_id"], dataset.genes["symbol"], strict=True
            )
        ]
        texts = {
            "barcodes.tsv": "\n".join(dataset.cells["cell_id"]) + "\n",
            ("features.tsv" if version == "v3" else "genes.tsv"): "\n".join(rows)
            + "\n",
        }
        if version == "v3":
            mtx = directory / "matrix.mtx"
            (directory / "matrix.mtx.gz").write_bytes(gzip.compress(mtx.read_bytes()))
            mtx.unlink()
        for name, text in texts.items():
            target = directory / f"{name}{suffix}"
            if suffix:
                target.write_bytes(gzip.compress(text.encode()))
            else:
                target.write_text(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read: Any = sc.read_10x_mtx(directory, var_names="gene_ids", gex_only=False)
        var = read.var.rename(
            columns={"gene_symbols": "symbol", "feature_types": "feature_type"}
        )
        case = f"{dataset.name}_mtx_{version}"
        _expected(
            out / "expected" / case,
            {"counts": read.X.astype(np.int64)},
            read.obs_names,
            read.obs,
            read.var_names,
            var,
        )
        cases.append({"case": case, "input": str(directory), "kind": "mtx"})
    return cases


def main(out: Path) -> int:
    """Write every case under ``out`` and its ``manifest.json``.

    Parameters
    ----------
    out : Path
        An empty or missing directory.

    Returns
    -------
    int
        0 on success, :data:`SKIP_EXIT_CODE` without anndata or scanpy.
    """
    try:
        import anndata  # noqa: F401  # pyright: ignore[reportUnusedImport]
        import scanpy  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        return SKIP_EXIT_CODE
    (out / "inputs").mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    for name in FIXTURE_NAMES:
        dataset = load_fixture(name)
        cases += _h5ad_cases(dataset, out)
        cases += _tenx_cases(dataset, out)
        cases += _mtx_cases(dataset, out)
    (out / "manifest.json").write_text(json.dumps(cases, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
