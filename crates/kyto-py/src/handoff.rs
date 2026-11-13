//! Handing data to Python (task 7.3, the part of 7.5's hand-off it needs): Polars frames
//! through Arrow IPC, NumPy arrays through the `numpy` crate, SciPy CSR matrices from
//! those arrays.
//!
//! Does: write a Rust Polars frame as Arrow IPC bytes and read them with Python Polars
//! (`pl.read_ipc`: one copy into Python's buffers, every dtype kept, `Enum` included);
//! move a Rust `Vec` into a NumPy array (no copy: the array owns the buffer); build a
//! `scipy.sparse.csr_matrix` from a layer's CSR (`int32` index arrays when they fit,
//! `int64` otherwise, as SciPy picks), also for graphs; a graph's edge list as a frame;
//! densify CSR row-major, absent entries 0 (R9), only within `max_bytes` (R8,
//! `KY-E402`).
//! Every object returned is new and owned by the caller (R2).
//! Does not: hand over without a copy through `pyo3-polars` or DLPack (7.5, 9.5), or
//! stream: each terminal holds its whole result in memory.

use std::io::Cursor;

use kyto_error::KytoError;
use kyto_store::CsrMatrix;
use numpy::ndarray::Array2;
use numpy::{Element, PyArray1, PyArray2};
use polars::prelude::{Column, DataFrame, IpcWriter, SerWriter};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::errors::to_py_err;

/// `frame` as a new Python `polars.DataFrame` (through Arrow IPC bytes).
pub fn to_polars<'py>(
    py: Python<'py>,
    mut frame: DataFrame,
    store: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let mut bytes = Cursor::new(Vec::new());
    IpcWriter::new(&mut bytes)
        .finish(&mut frame)
        .map_err(|error| {
            let error = KytoError::StoreIo {
                operation: "hand a table to Polars as Arrow IPC".to_owned(),
                path: store.to_owned(),
                detail: error.to_string(),
            };
            to_py_err(py, &error)
        })?;
    let bytes = PyBytes::new(py, bytes.get_ref());
    py.import("polars")?.call_method1("read_ipc", (bytes,))
}

/// A CSR matrix's index arrays in the dtype SciPy would pick: `int32` when every index
/// and `nnz` fit, `int64` otherwise.
fn index_arrays<'py>(
    py: Python<'py>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    n_columns: u64,
) -> (Bound<'py, PyAny>, Bound<'py, PyAny>) {
    let nnz = indices.len() as u64;
    let fits = |value: u64| i32::try_from(value).is_ok();
    if fits(nnz) && fits(n_columns) {
        // Every value is at most nnz or below n_columns: the casts are exact.
        let indptr: Vec<i32> = indptr.into_iter().map(|value| value as i32).collect();
        let indices: Vec<i32> = indices.into_iter().map(|value| value as i32).collect();
        (
            PyArray1::from_vec(py, indptr).into_any(),
            PyArray1::from_vec(py, indices).into_any(),
        )
    } else {
        // nnz is a length, below isize::MAX, and indices are u32: both fit i64.
        let indptr: Vec<i64> = indptr.into_iter().map(|value| value as i64).collect();
        let indices: Vec<i64> = indices.into_iter().map(i64::from).collect();
        (
            PyArray1::from_vec(py, indptr).into_any(),
            PyArray1::from_vec(py, indices).into_any(),
        )
    }
}

/// `matrix` as a new `scipy.sparse.csr_matrix` of its value dtype.
pub fn to_scipy<'py, T: Element>(
    py: Python<'py>,
    matrix: CsrMatrix<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let CsrMatrix {
        n_cells,
        n_genes,
        indptr,
        indices,
        data,
    } = matrix;
    let (indptr, indices) = index_arrays(py, indptr, indices, n_genes);
    let data = PyArray1::from_vec(py, data).into_any();
    let keywords = PyDict::new(py);
    keywords.set_item("shape", (n_cells, n_genes))?;
    // The arrays are new and of SciPy's dtypes: nothing to copy again.
    keywords.set_item("copy", false)?;
    py.import("scipy.sparse")?
        .getattr("csr_matrix")?
        .call(((data, indices, indptr),), Some(&keywords))
}

/// `KY-E402` for a dense `rows x columns` array of `bytes` bytes that `method` cannot
/// allocate, for `problem`.
pub fn dense_too_large(
    method: &str,
    rows: u64,
    columns: u64,
    bytes: u128,
    problem: String,
) -> KytoError {
    KytoError::DenseTooLarge {
        method: method.to_owned(),
        n_rows: rows,
        n_columns: columns,
        required_bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        problem,
    }
}

/// Bytes of a dense `rows x columns` array of `T`, saturating.
pub fn dense_bytes<T>(rows: u64, columns: u64) -> u128 {
    (u128::from(rows) * u128::from(columns)).saturating_mul(std::mem::size_of::<T>() as u128)
}

/// `matrix` densified to a new row-major `n_cells x n_genes` NumPy array, absent
/// entries 0 (R9), for the terminal `method`.
///
/// # Errors
/// `KY-E402` if the array is larger than `max_bytes` (checked before allocating) or
/// cannot be allocated.
pub fn to_dense<'py, T: Element + Copy + Default>(
    py: Python<'py>,
    method: &str,
    matrix: &CsrMatrix<T>,
    max_bytes: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_rows, n_columns) = (matrix.n_cells, matrix.n_genes);
    let bytes = dense_bytes::<T>(n_rows, n_columns);
    if bytes > u128::from(max_bytes) {
        let error = dense_too_large(
            method,
            n_rows,
            n_columns,
            bytes,
            format!("max_bytes is {max_bytes}"),
        );
        return Err(to_py_err(py, &error));
    }
    // Below max_bytes, a u64: the shape and the length fit usize on 64-bit systems; a
    // 32-bit system that cannot hold them fails the allocation below.
    let rows = usize::try_from(n_rows).unwrap_or(usize::MAX);
    let columns = usize::try_from(n_columns).unwrap_or(usize::MAX);
    let length = rows.checked_mul(columns);
    let mut values: Vec<T> = Vec::new();
    let reserved = length.is_some_and(|length| values.try_reserve_exact(length).is_ok());
    let (Some(length), true) = (length, reserved) else {
        let error = dense_too_large(
            method,
            n_rows,
            n_columns,
            bytes,
            "the system has no memory for it".to_owned(),
        );
        return Err(to_py_err(py, &error));
    };
    values.resize(length, T::default());
    for (row, bounds) in matrix.indptr.windows(2).enumerate() {
        let (start, end) = (bounds[0] as usize, bounds[1] as usize);
        let offset = row * columns;
        for (column, value) in matrix.indices[start..end]
            .iter()
            .zip(&matrix.data[start..end])
        {
            values[offset + *column as usize] = *value;
        }
    }
    let Ok(array) = Array2::from_shape_vec((rows, columns), values) else {
        unreachable!("a vector of rows * columns values has the shape (rows, columns)")
    };
    Ok(PyArray2::from_owned_array(py, array).into_any())
}

/// `values`, row-major, as a new `rows x columns` NumPy array (moved, no copy).
pub fn to_array2<'py, T: Element>(
    py: Python<'py>,
    rows: u64,
    columns: u64,
    values: Vec<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = (
        usize::try_from(rows).unwrap_or(usize::MAX),
        usize::try_from(columns).unwrap_or(usize::MAX),
    );
    let Ok(array) = Array2::from_shape_vec(shape, values) else {
        unreachable!("a stored embedding holds rows x width values (kyto-store checks it)")
    };
    Ok(PyArray2::from_owned_array(py, array).into_any())
}

/// A graph's edges as a frame: `source` and `target` cell positions (`UInt32`) and
/// `weight` in the stored dtype, row by row.
pub fn edges(graph: kyto_store::Graph) -> Result<DataFrame, KytoError> {
    let mut sources: Vec<u32> = Vec::with_capacity(graph.indices.len());
    for (row, bounds) in graph.indptr.windows(2).enumerate() {
        // A graph's cells are below u32::MAX: its indices are u32 (format.md section 7).
        let row = row as u32;
        sources.extend(std::iter::repeat_n(row, (bounds[1] - bounds[0]) as usize));
    }
    let weight = match graph.weights {
        kyto_store::FloatValues::F32(values) => Column::new("weight".into(), values),
        kyto_store::FloatValues::F64(values) => Column::new("weight".into(), values),
    };
    let height = sources.len();
    DataFrame::new(
        height,
        vec![
            Column::new("source".into(), sources),
            Column::new("target".into(), graph.indices),
            weight,
        ],
    )
    .map_err(|error| KytoError::CorruptMember {
        kind: "graph".to_owned(),
        name: graph.name,
        problem: error.to_string(),
    })
}
