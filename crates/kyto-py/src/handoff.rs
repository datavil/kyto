//! Handing data to Python (tasks 7.3 and 7.5): Polars frames through the Arrow C stream
//! interface, NumPy arrays through the `numpy` crate, SciPy CSR matrices from those
//! arrays.
//!
//! Does: hand a Rust Polars frame to Python Polars without copying its buffers (the
//! Arrow PyCapsule interface, `__arrow_c_stream__`, read by `pl.DataFrame`: Python
//! takes over the Rust buffers and frees them through Arrow's release callback, every
//! dtype kept, `Enum` included); move a Rust `Vec` into a NumPy array (no copy: the
//! array owns the buffer); build a `scipy.sparse.csr_matrix` from a layer's CSR
//! (`int32` index arrays when they fit, `int64` otherwise, as SciPy picks: the values
//! moved, the `int32` indices or `int64` offsets reinterpreted in place, the other
//! index array converted), also for graphs; a graph's edge list as a frame (its
//! vectors moved into the columns); densify CSR row-major, absent entries 0 (R9),
//! only within `max_bytes` (R8, `KY-E402`).
//! Every object returned is new and owned by the caller alone: nothing kyto keeps
//! shares its memory (R2).
//! Does not: hand over a GPU tensor (DLPack, 9.5), use `pyo3-polars` (it pins an older
//! pyo3; the C stream interface it uses is called directly here), or stream: each
//! terminal holds its whole result in memory.

use kyto_error::KytoError;
use kyto_store::CsrMatrix;
use numpy::ndarray::Array2;
use numpy::{Element, PyArray1, PyArray2};
use polars::prelude::{
    Column, CompatLevel, DataFrame, Float32Chunked, Float64Chunked, IntoColumn, IntoSeries, Series,
    UInt32Chunked,
};
use polars_arrow::ffi::export_iterator;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::errors::to_py_err;

/// A frame offered to Python Polars as an Arrow C stream (`pl.DataFrame(frame)`).
///
/// Private: a temporary that `to_polars` drops as soon as Polars has read it, so it
/// never reaches the caller and has no stub (R12 covers the public surface). It holds
/// the frame's buffers by reference count, not by copy.
#[pyclass(frozen, module = "kyto._kyto", name = "_ArrowStream")]
struct ArrowStream {
    /// One struct column whose chunks are the stream's batches (`pl.DataFrame` unnests
    /// it).
    series: Series,
}

#[pymethods]
impl ArrowStream {
    /// The frame as an `arrow_array_stream` PyCapsule (the Arrow PyCapsule interface),
    /// a new stream on each call. The consumer moves the stream out of the capsule; a
    /// capsule dropped unread releases it.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // The schema is the frame's own: a requested one is a hint the interface lets
        // producers ignore.
        let _ = requested_schema;
        let compat = CompatLevel::newest();
        let field = self.series.field().to_arrow(compat);
        let batches: Vec<_> = (0..self.series.n_chunks())
            .map(|chunk| Ok(self.series.to_arrow(chunk, compat)))
            .collect();
        let stream = export_iterator(Box::new(batches.into_iter()), field);
        PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
    }
}

/// `frame` as a new Python `polars.DataFrame`, its buffers handed over without a copy.
pub fn to_polars<'py>(py: Python<'py>, frame: DataFrame) -> PyResult<Bound<'py, PyAny>> {
    // Every frame kyto hands over has a column (a table's ID, an edge list's three, the
    // DE frame's six), so the struct below carries the height.
    debug_assert!(frame.width() > 0, "a handed-over frame has a column");
    // `into_struct` aligns the columns' chunks, so no copy for aligned ones.
    let series = frame.into_struct("".into()).into_series();
    let stream = Bound::new(py, ArrowStream { series })?;
    py.import("polars")?.getattr("DataFrame")?.call1((stream,))
}

/// A CSR matrix's index arrays in the dtype SciPy would pick: `int32` when every index
/// and `nnz` fit, `int64` otherwise. The indices (`int32`) and the offsets (`int64`)
/// keep their buffer, reinterpreted in place; the other array is converted.
fn index_arrays<'py>(
    py: Python<'py>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    n_columns: u64,
) -> (Bound<'py, PyAny>, Bound<'py, PyAny>) {
    let nnz = indices.len() as u64;
    let fits = |value: u64| i32::try_from(value).is_ok();
    if fits(nnz) && fits(n_columns) {
        // Every value is at most nnz or below n_columns: the cast and the
        // reinterpretation (same bits below 2^31) are exact.
        let indptr: Vec<i32> = indptr.into_iter().map(|value| value as i32).collect();
        let indices: Vec<i32> = bytemuck::cast_vec(indices);
        (
            PyArray1::from_vec(py, indptr).into_any(),
            PyArray1::from_vec(py, indices).into_any(),
        )
    } else {
        // Offsets are at most nnz, a length below isize::MAX (same bits as i64), and
        // indices are u32: both fit i64.
        let indptr: Vec<i64> = bytemuck::cast_vec(indptr);
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
    // `from_vec` moves each vector into its column (`Column::new` would copy it).
    let weight = match graph.weights {
        kyto_store::FloatValues::F32(values) => {
            Float32Chunked::from_vec("weight".into(), values).into_column()
        }
        kyto_store::FloatValues::F64(values) => {
            Float64Chunked::from_vec("weight".into(), values).into_column()
        }
    };
    let height = sources.len();
    DataFrame::new(
        height,
        vec![
            UInt32Chunked::from_vec("source".into(), sources).into_column(),
            UInt32Chunked::from_vec("target".into(), graph.indices).into_column(),
            weight,
        ],
    )
    .map_err(|error| KytoError::CorruptMember {
        kind: "graph".to_owned(),
        name: graph.name,
        problem: error.to_string(),
    })
}

/// Test hook: a frame built in Rust (`x` `Float64`, `n` `UInt32`, `s` `String`, 1000
/// rows) handed to Polars as the terminals do, with the addresses of the frame's value
/// buffers of `x` and `n`, so a test can check that Python sees the same memory.
#[pyfunction]
pub fn handoff_example(py: Python<'_>) -> PyResult<(Bound<'_, PyAny>, usize, usize)> {
    let values: Vec<f64> = (0..1000).map(f64::from).collect();
    let integers: Vec<u32> = (0..1000).collect();
    let Ok(frame) = DataFrame::new(
        1000,
        vec![
            Column::new("x".into(), values),
            Column::new("n".into(), integers),
            Column::new("s".into(), vec!["a"; 1000]),
        ],
    ) else {
        unreachable!("three columns of 1000 rows with distinct names form a frame")
    };
    let address = |name: &str| -> Option<usize> {
        let column = frame.column(name).ok()?;
        match column.dtype() {
            polars::prelude::DataType::Float64 => {
                Some(column.f64().ok()?.cont_slice().ok()?.as_ptr() as usize)
            }
            _ => Some(column.u32().ok()?.cont_slice().ok()?.as_ptr() as usize),
        }
    };
    let (Some(x), Some(n)) = (address("x"), address("n")) else {
        unreachable!("x and n are single-chunk Float64 and UInt32 columns without nulls")
    };
    Ok((to_polars(py, frame)?, x, n))
}
