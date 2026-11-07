//! The executor's in-memory batch of CSR rows over the selected genes (plan 5.1,
//! format.md §4.8).
//!
//! Does: hold one block of rows as `indptr: u64`, `indices: u32` and `data: f32 | u32`,
//! read through slices.
//! Does not: pool or align its buffers (task 5.1 adds that behind the same accessors),
//! or know which cells its rows are (the tile reader's `SelectedBlock` says).
//!
//! It lives in kyto-store because the tile reader (2.4) builds it and the streaming
//! writer (2.5) consumes it; the executor (kyto-exec) depends on kyto-store, not the
//! other way round.

/// The values of a [`CellBlock`], borrowed: `float32` or `uint32`, as the layer stores
/// them (format.md §4.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellValues<'a> {
    /// Values of a `float32` layer.
    F32(&'a [f32]),
    /// Values of a `uint32` layer.
    U32(&'a [u32]),
}

impl CellValues<'_> {
    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    /// True when there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Owned values of a [`CellBlock`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CellData {
    /// `float32` values.
    F32(Vec<f32>),
    /// `uint32` values.
    U32(Vec<u32>),
}

/// A batch of CSR rows over a gene selection (plan 5.1).
///
/// Row `r` holds entries `indptr[r]..indptr[r + 1]` of `indices` and `data`. `indices`
/// are positions in the gene selection (not table positions), strictly increasing
/// within each row and below [`CellBlock::n_genes`] (format.md §4.8). An absent entry
/// is 0 (R9); no stored value is 0 (format.md §4.6).
#[derive(Debug, Clone, PartialEq)]
pub struct CellBlock {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    data: CellData,
    n_genes: u64,
}

impl CellBlock {
    /// Build a block from valid parts: `indptr` starts at 0, does not decrease and ends
    /// at `indices.len()`, which equals the number of values.
    pub(crate) fn new(indptr: Vec<u64>, indices: Vec<u32>, data: CellData, n_genes: u64) -> Self {
        Self {
            indptr,
            indices,
            data,
            n_genes,
        }
    }

    /// Row pointers, length [`CellBlock::n_rows`] + 1, starting at 0.
    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    /// Position in the gene selection of each entry.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Value of each entry.
    pub fn data(&self) -> CellValues<'_> {
        match &self.data {
            CellData::F32(values) => CellValues::F32(values),
            CellData::U32(values) => CellValues::U32(values),
        }
    }

    /// Number of rows (cells).
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Number of columns: the length of the gene selection.
    pub fn n_genes(&self) -> u64 {
        self.n_genes
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> u64 {
        self.indices.len() as u64
    }
}
