//! The executor's in-memory batch of CSR rows over the selected genes (plan 5.1,
//! format.md §4.8).
//!
//! Does: hold one block of rows as `indptr: u64`, `indices: u32` and `data: f32 | u32`
//! in 64-byte aligned buffers ([`AlignedBuffer`]), read through slices; hand its
//! buffers back ([`CellBlock::into_parts`]) and take filled ones
//! ([`CellBlock::from_parts`]), so a pool (kyto-exec `CellBlockPool`, task 5.1) reuses
//! them block after block; densify it, or one gene of it, with every absent entry 0 (R9,
//! task 3.7). Every dense read of layer values goes through these, so the rule lives in
//! one place.
//! Does not: pool buffers itself (kyto-exec does), or know which cells its rows are (the
//! tile reader's `SelectedBlock` says).
//!
//! It lives in kyto-store because the tile reader (2.4) builds it and the streaming
//! writer (2.5) consumes it; the executor (kyto-exec) depends on kyto-store, not the
//! other way round.

use kyto_error::KytoError;

use crate::aligned_buffer::AlignedBuffer;
use crate::layer_open::ValueDtype;

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

/// Dense values densified from a [`CellBlock`], in the layer's dtype. Absent entries
/// are 0, never null or `NaN` (R9).
#[derive(Debug, Clone, PartialEq)]
pub enum DenseValues {
    /// Values of a `float32` layer.
    F32(Vec<f32>),
    /// Values of a `uint32` layer.
    U32(Vec<u32>),
}

impl DenseValues {
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

/// Owned values of a [`CellBlock`], in a 64-byte aligned buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum CellData {
    /// `float32` values.
    F32(AlignedBuffer<f32>),
    /// `uint32` values.
    U32(AlignedBuffer<u32>),
}

impl CellData {
    /// An empty buffer of `dtype` values; allocates nothing.
    pub fn new(dtype: ValueDtype) -> Self {
        match dtype {
            ValueDtype::F32 => Self::F32(AlignedBuffer::new()),
            ValueDtype::U32 => Self::U32(AlignedBuffer::new()),
        }
    }

    /// The dtype of the values.
    pub fn dtype(&self) -> ValueDtype {
        match self {
            Self::F32(_) => ValueDtype::F32,
            Self::U32(_) => ValueDtype::U32,
        }
    }

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

    /// The same allocation, emptied, holding `dtype` values (no copy, no allocation).
    pub fn into_cleared(self, dtype: ValueDtype) -> Self {
        let storage: AlignedBuffer<u32> = match self {
            Self::F32(values) => values.into_cleared(),
            Self::U32(values) => values.into_cleared(),
        };
        match dtype {
            ValueDtype::F32 => Self::F32(storage.into_cleared()),
            ValueDtype::U32 => Self::U32(storage),
        }
    }
}

/// The buffers of a [`CellBlock`], to fill and turn into one with
/// [`CellBlock::from_parts`], or taken back with [`CellBlock::into_parts`].
#[derive(Debug, Clone, PartialEq)]
pub struct CellBlockParts {
    /// Row pointers: `n_rows + 1` entries, starting at 0, ending at the nnz.
    pub indptr: AlignedBuffer<u64>,
    /// Position in the gene selection of each entry.
    pub indices: AlignedBuffer<u32>,
    /// Value of each entry.
    pub data: CellData,
}

impl CellBlockParts {
    /// Empty buffers for a block of `dtype` values; allocates nothing.
    pub fn new(dtype: ValueDtype) -> Self {
        Self {
            indptr: AlignedBuffer::new(),
            indices: AlignedBuffer::new(),
            data: CellData::new(dtype),
        }
    }
}

/// A batch of CSR rows over a gene selection (plan 5.1).
///
/// Its three buffers start on a 64-byte boundary ([`AlignedBuffer`]), and
/// [`CellBlock::into_parts`] gives them back for reuse.
///
/// Row `r` holds entries `indptr[r]..indptr[r + 1]` of `indices` and `data`. `indices`
/// are positions in the gene selection (not table positions), strictly increasing
/// within each row and below [`CellBlock::n_genes`] (format.md §4.8). An absent entry
/// is 0 (R9). Blocks from the tile reader hold no zeros (format.md §4.6); a block built
/// with [`CellBlock::from_f32`] or [`CellBlock::from_u32`] may (for example after a
/// subtraction), and the store writer drops them.
#[derive(Debug, Clone, PartialEq)]
pub struct CellBlock {
    indptr: AlignedBuffer<u64>,
    indices: AlignedBuffer<u32>,
    data: CellData,
    n_genes: u64,
}

impl CellBlock {
    /// Build a block from valid parts: `indptr` starts at 0, does not decrease and ends
    /// at `indices.len()`, which equals the number of values.
    pub(crate) fn new(
        indptr: AlignedBuffer<u64>,
        indices: AlignedBuffer<u32>,
        data: CellData,
        n_genes: u64,
    ) -> Self {
        Self {
            indptr,
            indices,
            data,
            n_genes,
        }
    }

    /// Build a `float32` block from CSR parts, checking them.
    ///
    /// Copies the vectors into 64-byte aligned buffers; [`CellBlock::from_parts`] takes
    /// filled buffers without a copy.
    ///
    /// # Errors
    /// `KY-E117` if `indptr` is empty, does not start at 0, decreases or does not end
    /// at `indices.len()`, if `data` has another length, or if a row's `indices` are not
    /// strictly increasing and below `n_genes`.
    pub fn from_f32(
        indptr: Vec<u64>,
        indices: Vec<u32>,
        data: Vec<f32>,
        n_genes: u64,
    ) -> Result<Self, KytoError> {
        check_parts(
            "CellBlock::from_f32",
            &indptr,
            &indices,
            data.len(),
            n_genes,
        )?;
        Ok(Self::new(
            AlignedBuffer::from(&indptr[..]),
            AlignedBuffer::from(&indices[..]),
            CellData::F32(AlignedBuffer::from(&data[..])),
            n_genes,
        ))
    }

    /// Build a `uint32` block from CSR parts, checking them.
    ///
    /// Copies the vectors, as [`CellBlock::from_f32`] does.
    ///
    /// # Errors
    /// `KY-E117` as for [`CellBlock::from_f32`].
    pub fn from_u32(
        indptr: Vec<u64>,
        indices: Vec<u32>,
        data: Vec<u32>,
        n_genes: u64,
    ) -> Result<Self, KytoError> {
        check_parts(
            "CellBlock::from_u32",
            &indptr,
            &indices,
            data.len(),
            n_genes,
        )?;
        Ok(Self::new(
            AlignedBuffer::from(&indptr[..]),
            AlignedBuffer::from(&indices[..]),
            CellData::U32(AlignedBuffer::from(&data[..])),
            n_genes,
        ))
    }

    /// Build a block from filled buffers, checking them as [`CellBlock::from_f32`] does.
    ///
    /// Takes ownership of the buffers (no copy, no allocation), so a pool's buffers
    /// pass through without allocating.
    ///
    /// # Errors
    /// `KY-E117` as for [`CellBlock::from_f32`].
    pub fn from_parts(parts: CellBlockParts, n_genes: u64) -> Result<Self, KytoError> {
        check_parts(
            "CellBlock::from_parts",
            parts.indptr.as_slice(),
            parts.indices.as_slice(),
            parts.data.len(),
            n_genes,
        )?;
        Ok(Self::new(parts.indptr, parts.indices, parts.data, n_genes))
    }

    /// The block's buffers, values kept (no copy), to reuse or rebuild a block from.
    pub fn into_parts(self) -> CellBlockParts {
        CellBlockParts {
            indptr: self.indptr,
            indices: self.indices,
            data: self.data,
        }
    }

    /// Row pointers, length [`CellBlock::n_rows`] + 1, starting at 0.
    pub fn indptr(&self) -> &[u64] {
        self.indptr.as_slice()
    }

    /// Position in the gene selection of each entry.
    pub fn indices(&self) -> &[u32] {
        self.indices.as_slice()
    }

    /// Value of each entry.
    pub fn data(&self) -> CellValues<'_> {
        match &self.data {
            CellData::F32(values) => CellValues::F32(values.as_slice()),
            CellData::U32(values) => CellValues::U32(values.as_slice()),
        }
    }

    /// The dtype of the values.
    pub fn dtype(&self) -> ValueDtype {
        self.data.dtype()
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

    /// The block as a dense row-major `n_rows x n_genes` array: the value of row `r` and
    /// gene position `g` is at `r * n_genes + g`, and every entry the block does not store
    /// is 0 (R9). Stored values, `NaN` included, are copied as they are.
    ///
    /// Allocates the whole array: callers bound its size first (R8; `to_dense(max_bytes=)`
    /// is task 7.3, block sizes are the physical plan's, task 4.5).
    ///
    /// # Errors
    /// `KY-E117` if `n_rows x n_genes` values are more bytes than this platform can
    /// address (`isize::MAX`), or if the allocation fails.
    pub fn to_dense(&self) -> Result<DenseValues, KytoError> {
        Ok(match &self.data {
            CellData::F32(values) => DenseValues::F32(self.scatter(values.as_slice())?),
            CellData::U32(values) => DenseValues::U32(self.scatter(values.as_slice())?),
        })
    }

    /// The values of gene position `position` (in the gene selection) for every row, in
    /// row order; a row that does not store the gene gives 0 (R9).
    ///
    /// # Errors
    /// `KY-E117` if `position` is not below [`CellBlock::n_genes`].
    pub fn gene_values(&self, position: u32) -> Result<DenseValues, KytoError> {
        if u64::from(position) >= self.n_genes {
            return Err(KytoError::InvalidCellBlock {
                context: "CellBlock::gene_values".to_owned(),
                problem: format!(
                    "gene position {position} is not below n_genes = {}",
                    self.n_genes
                ),
            });
        }
        Ok(match &self.data {
            CellData::F32(values) => DenseValues::F32(self.gather(values.as_slice(), position)),
            CellData::U32(values) => DenseValues::U32(self.gather(values.as_slice(), position)),
        })
    }

    /// The bounds of row `row` in `indices` and `data`.
    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        // indptr was checked: it does not decrease and ends at indices.len().
        let indptr = self.indptr.as_slice();
        indptr[row] as usize..indptr[row + 1] as usize
    }

    /// `values` written into a zeroed row-major `n_rows x n_genes` array.
    fn scatter<T: Copy + Default>(&self, values: &[T]) -> Result<Vec<T>, KytoError> {
        let too_large = |reason: &str| KytoError::InvalidCellBlock {
            context: "CellBlock::to_dense".to_owned(),
            problem: format!(
                "{} rows x {} genes of {} bytes each {reason}; densify fewer rows or genes \
                 at a time",
                self.n_rows(),
                self.n_genes,
                size_of::<T>()
            ),
        };
        let n_genes = usize::try_from(self.n_genes).ok();
        let length = n_genes.and_then(|n_genes| n_genes.checked_mul(self.n_rows()));
        let bytes = length.and_then(|length| length.checked_mul(size_of::<T>()));
        let (Some(n_genes), Some(length), Some(bytes)) = (n_genes, length, bytes) else {
            return Err(too_large("are more bytes than this platform can address"));
        };
        if isize::try_from(bytes).is_err() {
            return Err(too_large("are more bytes than this platform can address"));
        }
        let mut dense = Vec::new();
        dense
            .try_reserve_exact(length)
            .map_err(|_| too_large("could not be allocated"))?;
        dense.resize(length, T::default());
        for row in 0..self.n_rows() {
            let range = self.row_range(row);
            let start = row * n_genes;
            for (&gene, &value) in self.indices.as_slice()[range.clone()]
                .iter()
                .zip(&values[range])
            {
                dense[start + gene as usize] = value;
            }
        }
        Ok(dense)
    }

    /// The value at gene `position` of each row, 0 where the row does not store it.
    fn gather<T: Copy + Default>(&self, values: &[T], position: u32) -> Vec<T> {
        (0..self.n_rows())
            .map(|row| {
                let range = self.row_range(row);
                // Indices are strictly increasing within a row.
                match self.indices.as_slice()[range.clone()].binary_search(&position) {
                    Ok(offset) => values[range.start + offset],
                    Err(_) => T::default(),
                }
            })
            .collect()
    }
}

/// Check CSR parts against the invariants [`CellBlock`] documents.
fn check_parts(
    context: &str,
    indptr: &[u64],
    indices: &[u32],
    n_values: usize,
    n_genes: u64,
) -> Result<(), KytoError> {
    let invalid = |problem: String| {
        Err(KytoError::InvalidCellBlock {
            context: context.to_owned(),
            problem,
        })
    };
    if indptr.first() != Some(&0) {
        return invalid("indptr must start with 0 (it has n_rows + 1 entries)".to_owned());
    }
    if indptr.last() != Some(&(indices.len() as u64)) {
        return invalid(format!(
            "indptr ends at {:?} but there are {} indices",
            indptr.last(),
            indices.len()
        ));
    }
    if n_values != indices.len() {
        return invalid(format!(
            "there are {} indices but {n_values} values",
            indices.len()
        ));
    }
    for (row, bounds) in indptr.windows(2).enumerate() {
        if bounds[1] < bounds[0] {
            return invalid(format!(
                "indptr decreases at row {row}: {} then {}",
                bounds[0], bounds[1]
            ));
        }
    }
    // indptr is non-decreasing and ends at indices.len(), so every bound is in range.
    for (row, bounds) in indptr.windows(2).enumerate() {
        let row_indices = &indices[bounds[0] as usize..bounds[1] as usize];
        for (position, &gene) in row_indices.iter().enumerate() {
            if u64::from(gene) >= n_genes {
                return invalid(format!(
                    "row {row}: gene position {gene} is not below n_genes = {n_genes}"
                ));
            }
            if position > 0 && gene <= row_indices[position - 1] {
                return invalid(format!(
                    "row {row}: gene position {gene} follows {}; positions must be strictly increasing",
                    row_indices[position - 1]
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CellBlock;

    fn problem(indptr: &[u64], indices: &[u32], n_values: usize, n_genes: u64) -> String {
        let error = CellBlock::from_u32(
            indptr.to_vec(),
            indices.to_vec(),
            vec![1; n_values],
            n_genes,
        )
        .unwrap_err();
        assert_eq!(error.code(), "KY-E117");
        error.to_string()
    }

    #[test]
    fn valid_parts_build_a_block() {
        let block =
            CellBlock::from_f32(vec![0, 2, 2, 3], vec![0, 4, 1], vec![1.0, 0.0, 2.0], 5).unwrap();
        assert_eq!((block.n_rows(), block.nnz(), block.n_genes()), (3, 3, 5));
        let empty = CellBlock::from_u32(vec![0], Vec::new(), Vec::new(), 0).unwrap();
        assert_eq!(empty.n_rows(), 0);
    }

    #[test]
    fn each_problem_is_reported() {
        assert!(problem(&[], &[], 0, 3).contains("start with 0"));
        assert!(problem(&[1, 1], &[0], 1, 3).contains("start with 0"));
        assert!(problem(&[0, 1], &[0, 1], 2, 3).contains("ends at Some(1)"));
        assert!(problem(&[0, 1], &[0], 2, 3).contains("2 values"));
        assert!(problem(&[0, 2, 1, 2], &[0, 1], 2, 3).contains("decreases at row 1"));
        assert!(problem(&[0, 1], &[3], 1, 3).contains("not below n_genes = 3"));
        assert!(problem(&[0, 2], &[2, 1], 2, 3).contains("follows 2"));
        assert!(problem(&[0, 2], &[1, 1], 2, 3).contains("follows 1"));
    }
}
