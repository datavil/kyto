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

use kyto_error::KytoError;

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
/// is 0 (R9). Blocks from the tile reader hold no zeros (format.md §4.6); a block built
/// with [`CellBlock::from_f32`] or [`CellBlock::from_u32`] may (for example after a
/// subtraction), and the store writer drops them.
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

    /// Build a `float32` block from CSR parts, checking them.
    ///
    /// Takes ownership of the vectors (no copy).
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
        Ok(Self::new(indptr, indices, CellData::F32(data), n_genes))
    }

    /// Build a `uint32` block from CSR parts, checking them.
    ///
    /// Takes ownership of the vectors (no copy).
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
        Ok(Self::new(indptr, indices, CellData::U32(data), n_genes))
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
