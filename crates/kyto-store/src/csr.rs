//! A whole layer in memory as plain CSR (cells are rows, genes are columns).
//!
//! Does: hold the matrix a layer writer takes and a layer reader returns, and check
//! that it is valid CSR of its stated shape.
//! Does not: tile, compress or touch storage (see `tile_encode`, `layer_io`). It is not
//! the executor's `CellBlock` (plan 5.1).

use kyto_error::KytoError;

/// A sparse `n_cells x n_genes` matrix in CSR form, with global gene indices.
///
/// Row `i` holds entries `indptr[i]..indptr[i + 1]` of `indices` and `data`. A valid
/// matrix has `indptr.len() == n_cells + 1`, `indptr[0] == 0`, a non-decreasing
/// `indptr` ending at `indices.len() == data.len()`, and in each row strictly
/// increasing gene indices below `n_genes`.
///
/// Matrices returned by [`crate::read_layer`] are always valid and canonical: no
/// stored zeros (format.md §4.6). A matrix given to [`crate::write_layer`] may hold
/// zeros; the writer drops them.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    /// Number of rows (cells).
    pub n_cells: u64,
    /// Number of columns (genes).
    pub n_genes: u64,
    /// Row pointers, length `n_cells + 1`.
    pub indptr: Vec<u64>,
    /// Gene index of each entry.
    pub indices: Vec<u32>,
    /// Value of each entry.
    pub data: Vec<T>,
}

impl<T> CsrMatrix<T> {
    /// Number of stored entries.
    pub fn nnz(&self) -> u64 {
        self.indices.len() as u64
    }

    /// Check that the matrix is valid CSR of its stated shape.
    ///
    /// # Errors
    /// `KY-E106` ([`KytoError::InvalidSparseMatrix`]) naming the first problem found.
    pub fn validate(&self, layer: &str) -> Result<(), KytoError> {
        let invalid = |problem: String| KytoError::InvalidSparseMatrix {
            layer: layer.to_owned(),
            problem,
        };
        let expected_len = self.n_cells.checked_add(1);
        if expected_len != Some(self.indptr.len() as u64) {
            return Err(invalid(format!(
                "indptr has {} entries but n_cells + 1 is {}",
                self.indptr.len(),
                u128::from(self.n_cells) + 1
            )));
        }
        if self.indices.len() != self.data.len() {
            return Err(invalid(format!(
                "indices has {} entries but data has {}",
                self.indices.len(),
                self.data.len()
            )));
        }
        if self.indptr.first() != Some(&0) {
            return Err(invalid("indptr[0] must be 0".to_owned()));
        }
        if self.indptr.last() != Some(&self.nnz()) {
            return Err(invalid(format!(
                "indptr ends at {:?} but there are {} entries",
                self.indptr.last(),
                self.nnz()
            )));
        }
        for (row, bounds) in self.indptr.windows(2).enumerate() {
            if bounds[1] < bounds[0] {
                return Err(invalid(format!(
                    "indptr decreases at row {row}: {} then {}",
                    bounds[0], bounds[1]
                )));
            }
        }
        // indptr is now non-decreasing and ends at nnz, so every bound is <= nnz.
        for (row, bounds) in self.indptr.windows(2).enumerate() {
            let (start, end) = (bounds[0], bounds[1]);
            let row_indices = &self.indices[start as usize..end as usize];
            let mut previous: Option<u32> = None;
            for &gene in row_indices {
                if u64::from(gene) >= self.n_genes {
                    return Err(invalid(format!(
                        "row {row}: gene index {gene} is not below n_genes = {}",
                        self.n_genes
                    )));
                }
                if let Some(before) = previous
                    && gene <= before
                {
                    return Err(invalid(format!(
                        "row {row}: gene index {gene} follows {before}; indices must be strictly increasing"
                    )));
                }
                previous = Some(gene);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CsrMatrix;

    fn matrix(n_cells: u64, n_genes: u64, indptr: &[u64], indices: &[u32]) -> CsrMatrix<u32> {
        CsrMatrix {
            n_cells,
            n_genes,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            data: vec![1; indices.len()],
        }
    }

    fn problem(matrix: &CsrMatrix<u32>) -> String {
        let error = matrix.validate("counts").unwrap_err();
        assert_eq!(error.code(), "KY-E106");
        error.to_string()
    }

    #[test]
    fn valid_matrices_pass() {
        matrix(0, 0, &[0], &[]).validate("counts").unwrap();
        matrix(2, 3, &[0, 2, 3], &[0, 2, 1])
            .validate("counts")
            .unwrap();
        matrix(2, 0, &[0, 0, 0], &[]).validate("counts").unwrap();
    }

    #[test]
    fn each_problem_is_reported() {
        assert!(problem(&matrix(2, 3, &[0, 1], &[0])).contains("n_cells + 1 is 3"));
        assert!(problem(&matrix(1, 3, &[1, 1], &[0])).contains("indptr[0] must be 0"));
        assert!(problem(&matrix(1, 3, &[0, 1], &[0, 1])).contains("ends at"));
        assert!(problem(&matrix(2, 3, &[0, 2, 1], &[0])).contains("decreases at row 1"));
        assert!(problem(&matrix(2, 3, &[0, 2, 1, 2], &[0, 1])).contains("n_cells + 1"));
        assert!(problem(&matrix(3, 3, &[0, 2, 1, 2], &[0, 1])).contains("decreases at row 1"));
        assert!(problem(&matrix(1, 3, &[0, 1], &[3])).contains("not below n_genes = 3"));
        assert!(problem(&matrix(1, 3, &[0, 2], &[2, 1])).contains("follows 2"));
        assert!(problem(&matrix(1, 3, &[0, 2], &[1, 1])).contains("follows 1"));
        let mut lengths = matrix(1, 3, &[0, 1], &[0]);
        lengths.data.push(2);
        assert!(problem(&lengths).contains("data has 2"));
    }
}
