//! Canonical rows for the importers (task 2.6, format.md §4.3, §4.6, §12).
//!
//! Does: [`BlockBuilder`] turns the entries of each input row, in any order and with
//! repeated genes, into a canonical [`CellBlock`]: genes sorted, repeats summed (as
//! scipy's `sum_duplicates`), zeros dropped, values converted to the layer dtype.
//! [`RowSource`] is what every input matrix (dense, CSR, CSC, 10x, mtx) offers the
//! import driver: its rows, one cell block at a time.
//! Does not: read files (see the `import_*` input modules) or write the store.
//!
//! Values arrive as `f64`, which holds every `float32` and every integer up to `2^53`
//! exactly. Repeated entries are summed as scipy sums them: in `float32` for `float32`
//! layers read from `float32` (or Matrix Market `real`) inputs, in `f64` otherwise.

use kyto_error::KytoError;

use crate::cell_block::CellBlock;
use crate::layer_open::ValueDtype;

/// An input matrix, read by rows in increasing order.
pub(crate) trait RowSource {
    /// The value dtype its layer stores.
    fn dtype(&self) -> ValueDtype;

    /// Rows `start..end` over all genes, canonical. Calls come in increasing,
    /// adjacent ranges starting at 0.
    fn read_rows(&mut self, start: u64, end: u64) -> Result<CellBlock, KytoError>;
}

/// Advice for values that do not fit a `uint32` layer.
const UINT32_ADVICE: &str = "integer matrices become uint32 layers (docs/format.md section 4.2); cast the matrix to float32 before converting to keep negative, fractional or larger values";

/// Builds one canonical CellBlock, row by row.
pub(crate) struct BlockBuilder<'a> {
    dtype: ValueDtype,
    n_genes: u64,
    next_row: u64,
    source: &'a str,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    floats: Vec<f32>,
    integers: Vec<u32>,
    f32_sums: bool,
}

impl<'a> BlockBuilder<'a> {
    /// A builder whose first row is matrix row `first_row`; `source` names the input
    /// in errors.
    pub(crate) fn new(dtype: ValueDtype, n_genes: u64, first_row: u64, source: &'a str) -> Self {
        Self {
            dtype,
            n_genes,
            next_row: first_row,
            source,
            indptr: vec![0],
            indices: Vec::new(),
            floats: Vec::new(),
            integers: Vec::new(),
            f32_sums: false,
        }
    }

    /// Sum repeated entries in `float32` (scipy's `sum_duplicates` on `float32` data)
    /// when `yes` and the layer is `float32`.
    pub(crate) fn with_f32_sums(mut self, yes: bool) -> Self {
        self.f32_sums = yes && self.dtype == ValueDtype::F32;
        self
    }

    /// Add the next row from `(gene, value)` entries in any order; `entries` is
    /// sorted in place (stable, so repeats are summed in input order). Genes must be
    /// below `n_genes` (the input modules check them).
    ///
    /// # Errors
    /// `KY-E121` if a value does not fit the layer dtype.
    pub(crate) fn push_row(&mut self, entries: &mut [(u32, f64)]) -> Result<(), KytoError> {
        if !entries.is_sorted_by_key(|entry| entry.0) {
            entries.sort_by_key(|entry| entry.0);
        }
        let mut position = 0;
        while position < entries.len() {
            let gene = entries[position].0;
            let mut sum = entries[position].1;
            let mut narrow = sum as f32;
            position += 1;
            while position < entries.len() && entries[position].0 == gene {
                sum += entries[position].1;
                narrow += entries[position].1 as f32;
                position += 1;
            }
            if self.f32_sums {
                sum = f64::from(narrow);
            }
            // Zeros, `-0.0` included, are never stored (format.md §4.6).
            if sum == 0.0 {
                continue;
            }
            self.push_value(gene, sum)?;
        }
        self.indptr.push(self.indices.len() as u64);
        self.next_row += 1;
        Ok(())
    }

    fn push_value(&mut self, gene: u32, value: f64) -> Result<(), KytoError> {
        match self.dtype {
            ValueDtype::F32 => {
                let narrowed = value as f32;
                if value.is_finite() && narrowed.is_infinite() {
                    return Err(KytoError::InvalidInput {
                        path: self.source.to_owned(),
                        location: format!("cell {}, gene {gene}", self.next_row),
                        problem: format!("value {value} is beyond the float32 range"),
                        advice: "float matrices become float32 layers (docs/format.md section 4.2); rescale the values below 3.4e38 before converting".to_owned(),
                    });
                }
                self.floats.push(narrowed);
            }
            ValueDtype::U32 => {
                let fits = value.is_finite()
                    && value >= 0.0
                    && value <= f64::from(u32::MAX)
                    && value.fract() == 0.0;
                if !fits {
                    return Err(KytoError::InvalidInput {
                        path: self.source.to_owned(),
                        location: format!("cell {}, gene {gene}", self.next_row),
                        problem: format!("value {value} is not an integer from 0 to {}", u32::MAX),
                        advice: UINT32_ADVICE.to_owned(),
                    });
                }
                self.integers.push(value as u32);
            }
        }
        self.indices.push(gene);
        Ok(())
    }

    /// The finished block.
    ///
    /// # Errors
    /// `KY-E117` only on an internal bug (the parts are canonical by construction).
    pub(crate) fn finish(self) -> Result<CellBlock, KytoError> {
        match self.dtype {
            ValueDtype::F32 => {
                CellBlock::from_f32(self.indptr, self.indices, self.floats, self.n_genes)
            }
            ValueDtype::U32 => {
                CellBlock::from_u32(self.indptr, self.indices, self.integers, self.n_genes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BlockBuilder;
    use crate::cell_block::CellValues;
    use crate::layer_open::ValueDtype;

    #[test]
    fn rows_are_sorted_summed_and_zero_free() {
        let mut builder = BlockBuilder::new(ValueDtype::F32, 5, 0, "input");
        builder
            .push_row(&mut [(3, 1.0), (1, 2.0), (3, 0.5), (4, 0.0), (0, -0.0)])
            .unwrap();
        builder.push_row(&mut []).unwrap();
        builder
            .push_row(&mut [(2, 1.0), (2, -1.0), (0, 7.0)])
            .unwrap();
        let block = builder.finish().unwrap();
        assert_eq!(block.indptr(), &[0, 2, 2, 3]);
        assert_eq!(block.indices(), &[1, 3, 0]);
        assert!(matches!(block.data(), CellValues::F32(data) if data == [2.0, 1.5, 7.0]));
    }

    #[test]
    fn float32_sums_and_overflow() {
        let entries = || [(0_u32, 16_777_216.0), (0, 1.0), (0, 1.0)];
        let mut wide = BlockBuilder::new(ValueDtype::F32, 1, 0, "input");
        wide.push_row(&mut entries()).unwrap();
        let mut narrow = BlockBuilder::new(ValueDtype::F32, 1, 0, "input").with_f32_sums(true);
        narrow.push_row(&mut entries()).unwrap();
        // scipy on float32: 16777216 + 1 + 1 stays 16777216; in f64 it is 16777218.
        let first = |builder: BlockBuilder<'_>| match builder.finish().unwrap().data() {
            CellValues::F32(data) => data[0],
            CellValues::U32(_) => panic!("float32 expected"),
        };
        assert_eq!(first(wide), 16_777_218.0);
        assert_eq!(first(narrow), 16_777_216.0);
        let mut builder = BlockBuilder::new(ValueDtype::F32, 1, 0, "input");
        let error = builder.push_row(&mut [(0, 1e300)]).unwrap_err();
        assert_eq!(error.code(), "KY-E121");
        builder.push_row(&mut [(0, f64::INFINITY)]).unwrap();
    }

    #[test]
    fn uint32_rejects_negative_fractional_and_large_values() {
        for value in [-1.0, 0.5, 4_294_967_296.0, f64::NAN] {
            let mut builder = BlockBuilder::new(ValueDtype::U32, 3, 7, "x.h5ad:/X");
            let error = builder.push_row(&mut [(2, value)]).unwrap_err();
            assert_eq!(error.code(), "KY-E121");
            assert!(error.to_string().contains("cell 7, gene 2"), "{error}");
        }
        let mut builder = BlockBuilder::new(ValueDtype::U32, 3, 0, "input");
        builder.push_row(&mut [(0, 4_294_967_295.0)]).unwrap();
        let block = builder.finish().unwrap();
        assert!(matches!(block.data(), CellValues::U32(data) if data == [u32::MAX]));
    }
}
