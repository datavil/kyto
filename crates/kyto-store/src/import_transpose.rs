//! Rows from inputs that are not stored by cell: CSC `.h5ad` matrices and Matrix
//! Market files (task 2.6).
//!
//! Does: [`TransposedRows`] serves cell rows of an input that can only be scanned
//! whole ([`EntryScan`]), within a memory budget. A first scan counts the entries of
//! every cell. Each later scan collects the entries of the next range of cells that
//! fits the budget (whole cell blocks, at least one), and the rows of that range are
//! served from memory.
//! Does not: spill to disk. A large input stored by gene is read
//! `1 + ceil(entries / budget)` times; that is the price of bounded memory (R8).
//!
//! Memory: 8 bytes per cell (the counts) plus 16 bytes per buffered entry, at most
//! `budget_bytes` or one cell block's entries if that is more.

use kyto_error::KytoError;

use crate::cell_block::CellBlock;
use crate::import_rows::{BlockBuilder, RowSource};
use crate::layer_open::ValueDtype;

/// Bytes one buffered entry takes: a gene position and an `f64` value, aligned.
const ENTRY_BYTES: u64 = 16;

/// A run of input entries: `cells[i]`, `genes[i]` and `values[i]` describe entry `i`.
pub(crate) struct EntryChunk<'a> {
    pub cells: &'a [u32],
    pub genes: &'a [u32],
    pub values: &'a [f64],
}

/// An input whose entries can be visited, all of them, in a fixed order.
pub(crate) trait EntryScan {
    /// Visit every entry once, in chunks, in the same order on every call. Every cell
    /// and gene passed to `visit` is below the matrix shape (implementations check).
    fn scan(
        &mut self,
        visit: &mut dyn FnMut(EntryChunk<'_>) -> Result<(), KytoError>,
    ) -> Result<(), KytoError>;
}

/// The entries of the cells `start..end`, grouped by cell, in scan order per cell.
struct Range {
    start: u64,
    end: u64,
    offsets: Vec<usize>,
    entries: Vec<(u32, f64)>,
}

/// Cell rows of an [`EntryScan`] input (see the module notes).
pub(crate) struct TransposedRows<S> {
    scan: S,
    dtype: ValueDtype,
    n_genes: u64,
    source: String,
    counts: Vec<u64>,
    budget_entries: u64,
    block_rows: u64,
    f32_sums: bool,
    range: Option<Range>,
}

impl<S: EntryScan> TransposedRows<S> {
    /// Count the entries of every cell (the first scan of the input).
    ///
    /// `block_rows` is the row count of the driver's requests (the store's cell block
    /// size); ranges are whole multiples of it.
    pub(crate) fn new(
        mut scan: S,
        dtype: ValueDtype,
        n_cells: u64,
        n_genes: u64,
        source: String,
        budget_bytes: u64,
        block_rows: u64,
    ) -> Result<Self, KytoError> {
        let mut counts = vec![0_u64; n_cells as usize];
        scan.scan(&mut |chunk| {
            for &cell in chunk.cells {
                counts[cell as usize] += 1;
            }
            Ok(())
        })?;
        Ok(Self {
            scan,
            dtype,
            n_genes,
            source,
            counts,
            budget_entries: (budget_bytes / ENTRY_BYTES).max(1),
            block_rows: block_rows.max(1),
            f32_sums: false,
            range: None,
        })
    }

    /// Sum repeated entries in `float32` (as scipy does for `float32` data).
    pub(crate) fn with_f32_sums(mut self, yes: bool) -> Self {
        self.f32_sums = yes;
        self
    }

    /// Collect the entries of the cells from `start`: at least up to `end`, then whole
    /// blocks while they fit the budget.
    fn fill(&mut self, start: u64, end: u64) -> Result<(), KytoError> {
        let n_cells = self.counts.len() as u64;
        let sum =
            |from: u64, to: u64| -> u64 { self.counts[from as usize..to as usize].iter().sum() };
        let mut range_end = end;
        let mut total = sum(start, end);
        while range_end < n_cells {
            let next = (range_end + self.block_rows).min(n_cells);
            let more = sum(range_end, next);
            if total + more > self.budget_entries {
                break;
            }
            total += more;
            range_end = next;
        }
        self.range = None;
        let mut offsets = Vec::with_capacity((range_end - start) as usize + 1);
        offsets.push(0_usize);
        for &count in &self.counts[start as usize..range_end as usize] {
            let last = offsets[offsets.len() - 1];
            offsets.push(last + count as usize);
        }
        let mut cursors = offsets[..offsets.len() - 1].to_vec();
        let mut entries = vec![(0_u32, 0.0_f64); total as usize];
        self.scan.scan(&mut |chunk| {
            for ((&cell, &gene), &value) in chunk.cells.iter().zip(chunk.genes).zip(chunk.values) {
                let cell = u64::from(cell);
                if cell >= start && cell < range_end {
                    let cursor = &mut cursors[(cell - start) as usize];
                    entries[*cursor] = (gene, value);
                    *cursor += 1;
                }
            }
            Ok(())
        })?;
        self.range = Some(Range {
            start,
            end: range_end,
            offsets,
            entries,
        });
        Ok(())
    }
}

impl<S: EntryScan> RowSource for TransposedRows<S> {
    fn dtype(&self) -> ValueDtype {
        self.dtype
    }

    fn read_rows(&mut self, start: u64, end: u64) -> Result<CellBlock, KytoError> {
        let covered = self
            .range
            .as_ref()
            .is_some_and(|range| range.start <= start && end <= range.end);
        if !covered {
            self.fill(start, end)?;
        }
        let mut builder = BlockBuilder::new(self.dtype, self.n_genes, start, &self.source)
            .with_f32_sums(self.f32_sums);
        if let Some(range) = self.range.as_mut() {
            for row in start..end {
                let local = (row - range.start) as usize;
                let (from, to) = (range.offsets[local], range.offsets[local + 1]);
                builder.push_row(&mut range.entries[from..to])?;
            }
        }
        builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use kyto_error::KytoError;

    use super::{EntryChunk, EntryScan, TransposedRows};
    use crate::cell_block::CellValues;
    use crate::import_rows::RowSource;
    use crate::layer_open::ValueDtype;

    /// Entries in a fixed order, handed out two at a time; counts its scans.
    struct Entries {
        cells: Vec<u32>,
        genes: Vec<u32>,
        values: Vec<f64>,
        scans: usize,
    }

    impl EntryScan for Entries {
        fn scan(
            &mut self,
            visit: &mut dyn FnMut(EntryChunk<'_>) -> Result<(), KytoError>,
        ) -> Result<(), KytoError> {
            self.scans += 1;
            for start in (0..self.cells.len()).step_by(2) {
                let end = (start + 2).min(self.cells.len());
                visit(EntryChunk {
                    cells: &self.cells[start..end],
                    genes: &self.genes[start..end],
                    values: &self.values[start..end],
                })?;
            }
            Ok(())
        }
    }

    fn entries() -> Entries {
        // A 5 x 4 matrix stored by gene (CSC order), cell 3 empty, a repeated (4, 1).
        Entries {
            cells: vec![0, 4, 1, 4, 2, 4, 0],
            genes: vec![0, 0, 1, 1, 2, 1, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            scans: 0,
        }
    }

    fn rows(budget_bytes: u64) -> (Vec<u64>, Vec<u32>, Vec<f32>, usize) {
        let mut source = TransposedRows::new(
            entries(),
            ValueDtype::F32,
            5,
            4,
            "x".into(),
            budget_bytes,
            2,
        )
        .unwrap();
        let (mut indptr, mut indices, mut data) = (vec![0_u64], Vec::new(), Vec::new());
        for start in (0..5).step_by(2) {
            let block = source.read_rows(start, (start + 2).min(5)).unwrap();
            let base = indptr[indptr.len() - 1];
            indptr.extend(block.indptr()[1..].iter().map(|offset| offset + base));
            indices.extend_from_slice(block.indices());
            let CellValues::F32(values) = block.data() else {
                panic!("float32 expected")
            };
            data.extend_from_slice(values);
        }
        (indptr, indices, data, source.scan.scans)
    }

    #[test]
    fn rows_do_not_depend_on_the_budget() {
        let expected = (
            vec![0, 2, 3, 4, 4, 6],
            vec![0, 3, 1, 2, 0, 1],
            vec![1.0, 7.0, 3.0, 5.0, 2.0, 10.0],
        );
        // Everything in one range: one counting scan and one collecting scan.
        let (indptr, indices, data, scans) = rows(1 << 20);
        assert_eq!((indptr, indices, data), expected);
        assert_eq!(scans, 2);
        // One entry of budget: one range per cell block (3 blocks of 2 rows).
        let (indptr, indices, data, scans) = rows(1);
        assert_eq!((indptr, indices, data), expected);
        assert_eq!(scans, 4);
    }
}
