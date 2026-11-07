//! Matrices inside HDF5 inputs, read by cell rows (task 2.6).
//!
//! Does: [`CsrRows`] reads row-compressed matrices (anndata `csr_matrix`, and 10x
//! matrices, which are genes x cells stored by cell) one cell block at a time;
//! [`DenseRows`] reads dense 2-d arrays in row slabs; [`CscEntries`] scans
//! column-compressed matrices (anndata `csc_matrix`) for [`TransposedRows`].
//! Does not: interpret encodings or shapes (the format modules do).
//!
//! Memory: `CsrRows` holds one cell block's entries (16 bytes each); `DenseRows` one
//! slab of at most `budget_bytes` (8 bytes per value, at least one row); `CscEntries`
//! its `indptr` (8 bytes per gene) and one chunk of `READ_CHUNK` entries.
//!
//! [`TransposedRows`]: crate::import_transpose::TransposedRows

use hdf5_metno::Dataset;
use kyto_error::KytoError;

use crate::cell_block::CellBlock;
use crate::import_h5::{H5File, READ_CHUNK, length_1d, read_all, read_slice};
use crate::import_rows::{BlockBuilder, RowSource};
use crate::import_transpose::{EntryChunk, EntryScan};
use crate::layer_open::ValueDtype;

/// Advice for structurally broken sparse matrices.
const SPARSE_ADVICE: &str =
    "the sparse matrix breaks the scipy format; re-save it with scipy, anndata or Cell Ranger";

/// The `indptr`, `indices` and `data` datasets of a compressed sparse group.
pub(crate) struct SparseParts {
    /// Member path of the group, for messages.
    pub member: String,
    pub indptr: Dataset,
    pub indices: Dataset,
    pub data: Dataset,
    /// Length of the other axis: indices must be below it.
    pub n_minor: u64,
}

impl SparseParts {
    /// Open the three datasets of `member` and check their lengths.
    ///
    /// # Errors
    /// `KY-E120` if one is missing or not 1-d, `KY-E121` if `indptr` does not have
    /// `n_major + 1` entries or `indices` and `data` differ in length.
    pub(crate) fn open(
        file: &H5File,
        member: &str,
        n_major: u64,
        n_minor: u64,
    ) -> Result<Self, KytoError> {
        let accepted = "a compressed sparse matrix has 1-d datasets indptr, indices and data";
        let open = |name: &str| -> Result<(Dataset, u64), KytoError> {
            let path = format!("{member}/{name}");
            let dataset = file.dataset(&path, accepted)?;
            let length = length_1d(file, &path, &dataset)?;
            Ok((dataset, length))
        };
        let (indptr, indptr_length) = open("indptr")?;
        let (indices, indices_length) = open("indices")?;
        let (data, data_length) = open("data")?;
        let invalid = |problem: String| KytoError::InvalidInput {
            path: file.label(member),
            location: "the whole matrix".to_owned(),
            problem,
            advice: SPARSE_ADVICE.to_owned(),
        };
        if indptr_length != n_major + 1 {
            return Err(invalid(format!(
                "indptr has {indptr_length} entries, but the shape needs {}",
                n_major + 1
            )));
        }
        if indices_length != data_length {
            return Err(invalid(format!(
                "indices has {indices_length} entries but data has {data_length}"
            )));
        }
        Ok(Self {
            member: member.to_owned(),
            indptr,
            indices,
            data,
            n_minor,
        })
    }

    /// Entries stored (the length of `data`).
    fn stored(&self) -> u64 {
        self.data.size() as u64
    }
}

/// Check an `indptr` slice that starts at major position `first`: non-decreasing,
/// within the stored entries, starting at 0 when `first` is 0.
fn check_indptr(
    file: &H5File,
    parts: &SparseParts,
    first: u64,
    indptr: &[i64],
) -> Result<(), KytoError> {
    let invalid = |position: u64, problem: String| KytoError::InvalidInput {
        path: file.label(&parts.member),
        location: format!("indptr[{position}]"),
        problem,
        advice: SPARSE_ADVICE.to_owned(),
    };
    if first == 0 && indptr.first().is_some_and(|&start| start != 0) {
        return Err(invalid(0, format!("indptr starts at {}, not 0", indptr[0])));
    }
    for (offset, pair) in indptr.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(invalid(
                first + offset as u64 + 1,
                format!("indptr decreases from {} to {}", pair[0], pair[1]),
            ));
        }
    }
    let stored = parts.stored();
    if let Some(&last) = indptr.last()
        && (last < 0 || last as u64 > stored)
    {
        return Err(invalid(
            first + indptr.len() as u64 - 1,
            format!("indptr points to entry {last}, but {stored} entries are stored"),
        ));
    }
    Ok(())
}

/// A minor-axis index as a `u32` position, or `KY-E121`.
fn check_index(
    file: &H5File,
    parts: &SparseParts,
    index: i64,
    location: impl FnOnce() -> String,
) -> Result<u32, KytoError> {
    if index < 0 || index as u64 >= parts.n_minor {
        return Err(KytoError::InvalidInput {
            path: file.label(&parts.member),
            location: location(),
            problem: format!("index {index} is outside 0..{}", parts.n_minor),
            advice: SPARSE_ADVICE.to_owned(),
        });
    }
    Ok(index as u32)
}

/// Cell rows of a matrix stored by cell (see the module notes).
pub(crate) struct CsrRows<'a> {
    file: &'a H5File,
    parts: SparseParts,
    dtype: ValueDtype,
    label: String,
    f32_sums: bool,
    entries: Vec<(u32, f64)>,
}

impl<'a> CsrRows<'a> {
    /// Rows of `parts` (cells on the compressed axis), stored as `dtype`.
    pub(crate) fn new(file: &'a H5File, parts: SparseParts, dtype: ValueDtype) -> Self {
        let label = file.label(&parts.member);
        Self {
            file,
            parts,
            dtype,
            label,
            f32_sums: false,
            entries: Vec::new(),
        }
    }

    /// Sum repeated entries in `float32` (for `float32` data, as scipy does).
    pub(crate) fn with_f32_sums(mut self, yes: bool) -> Self {
        self.f32_sums = yes;
        self
    }
}

impl RowSource for CsrRows<'_> {
    fn dtype(&self) -> ValueDtype {
        self.dtype
    }

    fn read_rows(&mut self, start: u64, end: u64) -> Result<CellBlock, KytoError> {
        let (file, parts) = (self.file, &self.parts);
        let indptr = read_slice::<i64>(
            file,
            &parts.member,
            &parts.indptr,
            start as usize..end as usize + 1,
        )?;
        check_indptr(file, parts, start, &indptr)?;
        let (first, last) = (indptr[0] as usize, indptr[indptr.len() - 1] as usize);
        let indices = read_slice::<i64>(file, &parts.member, &parts.indices, first..last)?;
        let data = read_slice::<f64>(file, &parts.member, &parts.data, first..last)?;
        let mut builder = BlockBuilder::new(self.dtype, parts.n_minor, start, &self.label)
            .with_f32_sums(self.f32_sums);
        for (row, pair) in indptr.windows(2).enumerate() {
            let (from, to) = (pair[0] as usize - first, pair[1] as usize - first);
            self.entries.clear();
            for position in from..to {
                let gene = check_index(file, parts, indices[position], || {
                    format!("cell {}, entry {}", start + row as u64, first + position)
                })?;
                self.entries.push((gene, data[position]));
            }
            builder.push_row(&mut self.entries)?;
        }
        builder.finish()
    }
}

/// Cell rows of a dense 2-d `cells x genes` dataset.
pub(crate) struct DenseRows<'a> {
    file: &'a H5File,
    member: String,
    label: String,
    dataset: Dataset,
    dtype: ValueDtype,
    n_genes: u64,
    slab_rows: u64,
    entries: Vec<(u32, f64)>,
}

impl<'a> DenseRows<'a> {
    /// Rows of `dataset` (shape already checked), read in slabs of at most
    /// `budget_bytes`.
    pub(crate) fn new(
        file: &'a H5File,
        member: &str,
        dataset: Dataset,
        dtype: ValueDtype,
        n_genes: u64,
        budget_bytes: u64,
    ) -> Self {
        let row_bytes = (n_genes * 8).max(1);
        Self {
            file,
            member: member.to_owned(),
            label: file.label(member),
            dataset,
            dtype,
            n_genes,
            slab_rows: (budget_bytes / row_bytes).max(1),
            entries: Vec::new(),
        }
    }
}

impl RowSource for DenseRows<'_> {
    fn dtype(&self) -> ValueDtype {
        self.dtype
    }

    fn read_rows(&mut self, start: u64, end: u64) -> Result<CellBlock, KytoError> {
        let mut builder = BlockBuilder::new(self.dtype, self.n_genes, start, &self.label);
        let mut slab_start = start;
        while slab_start < end {
            let slab_end = (slab_start + self.slab_rows).min(end);
            let width = self.n_genes as usize;
            let values = if width == 0 {
                Vec::new()
            } else {
                self.dataset
                    .read_slice_2d::<f64, _>((slab_start as usize..slab_end as usize, 0..width))
                    // HDF5 fills the array in C order: take its buffer, no copy.
                    .map(|array| array.into_raw_vec_and_offset().0)
                    .map_err(|error| self.file.read_error(&self.member, error))?
            };
            for row in 0..(slab_end - slab_start) as usize {
                self.entries.clear();
                let row_values = &values[row * width..(row + 1) * width];
                self.entries.extend(
                    row_values
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| **value != 0.0)
                        .map(|(gene, &value)| (gene as u32, value)),
                );
                builder.push_row(&mut self.entries)?;
            }
            slab_start = slab_end;
        }
        builder.finish()
    }
}

/// The entries of a matrix stored by gene (`csc_matrix`), in storage order.
pub(crate) struct CscEntries<'a> {
    file: &'a H5File,
    parts: SparseParts,
    indptr: Vec<i64>,
}

impl<'a> CscEntries<'a> {
    /// Read and check `indptr` (genes on the compressed axis).
    pub(crate) fn new(file: &'a H5File, parts: SparseParts) -> Result<Self, KytoError> {
        let indptr = read_all::<i64>(file, &parts.member, &parts.indptr)?;
        check_indptr(file, &parts, 0, &indptr)?;
        Ok(Self {
            file,
            parts,
            indptr,
        })
    }
}

impl EntryScan for CscEntries<'_> {
    fn scan(
        &mut self,
        visit: &mut dyn FnMut(EntryChunk<'_>) -> Result<(), KytoError>,
    ) -> Result<(), KytoError> {
        let (file, parts) = (self.file, &self.parts);
        let end = self.indptr.last().copied().unwrap_or(0) as usize;
        let mut gene = 0_usize;
        let mut start = 0_usize;
        let (mut cells, mut genes) = (Vec::new(), Vec::new());
        while start < end {
            let stop = (start + READ_CHUNK).min(end);
            let indices = read_slice::<i64>(file, &parts.member, &parts.indices, start..stop)?;
            let values = read_slice::<f64>(file, &parts.member, &parts.data, start..stop)?;
            cells.clear();
            genes.clear();
            for (offset, &index) in indices.iter().enumerate() {
                let position = start + offset;
                while self.indptr[gene + 1] as usize <= position {
                    gene += 1;
                }
                let cell = check_index(file, parts, index, || {
                    format!("gene {gene}, entry {position}")
                })?;
                cells.push(cell);
                genes.push(gene as u32);
            }
            visit(EntryChunk {
                cells: &cells,
                genes: &genes,
                values: &values,
            })?;
            start = stop;
        }
        Ok(())
    }
}
