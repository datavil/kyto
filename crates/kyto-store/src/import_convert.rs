//! `convert`: import an `.h5ad`, 10x `.h5` or Matrix Market directory into a new store
//! (task 2.6).
//!
//! Does: detect the input kind, read its tables, apply the duplicate-ID policy before
//! any matrix is read, then stream every matrix through [`StoreWriter`] one cell block
//! at a time (all layers in step) and record the input in `provenance.json` (§9).
//! Does not: keep a matrix in memory. Memory: the two tables, one cell block per layer
//! and, for inputs not stored by cell (CSC `.h5ad`, `.mtx`), up to
//! `memory_budget_bytes` of buffered entries per such layer (see `import_transpose`).

use std::io::Read;
use std::path::Path;

use kyto_error::KytoError;
use polars::prelude::DataFrame;
use sha2::{Digest, Sha256};

use crate::import_10x::read_10x;
use crate::import_h5::H5File;
use crate::import_h5ad::read_h5ad;
use crate::import_ids::{OnDuplicateIds, suffix_ids};
use crate::import_mtx::{MtxFiles, read_mtx};
use crate::import_rows::RowSource;
use crate::store_json::ImportSource;
use crate::store_writer::{LayerSpec, StoreWriter};
use crate::table_axis::TableAxis;
use crate::table_check::{PlainStrings, check_table};
use crate::tiling::Tiling;
use crate::write_options::LayerWriteOptions;

/// Default of [`ConvertOptions::memory_budget_bytes`]: 256 MiB.
pub const DEFAULT_IMPORT_BUDGET_BYTES: u64 = 256 << 20;

/// Options of [`convert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertOptions {
    /// Layer name of the main matrix: `.h5ad` `X`, the 10x or `.mtx` counts. `.h5ad`
    /// `layers/*` keep their own names.
    pub layer: String,
    /// What to do with repeated cell or gene IDs.
    pub on_duplicate_ids: OnDuplicateIds,
    /// Replace an existing kyto store at the target (never anything else).
    pub overwrite: bool,
    /// Tile sizes of the store.
    pub tiling: Tiling,
    /// Chunking of every layer's arrays.
    pub layer_options: LayerWriteOptions,
    /// Bytes of buffered entries per layer for inputs not stored by cell, and of a
    /// dense slab. Larger budgets read such inputs fewer times.
    pub memory_budget_bytes: u64,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            layer: "counts".to_owned(),
            on_duplicate_ids: OnDuplicateIds::Error,
            overwrite: false,
            tiling: Tiling::default(),
            layer_options: LayerWriteOptions::default(),
            memory_budget_bytes: DEFAULT_IMPORT_BUDGET_BYTES,
        }
    }
}

/// One matrix of an input and the layer it becomes.
pub(crate) struct ImportLayer<'a> {
    pub name: String,
    pub source: Box<dyn RowSource + 'a>,
}

/// An input, read up to its matrices.
pub(crate) struct Import<'a> {
    /// `ImportSource::kind`: `h5ad`, `10x_h5` or `mtx`.
    pub kind: &'static str,
    /// Cell table, `cell_id` first, one row per matrix row.
    pub cells: DataFrame,
    /// Gene table, `gene_id` first, one row per matrix column.
    pub genes: DataFrame,
    /// Enum columns whose categories are ordered.
    pub ordered: Vec<(TableAxis, String)>,
    pub layers: Vec<ImportLayer<'a>>,
}

/// The input kinds [`convert`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputKind {
    H5ad,
    TenxH5,
    Mtx,
}

pub(crate) fn input_kind(source: &Path) -> Result<InputKind, KytoError> {
    let text = source.display().to_string();
    if let Err(error) = std::fs::metadata(source) {
        return Err(KytoError::StoreIo {
            operation: "open input".to_owned(),
            path: text,
            detail: error.to_string(),
        });
    }
    if source.is_dir() {
        return Ok(InputKind::Mtx);
    }
    let name = source
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.ends_with(".h5ad") {
        Ok(InputKind::H5ad)
    } else if name.ends_with(".h5") {
        Ok(InputKind::TenxH5)
    } else if name.ends_with(".mtx") || name.ends_with(".mtx.gz") {
        Ok(InputKind::Mtx)
    } else {
        Err(KytoError::UnsupportedInput {
            path: text,
            problem: "the file name does not end in .h5ad, .h5, .mtx or .mtx.gz".to_owned(),
            accepted:
                "pass an .h5ad file, a 10x .h5 file, or a 10x matrix directory (or its matrix.mtx)"
                    .to_owned(),
        })
    }
}

/// Import `source` into a new store at `target` (format.md §10.1: nothing exists at
/// `target` unless the whole import succeeds).
///
/// `source` is an `.h5ad` file (anndata >= 0.8), a 10x `.h5` file (Cell Ranger v2 or
/// v3), or a 10x matrix directory (`matrix.mtx`, `barcodes.tsv`, `features.tsv` or
/// `genes.tsv`, optionally gzipped) or its `.mtx` file. Integer matrices become
/// `uint32` layers, float matrices `float32` (format.md §4.2).
///
/// # Errors
/// - `KY-E110` if `source` cannot be read or the store cannot be written.
/// - `KY-E120` for an unknown input kind, encoding or dtype; `KY-E121` for invalid
///   values or structure (negative counts in an integer matrix, indices out of range).
/// - `KY-E101` for repeated IDs with [`OnDuplicateIds::Error`], raised before any
///   matrix is read; `KY-E111`, `KY-E112` for null IDs or bad columns.
/// - `KY-E102`, `KY-E103` for bad layer names; `KY-E104`, `KY-E105` for bad options or
///   inputs above the format limits; `KY-E118` if `target` exists.
pub fn convert(source: &Path, target: &Path, options: &ConvertOptions) -> Result<(), KytoError> {
    let kind = input_kind(source)?;
    let budget = options.memory_budget_bytes;
    let block_rows = options.tiling.cell_block_size;
    match kind {
        InputKind::H5ad => {
            let file = H5File::open(source)?;
            let import = read_h5ad(&file, &options.layer, budget, block_rows)?;
            write_import(import, source, source, target, options)
        }
        InputKind::TenxH5 => {
            let file = H5File::open(source)?;
            let import = read_10x(&file, &options.layer)?;
            write_import(import, source, source, target, options)
        }
        InputKind::Mtx => {
            let files = MtxFiles::find(source)?;
            let import = read_mtx(&files, &options.layer, budget, block_rows)?;
            write_import(import, source, &files.matrix, target, options)
        }
    }
}

/// Write an input to the store: tables checked first, then the matrices streamed.
fn write_import(
    mut import: Import<'_>,
    source: &Path,
    hashed: &Path,
    target: &Path,
    options: &ConvertOptions,
) -> Result<(), KytoError> {
    if options.on_duplicate_ids == OnDuplicateIds::Suffix {
        suffix_ids(&mut import.cells, TableAxis::Cells)?;
        suffix_ids(&mut import.genes, TableAxis::Genes)?;
    }
    // Fail early (before reading any matrix): the gene table is checked by `create`.
    check_table(TableAxis::Cells, &import.cells, PlainStrings::Allowed)?;
    let specs: Vec<LayerSpec> = import
        .layers
        .iter()
        .map(|layer| LayerSpec {
            name: layer.name.clone(),
            dtype: layer.source.dtype(),
            options: options.layer_options,
        })
        .collect();
    let mut writer = StoreWriter::create(
        target,
        &import.genes,
        &specs,
        options.tiling,
        options.overwrite,
    )?;
    for (axis, column) in &import.ordered {
        writer.mark_ordered(*axis, column);
    }
    if !import.layers.is_empty() {
        let n_cells = import.cells.height() as u64;
        let step = options.tiling.cell_block_size.max(1);
        let mut start = 0;
        while start < n_cells {
            let end = (start + step).min(n_cells);
            let blocks = import
                .layers
                .iter_mut()
                .map(|layer| layer.source.read_rows(start, end))
                .collect::<Result<Vec<_>, KytoError>>()?;
            writer.push(&blocks)?;
            start = end;
        }
    }
    let provenance = ImportSource {
        kind: import.kind.to_owned(),
        path: source.display().to_string(),
        sha256: sha256_file(hashed)?,
    };
    writer.finish(&import.cells, Some(&provenance))
}

/// Lowercase hex SHA-256 of a file, read in 1 MiB chunks.
fn sha256_file(path: &Path) -> Result<String, KytoError> {
    let failed = |error: std::io::Error| KytoError::StoreIo {
        operation: "hash input".to_owned(),
        path: path.display().to_string(),
        detail: error.to_string(),
    };
    let mut file = std::fs::File::open(path).map_err(failed)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(failed)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
