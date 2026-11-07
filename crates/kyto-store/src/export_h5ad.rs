//! `sink_h5ad`: export a store as an `.h5ad` file for scanpy and anndata (task 2.7).
//!
//! Does: check the target and the options before reading the matrix, then write the
//! anndata 0.13 layout (encodings as anndata's h5py writer, the ones the 2.6 importer
//! reads): `obs` and `var` dataframes from the tables (`export_h5ad_table`), the chosen
//! layer as `X` and every other layer under `layers/`, all as `csr_matrix` 0.1.0, and
//! empty `obsm`, `varm`, `obsp`, `varp` and `uns`. Each layer is streamed one cell
//! block at a time through the tile reader (R8): memory is the two tables and
//! `blocks_in_flight` cell blocks. The file is written next to the target and renamed
//! into place only when complete (`atomic_file`).
//! Does not: export embeddings or graphs (not stored yet), compress, or write dense
//! matrices (R8; anndata reads CSR as scipy CSR).

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use hdf5_metno::{Dataset, H5Type};
use kyto_error::KytoError;
use zarrs::filesystem::FilesystemStore;

use crate::atomic_file::AtomicFile;
use crate::cell_block::CellValues;
use crate::export_h5::H5Out;
use crate::export_h5ad_table::write_table;
use crate::layer_open::ValueDtype;
use crate::selection::AxisSelection;
use crate::store_schema::{SchemaLayer, StoreSchema, read_schema};
use crate::store_version::check_format_version;
use crate::table_axis::TableAxis;
use crate::table_io::read_table;
use crate::tile_reader::LayerReader;
use crate::zarr_arrays::store_io;

/// Default of [`SinkH5adOptions::blocks_in_flight`].
pub const DEFAULT_EXPORT_BLOCKS_IN_FLIGHT: usize = 2;

/// Options of [`sink_h5ad`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkH5adOptions {
    /// The layer written as `X`; every other layer goes to `layers/`. `None` writes no
    /// `X` (anndata then has `X = None`) and every layer under `layers/`.
    pub x_layer: Option<String>,
    /// Replace an existing file at the target (never a directory).
    pub overwrite: bool,
    /// Cell blocks read at once per layer; memory grows with it.
    pub blocks_in_flight: NonZeroUsize,
}

impl Default for SinkH5adOptions {
    fn default() -> Self {
        Self {
            x_layer: None,
            overwrite: false,
            blocks_in_flight: NonZeroUsize::MIN.saturating_add(DEFAULT_EXPORT_BLOCKS_IN_FLIGHT - 1),
        }
    }
}

/// Export the store at `store` as an `.h5ad` file at `target`.
///
/// `anndata.read_h5ad(target)` gives one `obs` row per cell (index `cell_id`), one
/// `var` row per gene (index `gene_id`), the layer `options.x_layer` as `X` and the
/// other layers in `layers`, all as scipy CSR with the layer's dtype (`float32` or
/// `uint32`). `Enum` columns become categoricals (category order and `ordered` kept),
/// integer and boolean columns with nulls nullable arrays. Nothing exists at `target`
/// unless the whole export succeeds.
///
/// # Errors
/// - `KY-E116`, `KY-E115` if `store` is not a kyto store of a supported version;
///   `KY-E122` if its `schema.json` is invalid; `KY-E113`, `KY-E107` for a corrupt table
///   or layer.
/// - `KY-E109` if `options.x_layer` is not a layer of the store.
/// - `KY-E123` if `target` exists (and `overwrite` is off, or it is a directory).
/// - `KY-E124` for a column name or string h5ad cannot hold.
/// - `KY-E110` if the store cannot be read or the file cannot be written.
pub fn sink_h5ad(store: &Path, target: &Path, options: &SinkH5adOptions) -> Result<(), KytoError> {
    let store_text = store.display().to_string();
    let storage = Arc::new(
        FilesystemStore::new(store).map_err(|error| store_io("open store", &store_text, error))?,
    );
    check_format_version(&*storage, &store_text)?;
    let schema = read_schema(store)?;
    if let Some(name) = &options.x_layer
        && !schema.layers.iter().any(|layer| &layer.name == name)
    {
        return Err(KytoError::MissingMember {
            kind: "layer".to_owned(),
            name: name.clone(),
            store: store_text,
        });
    }
    // Fail early: the target before anything is read or written.
    check_outside_store(store, target)?;
    let file = AtomicFile::create(target, options.overwrite)?;
    let cells = read_table(store, TableAxis::Cells)?;
    let genes = read_table(store, TableAxis::Genes)?;
    let out = H5Out::create(file.path(), target)?;
    write_file(
        &out,
        &storage,
        &store_text,
        &schema,
        &cells,
        &genes,
        options,
    )?;
    let H5Out { file: h5, target } = out;
    h5.close().map_err(|error| KytoError::StoreIo {
        operation: "close".to_owned(),
        path: target,
        detail: error.to_string(),
    })?;
    file.commit()
}

/// `KY-E123` if `target` lies inside the store: a writer adds no members (format.md §2).
fn check_outside_store(store: &Path, target: &Path) -> Result<(), KytoError> {
    let parent = match target.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let (Ok(store), Ok(parent)) = (store.canonicalize(), parent.canonicalize()) else {
        return Ok(()); // A missing parent is reported by `AtomicFile::create`.
    };
    if parent.starts_with(&store) {
        return Err(KytoError::OutputExists {
            path: target.display().to_string(),
            problem: "it is inside the store being exported".to_owned(),
        });
    }
    Ok(())
}

/// Write every member of the file.
fn write_file(
    out: &H5Out,
    storage: &Arc<FilesystemStore>,
    store: &str,
    schema: &StoreSchema,
    cells: &polars::prelude::DataFrame,
    genes: &polars::prelude::DataFrame,
    options: &SinkH5adOptions,
) -> Result<(), KytoError> {
    out.encode("", &out.file, ("anndata", "0.1.0"))?;
    let ordered = |axis: TableAxis| -> Vec<&str> {
        schema
            .ordered
            .iter()
            .filter(|(column_axis, _)| *column_axis == axis)
            .map(|(_, name)| name.as_str())
            .collect()
    };
    write_table(
        out,
        "obs",
        TableAxis::Cells,
        cells,
        &ordered(TableAxis::Cells),
    )?;
    write_table(
        out,
        "var",
        TableAxis::Genes,
        genes,
        &ordered(TableAxis::Genes),
    )?;
    let x_layer = options.x_layer.as_deref();
    for layer in schema
        .layers
        .iter()
        .filter(|layer| Some(layer.name.as_str()) == x_layer)
    {
        write_layer(out, "X", storage, store, schema, layer, options)?;
    }
    out.group("layers", ("dict", "0.1.0"))?;
    for layer in schema
        .layers
        .iter()
        .filter(|layer| Some(layer.name.as_str()) != x_layer)
    {
        let member = format!("layers/{}", layer.name);
        write_layer(out, &member, storage, store, schema, layer, options)?;
    }
    for member in ["obsm", "varm", "obsp", "varp", "uns"] {
        out.group(member, ("dict", "0.1.0"))?;
    }
    Ok(())
}

/// The CSR datasets of one matrix, written block by block.
struct CsrOut<'a> {
    out: &'a H5Out,
    member: String,
    data: Dataset,
    indices: Dataset,
    indptr: Dataset,
    /// int32 indices (`n_genes` and `nnz` fit), as scipy uses when it can.
    narrow_indices: bool,
    /// int32 indptr (same rule).
    narrow_indptr: bool,
}

impl CsrOut<'_> {
    fn write_index<T: H5Type>(
        &self,
        name: &str,
        dataset: &Dataset,
        offset: u64,
        values: &[T],
    ) -> Result<(), KytoError> {
        self.out
            .write_at(&format!("{}/{name}", self.member), dataset, offset, values)
    }

    /// Indices or indptr values at `offset`, in the dataset's width.
    fn write_positions(
        &self,
        name: &str,
        dataset: &Dataset,
        narrow: bool,
        offset: u64,
        values: impl Iterator<Item = u64>,
    ) -> Result<(), KytoError> {
        if narrow {
            let values: Vec<i32> = values.map(|value| value as i32).collect();
            self.write_index(name, dataset, offset, &values)
        } else {
            let values: Vec<i64> = values.map(|value| value as i64).collect();
            self.write_index(name, dataset, offset, &values)
        }
    }
}

/// Stream layer `layer` into the `csr_matrix` group `member`.
fn write_layer(
    out: &H5Out,
    member: &str,
    storage: &Arc<FilesystemStore>,
    store: &str,
    schema: &StoreSchema,
    layer: &SchemaLayer,
    options: &SinkH5adOptions,
) -> Result<(), KytoError> {
    let grid = schema.grid;
    let (n_cells, n_genes) = (grid.n_cells(), grid.n_genes());
    let reader = LayerReader::open(Arc::clone(storage), store, &layer.name, grid)?;
    let group = out.group(member, ("csr_matrix", "0.1.0"))?;
    out.attr_array(member, &group, "shape", &[n_cells as i64, n_genes as i64])?;
    let fits = |value: u64| value <= i32::MAX as u64;
    let data_member = format!("{member}/data");
    let data = match layer.dtype {
        ValueDtype::F32 => out.empty_dataset::<f32>(&data_member, layer.nnz)?,
        ValueDtype::U32 => out.empty_dataset::<u32>(&data_member, layer.nnz)?,
    };
    // One width for both, as scipy keeps them: int32 when every index fits.
    let narrow_indices = fits(n_genes) && fits(layer.nnz);
    let narrow_indptr = narrow_indices;
    let indices_member = format!("{member}/indices");
    let indptr_member = format!("{member}/indptr");
    let indices = if narrow_indices {
        out.empty_dataset::<i32>(&indices_member, layer.nnz)?
    } else {
        out.empty_dataset::<i64>(&indices_member, layer.nnz)?
    };
    let indptr = if narrow_indptr {
        out.empty_dataset::<i32>(&indptr_member, n_cells + 1)?
    } else {
        out.empty_dataset::<i64>(&indptr_member, n_cells + 1)?
    };
    let csr = CsrOut {
        out,
        member: member.to_owned(),
        data,
        indices,
        indptr,
        narrow_indices,
        narrow_indptr,
    };
    csr.write_positions("indptr", &csr.indptr, narrow_indptr, 0, std::iter::once(0))?;

    let corrupt = |problem: String| KytoError::CorruptLayer {
        layer: layer.name.clone(),
        problem,
    };
    let mut entries = 0_u64;
    let mut rows = 0_u64;
    for selected in reader.cell_blocks(
        &AxisSelection::All,
        &AxisSelection::All,
        options.blocks_in_flight,
    )? {
        let block = selected?.block;
        let block_nnz = block.nnz();
        if entries + block_nnz > layer.nnz {
            return Err(corrupt(format!(
                "it holds more entries than the {} schema.json gives",
                layer.nnz
            )));
        }
        match block.data() {
            CellValues::F32(values) => csr.write_index("data", &csr.data, entries, values)?,
            CellValues::U32(values) => csr.write_index("data", &csr.data, entries, values)?,
        }
        csr.write_positions(
            "indices",
            &csr.indices,
            csr.narrow_indices,
            entries,
            block.indices().iter().map(|&index| u64::from(index)),
        )?;
        csr.write_positions(
            "indptr",
            &csr.indptr,
            csr.narrow_indptr,
            rows + 1,
            block.indptr()[1..].iter().map(|&offset| entries + offset),
        )?;
        entries += block_nnz;
        rows += block.n_rows() as u64;
    }
    if entries != layer.nnz || rows != n_cells {
        return Err(corrupt(format!(
            "it holds {entries} entries in {rows} rows, but schema.json gives {} entries in {n_cells} cells",
            layer.nnz
        )));
    }
    Ok(())
}
