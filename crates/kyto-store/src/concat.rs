//! `concat`: combine several stores or importer inputs into one new store (task 2.9),
//! as `anndata.concat(..., label=..., keys=..., join=...)`.
//!
//! Does: open every source (a kyto store, or an input [`crate::convert`] reads), join
//! the gene axes ([`Join`]), build the combined cell and gene tables (`concat_tables`)
//! and check them before any matrix is read, then stream each source's cells in order,
//! one cell block at a time, remapping its genes onto the combined gene axis, through
//! [`StoreWriter`]. Layers: those of every source (`"inner"`) or of any (`"outer"`, a
//! source without one gives absent values, R9); `uint32` mixed with `float32` gives
//! `float32`.
//! Does not: keep a matrix in memory or read sources in parallel. Memory: every
//! source's tables, the combined tables, and one cell block per layer of the source
//! being read (plus an input's own buffers, see `import_convert`).

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kyto_error::KytoError;
use polars::prelude::DataFrame;
use zarrs::filesystem::FilesystemStore;

use crate::aligned_buffer::AlignedBuffer;
use crate::cell_block::{CellBlock, CellBlockParts, CellData, CellValues};
use crate::concat_tables::{
    Join, SourceTable, concat_cell_tables, join_ids, merge_gene_table, source_rows,
};
use crate::import_10x::read_10x;
use crate::import_convert::{ConvertOptions, Import, ImportLayer, InputKind, input_kind};
use crate::import_h5::H5File;
use crate::import_h5ad::read_h5ad;
use crate::import_ids::{OnDuplicateIds, suffix_ids};
use crate::import_mtx::{MtxFiles, read_mtx};
use crate::import_rows::RowSource;
use crate::layer_open::ValueDtype;
use crate::selection::AxisSelection;
use crate::store_schema::read_schema;
use crate::store_writer::{LayerSpec, StoreWriter};
use crate::table_axis::TableAxis;
use crate::table_check::{PlainStrings, check_table};
use crate::table_io::read_table;
use crate::tile_reader::LayerReader;
use crate::zarr_arrays::store_io;

/// Options of [`concat`]; the importer settings (layer name of an input's main matrix,
/// tiling, chunking, memory budget) are those of [`ConvertOptions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatOptions {
    /// Name of the cell column naming each cell's source.
    pub label: String,
    /// The label of each source, in source order; `None` takes the source names (file
    /// or directory name without a known extension).
    pub keys: Option<Vec<String>>,
    /// How the gene axes are joined.
    pub join: Join,
    /// Repeated cell IDs across sources (and repeated IDs inside an input, as for
    /// [`crate::convert`]).
    pub on_duplicate_ids: OnDuplicateIds,
    /// Replace an existing kyto store at the target (never anything else).
    pub overwrite: bool,
    /// How inputs are imported and the store is written.
    pub convert: ConvertOptions,
}

impl Default for ConcatOptions {
    fn default() -> Self {
        Self {
            label: "sample".to_owned(),
            keys: None,
            join: Join::Inner,
            on_duplicate_ids: OnDuplicateIds::Error,
            overwrite: false,
            convert: ConvertOptions::default(),
        }
    }
}

/// A source opened up to its matrices.
enum Opened {
    Store(PathBuf),
    H5(H5File, InputKind),
    Mtx(MtxFiles),
}

/// A source read up to its matrices: tables and one row source per layer.
struct Source<'a> {
    cells: DataFrame,
    genes: DataFrame,
    ordered: Vec<(TableAxis, String)>,
    layers: Vec<ImportLayer<'a>>,
    /// Rows read per step: a store's cell block size, else the output's.
    step: u64,
}

/// Combine `sources` into a new store at `target` (format.md §10.1: nothing exists at
/// `target` unless the whole write succeeds), as
/// `anndata.concat(sources, label=label, keys=keys, join=join)`.
///
/// Each source is a kyto store (a directory with a root `zarr.json`) or an input
/// [`crate::convert`] reads. Cells come in source order; the cell table holds the
/// sources' columns (see `concat_tables`) and `label`, an `Enum` of `keys`. Genes are
/// matched by `gene_id`.
///
/// # Errors
/// - `KY-E104` for no sources, sources with layers but none in common under
///   `"inner"`, a `keys` of another length, repeated or default keys
///   that repeat (pass `keys`), or a `label` that is the `cell_id` column.
/// - `KY-E101` for a cell ID in more than one source with [`OnDuplicateIds::Error`],
///   before any matrix is read.
/// - `KY-E118` if `target` exists and may not be replaced, or is a source.
/// - `KY-E112` for a cell column whose dtypes have no common type.
/// - Those of [`crate::convert`] for an input, of the tile reader for a store.
pub fn concat(
    sources: &[PathBuf],
    target: &Path,
    options: &ConcatOptions,
) -> Result<(), KytoError> {
    let keys = source_keys(sources, options.keys.as_deref())?;
    if options.label == TableAxis::Cells.id_column() {
        return Err(KytoError::InvalidParameter {
            parameter: "label".to_owned(),
            value: options.label.clone(),
            accepted: "a column name other than 'cell_id'".to_owned(),
        });
    }
    check_target(sources, target)?;
    let opened = sources
        .iter()
        .map(|source| open(source))
        .collect::<Result<Vec<_>, KytoError>>()?;
    let mut read = opened
        .iter()
        .map(|opened| read_source(opened, options))
        .collect::<Result<Vec<_>, KytoError>>()?;

    let tables = |axis: TableAxis| -> Vec<SourceTable<'_>> {
        read.iter()
            .map(|source| SourceTable {
                frame: match axis {
                    TableAxis::Cells => &source.cells,
                    TableAxis::Genes => &source.genes,
                },
                ordered: source
                    .ordered
                    .iter()
                    .filter(|(column_axis, _)| *column_axis == axis)
                    .map(|(_, name)| name.clone())
                    .collect(),
            })
            .collect()
    };
    let gene_ids = read
        .iter()
        .map(|source| string_ids(&source.genes, TableAxis::Genes))
        .collect::<Result<Vec<_>, KytoError>>()?;
    let merged = join_ids(&gene_ids, options.join);
    let rows: Vec<Vec<Option<u64>>> = gene_ids
        .iter()
        .map(|ids| source_rows(&merged, ids))
        .collect();
    let (genes, gene_ordered) = merge_gene_table(&tables(TableAxis::Genes), &merged, &rows)?;
    let (mut cells, cell_ordered) = concat_cell_tables(
        &tables(TableAxis::Cells),
        options.join,
        &options.label,
        &keys,
    )?;
    if options.on_duplicate_ids == OnDuplicateIds::Suffix {
        suffix_ids(&mut cells, TableAxis::Cells)?;
    }
    // Fail early (before reading any matrix): the gene table is checked by `create`.
    check_table(TableAxis::Cells, &cells, PlainStrings::Allowed)?;

    let specs = layer_specs(&read, options)?;
    let maps: Vec<GeneMap> = gene_ids
        .iter()
        .zip(&merged_positions(&merged, &gene_ids))
        .map(|(ids, positions)| GeneMap::new(positions.clone(), ids.len(), merged.len()))
        .collect();
    let mut writer = StoreWriter::create(
        target,
        &genes,
        &specs,
        options.convert.tiling,
        options.overwrite,
    )?;
    for column in gene_ordered {
        writer.mark_ordered(TableAxis::Genes, &column);
    }
    for column in cell_ordered {
        writer.mark_ordered(TableAxis::Cells, &column);
    }
    for (source, map) in read.iter_mut().zip(&maps) {
        write_source(&mut writer, source, map, &specs)?;
    }
    writer.finish(&cells, None)
}

/// The label of each source: `keys`, or the source names.
fn source_keys(sources: &[PathBuf], keys: Option<&[String]>) -> Result<Vec<String>, KytoError> {
    if sources.is_empty() {
        return Err(KytoError::InvalidParameter {
            parameter: "sources".to_owned(),
            value: "[]".to_owned(),
            accepted: "at least one store or input file".to_owned(),
        });
    }
    let keys: Vec<String> = match keys {
        Some(keys) if keys.len() != sources.len() => {
            return Err(KytoError::InvalidParameter {
                parameter: "keys".to_owned(),
                value: format!("{} keys", keys.len()),
                accepted: format!("one key per source ({})", sources.len()),
            });
        }
        Some(keys) => keys.to_vec(),
        None => sources.iter().map(|source| source_name(source)).collect(),
    };
    let mut seen = HashSet::new();
    if let Some(repeated) = keys.iter().find(|key| !seen.insert(key.as_str())) {
        return Err(KytoError::InvalidParameter {
            parameter: "keys".to_owned(),
            value: format!("'{repeated}' (twice)"),
            accepted:
                "a different key for every source; pass keys=[...] when sources share a file name"
                    .to_owned(),
        });
    }
    Ok(keys)
}

/// A source's file or directory name without a known extension.
fn source_name(source: &Path) -> String {
    let name = source
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| source.display().to_string());
    let lower = name.to_lowercase();
    for extension in [".h5ad", ".mtx.gz", ".mtx", ".h5", ".kyto", ".zarr"] {
        if lower.ends_with(extension) && lower.len() > extension.len() {
            return name[..name.len() - extension.len()].to_owned();
        }
    }
    name
}

/// `KY-E118` if `target` is one of the sources, lies inside one or holds one.
fn check_target(sources: &[PathBuf], target: &Path) -> Result<(), KytoError> {
    let parent = match target.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let (Ok(parent), Some(name)) = (parent.canonicalize(), target.file_name()) else {
        return Ok(()); // A missing parent is reported by `StoreWriter::create`.
    };
    let target = parent.join(name);
    // The target itself, if it exists (a symlink resolves to what it replaces).
    let target = target.canonicalize().unwrap_or(target);
    for source in sources {
        if let Ok(source) = source.canonicalize()
            && (target.starts_with(&source) || source.starts_with(&target))
        {
            return Err(KytoError::StoreExists {
                path: target.display().to_string(),
                problem: format!(
                    "it is the source '{}', lies inside it or holds it",
                    source.display()
                ),
            });
        }
    }
    Ok(())
}

/// Open a source: a kyto store or an importer input.
fn open(source: &Path) -> Result<Opened, KytoError> {
    if source.is_dir() && source.join("zarr.json").is_file() {
        return Ok(Opened::Store(source.to_path_buf()));
    }
    Ok(match input_kind(source)? {
        kind @ (InputKind::H5ad | InputKind::TenxH5) => Opened::H5(H5File::open(source)?, kind),
        InputKind::Mtx => Opened::Mtx(MtxFiles::find(source)?),
    })
}

/// Read a source's tables and open its layers.
fn read_source<'a>(opened: &'a Opened, options: &ConcatOptions) -> Result<Source<'a>, KytoError> {
    let convert = &options.convert;
    let budget = convert.memory_budget_bytes;
    let block_rows = convert.tiling.cell_block_size;
    let import = match opened {
        Opened::Store(store) => return read_store(store),
        Opened::H5(file, InputKind::H5ad) => read_h5ad(file, &convert.layer, budget, block_rows)?,
        Opened::H5(file, _) => read_10x(file, &convert.layer)?,
        Opened::Mtx(files) => read_mtx(files, &convert.layer, budget, block_rows)?,
    };
    let Import {
        mut cells,
        mut genes,
        ordered,
        layers,
        ..
    } = import;
    if options.on_duplicate_ids == OnDuplicateIds::Suffix {
        suffix_ids(&mut cells, TableAxis::Cells)?;
        suffix_ids(&mut genes, TableAxis::Genes)?;
    }
    Ok(Source {
        cells,
        genes,
        ordered,
        layers,
        step: block_rows.max(1),
    })
}

/// A kyto store as a source: its tables and a tile reader per layer.
fn read_store<'a>(store: &Path) -> Result<Source<'a>, KytoError> {
    let store_text = store.display().to_string();
    let storage = Arc::new(
        FilesystemStore::new(store).map_err(|error| store_io("open store", &store_text, error))?,
    );
    let schema = read_schema(store)?;
    let cells = read_table(store, TableAxis::Cells)?;
    let genes = read_table(store, TableAxis::Genes)?;
    let layers = schema
        .layers
        .iter()
        .map(|layer| {
            let reader = LayerReader::open(storage.clone(), &store_text, &layer.name, schema.grid)?;
            Ok(ImportLayer {
                name: layer.name.clone(),
                source: Box::new(StoreRows {
                    reader,
                    dtype: layer.dtype,
                }) as Box<dyn RowSource>,
            })
        })
        .collect::<Result<Vec<_>, KytoError>>()?;
    Ok(Source {
        cells,
        genes,
        ordered: schema.ordered,
        layers,
        step: schema.grid.cell_block_size().max(1),
    })
}

/// The `String` IDs of a table (its first column, checked by its reader).
fn string_ids(frame: &DataFrame, axis: TableAxis) -> Result<Vec<String>, KytoError> {
    check_table(axis, frame, PlainStrings::Allowed)?;
    let column = frame
        .columns()
        .first()
        .and_then(|column| column.str().ok())
        .ok_or_else(|| KytoError::InvalidTableColumn {
            axis: axis.as_str().to_owned(),
            column: axis.id_column().to_owned(),
            problem: "is missing or not String".to_owned(),
            accepted: "a String ID column first".to_owned(),
        })?;
    Ok(column.iter().flatten().map(str::to_owned).collect())
}

/// For each source, the combined position of each of its genes (`None` if dropped).
fn merged_positions(merged: &[String], sources: &[Vec<String>]) -> Vec<Vec<Option<u32>>> {
    let positions: std::collections::HashMap<&str, u32> = merged
        .iter()
        .enumerate()
        .map(|(position, id)| (id.as_str(), position as u32))
        .collect();
    sources
        .iter()
        .map(|ids| {
            ids.iter()
                .map(|id| positions.get(id.as_str()).copied())
                .collect()
        })
        .collect()
}

/// The combined store's layers: in every source (`"inner"`) or any (`"outer"`), in
/// the first source's order then order of appearance; `float32` if any source stores
/// it as `float32`.
fn layer_specs(
    sources: &[Source<'_>],
    options: &ConcatOptions,
) -> Result<Vec<LayerSpec>, KytoError> {
    let mut specs: Vec<LayerSpec> = Vec::new();
    for source in sources {
        for layer in &source.layers {
            let dtype = layer.source.dtype();
            match specs.iter_mut().find(|spec| spec.name == layer.name) {
                Some(spec) => {
                    if dtype == ValueDtype::F32 {
                        spec.dtype = ValueDtype::F32;
                    }
                }
                None => specs.push(LayerSpec {
                    name: layer.name.clone(),
                    dtype,
                    options: options.convert.layer_options,
                }),
            }
        }
    }
    let any_layer = !specs.is_empty();
    if options.join == Join::Inner {
        specs.retain(|spec| {
            sources
                .iter()
                .all(|source| source.layers.iter().any(|layer| layer.name == spec.name))
        });
    }
    // Sources with matrices but no layer name in common would give a store without
    // any matrix: refuse instead of dropping every value silently.
    if any_layer && specs.is_empty() {
        return Err(KytoError::InvalidParameter {
            parameter: "sources".to_owned(),
            value: "no layer name in every source".to_owned(),
            accepted: format!(
                "sources sharing at least one layer name; pass layer=... so an input's main matrix gets a store's layer name (now '{}'), or join='outer'",
                options.convert.layer
            ),
        });
    }
    Ok(specs)
}

/// Stream one source's cells into the writer, one step of rows at a time.
fn write_source(
    writer: &mut StoreWriter,
    source: &mut Source<'_>,
    map: &GeneMap,
    specs: &[LayerSpec],
) -> Result<(), KytoError> {
    let n_cells = source.cells.height() as u64;
    if specs.is_empty() {
        return Ok(());
    }
    let mut start = 0;
    while start < n_cells {
        let end = (start + source.step).min(n_cells);
        let mut blocks = Vec::with_capacity(specs.len());
        for spec in specs {
            let block = match source
                .layers
                .iter_mut()
                .find(|layer| layer.name == spec.name)
            {
                Some(layer) => map.apply(&layer.source.read_rows(start, end)?, spec.dtype)?,
                None => empty_block(end - start, map.n_out, spec.dtype),
            };
            blocks.push(block);
        }
        writer.push(&blocks)?;
        start = end;
    }
    Ok(())
}

/// A source's genes onto the combined gene axis.
struct GeneMap {
    /// Combined position of each source gene, `None` if the join drops it.
    positions: Vec<Option<u32>>,
    /// Whether kept genes keep their order (no per-row sort needed).
    monotone: bool,
    /// Whether the map is the identity (blocks pass through).
    identity: bool,
    n_out: u64,
}

impl GeneMap {
    fn new(positions: Vec<Option<u32>>, n_source: usize, n_out: usize) -> Self {
        let kept: Vec<u32> = positions.iter().flatten().copied().collect();
        let monotone = kept.windows(2).all(|pair| pair[0] < pair[1]);
        let identity = n_source == n_out
            && positions
                .iter()
                .enumerate()
                .all(|(gene, position)| *position == Some(gene as u32));
        Self {
            positions,
            monotone,
            identity,
            n_out: n_out as u64,
        }
    }

    /// `block` over the source's genes as a block over the combined genes, values in
    /// `dtype` (`uint32` to `float32` is the only conversion).
    fn apply(&self, block: &CellBlock, dtype: ValueDtype) -> Result<CellBlock, KytoError> {
        if self.identity && block.dtype() == dtype {
            return Ok(block.clone());
        }
        match (block.data(), dtype) {
            (CellValues::U32(values), ValueDtype::U32) => {
                let (indptr, indices, data) = self.remap(block, |entry| values[entry]);
                checked_block(indptr, indices, CellData::U32(data), self.n_out)
            }
            (CellValues::U32(values), ValueDtype::F32) => {
                let (indptr, indices, data) = self.remap(block, |entry| values[entry] as f32);
                checked_block(indptr, indices, CellData::F32(data), self.n_out)
            }
            (CellValues::F32(values), _) => {
                let (indptr, indices, data) = self.remap(block, |entry| values[entry]);
                checked_block(indptr, indices, CellData::F32(data), self.n_out)
            }
        }
    }

    /// The entries of every row on combined positions, increasing within each row.
    fn remap<T: bytemuck::Pod>(
        &self,
        block: &CellBlock,
        value: impl Fn(usize) -> T,
    ) -> (AlignedBuffer<u64>, AlignedBuffer<u32>, AlignedBuffer<T>) {
        let source_indptr = block.indptr();
        let source_indices = block.indices();
        let mut indptr = AlignedBuffer::with_capacity(source_indptr.len());
        let mut indices = AlignedBuffer::with_capacity(source_indices.len());
        let mut data = AlignedBuffer::with_capacity(source_indices.len());
        let mut row_entries: Vec<(u32, T)> = Vec::new();
        indptr.push(0);
        for row in source_indptr.windows(2) {
            let (start, end) = (row[0] as usize, row[1] as usize);
            row_entries.clear();
            for (entry, gene) in source_indices.iter().enumerate().take(end).skip(start) {
                if let Some(Some(position)) = self.positions.get(*gene as usize) {
                    row_entries.push((*position, value(entry)));
                }
            }
            if !self.monotone {
                row_entries.sort_unstable_by_key(|entry| entry.0);
            }
            for (position, value) in &row_entries {
                indices.push(*position);
                data.push(*value);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, data)
    }
}

/// A block from buffers, checked (`KY-E117` would be a bug here).
fn checked_block(
    indptr: AlignedBuffer<u64>,
    indices: AlignedBuffer<u32>,
    data: CellData,
    n_genes: u64,
) -> Result<CellBlock, KytoError> {
    CellBlock::from_parts(
        CellBlockParts {
            indptr,
            indices,
            data,
        },
        n_genes,
    )
}

/// A block of `n_rows` rows without entries.
fn empty_block(n_rows: u64, n_genes: u64, dtype: ValueDtype) -> CellBlock {
    let indptr = AlignedBuffer::from(&vec![0_u64; n_rows as usize + 1][..]);
    CellBlock::new(indptr, AlignedBuffer::new(), CellData::new(dtype), n_genes)
}

/// One layer of a kyto store, read by rows.
struct StoreRows {
    reader: LayerReader<FilesystemStore>,
    dtype: ValueDtype,
}

impl RowSource for StoreRows {
    fn dtype(&self) -> ValueDtype {
        self.dtype
    }

    fn read_rows(&mut self, start: u64, end: u64) -> Result<CellBlock, KytoError> {
        let cells = AxisSelection::Range(start..end);
        let mut blocks = self
            .reader
            .cell_blocks(&cells, &AxisSelection::All, NonZeroUsize::MIN)?
            .map(|selected| selected.map(|selected| selected.block))
            .collect::<Result<Vec<_>, KytoError>>()?;
        // `concat` reads one store cell block per call.
        if blocks.len() == 1
            && let Some(block) = blocks.pop()
        {
            return Ok(block);
        }
        let n_genes = self.reader.grid().n_genes();
        let mut indptr = AlignedBuffer::from(&[0_u64][..]);
        let mut indices = AlignedBuffer::new();
        let mut data = CellData::new(self.dtype);
        for block in &blocks {
            let offset = indices.len() as u64;
            for pointer in &block.indptr()[1..] {
                indptr.push(offset + pointer);
            }
            indices.extend_from_slice(block.indices());
            match (&mut data, block.data()) {
                (CellData::F32(out), CellValues::F32(values)) => out.extend_from_slice(values),
                (CellData::U32(out), CellValues::U32(values)) => out.extend_from_slice(values),
                _ => {}
            }
        }
        checked_block(indptr, indices, data, n_genes)
    }
}
