//! Write a whole store from a stream of `CellBlock`s, atomically (format.md §10).
//!
//! Does: [`StoreWriter::create`] checks the gene table, tiling and layers and opens a
//! temporary directory next to the target; [`StoreWriter::push`] takes one CellBlock per
//! layer, the same rows in each, and writes the cell blocks they complete;
//! [`StoreWriter::finish`] writes the rest in the order of format.md §10.2 (array
//! metadata and `tile_offsets`, group `zarr.json`s, the tables, `provenance.json`,
//! `schema.json`, the root `zarr.json` last), fsyncs, and renames the directory to the
//! target. A failed or dropped writer deletes its temporary directory, so the target is
//! never a half store (format.md §10.1).
//! Embeddings and graphs given with [`StoreWriter::add_embedding`] and
//! [`StoreWriter::add_graph`] are held whole and written by `finish` (format.md §6, §7;
//! task 5.11), and [`StoreWriter::set_plan`] records the plan JSON that wrote the store
//! in `provenance.json` (§9).
//! Does not: fill in the seeds or timings of `provenance.json` (8.5), or stream the
//! cell table, embeddings or graphs (given whole, the buffered path of §10.2).
//!
//! Memory: per layer, fewer than `C` buffered rows, fewer than `S` unwritten entries and
//! 8 bytes per tile between calls; during a push, a copy of the pushed rows and one
//! `indptr` shard (see `layer_stream`). The gene table is held until `finish`.

use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use kyto_core::{Axis, LayerSchema, Lineage};
use kyto_error::KytoError;
use polars::prelude::DataFrame;
use rayon::prelude::*;
use zarrs::filesystem::FilesystemStore;

use crate::atomic_dir::AtomicDir;
use crate::cell_block::{CellBlock, CellValues};
use crate::commit_check::{CommitCheck, proceed};
use crate::embedding_io::{Embedding, write_embedding};
use crate::graph_io::{Graph, write_graph};
use crate::layer_open::{LAYER_KIND, ValueDtype};
use crate::layer_stream::LayerStream;
use crate::names::validate_member_name;
use crate::store_json::{ImportSource, StoreMembers, atlas_schema, provenance_json};
use crate::store_version::FORMAT_VERSION;
use crate::table_axis::TableAxis;
use crate::table_check::{PlainStrings, check_table};
use crate::table_columns::{ColumnDtype, ColumnSchema};
use crate::table_io::write_table_columns;
use crate::tiling::{TileGrid, Tiling};
use crate::write_options::LayerWriteOptions;
use crate::zarr_arrays::{store_group, store_io};

/// A layer to write: its name, value dtype and chunking (format.md §4.2, §4.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    /// Layer name (format.md §2.1).
    pub name: String,
    /// Value dtype of the layer; every block pushed for it must have it.
    pub dtype: ValueDtype,
    /// `K`, `S` and the zstd level of its arrays.
    pub options: LayerWriteOptions,
}

/// One layer's stream, by value dtype.
enum LayerState {
    F32(LayerStream<f32>),
    U32(LayerStream<u32>),
}

/// A store being written from a stream of CellBlocks (format.md §10).
///
/// Nothing exists at the target until [`StoreWriter::finish`] succeeds. Dropping the
/// writer, or any error, deletes its temporary directory. A process killed midway leaves
/// the target absent and a `.<name>.kyto-tmp-<random>` directory, deleted by the next
/// write to the same target.
pub struct StoreWriter {
    dir: AtomicDir,
    storage: Arc<FilesystemStore>,
    tiling: Tiling,
    genes: DataFrame,
    names: Vec<String>,
    dtypes: Vec<ValueDtype>,
    layers: Vec<LayerState>,
    n_cells: u64,
    started_at: SystemTime,
    /// Enum columns to record as ordered in `schema.json` (format.md §8).
    ordered: Vec<(TableAxis, String)>,
    /// Lineage to record for derived columns in `schema.json` (format.md §8).
    lineage: Vec<(TableAxis, String, Lineage)>,
    /// Embeddings and graphs to write at the end (format.md §6, §7).
    embeddings: Vec<Embedding>,
    graphs: Vec<Graph>,
    /// The plan JSON that wrote the store, for `provenance.json` (§9).
    plan: Option<serde_json::Value>,
    /// Set when a write failed: the arrays are then incomplete, so later calls fail.
    failed: bool,
}

impl StoreWriter {
    /// Check the inputs and start writing a store to `target` (format.md §10.1 step 1).
    ///
    /// `genes` is the gene table (format.md §5); its rows fix the gene axis of every
    /// layer before the first block (§10.2). `layers` lists the layers in the order
    /// [`StoreWriter::push`] takes their blocks; it may be empty. `tiling` is the store's
    /// tile sizes (§4.1). With `overwrite`, an existing kyto store at `target` is
    /// replaced once the new one is complete; anything else at `target` is never
    /// replaced. The gene table is cloned (Polars shares its buffers; no data copy).
    ///
    /// # Errors
    /// - `KY-E111`, `KY-E112` or `KY-E101` if `genes` breaks format.md §5.
    /// - `KY-E104` or `KY-E105` if `tiling` or a layer's options are out of range, or
    ///   `genes` has more than `2^32 - 1` rows.
    /// - `KY-E102` for a bad layer name, `KY-E103` for two layers whose names differ
    ///   only in case.
    /// - `KY-E118` if `target` exists and may not be replaced.
    /// - `KY-E110` if the file system fails.
    pub fn create(
        target: &Path,
        genes: &DataFrame,
        layers: &[LayerSpec],
        tiling: Tiling,
        overwrite: bool,
    ) -> Result<Self, KytoError> {
        let started_at = SystemTime::now();
        check_table(TableAxis::Genes, genes, PlainStrings::Allowed)?;
        let n_genes = genes.height() as u64;
        TileGrid::new(0, n_genes, tiling)?;
        for (position, layer) in layers.iter().enumerate() {
            validate_member_name("layer", &layer.name)?;
            layer.options.validate()?;
            if let Some(earlier) = layers[..position]
                .iter()
                .find(|earlier| earlier.name.eq_ignore_ascii_case(&layer.name))
            {
                return Err(KytoError::MemberExists {
                    kind: "layer".to_owned(),
                    name: layer.name.clone(),
                    existing: earlier.name.clone(),
                });
            }
        }
        let dir = AtomicDir::create(target, overwrite)?;
        let storage = FilesystemStore::new(dir.path())
            .map(Arc::new)
            .map_err(|error| store_io("open store", &dir.path().display().to_string(), error))?;
        let streams = layers
            .iter()
            .map(|layer| {
                let storage = storage.clone();
                Ok(match layer.dtype {
                    ValueDtype::F32 => LayerState::F32(LayerStream::new(
                        storage,
                        &layer.name,
                        n_genes,
                        tiling,
                        layer.options,
                    )?),
                    ValueDtype::U32 => LayerState::U32(LayerStream::new(
                        storage,
                        &layer.name,
                        n_genes,
                        tiling,
                        layer.options,
                    )?),
                })
            })
            .collect::<Result<Vec<_>, KytoError>>()?;
        Ok(Self {
            dir,
            storage,
            tiling,
            genes: genes.clone(),
            names: layers.iter().map(|layer| layer.name.clone()).collect(),
            dtypes: layers.iter().map(|layer| layer.dtype).collect(),
            layers: streams,
            n_cells: 0,
            started_at,
            ordered: Vec::new(),
            lineage: Vec::new(),
            embeddings: Vec::new(),
            graphs: Vec::new(),
            plan: None,
            failed: false,
        })
    }

    /// Record the `Enum` column `column` of the `axis` table as ordered in
    /// `schema.json` (format.md §5.4, §8), as anndata's `ordered=True` categoricals.
    /// Polars `Enum` has no such flag, so the writer keeps it; without this call every
    /// Enum is unordered. Checked by [`StoreWriter::finish`].
    pub fn mark_ordered(&mut self, axis: TableAxis, column: &str) {
        self.ordered.push((axis, column.to_owned()));
    }

    /// Record `lineage` for the column `column` of the `axis` table in `schema.json`
    /// (format.md §8): a stored derived column, as the executor's sink carries it over
    /// from the store it read (task 5.9). Without this call no column has lineage.
    /// Checked by [`StoreWriter::finish`].
    pub fn set_lineage(&mut self, axis: TableAxis, column: &str, lineage: Lineage) {
        self.lineage.push((axis, column.to_owned(), lineage));
    }

    /// Write `embedding` with the store (format.md §6): one row per cell of the store
    /// (as many as the cell table given to `finish`) or per gene. Held until `finish`,
    /// which checks it.
    pub fn add_embedding(&mut self, embedding: Embedding) {
        self.embeddings.push(embedding);
    }

    /// Write `graph` with the store (format.md §7), over the store's cells. Held until
    /// `finish`, which checks it.
    pub fn add_graph(&mut self, graph: Graph) {
        self.graphs.push(graph);
    }

    /// Record `plan` (the plan JSON of task 4.3, as a JSON value) as the `plan` of
    /// `provenance.json` (format.md §9): the executor's sink records the plan that
    /// wrote the store, so a recomputed column finds its recipe (task 5.11).
    pub fn set_plan(&mut self, plan: serde_json::Value) {
        self.plan = Some(plan);
    }

    /// Matrix rows pushed so far.
    pub fn n_cells(&self) -> u64 {
        self.n_cells
    }

    /// Write the next rows: `blocks[l]` holds them for layer `l` of
    /// [`StoreWriter::create`], over all genes of the store, so its `indices` are gene
    /// table positions. Blocks may have any number of rows, the same in every layer;
    /// they are re-blocked into cell blocks of `C` rows (format.md §10.2). Zeros are
    /// dropped (§4.6). Layers are written in parallel on the current rayon pool; the
    /// bytes do not depend on the thread count or on how rows are split into blocks
    /// (§10.3, R11). The blocks are read, not kept.
    ///
    /// # Errors
    /// - `KY-E117`, before writing anything, if the number of blocks is not the number of
    ///   layers, or a block has another gene count, row count or dtype than it must.
    /// - `KY-E105` if the total row count would pass `2^32 - 1` (format.md §4.5).
    /// - `KY-E110` if a write fails, or an earlier one did; the writer is then unusable.
    pub fn push(&mut self, blocks: &[CellBlock]) -> Result<(), KytoError> {
        self.check_usable()?;
        let n_rows = self.check_blocks(blocks)?;
        let total = self.n_cells + n_rows;
        if total > u64::from(u32::MAX) {
            return Err(KytoError::LimitExceeded {
                quantity: "n_cells".to_owned(),
                value: total,
                limit: u64::from(u32::MAX),
                advice: "split the cells into several stores".to_owned(),
            });
        }
        let results: Vec<Result<(), KytoError>> = self
            .layers
            .par_iter_mut()
            .zip(blocks)
            .map(|(layer, block)| match (layer, block.data()) {
                (LayerState::F32(stream), CellValues::F32(data)) => {
                    stream.push(block.indptr(), block.indices(), data)
                }
                (LayerState::U32(stream), CellValues::U32(data)) => {
                    stream.push(block.indptr(), block.indices(), data)
                }
                // check_blocks matched every dtype.
                _ => Ok(()),
            })
            .collect();
        // The first error in layer order, whatever the thread count (R11).
        if let Some(error) = results.into_iter().find_map(Result::err) {
            self.failed = true;
            return Err(error);
        }
        self.n_cells = total;
        Ok(())
    }

    /// Finish the store and move it to the target (format.md §10.1, §10.2).
    ///
    /// `cells` is the cell table, one row per pushed row (format.md §5.1); with no
    /// layers, its height is `n_cells`. `source` is recorded in `provenance.json` for an
    /// import (§9). On success the target holds a complete store; on error the target is
    /// as it was and the temporary directory is deleted.
    ///
    /// # Errors
    /// - `KY-E111`, `KY-E112` or `KY-E101` if `cells` breaks format.md §5.
    /// - `KY-E119` if `cells` does not have one row per pushed row.
    /// - `KY-E102` or `KY-E103` for a bad or repeated embedding or graph name, `KY-E104`
    ///   for an embedding without one row per cell (gene) or width 0, or a graph that
    ///   breaks format.md §7 over the store's cells.
    /// - `KY-E105` if `n_cells` is above `2^32 - 1`.
    /// - `KY-E118` if the target appeared meanwhile and may not be replaced.
    /// - `KY-E110` if the file system fails, or an earlier write did.
    pub fn finish(self, cells: &DataFrame, source: Option<&ImportSource>) -> Result<(), KytoError> {
        self.finish_checked(cells, source, &proceed)
    }

    /// [`StoreWriter::finish`] with a last check: `before_commit` runs once the whole
    /// store is written and synced, right before it is renamed to the target. If it
    /// fails, the target is as it was and the temporary directory is deleted (the
    /// executor passes its cancel check, so a cancelled sink leaves no target).
    ///
    /// # Errors
    /// Those of [`StoreWriter::finish`], and the error of `before_commit`.
    pub fn finish_checked(
        mut self,
        cells: &DataFrame,
        source: Option<&ImportSource>,
        before_commit: CommitCheck<'_>,
    ) -> Result<(), KytoError> {
        self.check_usable()?;
        check_table(TableAxis::Cells, cells, PlainStrings::Allowed)?;
        let n_cells = if self.layers.is_empty() {
            cells.height() as u64
        } else {
            self.n_cells
        };
        if cells.height() as u64 != n_cells {
            return Err(KytoError::TableLength {
                axis: TableAxis::Cells.as_str().to_owned(),
                n_rows: cells.height() as u64,
                n_axis: n_cells,
            });
        }
        let n_genes = self.genes.height() as u64;
        TileGrid::new(n_cells, n_genes, self.tiling)?;
        self.check_members(n_cells, n_genes)?;

        // Format.md §10.2, at the end: arrays' zarr.json and tile_offsets, ...
        let mut summaries = Vec::with_capacity(self.layers.len());
        for ((layer, name), dtype) in std::mem::take(&mut self.layers)
            .into_iter()
            .zip(&self.names)
            .zip(&self.dtypes)
        {
            let nnz = match layer {
                LayerState::F32(stream) => stream.finish(n_cells)?,
                LayerState::U32(stream) => stream.finish(n_cells)?,
            };
            summaries.push(LayerSchema {
                name: name.clone(),
                dtype: (*dtype).into(),
                nnz,
            });
        }
        // ... the zarr.json of every layer group and of layers/, embeddings/, graphs/, ...
        let storage = &self.storage;
        for name in &self.names {
            let kind = serde_json::json!({ "kind": LAYER_KIND });
            store_group(storage, &format!("/layers/{name}"), Some(kind))?;
        }
        let options = LayerWriteOptions::default();
        for embedding in &self.embeddings {
            let block_rows = match embedding.axis {
                Axis::Cells => self.tiling.cell_block_size,
                Axis::Genes => self.tiling.gene_block_size,
            };
            write_embedding(storage, embedding, block_rows, options.zstd_level)?;
        }
        for graph in &self.graphs {
            write_graph(storage, graph, &options)?;
        }
        for group in ["/layers", "/embeddings", "/graphs"] {
            store_group(storage, group, None)?;
        }
        // ... the tables, provenance.json, schema.json, ...
        let store = self.dir.path().to_path_buf();
        let mut gene_columns =
            write_table_columns(&store, TableAxis::Genes, &self.genes, self.tiling)?;
        let mut cell_columns = write_table_columns(&store, TableAxis::Cells, cells, self.tiling)?;
        for (axis, name) in &self.ordered {
            let columns = match axis {
                TableAxis::Cells => &mut cell_columns,
                TableAxis::Genes => &mut gene_columns,
            };
            set_ordered(*axis, columns, name)?;
        }
        for (axis, name, lineage) in &self.lineage {
            let columns = match axis {
                TableAxis::Cells => &mut cell_columns,
                TableAxis::Genes => &mut gene_columns,
            };
            set_lineage(*axis, columns, name, lineage)?;
        }
        let provenance =
            provenance_json(source, self.plan.take(), self.started_at, SystemTime::now());
        write_json(&store.join("provenance.json"), &provenance)?;
        let schema = atlas_schema(
            n_cells,
            n_genes,
            self.tiling,
            cell_columns,
            gene_columns,
            StoreMembers {
                layers: summaries,
                embeddings: self.embeddings.iter().map(Embedding::schema).collect(),
                graphs: self.graphs.iter().map(Graph::schema).collect(),
            },
        );
        write_json(&store.join("schema.json"), &schema)?;
        // ... and the root zarr.json last (format.md §3).
        let kyto = serde_json::json!({ "format_version": FORMAT_VERSION });
        store_group(storage, "/", Some(kyto))?;
        self.dir.commit(before_commit)
    }

    /// `KY-E110` if an earlier write failed.
    fn check_usable(&self) -> Result<(), KytoError> {
        if self.failed {
            return Err(store_io(
                "continue writing",
                &self.dir.path().display().to_string(),
                "an earlier write to this store failed; start a new writer",
            ));
        }
        Ok(())
    }

    /// Check the embeddings and graphs against the store's `n_cells` and `n_genes`.
    fn check_members(&self, n_cells: u64, n_genes: u64) -> Result<(), KytoError> {
        let invalid = |parameter: String, value: String, accepted: String| {
            Err(KytoError::InvalidParameter {
                parameter,
                value,
                accepted,
            })
        };
        let names = self
            .embeddings
            .iter()
            .map(|embedding| ("embedding", embedding.name.as_str()))
            .chain(
                self.graphs
                    .iter()
                    .map(|graph| ("graph", graph.name.as_str())),
            );
        let mut seen: Vec<(&str, &str)> = Vec::new();
        for (kind, name) in names {
            validate_member_name(kind, name)?;
            if let Some((_, earlier)) = seen.iter().find(|(seen_kind, earlier)| {
                *seen_kind == kind && earlier.eq_ignore_ascii_case(name)
            }) {
                return Err(KytoError::MemberExists {
                    kind: kind.to_owned(),
                    name: name.to_owned(),
                    existing: (*earlier).to_owned(),
                });
            }
            seen.push((kind, name));
        }
        for embedding in &self.embeddings {
            let (rows, axis) = match embedding.axis {
                Axis::Cells => (n_cells, "cell"),
                Axis::Genes => (n_genes, "gene"),
            };
            let values = embedding.values.len() as u64;
            if embedding.width == 0 || values != rows * embedding.width {
                return invalid(
                    format!("embedding '{}'", embedding.name),
                    format!("{values} values of width {}", embedding.width),
                    format!("width at least 1 and one row per {axis} ({rows} rows)"),
                );
            }
        }
        for graph in &self.graphs {
            if let Some(problem) = graph.check(n_cells) {
                return invalid(
                    format!("graph '{}'", graph.name),
                    problem,
                    format!("CSR over the store's {n_cells} cells (format.md section 7)"),
                );
            }
        }
        Ok(())
    }

    /// Check one push and return its row count (0 for a store without layers).
    fn check_blocks(&self, blocks: &[CellBlock]) -> Result<u64, KytoError> {
        let invalid = |context: String, problem: String| {
            Err(KytoError::InvalidCellBlock { context, problem })
        };
        if blocks.len() != self.layers.len() {
            return invalid(
                "StoreWriter::push".to_owned(),
                format!(
                    "got {} blocks for {} layers; push one block per layer, in layer order",
                    blocks.len(),
                    self.layers.len()
                ),
            );
        }
        let n_genes = self.genes.height() as u64;
        let n_rows = blocks.first().map_or(0, CellBlock::n_rows);
        for ((block, name), dtype) in blocks.iter().zip(&self.names).zip(&self.dtypes) {
            if block.n_genes() != n_genes {
                return invalid(
                    format!("StoreWriter::push, layer '{name}'"),
                    format!(
                        "the block has {} genes but the store has {n_genes}",
                        block.n_genes()
                    ),
                );
            }
            if block.n_rows() != n_rows {
                return invalid(
                    format!("StoreWriter::push, layer '{name}'"),
                    format!(
                        "the block has {} rows but the block of layer '{}' has {n_rows}",
                        block.n_rows(),
                        self.names[0]
                    ),
                );
            }
            let block_dtype = match block.data() {
                CellValues::F32(_) => ValueDtype::F32,
                CellValues::U32(_) => ValueDtype::U32,
            };
            if block_dtype != *dtype {
                return invalid(
                    format!("StoreWriter::push, layer '{name}'"),
                    format!(
                        "the layer stores {} but the block holds {}",
                        dtype.zarr_name(),
                        block_dtype.zarr_name()
                    ),
                );
            }
        }
        Ok(n_rows as u64)
    }
}

/// Mark the Enum column `name` ordered.
///
/// # Errors
/// `KY-E112` if the table has no such column or it is not an `Enum`.
fn set_ordered(axis: TableAxis, columns: &mut [ColumnSchema], name: &str) -> Result<(), KytoError> {
    let problem = match columns.iter_mut().find(|column| column.name == name) {
        Some(ColumnSchema {
            dtype: ColumnDtype::Enum { ordered, .. },
            ..
        }) => {
            *ordered = true;
            return Ok(());
        }
        Some(column) => format!("has dtype {}, so it cannot be ordered", column.dtype.name()),
        None => "does not exist, so it cannot be ordered".to_owned(),
    };
    Err(KytoError::InvalidTableColumn {
        axis: axis.as_str().to_owned(),
        column: name.to_owned(),
        problem,
        accepted: "mark only Enum columns of the table as ordered".to_owned(),
    })
}

/// Record `lineage` for the column `name`.
///
/// # Errors
/// `KY-E112` if the table has no such column.
fn set_lineage(
    axis: TableAxis,
    columns: &mut [ColumnSchema],
    name: &str,
    lineage: &Lineage,
) -> Result<(), KytoError> {
    match columns.iter_mut().find(|column| column.name == name) {
        Some(column) => {
            column.lineage = Some(lineage.clone());
            Ok(())
        }
        None => Err(KytoError::InvalidTableColumn {
            axis: axis.as_str().to_owned(),
            column: name.to_owned(),
            problem: "does not exist, so it cannot record lineage".to_owned(),
            accepted: "record lineage only for columns of the table".to_owned(),
        }),
    }
}

/// Write a JSON file with keys in insertion order (format.md §10.3) and a final newline.
fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), KytoError> {
    let path_text = path.display().to_string();
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|error| store_io("write", &path_text, error))?;
    bytes.push(b'\n');
    std::fs::write(path, bytes).map_err(|error| store_io("write", &path_text, error))
}
