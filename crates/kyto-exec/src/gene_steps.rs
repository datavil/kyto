//! Gene steps of a run (task 5.8): `filter_genes`, `select_genes` and `with_genes`,
//! computed once, in plan order, before the output's stream starts.
//!
//! Does: keep the genes present at each step of the stream and their table
//! ([`Genes`]: the store's gene table at the kept genes, in their order, with the
//! columns `with_genes` computed); evaluate a gene predicate or column over that table,
//! with the profile of a cell (its values across the genes present) collected by one
//! pass over the stream below the step ([`collect_profiles`], the statistics pass the
//! physical plan gives such a step); and turn a change of genes into a
//! [`GeneSelection`], the operator that remaps every later block's genes (dropping
//! the others, in the selection's order). `select_genes` keeps the order it lists
//! (4.1 Open (2)); a filter keeps the table order.
//! Does not: read gene embeddings (`KY-E601`, plan 5.11), or check names (the plan did;
//! a gene a filter removed is `KY-E210` here, at run time).

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use kyto_core::{ColumnDtype, Source};
use kyto_error::KytoError;
use kyto_store::{CellBlock, CellData, CellValues, TableAxis, read_table};
use polars::prelude::{DataFrame, IdxCa, IdxSize, NamedFrom, Series};
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{StreamSource, Windows};
use crate::cell_block_pool::CellBlockPool;
use crate::expression::{Prepared, cast, mask};
use crate::pipeline_block::PipelineBlock;
use crate::statistics::HvgValues;

/// The genes present at a step of the stream, and their table.
#[derive(Debug, Clone)]
pub struct Genes {
    positions: Vec<u64>,
    table: DataFrame,
}

impl Genes {
    /// Every gene of the store at `store`, as stored.
    ///
    /// # Errors
    /// As [`read_table`].
    pub(crate) fn read(store: &str) -> Result<Self, KytoError> {
        let table = read_table(Path::new(store), TableAxis::Genes)?;
        Ok(Self {
            positions: (0..table.height() as u64).collect(),
            table,
        })
    }

    /// The store position of each gene present, in the stream's gene order (a block's
    /// gene position `j` is the store's gene `positions()[j]`).
    pub fn positions(&self) -> &[u64] {
        &self.positions
    }

    /// The gene table of the genes present, in the stream's gene order: the stored
    /// columns, with the columns `with_genes` steps computed added or replaced.
    pub fn table(&self) -> &DataFrame {
        &self.table
    }

    /// The number of genes present.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether no gene is present.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The gene IDs of the genes present, in the stream's gene order.
    ///
    /// # Errors
    /// `KY-E113` if the table has no string `gene_id` column.
    pub(crate) fn ids(&self, store: &str) -> Result<Vec<String>, KytoError> {
        let path = format!("{store}/genes.parquet");
        let ids = self
            .table
            .column("gene_id")
            .and_then(|column| column.str().cloned())
            .map_err(|error| corrupt(&path, error.to_string()))?;
        Ok(ids
            .iter()
            .map(|id| id.unwrap_or_default().to_owned())
            .collect())
    }

    /// Add (or replace) the gene columns `highly_variable_genes` computed (task 6.3).
    ///
    /// # Errors
    /// `KY-E113` if a column does not have one value per gene present.
    pub(crate) fn set_columns(
        &mut self,
        columns: &[(String, HvgValues)],
        store: &str,
    ) -> Result<(), KytoError> {
        let path = format!("{store}/genes.parquet");
        for (name, values) in columns {
            let series = match values {
                HvgValues::Boolean(values) => Series::new(name.into(), values),
                HvgValues::Float64(values) => Series::new(name.into(), values),
                HvgValues::Int64(values) => Series::new(name.into(), values),
            };
            self.table
                .with_column(series.into())
                .map_err(|error| corrupt(&path, error.to_string()))?;
        }
        Ok(())
    }

    /// The position among the genes present of the gene `gene_id`, if present.
    pub(crate) fn position(&self, gene_id: &str) -> Option<u32> {
        let ids = self.table.column("gene_id").ok()?.str().ok()?;
        ids.iter()
            .position(|id| id == Some(gene_id))
            .and_then(|position| u32::try_from(position).ok())
    }

    /// Keep the genes at `selected` (positions among the genes present, in the new
    /// order); the operator that does it to blocks, or `None` when nothing changes.
    fn select(
        &mut self,
        selected: Vec<u32>,
        path: &str,
    ) -> Result<Option<GeneSelection>, KytoError> {
        let unchanged = selected.len() == self.positions.len()
            && selected
                .iter()
                .enumerate()
                .all(|(new, old)| *old as usize == new);
        if unchanged {
            return Ok(None);
        }
        let indices = selected.iter().map(|old| *old as IdxSize).collect();
        self.table = self
            .table
            .take(&IdxCa::from_vec("genes".into(), indices))
            .map_err(|error| corrupt(path, error.to_string()))?;
        let old_count = self.positions.len();
        self.positions = selected
            .iter()
            .map(|old| self.positions[*old as usize])
            .collect();
        Ok(Some(GeneSelection::new(&selected, old_count)))
    }
}

/// What a gene step does.
#[derive(Debug, Clone)]
pub(crate) enum GeneStepKind {
    /// `filter_genes`: keep the genes where it is true.
    Filter(Prepared),
    /// `select_genes`: these gene IDs, in this order.
    Select(Vec<String>),
    /// `with_genes`: each column's name, expression and declared dtype.
    Columns(Vec<(String, Prepared, ColumnDtype)>),
}

impl GeneStepKind {
    /// The cell profiles its expressions read: (layer, cell ID), without repeats.
    pub(crate) fn profiles(&self) -> Vec<(String, String)> {
        let prepared: Vec<&Prepared> = match self {
            Self::Filter(predicate) => vec![predicate],
            Self::Select(_) => Vec::new(),
            Self::Columns(columns) => columns.iter().map(|(_, prepared, _)| prepared).collect(),
        };
        let mut profiles = Vec::new();
        for leaf in prepared.iter().flat_map(|prepared| &prepared.leaves) {
            if let Source::Profile {
                layer: Some(layer),
                cell_id,
            } = leaf
            {
                let profile = (layer.clone(), cell_id.clone());
                if !profiles.contains(&profile) {
                    profiles.push(profile);
                }
            }
        }
        profiles
    }

    /// Run the step on `genes`, with the profiles of [`GeneStepKind::profiles`]
    /// (`profiles[i]` the values of the `i`-th, one per gene present). Returns the
    /// operator for later blocks if the genes change.
    ///
    /// # Errors
    /// - `KY-E210` if `select_genes` lists a gene an earlier step removed.
    /// - `KY-E113` if an expression cannot be evaluated or cast over the gene table.
    pub(crate) fn apply(
        &self,
        genes: &mut Genes,
        profiles: &[((String, String), Series)],
        store: &str,
    ) -> Result<Option<GeneSelection>, KytoError> {
        let path = format!("{store}/genes.parquet");
        let evaluate = |prepared: &Prepared, genes: &Genes| {
            let leaves = prepared
                .leaves
                .iter()
                .map(|leaf| match leaf {
                    Source::Profile {
                        layer: Some(layer),
                        cell_id,
                    } => profiles
                        .iter()
                        .find(|((known_layer, known_cell), _)| {
                            known_layer == layer && known_cell == cell_id
                        })
                        .map(|(_, values)| values.clone())
                        .ok_or_else(|| corrupt(&path, format!("no profile of cell '{cell_id}'"))),
                    other => Err(corrupt(
                        &path,
                        format!("unexpected gene expression leaf {other:?}"),
                    )),
                })
                .collect::<Result<Vec<_>, KytoError>>()?;
            let frame = genes
                .table
                .select(prepared.columns.iter().map(String::as_str))
                .map_err(|error| corrupt(&path, error.to_string()))?;
            let frame = if frame.width() == 0 {
                DataFrame::empty_with_height(genes.len())
            } else {
                frame
            };
            prepared.evaluate(frame, leaves, &path)
        };
        match self {
            Self::Filter(predicate) => {
                let keep = mask(&evaluate(predicate, genes)?, &path)?;
                let selected = keep
                    .iter()
                    .enumerate()
                    .filter(|(_, kept)| **kept)
                    .map(|(position, _)| position as u32)
                    .collect();
                genes.select(selected, &path)
            }
            Self::Select(ids) => {
                let selected = ids
                    .iter()
                    .map(|id| {
                        genes.position(id).ok_or_else(|| KytoError::UnknownName {
                            kind: "gene".to_owned(),
                            name: id.clone(),
                            expression: format!("select_genes([{} IDs])", ids.len()),
                            problem: "an earlier step (a gene filter or selection) removed it"
                                .to_owned(),
                            suggestions: Box::default(),
                        })
                    })
                    .collect::<Result<Vec<_>, KytoError>>()?;
                genes.select(selected, &path)
            }
            Self::Columns(columns) => {
                let values = columns
                    .iter()
                    .map(|(name, prepared, dtype)| {
                        cast(evaluate(prepared, genes)?, name, dtype, &path)
                    })
                    .collect::<Result<Vec<_>, KytoError>>()?;
                for values in values {
                    genes
                        .table
                        .with_column(values.into())
                        .map_err(|error| corrupt(&path, error.to_string()))?;
                }
                Ok(None)
            }
        }
    }
}

fn corrupt(path: &str, problem: String) -> KytoError {
    KytoError::CorruptTable {
        path: path.to_owned(),
        problem,
    }
}

/// The profile of each cell of `wanted` ((layer, store position)), one value per gene
/// present (absent entries are 0, R9), read from one pass over `source`'s stream, in
/// the layer's dtype. `None` for a cell the stream does not carry.
///
/// # Errors
/// Any error of the stream.
pub(crate) fn collect_profiles<S>(
    source: &StreamSource<S>,
    thread_pool: &rayon::ThreadPool,
    wanted: &[(String, u64)],
) -> Result<Vec<Option<Series>>, KytoError>
where
    S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
{
    let pool = CellBlockPool::new(source.window.get());
    thread_pool.install(|| {
        let mut found: Vec<Option<Series>> = vec![None; wanted.len()];
        let mut windows = Windows::open(source, &pool)?;
        while let Some(items) = windows.next_window() {
            for item in items {
                let block = item?;
                for (slot, (layer, cell)) in found.iter_mut().zip(wanted) {
                    if let Some(row) = block.cells.iter().position(|known| known == cell) {
                        let values =
                            block
                                .layers
                                .get(layer)
                                .ok_or_else(|| KytoError::MissingMember {
                                    kind: "layer".to_owned(),
                                    name: layer.clone(),
                                    store: source.store.clone(),
                                    suggestions: Box::default(),
                                })?;
                        *slot = Some(dense_row(values, row));
                    }
                }
                recycle(&pool, block);
            }
        }
        Ok(found)
    })
}

fn recycle(pool: &CellBlockPool, block: PipelineBlock) {
    for (_, layer) in block.layers {
        pool.recycle(layer);
    }
}

/// Row `row` of `block` as a dense series over its genes.
fn dense_row(block: &CellBlock, row: usize) -> Series {
    let range = block.indptr()[row] as usize..block.indptr()[row + 1] as usize;
    let indices = &block.indices()[range.clone()];
    let n_genes = block.n_genes() as usize;
    match block.data() {
        CellValues::F32(values) => {
            let mut dense = vec![0.0_f32; n_genes];
            for (index, value) in indices.iter().zip(&values[range]) {
                dense[*index as usize] = *value;
            }
            Series::new("".into(), dense)
        }
        // `uint32` profiles are `Float64` for arithmetic (exact; see
        // `cell_steps::widen`).
        CellValues::U32(values) => {
            let mut dense = vec![0.0_f64; n_genes];
            for (index, value) in indices.iter().zip(&values[range]) {
                dense[*index as usize] = f64::from(*value);
            }
            Series::new("".into(), dense)
        }
    }
}

/// Remaps the genes of every block: keeps the selected genes, at their new positions.
#[derive(Debug, Clone)]
pub(crate) struct GeneSelection {
    /// The new position of each old gene; `u32::MAX` for a gene dropped.
    new_of_old: Arc<[u32]>,
    n_genes: u64,
    /// Whether the kept genes keep their relative order (no sort needed).
    in_order: bool,
}

impl GeneSelection {
    /// Keep the old genes `selected`, gene `selected[j]` becoming gene `j`; `old_count`
    /// genes before. `selected` has no repeats (the plan refuses repeated IDs).
    fn new(selected: &[u32], old_count: usize) -> Self {
        let mut new_of_old = vec![u32::MAX; old_count];
        for (new, old) in selected.iter().enumerate() {
            new_of_old[*old as usize] = new as u32;
        }
        Self {
            new_of_old: new_of_old.into(),
            n_genes: selected.len() as u64,
            in_order: selected.windows(2).all(|pair| pair[0] < pair[1]),
        }
    }

    /// `block` over the selected genes.
    ///
    /// # Errors
    /// `KY-E117` if a layer's genes do not match the selection (a kyto bug).
    pub(crate) fn apply(
        &self,
        mut block: PipelineBlock,
        pool: &CellBlockPool,
    ) -> Result<PipelineBlock, KytoError> {
        let layers = std::mem::take(&mut block.layers);
        for (name, layer) in layers {
            let selected = self.remap(&layer, pool)?;
            pool.recycle(layer);
            block.layers.insert(name, selected);
        }
        Ok(block)
    }

    fn remap(&self, layer: &CellBlock, pool: &CellBlockPool) -> Result<CellBlock, KytoError> {
        if layer.n_genes() != self.new_of_old.len() as u64 {
            return Err(KytoError::InvalidCellBlock {
                context: "select genes".to_owned(),
                problem: format!(
                    "a block of {} genes reaches a selection over {}",
                    layer.n_genes(),
                    self.new_of_old.len()
                ),
            });
        }
        let mut parts = pool.take(layer.dtype());
        parts.indptr.push(0);
        let indptr = layer.indptr();
        let indices = layer.indices();
        let mut row_entries: Vec<(u32, usize)> = Vec::new();
        for row in 0..layer.n_rows() {
            let range = indptr[row] as usize..indptr[row + 1] as usize;
            row_entries.clear();
            row_entries.extend(range.filter_map(|entry| {
                let new = self.new_of_old[indices[entry] as usize];
                (new != u32::MAX).then_some((new, entry))
            }));
            if !self.in_order {
                row_entries.sort_unstable_by_key(|(new, _)| *new);
            }
            for (new, _) in &row_entries {
                parts.indices.push(*new);
            }
            match (&mut parts.data, layer.data()) {
                (CellData::F32(out), CellValues::F32(values)) => {
                    for (_, entry) in &row_entries {
                        out.push(values[*entry]);
                    }
                }
                (CellData::U32(out), CellValues::U32(values)) => {
                    for (_, entry) in &row_entries {
                        out.push(values[*entry]);
                    }
                }
                _ => unreachable!("the pool's parts take the layer's dtype"),
            }
            parts.indptr.push(parts.indices.len() as u64);
        }
        CellBlock::from_parts(parts, self.n_genes)
    }
}

/// The layers `kind`'s profiles read.
pub(crate) fn profile_layers(profiles: &[(String, String)]) -> BTreeSet<String> {
    profiles.iter().map(|(layer, _)| layer.clone()).collect()
}
