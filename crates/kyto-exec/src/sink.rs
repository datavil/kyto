//! Sinks in the executor (task 5.9): write a plan's stream as a kyto store or an
//! `.h5ad` file.
//!
//! Does: [`SinkSpec`] is what a plan ending in `sink_zarr`/`sink_h5ad` writes, fixed
//! when the pipeline is lowered: every layer the plan carries (name order, the dtype
//! its schema declares), the cell and gene table columns of its schema in schema order
//! (stored columns at the kept cells, `with_cells`/`with_genes` columns added or
//! replacing stored ones), with lineage carried for stored derived columns still valid,
//! and stale ones (those a read would refuse with `KY-E301`, task 3.8) dropped.
//! [`Pipeline::sink`] drains the output's stream into 2.5's [`StoreWriter`] (each block
//! pushed as it arrives, the blocks recycled) and finishes it with the cell table of
//! the kept cells and the gene table of the genes present (5.8's [`crate::Genes`]). An
//! `.h5ad` is written as a store in a hidden temporary directory next to the target,
//! exported by 2.7's [`kyto_store::sink_h5ad`] (the plan's default layer as `X`), and
//! deleted. Any error or a cancel (`KY-E503`) leaves the target as it was: the writers
//! write to temporaries renamed on success.
//! Does not: write embeddings or graphs (`KY-E601`: the plan has nothing that writes
//! them yet), remote targets (the path is a local path), or count the tables it holds
//! in the estimate (5.10). Cluster labels (`String`) become `Enum` with 6.8.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kyto_core::{Axis, CellExpr, ColumnDtype, GeneExpr, Lineage};
use kyto_error::KytoError;
use kyto_plan::{LogicalNode, Plan, Sink};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, SinkH5adOptions, StoreWriter, TableAxis, Tiling,
    ValueDtype, read_table, sink_h5ad,
};
use polars::prelude::{Column, DataFrame, DataType, IdxCa, IdxSize, NamedFrom, Series};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

use crate::expression::cast;
use crate::pipeline::{ExecuteOptions, Pipeline};

/// One column a sink writes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WrittenColumn {
    pub(crate) name: String,
    /// Lineage carried over from the scanned store (a stored derived column).
    pub(crate) lineage: Option<Lineage>,
    /// An `Enum` recorded as ordered in the scanned store.
    pub(crate) ordered: bool,
    /// The dtype the plan declares (an empty computed column has it).
    pub(crate) dtype: ColumnDtype,
}

/// What a plan's sink writes, fixed when the pipeline is lowered.
#[derive(Debug, Clone)]
pub(crate) struct SinkSpec {
    pub(crate) target: Sink,
    /// The layers, in name order (the order of the stream's `output_layers`).
    pub(crate) layers: Vec<LayerSpec>,
    pub(crate) tiling: Tiling,
    /// The layer an `.h5ad` stores as `X`: the plan's default layer, if it carries it.
    pub(crate) x_layer: Option<String>,
    pub(crate) cells: Vec<WrittenColumn>,
    pub(crate) genes: Vec<WrittenColumn>,
}

impl SinkSpec {
    /// The spec of the sink node `plan`.
    ///
    /// # Errors
    /// `KY-E601` if the plan carries embeddings or graphs (nothing writes them yet).
    pub(crate) fn new(plan: &Plan) -> Result<Self, KytoError> {
        let Plan::Sink(node) = plan else {
            unreachable!("SinkSpec::new is called on sink nodes only")
        };
        let schema = plan.output_schema().atlas();
        if let Some(name) = schema
            .embeddings
            .iter()
            .map(|embedding| format!("the embedding '{}'", embedding.name))
            .chain(
                schema
                    .graphs
                    .iter()
                    .map(|graph| format!("the graph '{}'", graph.name)),
            )
            .next()
        {
            return Err(KytoError::NotImplementedYet {
                method: format!("{} writing {name} in the executor", node.params().method()),
                task: "5.11".to_owned(),
            });
        }
        let mut layers: Vec<LayerSpec> = schema
            .layers
            .iter()
            .map(|layer| LayerSpec {
                name: layer.name.clone(),
                dtype: ValueDtype::from(layer.dtype),
                options: LayerWriteOptions::default(),
            })
            .collect();
        layers.sort_by(|left, right| left.name.cmp(&right.name));
        let x_layer = plan
            .default_layer()
            .filter(|name| layers.iter().any(|layer| layer.name == *name))
            .map(str::to_owned);
        Ok(Self {
            target: node.params().clone(),
            layers,
            tiling: schema.tiling,
            x_layer,
            cells: written_columns(plan, Axis::Cells),
            genes: written_columns(plan, Axis::Genes),
        })
    }
}

/// The columns of `axis` the sink `plan` writes: its schema's, in order, without the
/// stored derived columns a read at the sink refuses as stale (`KY-E301`).
fn written_columns(plan: &Plan, axis: Axis) -> Vec<WrittenColumn> {
    plan.output_schema()
        .table(axis)
        .columns
        .iter()
        .filter(|column| {
            column.lineage.is_none()
                || match axis {
                    Axis::Cells => plan.resolve(&CellExpr::column(column.name.clone())).is_ok(),
                    Axis::Genes => plan.resolve(&GeneExpr::column(column.name.clone())).is_ok(),
                }
        })
        .map(|column| WrittenColumn {
            name: column.name.clone(),
            lineage: column.lineage.clone(),
            ordered: matches!(column.dtype, ColumnDtype::Enum { ordered: true, .. }),
            dtype: column.dtype,
        })
        .collect()
}

impl Pipeline {
    /// Run a plan ending in a sink and write its target; the number of cells written.
    /// The store is the scan's source, a local directory. With `overwrite`, an existing
    /// kyto store (`sink_zarr`) or file (`sink_h5ad`) at the target is replaced once the
    /// new one is complete; anything else there is never replaced.
    ///
    /// # Errors
    /// Those of [`Pipeline::sink_on`], and `KY-E110` if the store cannot be opened.
    pub fn sink(&self, options: ExecuteOptions, overwrite: bool) -> Result<u64, KytoError> {
        let storage =
            FilesystemStore::new(Path::new(&self.store)).map_err(|error| KytoError::StoreIo {
                operation: "open store".to_owned(),
                path: self.store.clone(),
                detail: error.to_string(),
            })?;
        self.sink_on(Arc::new(storage), options, overwrite)
    }

    /// [`Pipeline::sink`] reading the store from `storage`. Blocks until the target is
    /// written. On any error the target is as it was before.
    ///
    /// # Errors
    /// - `KY-E215` if the plan's output is not a sink.
    /// - `KY-E118` (`sink_zarr`) or `KY-E123` (`sink_h5ad`) if the target exists and
    ///   may not be replaced, or is the scanned store or lies inside it (checked before
    ///   anything is read; a store target is checked again when it is created).
    /// - Any error of [`Pipeline::run_on`] or the stream (`KY-E503` when cancelled).
    /// - The writers' errors ([`StoreWriter`], [`kyto_store::sink_h5ad`]): `KY-E124`
    ///   for names an `.h5ad` cannot hold, `KY-E110` for the file system.
    pub fn sink_on<S>(
        &self,
        storage: Arc<S>,
        options: ExecuteOptions,
        overwrite: bool,
    ) -> Result<u64, KytoError>
    where
        S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
    {
        let Some(spec) = &self.sink else {
            return Err(KytoError::InvalidArgument {
                function: "Pipeline::sink".to_owned(),
                parameter: "plan".to_owned(),
                value: "an output without a sink".to_owned(),
                problem: "the output does not end in sink_zarr or sink_h5ad".to_owned(),
                accepted: "a plan ending in a sink; stream other outputs with Pipeline::run"
                    .to_owned(),
            });
        };
        let target = PathBuf::from(spec.target.path());
        check_target(spec, &target, Path::new(&self.store), overwrite)?;
        match &spec.target {
            Sink::Zarr { .. } => self.write_store(spec, storage, options, &target, overwrite),
            Sink::H5ad { .. } => {
                let parent = match target.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => parent,
                    _ => Path::new("."),
                };
                // A hidden directory next to the target, deleted when dropped (on
                // success, error or panic; a killed process leaves it, as 2.7's file).
                let scratch = tempfile::Builder::new()
                    .prefix(".kyto-sink-")
                    .tempdir_in(parent)
                    .map_err(|error| KytoError::StoreIo {
                        operation: "create a temporary store".to_owned(),
                        path: parent.display().to_string(),
                        detail: error.to_string(),
                    })?;
                let store = scratch.path().join("store");
                let cells = self.write_store(spec, storage, options, &store, false)?;
                let export = SinkH5adOptions {
                    x_layer: spec.x_layer.clone(),
                    overwrite,
                    ..SinkH5adOptions::default()
                };
                sink_h5ad(&store, &target, &export)?;
                Ok(cells)
            }
        }
    }

    /// Stream the plan into a store at `target`; the cells written.
    fn write_store<S>(
        &self,
        spec: &SinkSpec,
        storage: Arc<S>,
        options: ExecuteOptions,
        target: &Path,
        overwrite: bool,
    ) -> Result<u64, KytoError>
    where
        S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
    {
        let mut stream = self.run_on(storage, options)?;
        let genes = match stream.genes() {
            Some(genes) => genes.table().clone(),
            None => read_table(Path::new(&self.store), TableAxis::Genes)?,
        };
        let genes = select_columns(&genes, &spec.genes, None, TableAxis::Genes, &self.store)?;
        let mut writer = StoreWriter::create(target, &genes, &spec.layers, spec.tiling, overwrite)?;
        for (axis, columns) in [
            (TableAxis::Cells, &spec.cells),
            (TableAxis::Genes, &spec.genes),
        ] {
            for column in columns {
                if column.ordered {
                    writer.mark_ordered(axis, &column.name);
                }
                if let Some(lineage) = &column.lineage {
                    writer.set_lineage(axis, &column.name, lineage.clone());
                }
            }
        }
        let mut positions: Vec<IdxSize> = Vec::new();
        let mut computed: Option<DataFrame> = None;
        while let Some(block) = stream.next() {
            let mut block = block?;
            for cell in &block.cells {
                positions.push(IdxSize::try_from(*cell).map_err(|_| {
                    corrupt(&self.store, "a cell position is beyond Polars' index")
                })?);
            }
            if block.columns.width() > 0 {
                let columns = std::mem::take(&mut block.columns);
                match &mut computed {
                    Some(frame) => {
                        frame
                            .vstack_mut_owned(columns)
                            .map_err(|error| corrupt(&self.store, &error.to_string()))?;
                    }
                    None => computed = Some(columns),
                }
            }
            let layers: Vec<CellBlock> = spec
                .layers
                .iter()
                .map(|layer| {
                    block.layers.remove(&layer.name).ok_or_else(|| {
                        corrupt(
                            &self.store,
                            "a streamed block misses a layer the sink writes",
                        )
                    })
                })
                .collect::<Result<_, KytoError>>()?;
            let pushed = writer.push(&layers);
            for layer in layers {
                stream.pool().recycle(layer);
            }
            stream.recycle(block);
            pushed?;
        }
        let stored = read_table(Path::new(&self.store), TableAxis::Cells)?
            .take(&IdxCa::new("cells".into(), positions))
            .map_err(|error| corrupt(&self.store, &error.to_string()))?;
        let cells = select_columns(
            &stored,
            &spec.cells,
            computed.as_ref(),
            TableAxis::Cells,
            &self.store,
        )?;
        let n_cells = cells.height() as u64;
        writer.finish(&cells, None)?;
        Ok(n_cells)
    }
}

/// The columns `columns` of `table`, in their order, each from `computed` when it holds
/// one of that name (a computed column replaces a stored one). Without rows, no block
/// computed anything: a column neither stored nor computed is empty, of its dtype.
fn select_columns(
    table: &DataFrame,
    columns: &[WrittenColumn],
    computed: Option<&DataFrame>,
    axis: TableAxis,
    store: &str,
) -> Result<DataFrame, KytoError> {
    let selected = columns
        .iter()
        .map(|column| {
            computed
                .and_then(|frame| frame.column(&column.name).ok())
                .or_else(|| table.column(&column.name).ok())
                .cloned()
                .map(Ok)
                .or_else(|| {
                    (table.height() == 0).then(|| {
                        let empty = Series::new_empty(column.name.as_str().into(), &DataType::Null);
                        cast(empty, &column.name, &column.dtype, store).map(Column::from)
                    })
                })
                .unwrap_or_else(|| {
                    Err(
                    corrupt(
                        store,
                        &format!(
                            "the {} column '{}' of the plan's schema is neither stored nor computed",
                            axis.as_str(),
                            column.name
                        ),
                    ))
                })
        })
        .collect::<Result<Vec<Column>, KytoError>>()?;
    DataFrame::new(table.height(), selected).map_err(|error| corrupt(store, &error.to_string()))
}

/// Fail before reading anything if the target may not be written: it exists and
/// `overwrite` is off, or it is the scanned store or lies inside it.
fn check_target(
    spec: &SinkSpec,
    target: &Path,
    store: &Path,
    overwrite: bool,
) -> Result<(), KytoError> {
    let exists = |problem: &str| match spec.target {
        Sink::Zarr { .. } => KytoError::StoreExists {
            path: target.display().to_string(),
            problem: problem.to_owned(),
        },
        Sink::H5ad { .. } => KytoError::OutputExists {
            path: target.display().to_string(),
            problem: problem.to_owned(),
        },
    };
    // The target's nearest existing ancestor (the target itself if it exists).
    let mut existing = Some(target);
    let resolved = loop {
        match existing {
            Some(path) if path.as_os_str().is_empty() => break std::env::current_dir().ok(),
            Some(path) => match path.canonicalize() {
                Ok(resolved) => break Some(resolved),
                Err(_) => existing = path.parent(),
            },
            None => break None,
        }
    };
    if let (Some(resolved), Ok(store)) = (resolved, store.canonicalize())
        && resolved.starts_with(&store)
    {
        return Err(exists(
            "it is the store the plan reads, or lies inside it; write elsewhere",
        ));
    }
    if target.symlink_metadata().is_ok() && !overwrite {
        return Err(exists("it exists; pass overwrite to replace it"));
    }
    Ok(())
}

fn corrupt(store: &str, problem: &str) -> KytoError {
    KytoError::CorruptTable {
        path: format!("{store}/cells.parquet"),
        problem: problem.to_owned(),
    }
}
