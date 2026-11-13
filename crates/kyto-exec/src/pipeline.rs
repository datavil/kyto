//! The pipeline executor (task 5.2): turns one output of a physical plan into a chain
//! of streamed operators and runs it.
//!
//! Does: walk from the output's node down to the scan, keep what the stream runs
//! today (the scan's layers, `with_default_layer`, per-cell `normalize_total`/`log1p`
//! maps, alone or fused, the cross-cell nodes of task 5.4: a median
//! `normalize_total`, alone or first in a fused pass, and `highly_variable_genes`
//! (with or without batches; its gene columns join the gene table, task 6.3) and
//! `scale` (its apply pass writes the scaled layer, task 6.4), `pca` (its embedding,
//! loadings and variances are statistics, task 6.5), and
//! the steps of task 5.8: cell and gene filters, `select_genes`, `take_cells`,
//! `with_cells` and `with_genes`), `neighbors` on the embedding of a `pca` below it
//! (task 6.6: searched once that node's passes are done, in memory, its graph a
//! statistic), `leiden`/`louvain` on a graph of a `neighbors` node below (task 6.8:
//! clustered once that graph is built, in memory, its labels a computed cell column on
//! every block after it), and refuse the rest with `KY-E601` naming the task
//! that adds it, before anything is read. Cell steps run on every block
//! (`cell_steps.rs`); gene steps run once, in plan order, between the statistics
//! passes (`gene_steps.rs`: a step reading a cell's profile streams the store once to
//! collect it), and leave a gene selection for the blocks after them. The cell and
//! gene tables are read (from the scan's local store) only when a step needs them.
//! [`Pipeline::run`] opens the scan's layers once, runs every
//! cross-cell node's statistics passes in plan order (each pass streams the node's
//! input again from the scan, through the stages below it; exactly the passes the
//! physical plan gives the node; its blocks reduced in a fixed tree in block order,
//! task 5.6), then starts the output's stream ([`BlockStream`],
//! carrying the [`Statistics`]) on a thread pool of exactly the threads asked for
//! ([`ExecuteOptions`], R7), with the plan's `blocks_in_flight` blocks read and
//! computed at once. A cross-cell node without an apply pass (HVG) leaves the blocks as
//! they are. The output's stream reads the store only when the output reads a layer
//! (4.6's final pass); otherwise it streams the cells' rows alone.
//! A plan ending in a sink keeps what it writes ([`crate::sink`], task 5.9;
//! [`Pipeline::sink`] runs it).
//! Does not: neighbours on an embedding read from the store or past a cell filter or
//! `take_cells` above its `pca`, clusterings of a graph read from the store or past a
//! cell filter or `take_cells` above its `neighbors` (5.11), expression leaves
//! reading embeddings or recomputing stored columns (see the error each one gives), or
//! memory limits (5.5). Tables of a remote store (`run_on` other storage)
//! are still read from the scan's path (plan 5.8 Open).

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use kyto_core::{Axis, ColumnDtype, Expr, ExprAxis, Source};
use kyto_error::KytoError;
use kyto_plan::{
    Blocks, Cluster, Columns, CrossCellOp, Member, Members, Metric, Neighbors, Operation,
    PhysicalPlan, Plan, Predicate, Umap,
};
use kyto_store::{LayerReader, TableAxis, TileGrid, read_table};
use polars::prelude::DataFrame;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{BlockStream, StreamSource};
use crate::cell_steps::{CellLeaf, CellStep, Evaluation, Take, is_block_leaf};
use crate::cluster_graph::CellGraph;
use crate::cluster_labels::ClusterLabels;
use crate::connectivities;
use crate::cross_cell::{CrossCellKind, CrossCellStage, starts_with_median};
use crate::elementwise::ElementwiseMaps;
use crate::expression::Prepared;
use crate::gene_steps::{GeneStepKind, Genes, collect_profiles, profile_layers};
use crate::hvg;
use crate::leiden::{self, ClusterStatistics};
use crate::neighbors::{self, NeighborStatistics};
use crate::operator::Operator;
use crate::qc::{CellQc, GeneSet};
use crate::reduction::reduce_pass;
use crate::run_control::RunControl;
use crate::sink::SinkSpec;
use crate::statistics::{NodeStatistics, Statistics};
use crate::umap::{self, UmapStatistics};

/// How a pipeline runs, apart from the plan.
#[derive(Debug, Clone)]
pub struct ExecuteOptions {
    /// Threads that read and compute blocks (a thread pool of its own per run; the
    /// caller's thread consumes the stream). The output is the same on any number (R11).
    pub threads: NonZeroUsize,
    /// The run's cancel flag and progress counters (task 5.7); keep a clone to cancel
    /// the run or read its progress from another thread.
    pub control: RunControl,
}

impl ExecuteOptions {
    /// Run on `threads` threads, with a control of its own (nobody else cancels it).
    pub fn new(threads: NonZeroUsize) -> Self {
        Self {
            threads,
            control: RunControl::new(),
        }
    }

    /// The same options, cancelled and followed through `control` (task 5.7).
    #[must_use]
    pub fn with_control(self, control: RunControl) -> Self {
        Self { control, ..self }
    }
}

/// One step of a lowered pipeline, from the scan up.
#[derive(Debug, Clone)]
pub(crate) enum Stage {
    /// Per-cell maps, run on every block as it streams by.
    Maps(ElementwiseMaps),
    /// A cross-cell node: statistics passes first.
    CrossCell(CrossCellStage),
    /// A cell filter, `take_cells` or `with_cells`, run on every block (task 5.8).
    Cells(CellStage),
    /// A gene filter, `select_genes` or `with_genes`, run once with the statistics
    /// passes the physical plan gives it (task 5.8).
    Genes { kind: GeneStepKind, passes: u32 },
    /// `neighbors` (tasks 6.6, 6.7), searched and weighted in memory once its embedding
    /// is computed.
    Neighbors(NeighborStage),
    /// `leiden`/`louvain` (task 6.8), clustered in memory once its graph is built.
    Cluster(ClusterStage),
    /// `umap` (task 6.9), laid out in memory once its graph is computed.
    Umap(UmapStage),
}

/// A clustering node and the `neighbors` node whose statistics hold its graph.
#[derive(Debug, Clone)]
pub(crate) struct ClusterStage {
    /// The node's index in the optimized plan.
    node: usize,
    params: Cluster,
    /// The `neighbors` node writing the graph (resolved by `Pipeline::new`).
    source: usize,
    /// Whether the graph is that node's distances (else its connectivities).
    distances: bool,
}

/// A `umap` node and the nodes whose statistics hold its graph and the embedding the
/// graph's neighbours were searched in.
#[derive(Debug, Clone)]
pub(crate) struct UmapStage {
    /// The node's index in the optimized plan.
    node: usize,
    params: Umap,
    /// The `neighbors` node writing the graph (resolved by `Pipeline::new`).
    source: usize,
}

/// A `neighbors` node and the node whose statistics hold its embedding.
#[derive(Debug, Clone)]
pub(crate) struct NeighborStage {
    /// The node's index in the optimized plan.
    node: usize,
    params: Neighbors,
    /// The `pca` node writing the embedding (resolved by `Pipeline::new`).
    source: usize,
}

/// A cell step as the plan gives it, before its leaves are resolved against the genes
/// present at its place.
#[derive(Debug, Clone)]
pub(crate) enum CellStage {
    Filter(Prepared),
    Take(Vec<String>),
    Columns(Vec<(String, Prepared, ColumnDtype)>),
}

/// One output of a physical plan, lowered to a scan and a chain of streamed operators.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub(crate) store: String,
    pub(crate) grid: TileGrid,
    /// The layers the scan reads.
    pub(crate) layers: Vec<String>,
    pub(crate) stages: Vec<Stage>,
    /// The layers the output reads; the stream drops the others at its end.
    pub(crate) output_layers: BTreeSet<String>,
    pub(crate) blocks_in_flight: NonZeroUsize,
    /// Cells per streamed block: the scan's execution's (the store's `C`, or fewer
    /// under a memory limit, task 5.5).
    pub(crate) block_cells: NonZeroU64,
    /// What the plan's sink writes, when the output ends in one (task 5.9).
    pub(crate) sink: Option<SinkSpec>,
}

impl Pipeline {
    /// Lower output `output` of `plan` (its position in
    /// `plan.optimized().outputs()`). Reads nothing.
    ///
    /// # Errors
    /// - `KY-E215` if there is no output `output`.
    /// - `KY-E601` for a node the executor does not run yet, naming the task that adds it.
    /// - `KY-E104` if the store's schema holds a tiling the reader rejects.
    pub fn new(plan: &PhysicalPlan, output: usize) -> Result<Self, KytoError> {
        let optimized = plan.optimized();
        let outputs = optimized.outputs();
        let Some(output_entry) = outputs.get(output) else {
            return Err(KytoError::InvalidArgument {
                function: "Pipeline::new".to_owned(),
                parameter: "output".to_owned(),
                value: output.to_string(),
                problem: format!("the plan has {} outputs", outputs.len()),
                accepted: format!("an output position below {}", outputs.len()),
            });
        };
        let nodes = optimized.nodes();
        let executions = plan.executions();
        let mut stages = Vec::new();
        let mut next = Some(output_entry.node());
        let mut scan = None;
        let mut block_cells = None;
        let mut sink = None;
        while let Some(index) = next {
            let node = &nodes[index];
            let passes = executions[index].statistics_passes();
            match node.operation() {
                Operation::Scan { plan, reads } => {
                    scan = Some((Arc::clone(plan), layer_names(reads).collect()));
                    if let Blocks::Cells(cells) = executions[index].blocks() {
                        block_cells = NonZeroU64::new(cells);
                    }
                }
                Operation::Fused { maps, writes } => {
                    if starts_with_median(maps) {
                        stages.push(Stage::CrossCell(CrossCellStage::new(
                            index,
                            passes,
                            CrossCellKind::MedianNormalize {
                                maps: maps.clone(),
                                keep: Some(writes.clone()),
                            },
                        )));
                    } else {
                        stages.push(Stage::Maps(ElementwiseMaps::new(
                            maps.clone(),
                            Some(writes),
                        )));
                    }
                }
                Operation::Node(logical) if matches!(**logical, Plan::Sink(_)) => {
                    sink = Some(SinkSpec::new(logical)?);
                }
                Operation::Node(logical) => {
                    if let Some(stage) = lower_node(logical, index, passes)? {
                        stages.push(stage);
                    }
                }
            }
            next = node.input();
        }
        // The arena's inputs end at a scan (OptimizedPlan's invariant).
        let Some((scan, layers)) = scan else {
            unreachable!("an optimized plan's nodes lead down to a scan")
        };
        stages.reverse();
        resolve_embeddings(&mut stages, nodes)?;
        let root = scan.root();
        let schema = root.schema();
        let grid = TileGrid::new(schema.n_cells, schema.n_genes, schema.tiling)?;
        // A scan streams in blocks of at most C cells (physical_plan.rs).
        let store_cells = NonZeroU64::new(grid.cell_block_size()).unwrap_or(NonZeroU64::MIN);
        let block_cells = block_cells.map_or(store_cells, |cells| cells.min(store_cells));
        // At most one window of every block (as 4.6's estimate counts them), so a huge
        // blocks_in_flight sizes nothing beyond the store; at least 1.
        let n_blocks = grid.n_cell_blocks().saturating_mul(
            grid.cell_block_size()
                .min(grid.n_cells())
                .div_ceil(block_cells.get()),
        );
        let blocks_in_flight = plan.options().blocks_in_flight.get().min(n_blocks);
        let blocks_in_flight = usize::try_from(blocks_in_flight)
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN);
        let output_layers = layer_names(output_entry.reads()).collect();
        Ok(Self {
            store: root.source().to_owned(),
            grid,
            layers,
            stages,
            output_layers,
            blocks_in_flight,
            block_cells,
            sink,
        })
    }

    /// Whether the output ends in a sink, written by [`Pipeline::sink`] (task 5.9).
    pub fn is_sink(&self) -> bool {
        self.sink.is_some()
    }

    /// The layers the scan reads, in name order.
    pub fn layers(&self) -> &[String] {
        &self.layers
    }

    /// The layers each block of the stream holds: those the output reads.
    pub fn output_layers(&self) -> &BTreeSet<String> {
        &self.output_layers
    }

    /// Cell blocks read and computed at once: the plan's `blocks_in_flight`, at most
    /// the store's cell blocks (at least 1).
    pub fn blocks_in_flight(&self) -> NonZeroUsize {
        self.blocks_in_flight
    }

    /// Cells per streamed block: the store's `C`, or fewer when the physical plan
    /// splits blocks (task 5.5, [`kyto_plan::PhysicalOptions::max_block_cells`]).
    pub fn block_cells(&self) -> NonZeroU64 {
        self.block_cells
    }

    /// Run on the store at the scan's source, a local directory.
    ///
    /// # Errors
    /// Those of [`Pipeline::run_on`], and `KY-E110` if the directory cannot be opened.
    pub fn run(&self, options: ExecuteOptions) -> Result<BlockStream, KytoError> {
        let storage =
            FilesystemStore::new(Path::new(&self.store)).map_err(|error| KytoError::StoreIo {
                operation: "open store".to_owned(),
                path: self.store.clone(),
                detail: error.to_string(),
            })?;
        self.run_on(Arc::new(storage), options)
    }

    /// Run on the store in `storage` (the scan's store, for example a remote one).
    /// Opens every layer the scan reads, runs the statistics passes of the cross-cell
    /// nodes (blocking: each streams the store once), then returns the output's stream,
    /// which runs in a thread of its own (see [`BlockStream`]).
    ///
    /// # Errors
    /// - Any error a statistics pass's stream gives (as the output's stream would).
    /// - `KY-E116`, `KY-E115`, `KY-E109`, `KY-E107` or `KY-E110` if a layer cannot be
    ///   opened ([`LayerReader::open`]).
    /// - `KY-E502` if the threads cannot be started.
    pub fn run_on<S>(
        &self,
        storage: Arc<S>,
        options: ExecuteOptions,
    ) -> Result<BlockStream, KytoError>
    where
        S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
    {
        let readers = Arc::new(
            self.layers
                .iter()
                .map(|layer| LayerReader::open(Arc::clone(&storage), &self.store, layer, self.grid))
                .collect::<Result<Vec<_>, _>>()?,
        );
        // Every statistics pass and the output's stream compute every cell once.
        let control = options.control.clone();
        let passes = self.stages.iter().fold(1_u64, |passes, stage| match stage {
            Stage::Maps(_)
            | Stage::Cells(_)
            | Stage::Neighbors(_)
            | Stage::Cluster(_)
            | Stage::Umap(_) => passes,
            Stage::CrossCell(cross_cell) => passes.saturating_add(u64::from(cross_cell.passes)),
            Stage::Genes { passes: more, .. } => passes.saturating_add(u64::from(*more)),
        });
        control.set_total(self.grid.n_cells().saturating_mul(passes));
        control.check()?;
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads.get())
            .thread_name(|index| format!("kyto-exec-{index}"))
            .build()
            .map_err(|error| KytoError::ThreadStart {
                threads: options.threads.get() as u64,
                problem: error.to_string(),
            })?;
        let thread_pool = Arc::new(thread_pool);
        let source = |readers: Arc<Vec<LayerReader<S>>>,
                      operators: &[Operator],
                      output_layers: BTreeSet<String>| {
            let layers = if readers.is_empty() {
                Vec::new()
            } else {
                self.layers.clone()
            };
            StreamSource {
                readers,
                layers,
                grid: self.grid,
                operators: operators.iter().map(Operator::fresh).collect(),
                output_layers,
                window: self.blocks_in_flight,
                block_cells: self.block_cells,
                store: self.store.clone(),
                control: control.clone(),
            }
        };

        // The stages below each cross-cell node or gene step run as operators in its
        // statistics passes.
        let mut operators = Vec::new();
        let mut statistics = Statistics::default();
        let mut tables = RunTables::new(&self.store, self.grid.n_genes());
        for stage in &self.stages {
            let cross_cell = match stage {
                Stage::Maps(maps) => {
                    operators.push(Operator::Maps(maps.clone()));
                    continue;
                }
                Stage::Cells(cell_stage) => {
                    operators.push(Operator::Cells(tables.cell_step(cell_stage)?));
                    continue;
                }
                Stage::Genes { kind, passes } => {
                    let profiles = kind.profiles();
                    let mut values = Vec::new();
                    if profiles.is_empty() {
                        // A pass the plan counts that this step does not need (a gene
                        // embedding leaf is refused when built): counted as done.
                        control.add_done(self.grid.n_cells().saturating_mul(u64::from(*passes)));
                    } else {
                        let wanted = profiles
                            .iter()
                            .map(|(layer, cell_id)| {
                                Ok((
                                    layer.clone(),
                                    tables.cell_position(cell_id, "a gene step's profile")?,
                                ))
                            })
                            .collect::<Result<Vec<_>, KytoError>>()?;
                        let found = collect_profiles(
                            &source(Arc::clone(&readers), &operators, profile_layers(&profiles)),
                            &thread_pool,
                            &wanted,
                        )?;
                        for (profile, found) in profiles.into_iter().zip(found) {
                            let Some(found) = found else {
                                return Err(KytoError::UnknownName {
                                    kind: "cell".to_owned(),
                                    name: profile.1.clone(),
                                    expression: format!(
                                        "profile({:?}, layer={:?})",
                                        profile.1, profile.0
                                    ),
                                    problem:
                                        "an earlier step (a cell filter or take_cells) removed it"
                                            .to_owned(),
                                    suggestions: Box::default(),
                                });
                            };
                            values.push((profile, found));
                        }
                    }
                    let genes = tables.genes()?;
                    if let Some(selection) = kind.apply(genes, &values, &self.store)? {
                        operators.push(Operator::Genes(selection));
                    }
                    continue;
                }
                Stage::Neighbors(stage) => {
                    let Some(NodeStatistics::Pca(pca)) = statistics.nodes.get(&stage.source) else {
                        unreachable!("Pipeline::new resolved the pca below neighbors")
                    };
                    let (graph, weights) = thread_pool.install(|| {
                        neighbors::search(
                            pca.embedding(),
                            pca.n_components(),
                            &stage.params,
                            &control,
                        )
                        .map(|graph| {
                            let weights = connectivities::connectivities(&graph);
                            (graph, weights)
                        })
                    })?;
                    let found = NeighborStatistics {
                        cells: Arc::clone(&pca.cells),
                        distances: graph,
                        connectivities: weights,
                    };
                    statistics
                        .nodes
                        .insert(stage.node, NodeStatistics::Neighbors(Box::new(found)));
                    continue;
                }
                Stage::Cluster(stage) => {
                    let Some(NodeStatistics::Neighbors(graphs)) =
                        statistics.nodes.get(&stage.source)
                    else {
                        unreachable!("Pipeline::new resolved the neighbors below the clustering")
                    };
                    let graph = if stage.distances {
                        CellGraph::from(graphs.distances())
                    } else {
                        CellGraph::from(graphs.connectivities())
                    };
                    let clustering = leiden::run(graph, &stage.params, &control)?;
                    operators.push(Operator::Labels(ClusterLabels::new(
                        &stage.params.into,
                        graphs.cells(),
                        clustering.labels(),
                        clustering.n_clusters(),
                    )?));
                    let found = ClusterStatistics {
                        cells: Arc::clone(&graphs.cells),
                        clustering,
                    };
                    statistics
                        .nodes
                        .insert(stage.node, NodeStatistics::Cluster(Box::new(found)));
                    continue;
                }
                Stage::Umap(stage) => {
                    let Some(NodeStatistics::Neighbors(neighbors)) =
                        statistics.nodes.get(&stage.source)
                    else {
                        unreachable!("Pipeline::new resolved the neighbors below umap")
                    };
                    let data = neighbor_data(&statistics, &self.stages, stage.source);
                    let layout = thread_pool.install(|| {
                        umap::layout(
                            neighbors.connectivities(),
                            data.as_ref()
                                .map(|(values, width, metric)| (&**values, *width, *metric)),
                            &stage.params,
                            &control,
                        )
                    })?;
                    let found = UmapStatistics {
                        cells: Arc::clone(&neighbors.cells),
                        layout,
                    };
                    statistics
                        .nodes
                        .insert(stage.node, NodeStatistics::Umap(Box::new(found)));
                    continue;
                }
                Stage::CrossCell(cross_cell) => cross_cell,
            };
            let n_genes = usize::try_from(tables.n_genes()).unwrap_or(usize::MAX);
            let cells = match cross_cell.batch_key() {
                Some(_) => Some(tables.cells()?),
                None => None,
            };
            let mut accumulator = cross_cell.accumulator(n_genes, cells, &self.store);
            let input = BTreeSet::from([cross_cell.input_layer().to_owned()]);
            for pass in 0..cross_cell.passes {
                if accumulator.skips(pass) {
                    // Nothing left to compute (pca's basis stopped growing): counted as done.
                    control.add_done(self.grid.n_cells());
                    continue;
                }
                // A tree in block order over the store's cell blocks (task 5.6, R11).
                let reduced = reduce_pass(
                    &source(Arc::clone(&readers), &operators, input.clone()),
                    &thread_pool,
                    cross_cell.input_layer(),
                    &accumulator,
                    pass,
                )?;
                accumulator.end_pass(pass, reduced)?;
            }
            let (mut node_statistics, apply) = cross_cell.finish(accumulator);
            if let (
                NodeStatistics::HighlyVariableGenes(hvg),
                CrossCellKind::HighlyVariableGenes { n_top_genes, .. },
            ) = (&mut node_statistics, &cross_cell.kind)
            {
                // The gene columns (task 6.3), in the gene table later steps read.
                let genes = tables.genes()?;
                let ids = genes.ids(&self.store)?;
                hvg.columns = hvg::columns(hvg, *n_top_genes, &ids)?;
                genes.set_columns(&hvg.columns, &self.store)?;
            }
            statistics.nodes.insert(cross_cell.node, node_statistics);
            operators.extend(apply);
        }

        // The output's stream reads the store only for the layers it outputs or a cell
        // step reads; without any, it streams the cells' rows through the cell steps
        // alone, with nothing to map.
        let reads_layers = !self.output_layers.is_empty()
            || operators.iter().any(|operator| match operator {
                Operator::Cells(step) => step.reads_layers(),
                _ => false,
            });
        let readers = if reads_layers {
            readers
        } else {
            operators.retain(|operator| !operator.reads_layers());
            Arc::new(Vec::new())
        };
        BlockStream::start(
            source(readers, &operators, self.output_layers.clone()),
            thread_pool,
            statistics,
            tables.into_genes(),
        )
    }
}

/// The stage of logical node `index` (with the plan's statistics `passes`), `None`
/// for one with no work at run time, or `KY-E601` for one the executor does not run
/// yet.
fn lower_node(plan: &Plan, index: usize, passes: u32) -> Result<Option<Stage>, KytoError> {
    match plan {
        Plan::WithDefaultLayer(_) => Ok(None),
        Plan::MapElementwise(node) => {
            let maps = vec![node.params().clone()];
            if starts_with_median(&maps) {
                let kind = CrossCellKind::MedianNormalize { maps, keep: None };
                return Ok(Some(Stage::CrossCell(CrossCellStage::new(
                    index, passes, kind,
                ))));
            }
            Ok(Some(Stage::Maps(ElementwiseMaps::new(maps, None))))
        }
        Plan::CrossCell(node) => match node.params().op() {
            CrossCellOp::HighlyVariableGenes {
                layer,
                flavor,
                n_top_genes,
                batch_key,
            } => {
                let kind = CrossCellKind::HighlyVariableGenes {
                    layer: layer.clone(),
                    flavor: *flavor,
                    n_top_genes: *n_top_genes,
                    batch_key: batch_key.clone(),
                };
                Ok(Some(Stage::CrossCell(CrossCellStage::new(
                    index, passes, kind,
                ))))
            }
            CrossCellOp::Scale {
                layer,
                into,
                zero_center,
                max_value,
            } => {
                let kind = CrossCellKind::Scale {
                    layer: layer.clone(),
                    into: into.clone(),
                    zero_center: *zero_center,
                    max_value: *max_value,
                };
                Ok(Some(Stage::CrossCell(CrossCellStage::new(
                    index, passes, kind,
                ))))
            }
            CrossCellOp::Pca {
                layer,
                n_components,
                seed,
                ..
            } => {
                let kind = CrossCellKind::Pca {
                    layer: layer.clone(),
                    n_components: *n_components,
                    seed: *seed,
                };
                Ok(Some(Stage::CrossCell(CrossCellStage::new(
                    index, passes, kind,
                ))))
            }
        },
        Plan::Neighbors(node) => Ok(Some(Stage::Neighbors(NeighborStage {
            node: index,
            params: node.params().clone(),
            source: usize::MAX,
        }))),
        Plan::Cluster(node) => Ok(Some(Stage::Cluster(ClusterStage {
            node: index,
            params: node.params().clone(),
            source: usize::MAX,
            distances: false,
        }))),
        Plan::Umap(node) => Ok(Some(Stage::Umap(UmapStage {
            node: index,
            params: node.params().clone(),
            source: usize::MAX,
        }))),
        Plan::Filter(node) => Ok(Some(match node.params().predicate() {
            Predicate::Cells(predicate) => Stage::Cells(CellStage::Filter(cell_expression(
                predicate,
                "filter_cells",
            )?)),
            Predicate::Genes(predicate) => Stage::Genes {
                kind: GeneStepKind::Filter(gene_expression(predicate, "filter_genes")?),
                passes,
            },
        })),
        Plan::SelectGenes(node) => Ok(Some(Stage::Genes {
            kind: GeneStepKind::Select(node.params().gene_ids().to_vec()),
            passes,
        })),
        Plan::TakeCells(node) => Ok(Some(Stage::Cells(CellStage::Take(
            node.params().cell_ids().to_vec(),
        )))),
        Plan::WithColumns(node) => {
            let schema = plan.output_schema();
            let dtype = |axis: Axis, name: &str| {
                schema
                    .column(axis, name)
                    .map(|column| column.dtype)
                    .ok_or_else(|| KytoError::InvalidArgument {
                        function: "Pipeline::new".to_owned(),
                        parameter: "plan".to_owned(),
                        value: name.to_owned(),
                        problem: "a computed column is not in its node's output schema".to_owned(),
                        accepted: "a plan built by kyto".to_owned(),
                    })
            };
            Ok(Some(match node.params().columns() {
                Columns::Cells(columns) => Stage::Cells(CellStage::Columns(
                    columns
                        .iter()
                        .map(|(name, expr)| {
                            Ok((
                                name.clone(),
                                cell_expression(expr, "with_cells")?,
                                dtype(Axis::Cells, name)?,
                            ))
                        })
                        .collect::<Result<_, KytoError>>()?,
                )),
                Columns::Genes(columns) => Stage::Genes {
                    kind: GeneStepKind::Columns(
                        columns
                            .iter()
                            .map(|(name, expr)| {
                                Ok((
                                    name.clone(),
                                    gene_expression(expr, "with_genes")?,
                                    dtype(Axis::Genes, name)?,
                                ))
                            })
                            .collect::<Result<_, KytoError>>()?,
                    ),
                    passes,
                },
            }))
        }
        // A sink is lowered by `Pipeline::new` (`SinkSpec`); a scan is the arena's
        // `Operation::Scan`.
        Plan::Scan(_) | Plan::Sink(_) => Ok(None),
    }
}

/// A resolved cell expression of `method`, whose leaves the stream computes: table
/// columns, gene values and QC metrics. `KY-E601` for embeddings and recomputed
/// columns (plan 5.11).
fn cell_expression<A: ExprAxis>(expr: &Expr<A>, method: &str) -> Result<Prepared, KytoError> {
    let prepared = Prepared::new(expr.node());
    for leaf in &prepared.leaves {
        let layered = match leaf {
            Source::Expression { layer, .. } | Source::Qc { layer, .. } => layer.is_some(),
            _ => false,
        };
        if !(is_block_leaf(leaf) && layered) {
            return Err(unsupported_leaf(leaf, method));
        }
    }
    Ok(prepared)
}

/// A resolved gene expression of `method`, whose leaves the run computes: table
/// columns and cell profiles. `KY-E601` for embeddings (plan 5.11).
fn gene_expression<A: ExprAxis>(expr: &Expr<A>, method: &str) -> Result<Prepared, KytoError> {
    let prepared = Prepared::new(expr.node());
    for leaf in &prepared.leaves {
        if !matches!(leaf, Source::Profile { layer: Some(_), .. }) {
            return Err(unsupported_leaf(leaf, method));
        }
    }
    Ok(prepared)
}

fn unsupported_leaf(leaf: &Source, method: &str) -> KytoError {
    let what = match leaf {
        Source::Embedding { .. } => "an embedding component",
        Source::Recompute { .. } => "a recomputed column (`.recompute()`)",
        _ => "an unresolved name",
    };
    not_yet(&format!("{what} in {method}"), "5.11")
}

/// The cell and gene tables of a run, read when a step first needs them, and the
/// genes present at the step being lowered.
struct RunTables {
    store: String,
    n_genes: u64,
    cells: Option<Arc<DataFrame>>,
    genes: Option<Genes>,
}

impl RunTables {
    fn new(store: &str, n_genes: u64) -> Self {
        Self {
            store: store.to_owned(),
            n_genes,
            cells: None,
            genes: None,
        }
    }

    /// The number of genes present.
    fn n_genes(&self) -> u64 {
        self.genes
            .as_ref()
            .map_or(self.n_genes, |genes| genes.len() as u64)
    }

    fn cells(&mut self) -> Result<Arc<DataFrame>, KytoError> {
        if let Some(cells) = &self.cells {
            return Ok(Arc::clone(cells));
        }
        let cells = Arc::new(read_table(Path::new(&self.store), TableAxis::Cells)?);
        self.cells = Some(Arc::clone(&cells));
        Ok(cells)
    }

    fn genes(&mut self) -> Result<&mut Genes, KytoError> {
        if self.genes.is_none() {
            self.genes = Some(Genes::read(&self.store)?);
        }
        match self.genes.as_mut() {
            Some(genes) => Ok(genes),
            None => unreachable!("set above"),
        }
    }

    fn into_genes(self) -> Option<Genes> {
        self.genes
    }

    /// The store positions of the cells `ids`, in their order; `KY-E210` for an ID
    /// the store does not have (`expression` names the step).
    fn cell_positions(&mut self, ids: &[String], expression: &str) -> Result<Vec<u64>, KytoError> {
        let cells = self.cells()?;
        let column = cells
            .column("cell_id")
            .and_then(|column| column.str().cloned())
            .map_err(|error| KytoError::CorruptTable {
                path: format!("{}/cells.parquet", self.store),
                problem: error.to_string(),
            })?;
        let wanted: std::collections::HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(order, id)| (id.as_str(), order))
            .collect();
        let mut positions = vec![None; ids.len()];
        for (position, id) in column.iter().enumerate() {
            if let Some(order) = id.and_then(|id| wanted.get(id)) {
                positions[*order] = Some(position as u64);
            }
        }
        positions
            .into_iter()
            .zip(ids)
            .map(|(position, id)| {
                position.ok_or_else(|| KytoError::UnknownName {
                    kind: "cell".to_owned(),
                    name: id.clone(),
                    expression: expression.to_owned(),
                    problem: "the store has no cell with this ID".to_owned(),
                    suggestions: Box::default(),
                })
            })
            .collect()
    }

    fn cell_position(&mut self, id: &str, expression: &str) -> Result<u64, KytoError> {
        Ok(self.cell_positions(&[id.to_owned()], expression)?[0])
    }

    /// `stage` with its leaves resolved against the genes present.
    fn cell_step(&mut self, stage: &CellStage) -> Result<CellStep, KytoError> {
        let table = self.cells()?;
        Ok(match stage {
            CellStage::Filter(predicate) => {
                CellStep::filter(self.evaluation(predicate)?, table, &self.store)
            }
            CellStage::Take(ids) => {
                let expression = format!("take_cells([{} IDs])", ids.len());
                let positions = self.cell_positions(ids, &expression)?;
                CellStep::take(Take::new(positions, ids), table, &self.store)
            }
            CellStage::Columns(columns) => {
                let columns = columns
                    .iter()
                    .map(|(name, prepared, dtype)| {
                        Ok((name.clone(), self.evaluation(prepared)?, *dtype))
                    })
                    .collect::<Result<_, KytoError>>()?;
                CellStep::columns(columns, table, &self.store)
            }
        })
    }

    /// How a block computes each leaf of `prepared`.
    fn evaluation(&mut self, prepared: &Prepared) -> Result<Evaluation, KytoError> {
        let store = self.store.clone();
        let leaves = prepared
            .leaves
            .iter()
            .map(|leaf| match leaf {
                Source::Expression {
                    layer: Some(layer),
                    gene_id,
                } => {
                    let position =
                        self.genes()?
                            .position(gene_id)
                            .ok_or_else(|| KytoError::UnknownName {
                                kind: "gene".to_owned(),
                                name: gene_id.clone(),
                                expression: format!("expression({gene_id:?}, layer={layer:?})"),
                                problem: "an earlier step (a gene filter or selection) removed it"
                                    .to_owned(),
                                suggestions: Box::default(),
                            })?;
                    Ok(CellLeaf::Gene {
                        layer: layer.clone(),
                        position,
                    })
                }
                Source::Qc {
                    layer: Some(layer),
                    metric,
                } => {
                    // The gene set over the genes present at this step (6.1 Open).
                    let gene_set = match metric.genes() {
                        Some(genes) => Some(GeneSet::evaluate(
                            self.genes()?.table(),
                            genes,
                            &format!("{store}/genes.parquet"),
                        )?),
                        None => None,
                    };
                    Ok(CellLeaf::Qc {
                        layer: layer.clone(),
                        qc: Box::new(CellQc::new(metric, gene_set)),
                    })
                }
                other => Err(unsupported_leaf(other, "a cell expression")),
            })
            .collect::<Result<Vec<_>, KytoError>>()?;
        Ok(Evaluation::new(prepared.clone(), leaves))
    }
}

/// The names of the layers among `members`, in name order.
fn layer_names(members: &Members) -> impl Iterator<Item = String> + '_ {
    members.iter().filter_map(|member| match member {
        Member::Layer(name) => Some(name.clone()),
        _ => None,
    })
}

/// Point every `neighbors` stage at the `pca` stage below it that writes its embedding
/// (the nearest one: a later `pca` into the same name replaces it), and every
/// clustering and `umap` at the `neighbors` stage below it that writes its graph, the
/// same way. `KY-E601` (5.11) for an embedding or graph no stage writes (one read from
/// the store), or a cell filter or `take_cells` between the two (the graph would need
/// the embedding's rows, the clustering or layout the graph's, of the kept cells).
/// `KY-E215` for a `umap` of a `distances` graph (umap lays out connectivities, as
/// scanpy's `tl.umap`).
fn resolve_embeddings(
    stages: &mut [Stage],
    nodes: &[kyto_plan::OptimizedNode],
) -> Result<(), KytoError> {
    // (embedding, pca node, whether cells were removed since)
    let mut written: Vec<(String, usize, bool)> = Vec::new();
    // (graph, neighbors node, whether it is the distances, whether cells were removed)
    let mut graphs: Vec<(String, usize, bool, bool)> = Vec::new();
    for stage in stages {
        match stage {
            Stage::CrossCell(cross_cell) => {
                if let Operation::Node(logical) = nodes[cross_cell.node].operation()
                    && let Plan::CrossCell(node) = &**logical
                    && let CrossCellOp::Pca { into, .. } = node.params().op()
                {
                    written.retain(|(name, _, _)| name != into);
                    written.push((into.clone(), cross_cell.node, false));
                }
            }
            Stage::Cells(CellStage::Filter(_) | CellStage::Take(_)) => {
                for entry in &mut written {
                    entry.2 = true;
                }
                for entry in &mut graphs {
                    entry.3 = true;
                }
            }
            Stage::Umap(umap) => {
                let graph = &umap.params.graph;
                match graphs.iter().find(|(name, ..)| name == graph) {
                    Some((_, source, false, false)) => umap.source = *source,
                    Some((_, _, true, _)) => {
                        return Err(KytoError::InvalidArgument {
                            function: "umap".to_owned(),
                            parameter: "graph".to_owned(),
                            value: format!("'{graph}'"),
                            problem: "it is the distances graph of a neighbors step".to_owned(),
                            accepted: "the connectivities graph of a neighbors step".to_owned(),
                        });
                    }
                    Some((_, _, false, true)) => {
                        return Err(not_yet(
                            "umap after a cell filter or take_cells above its neighbors",
                            "5.11",
                        ));
                    }
                    None => {
                        return Err(not_yet(
                            &format!("umap on graph '{graph}' read from the store"),
                            "5.11",
                        ));
                    }
                }
                // Its embedding replaces any of that name a pca wrote.
                written.retain(|(name, _, _)| *name != umap.params.into);
            }
            Stage::Neighbors(neighbors) => {
                let embedding = &neighbors.params.embedding;
                match written.iter().find(|(name, _, _)| name == embedding) {
                    Some((_, source, false)) => {
                        neighbors.source = *source;
                        let (distances, connectivities) = neighbors.params.graph_names();
                        graphs.retain(|(name, ..)| *name != distances && *name != connectivities);
                        graphs.push((distances, neighbors.node, true, false));
                        graphs.push((connectivities, neighbors.node, false, false));
                    }
                    Some((_, _, true)) => {
                        return Err(not_yet(
                            "neighbors after a cell filter or take_cells above its pca",
                            "5.11",
                        ));
                    }
                    None => {
                        return Err(not_yet(
                            &format!("neighbors on embedding '{embedding}' read from the store"),
                            "5.11",
                        ));
                    }
                }
            }
            Stage::Cluster(cluster) => {
                let graph = &cluster.params.graph;
                let method = cluster.params.method.name();
                match graphs.iter().find(|(name, ..)| name == graph) {
                    Some((_, source, distances, false)) => {
                        cluster.source = *source;
                        cluster.distances = *distances;
                    }
                    Some((.., true)) => {
                        return Err(not_yet(
                            &format!(
                                "{method} after a cell filter or take_cells above its neighbors"
                            ),
                            "5.11",
                        ));
                    }
                    None => {
                        return Err(not_yet(
                            &format!("{method} on graph '{graph}' read from the store"),
                            "5.11",
                        ));
                    }
                }
            }
            Stage::Maps(_) | Stage::Cells(_) | Stage::Genes { .. } => {}
        }
    }
    Ok(())
}

/// The embedding the `neighbors` node `node` searched (the leading columns of its
/// `pca`'s embedding it used) and its metric: umap places many graph components by
/// their centroids in it.
fn neighbor_data<'a>(
    statistics: &'a Statistics,
    stages: &[Stage],
    node: usize,
) -> Option<(Cow<'a, [f32]>, usize, Metric)> {
    let stage = stages.iter().find_map(|stage| match stage {
        Stage::Neighbors(neighbors) if neighbors.node == node => Some(neighbors),
        _ => None,
    })?;
    let Some(NodeStatistics::Pca(pca)) = statistics.nodes.get(&stage.source) else {
        return None;
    };
    let width = pca.n_components();
    let used = stage.params.n_components.map_or(width, |used| {
        usize::try_from(used).unwrap_or(width).min(width)
    });
    let values = if used == width {
        Cow::Borrowed(pca.embedding())
    } else {
        Cow::Owned(
            pca.embedding()
                .chunks(width)
                .flat_map(|row| row[..used].iter().copied())
                .collect(),
        )
    };
    Some((values, used, stage.params.metric))
}

fn not_yet(method: &str, task: &str) -> KytoError {
    KytoError::NotImplementedYet {
        method: format!("{method} in the executor"),
        task: task.to_owned(),
    }
}
