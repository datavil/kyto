//! The pipeline executor (task 5.2): turns one output of a physical plan into a chain
//! of streamed operators and runs it.
//!
//! Does: walk from the output's node down to the scan, keep what the stream runs
//! today (the scan's layers, `with_default_layer`, per-cell `normalize_total`/`log1p`
//! maps, alone or fused, and the cross-cell nodes of task 5.4: a median
//! `normalize_total`, alone or first in a fused pass, and `highly_variable_genes`
//! without batches), and refuse the rest with `KY-E601` naming the task that adds it,
//! before anything is read. [`Pipeline::run`] opens the scan's layers once, runs every
//! cross-cell node's statistics passes in plan order (each pass streams the node's
//! input again from the scan, through the stages below it; exactly the passes the
//! physical plan gives the node; its blocks reduced in a fixed tree in block order,
//! task 5.6), then starts the output's stream ([`BlockStream`],
//! carrying the [`Statistics`]) on a thread pool of exactly the threads asked for
//! ([`ExecuteOptions`], R7), with the plan's `blocks_in_flight` blocks read and
//! computed at once. A cross-cell node without an apply pass (HVG) leaves the blocks as
//! they are. The output's stream reads the store only when the output reads a layer
//! (4.6's final pass); otherwise it streams the cells' rows alone.
//! Does not: filters, gene selections, `take_cells`, computed columns, `scale`, `pca`,
//! HVG batches, sinks (see the error each one gives), or memory limits (5.5).

use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{Blocks, CrossCellOp, Member, Members, Operation, PhysicalPlan, Plan};
use kyto_store::{LayerReader, TileGrid};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{BlockStream, StreamSource};
use crate::cross_cell::{CrossCellKind, CrossCellStage, starts_with_median};
use crate::elementwise::ElementwiseMaps;
use crate::reduction::reduce_pass;
use crate::run_control::RunControl;
use crate::statistics::Statistics;

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
        })
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
            Stage::Maps(_) => passes,
            Stage::CrossCell(cross_cell) => passes.saturating_add(u64::from(cross_cell.passes)),
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
                      operators: &[ElementwiseMaps],
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
                operators: operators.to_vec(),
                output_layers,
                window: self.blocks_in_flight,
                block_cells: self.block_cells,
                store: self.store.clone(),
                control: control.clone(),
            }
        };

        // The stages below each cross-cell node run as maps in its statistics passes.
        let mut operators = Vec::new();
        let mut statistics = Statistics::default();
        let n_genes = usize::try_from(self.grid.n_genes()).unwrap_or(usize::MAX);
        for stage in &self.stages {
            let cross_cell = match stage {
                Stage::Maps(maps) => {
                    operators.push(maps.clone());
                    continue;
                }
                Stage::CrossCell(cross_cell) => cross_cell,
            };
            let mut accumulator = cross_cell.accumulator(n_genes);
            let input = BTreeSet::from([cross_cell.input_layer().to_owned()]);
            for pass in 0..cross_cell.passes {
                // A tree in block order over the store's cell blocks (task 5.6, R11).
                let reduced = reduce_pass(
                    &source(Arc::clone(&readers), &operators, input.clone()),
                    &thread_pool,
                    cross_cell.input_layer(),
                    &accumulator,
                    pass,
                )?;
                accumulator.end_pass(pass, reduced);
            }
            let (node_statistics, apply) = cross_cell.finish(accumulator);
            statistics.nodes.insert(cross_cell.node, node_statistics);
            operators.extend(apply);
        }

        // The output's stream reads the store only for the layers it outputs; without
        // any, it streams the cells' rows, with nothing to map.
        if self.output_layers.is_empty() {
            operators.clear();
        }
        let readers = if self.output_layers.is_empty() {
            Arc::new(Vec::new())
        } else {
            readers
        };
        BlockStream::start(
            source(readers, &operators, self.output_layers.clone()),
            thread_pool,
            statistics,
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
                batch_key: None,
                ..
            } => {
                let kind = CrossCellKind::HighlyVariableGenes {
                    layer: layer.clone(),
                    flavor: *flavor,
                };
                Ok(Some(Stage::CrossCell(CrossCellStage::new(
                    index, passes, kind,
                ))))
            }
            CrossCellOp::HighlyVariableGenes { .. } => {
                Err(not_yet("highly_variable_genes(batch_key=...)", "6.3"))
            }
            CrossCellOp::Scale { .. } => Err(not_yet(plan.node().method(), "6.4")),
            CrossCellOp::Pca { .. } => Err(not_yet(plan.node().method(), "6.5")),
        },
        Plan::Neighbors(_) => Err(not_yet(plan.node().method(), "6.6")),
        Plan::Cluster(_) => Err(not_yet(plan.node().method(), "6.8")),
        // Filters, selections, computed columns and sinks come with the terminals that
        // run plans from Python (7.2); plan 5.2 Open (1).
        Plan::Scan(_)
        | Plan::Filter(_)
        | Plan::SelectGenes(_)
        | Plan::TakeCells(_)
        | Plan::WithColumns(_)
        | Plan::Sink(_) => Err(not_yet(plan.node().method(), "7.2")),
    }
}

/// The names of the layers among `members`, in name order.
fn layer_names(members: &Members) -> impl Iterator<Item = String> + '_ {
    members.iter().filter_map(|member| match member {
        Member::Layer(name) => Some(name.clone()),
        _ => None,
    })
}

fn not_yet(method: &str, task: &str) -> KytoError {
    KytoError::NotImplementedYet {
        method: format!("{method} in the executor"),
        task: task.to_owned(),
    }
}
