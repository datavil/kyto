//! The pipeline executor (task 5.2): turns one output of a physical plan into a chain
//! of streamed operators and runs it.
//!
//! Does: walk from the output's node down to the scan, keep what the stream runs
//! today (the scan's layers, `with_default_layer`, per-cell `normalize_total`/`log1p`
//! maps, alone or fused), and refuse the rest with `KY-E601` naming the task that adds
//! it, before anything is read. [`Pipeline::run`] opens the scan's layers and starts
//! the stream ([`BlockStream`]) on a thread pool of exactly the threads asked for
//! ([`ExecuteOptions`], R7), with the plan's `blocks_in_flight` blocks read and
//! computed at once.
//! Does not: filters, gene selections, `take_cells`, computed columns, cross-cell
//! nodes, sinks (see the error each one gives), or memory limits (5.5).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{
    CellScope, CrossCellOp, LogicalNode, Member, Members, Operation, PhysicalPlan, Plan,
};
use kyto_store::{LayerReader, TileGrid};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::BlockStream;
use crate::elementwise::ElementwiseMaps;

/// How a pipeline runs, apart from the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecuteOptions {
    /// Threads that read and compute blocks (a thread pool of its own per run; the
    /// caller's thread consumes the stream). The output is the same on any number (R11).
    pub threads: NonZeroUsize,
}

impl ExecuteOptions {
    /// Run on `threads` threads.
    pub fn new(threads: NonZeroUsize) -> Self {
        Self { threads }
    }
}

/// One output of a physical plan, lowered to a scan and a chain of streamed operators.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub(crate) store: String,
    pub(crate) grid: TileGrid,
    /// The layers the scan reads.
    pub(crate) layers: Vec<String>,
    pub(crate) operators: Vec<ElementwiseMaps>,
    /// The layers the output reads; the stream drops the others at its end.
    pub(crate) output_layers: BTreeSet<String>,
    pub(crate) blocks_in_flight: NonZeroUsize,
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
        let mut operators = Vec::new();
        let mut next = Some(output_entry.node());
        let mut scan = None;
        while let Some(index) = next {
            let node = &nodes[index];
            match node.operation() {
                Operation::Scan { plan, reads } => {
                    scan = Some((Arc::clone(plan), layer_names(reads).collect()));
                }
                Operation::Fused { maps, writes } => {
                    if maps
                        .first()
                        .is_some_and(|first| first.cell_scope() == CellScope::CrossCell)
                    {
                        return Err(not_yet("normalize_total(target_sum=None)", "5.4"));
                    }
                    operators.push(ElementwiseMaps::new(maps.clone(), Some(writes)));
                }
                Operation::Node(logical) => {
                    if let Some(operator) = lower_node(logical)? {
                        operators.push(operator);
                    }
                }
            }
            next = node.input();
        }
        // The arena's inputs end at a scan (OptimizedPlan's invariant).
        let Some((scan, layers)) = scan else {
            unreachable!("an optimized plan's nodes lead down to a scan")
        };
        operators.reverse();
        let root = scan.root();
        let schema = root.schema();
        let grid = TileGrid::new(schema.n_cells, schema.n_genes, schema.tiling)?;
        // At most one window of every cell block (as 4.6's estimate counts them), so a
        // huge blocks_in_flight sizes nothing beyond the store; at least 1.
        let blocks_in_flight = plan
            .options()
            .blocks_in_flight
            .get()
            .min(grid.n_cell_blocks());
        let blocks_in_flight = usize::try_from(blocks_in_flight)
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN);
        let output_layers = layer_names(output_entry.reads()).collect();
        Ok(Self {
            store: root.source().to_owned(),
            grid,
            layers,
            operators,
            output_layers,
            blocks_in_flight,
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
    /// Opens every layer the scan reads before it returns, then streams in a thread of
    /// its own (see [`BlockStream`]).
    ///
    /// # Errors
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
        let readers = self
            .layers
            .iter()
            .map(|layer| LayerReader::open(Arc::clone(&storage), &self.store, layer, self.grid))
            .collect::<Result<Vec<_>, _>>()?;
        BlockStream::start(self, readers, options.threads)
    }
}

/// The operator of a logical node, `None` for one with no work at run time, or
/// `KY-E601` for one the executor does not run yet.
fn lower_node(plan: &Plan) -> Result<Option<ElementwiseMaps>, KytoError> {
    match plan {
        Plan::WithDefaultLayer(_) => Ok(None),
        Plan::MapElementwise(node) => {
            let map = node.params();
            if map.cell_scope() == CellScope::CrossCell {
                return Err(not_yet("normalize_total(target_sum=None)", "5.4"));
            }
            Ok(Some(ElementwiseMaps::new(vec![map.clone()], None)))
        }
        Plan::CrossCell(node) => {
            let task = match node.params().op() {
                CrossCellOp::Scale { .. } => "6.4",
                CrossCellOp::HighlyVariableGenes { .. } => "6.3",
                CrossCellOp::Pca { .. } => "6.5",
            };
            Err(not_yet(plan.node().method(), task))
        }
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
