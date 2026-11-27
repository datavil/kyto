//! The physical plan (task 4.5): how each node of an [`OptimizedPlan`] runs.
//!
//! Does: choose, for every optimized node, the device it runs on, how it meets the
//! data (streamed over blocks of cells, in memory, or not at all) and the number of
//! passes it makes over its input's blocks: statistics passes (over all cells, before
//! it can output anything) and an apply pass (it maps the stream of blocks). The
//! rules, one arm per node kind in [`execution`]:
//!
//! | Node | Statistics | Apply | Blocks |
//! | --- | --- | --- | --- |
//! | scan, `filter_cells`, `filter_genes` on metadata, `select_genes`, `take_cells`, `normalize_total(target_sum)`, `log1p`, `with_cells` reading the matrix, sink | 0 | 1 | cells |
//! | `normalize_total` (median), `scale`, `filter_genes` reading the matrix, a fused pass starting with a median `normalize_total` | 1 | 1 | cells |
//! | `highly_variable_genes` (`seurat`, `cell_ranger`), `with_genes` reading the matrix | 1 | 0 | cells |
//! | `highly_variable_genes` (`seurat_v3`: moments, then clipped variances) | 2 | 0 | cells |
//! | `pca` (streamed randomized block Krylov: `2 +` [`PCA_POWER_ITERATIONS`]) | 11 | 0 | cells |
//! | `with_cells`/`with_genes` on metadata only, `neighbors`, `leiden`/`louvain` | 0 | 0 | in memory |
//! | `with_default_layer`, `on` | 0 | 0 | none (resolved at build) |
//!
//! A pass counts one read of the node's input stream for one read of its output: a
//! node's work in its apply pass shares that pass with the per-cell nodes around it.
//!
//! **Held inputs (task 5.12).** A streamed node with two or more statistics passes
//! (`pca`, `highly_variable_genes` with `seurat_v3`) holds its input in memory as CSR
//! when [`PhysicalOptions::hold_inputs`] allows it ([`Execution::holds_input`]): the
//! input streams from the store once, through the nodes below, and every statistics
//! pass runs over the held blocks, with the same bits (kyto-exec's `held_input.rs`).
//! The input is held by default; under a memory limit it is held only if the plan
//! still fits (`memory_limit.rs`), else it streams on every pass. Nodes with one
//! statistics pass (`scale`, a median `normalize_total`, `scrublet`, which gathers its
//! rows in its one pass anyway) have no pass to save: their apply pass is the output's
//! stream.
//! Blocks hold the store's cell block size (`C`, `schema.json` `tiling`) rows, or
//! fewer with [`PhysicalOptions::max_block_cells`] (task 5.5), and every gene read.
//! With more blocks in flight than [`PhysicalOptions::store_blocks_in_flight`] (a run on
//! many threads, task 5.14), every scan's blocks shrink to `ceil(n x C / in flight)`
//! cells, so the cells in flight stay those of `n` store cell blocks; the executor
//! still reads whole cell blocks ([`PhysicalOptions::scan_read_cells`]) and cuts them
//! in memory, so reading costs what it costs at `C`.
//!
//! **Devices (task 9.1).** Each node's [`Placement`] follows the device the plan asks
//! for (`on`, [`OptimizedPlan::device_request`]) on the devices found
//! ([`PhysicalOptions::devices`]), by the rules in `device_placement.rs`; a node that
//! cannot run as asked makes [`PhysicalPlan::device_error`] an error, which the
//! executor raises before reading anything.
//! Does not: execute (5.x), choose blocks for a memory limit (`memory_limit.rs`), count the
//! passes over the store for the whole plan or estimate bytes and memory (4.6,
//! `estimate.rs`), find the devices (`kyto_gpu::probe`), or print the logical
//! tree or JSON (4.7, `explain.rs`).

use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_gpu::{Device, Devices};

use crate::cross_cell::{CrossCellOp, HvgFlavor};
use crate::device_placement::{DeviceRequest, Placement, operations, place};
use crate::filter::Predicate;
use crate::node::{CellScope, LogicalNode};
use crate::optimize::{OptimizeOptions, Output, optimize};
use crate::optimized_plan::{NodeLine, Operation, OptimizedNode, OptimizedPlan};
use crate::plan::Plan;
use crate::with_columns::Columns;

/// Power iterations of the streamed randomized PCA (task 6.5: a block Krylov method on
/// the genes' side, Halko et al. 2011), so its passes are `2 + PCA_POWER_ITERATIONS`:
/// one for the sketch `X^T Omega` (with the gene means for the implicit centring), one
/// per iteration (the centred covariance times the newest block, `X^T (X Q)`, which
/// grows the basis), one for the embedding `(X - mu) V`. Set on the fixtures
/// (`kyto-exec/tests/it/pca_parity.rs`, 50 components of pbmc3k's 2,000 HVGs): 7 gave
/// scanpy's variances to 1.2e-4, 9 to 1.3e-6 (the `float32` reference's precision).
pub const PCA_POWER_ITERATIONS: u32 = 9;

/// Cell blocks a stream reads at once in [`PhysicalOptions::default`] (as
/// `kyto_store::sink_h5ad`): what `explain` and `estimate` assume when given no thread
/// count. A run uses [`PhysicalOptions::for_threads`] (task 5.13), which keeps the cells
/// in flight of this many store cell blocks (task 5.14).
pub const DEFAULT_BLOCKS_IN_FLIGHT: NonZeroU64 = NonZeroU64::new(2).expect("2 is not zero");

/// The choices the physical plan is built with.
///
/// [`Default`] streams [`DEFAULT_BLOCKS_IN_FLIGHT`] blocks at once;
/// [`PhysicalOptions::for_threads`] one per thread of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalOptions {
    /// The devices found on the machine (`kyto_gpu::probe()`), among which each node's
    /// device is chosen (task 9.1). [`Default`] is the CPU alone, so a plan explains
    /// the same on every machine unless it is given the probe (R7: an argument).
    pub devices: Devices,
    /// Cell blocks a stream reads and holds at once (memory grows with it; task 4.6's
    /// estimate counts it, 5.2's executor uses it).
    pub blocks_in_flight: NonZeroU64,
    /// At most this many cells per streamed block; `None` streams the store's cell
    /// blocks of `C` cells whole. Smaller blocks split each of the store's cell blocks
    /// (task 5.5: [`PhysicalPlan::within_memory_limit`] picks it for a memory limit).
    pub max_block_cells: Option<NonZeroU64>,
    /// Whether a node with several statistics passes may hold its input in memory and
    /// read the store once (task 5.12); `false` streams every pass from the store.
    /// [`PhysicalPlan::within_memory_limit`] turns it off when holding does not fit.
    pub hold_inputs: bool,
    /// At most the cells of this many store cell blocks in flight (task 5.14): with
    /// `Some(n)` and more than `n` [`PhysicalOptions::blocks_in_flight`], every scan
    /// streams blocks of at most `ceil(n x C / blocks_in_flight)` cells (`C` the
    /// scan's store cell block size; the smaller of it and
    /// [`PhysicalOptions::max_block_cells`] wins). `None` keeps the blocks as they are.
    pub store_blocks_in_flight: Option<NonZeroU64>,
}

impl Default for PhysicalOptions {
    fn default() -> Self {
        Self {
            devices: Devices::CPU_ONLY,
            blocks_in_flight: DEFAULT_BLOCKS_IN_FLIGHT,
            max_block_cells: None,
            hold_inputs: true,
            store_blocks_in_flight: None,
        }
    }
}

impl PhysicalOptions {
    /// The options of a run on `threads` threads with no memory limit: one cell block
    /// in flight per thread (task 5.13), so every thread computes a block of its own,
    /// holding the cells of [`DEFAULT_BLOCKS_IN_FLIGHT`] store cell blocks (task 5.14:
    /// on more threads the blocks are smaller, so the peak memory does not grow with
    /// the threads); the rest as [`Default`] (so `for_threads(2)` builds the default's plan). Under a
    /// memory limit the fit starts from these and lowers the blocks in flight until the
    /// plan fits ([`PhysicalPlan::within_memory_limit`]). The output does not depend
    /// on it (R11).
    pub fn for_threads(threads: std::num::NonZeroUsize) -> Self {
        let threads = u64::try_from(threads.get()).unwrap_or(u64::MAX);
        Self {
            blocks_in_flight: NonZeroU64::new(threads).unwrap_or(NonZeroU64::MIN),
            store_blocks_in_flight: Some(DEFAULT_BLOCKS_IN_FLIGHT),
            ..Self::default()
        }
    }

    /// The cells a scan reads at once from a store cell block of `store_cells` cells:
    /// the store's or [`PhysicalOptions::max_block_cells`], the smaller (at least 1).
    /// Blocks smaller than this, for [`PhysicalOptions::store_blocks_in_flight`], are
    /// cut in memory from what was read (task 5.14), so reading costs what it costs
    /// for whole blocks.
    pub fn scan_read_cells(&self, store_cells: u64) -> u64 {
        let limited = self.max_block_cells.map_or(u64::MAX, NonZeroU64::get);
        store_cells.min(limited).max(1)
    }

    /// The most cells per block of a scan whose store cell blocks hold `store_cells`
    /// cells: the store's, [`PhysicalOptions::max_block_cells`] and
    /// [`PhysicalOptions::store_blocks_in_flight`]'s share, the smallest (at least 1).
    pub fn scan_block_cells(&self, store_cells: u64) -> u64 {
        let in_flight = self.blocks_in_flight.get();
        let shared = self
            .store_blocks_in_flight
            .filter(|store_blocks| in_flight > store_blocks.get())
            .map_or(u64::MAX, |store_blocks| {
                store_cells
                    .saturating_mul(store_blocks.get())
                    .div_ceil(in_flight)
                    .max(1)
            });
        let limited = self.max_block_cells.map_or(u64::MAX, NonZeroU64::get);
        store_cells.min(shared).min(limited)
    }
}

/// How a node meets the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocks {
    /// Streamed over blocks of at most this many cells, each with every gene read.
    Cells(u64),
    /// On a whole table, embedding or graph held in memory; no matrix blocks.
    InMemory,
    /// Nothing at run time (resolved when the plan was built).
    None,
}

/// How one optimized node runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Execution {
    placement: Placement,
    statistics_passes: u32,
    apply_pass: bool,
    blocks: Blocks,
    holds_input: bool,
}

impl Execution {
    /// The device it runs on, `None` when it cannot run as asked (`KY-E501`,
    /// `KY-E504`: [`Execution::placement`] says why).
    pub fn device(&self) -> Option<Device> {
        self.placement.device()
    }

    /// Where it runs, and why when not where the plan asked (task 9.1).
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// Passes over all of its input's blocks before it outputs anything.
    pub fn statistics_passes(&self) -> u32 {
        self.statistics_passes
    }

    /// Whether it maps the stream of blocks, in one more pass.
    pub fn apply_pass(&self) -> bool {
        self.apply_pass
    }

    /// Its passes over its input's blocks: the statistics passes, plus the apply pass.
    pub fn passes(&self) -> u32 {
        self.statistics_passes + u32::from(self.apply_pass)
    }

    /// How it meets the data.
    pub fn blocks(&self) -> Blocks {
        self.blocks
    }

    /// Whether it holds its input in memory (task 5.12): the input streams from the
    /// store once and its statistics passes run over the held blocks.
    pub fn holds_input(&self) -> bool {
        self.holds_input
    }

    /// Passes over the store this node's statistics passes save by holding its input:
    /// its statistics passes but one, 0 when it streams them.
    pub fn passes_saved(&self) -> u32 {
        if self.holds_input {
            self.statistics_passes.saturating_sub(1)
        } else {
            0
        }
    }
}

/// `device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells`, then ` input held`
/// for a node holding its input (task 5.12); the device with its note when it does not
/// run where asked (`device=cpu (auto: no gpu kernel)`, task 9.1).
impl fmt::Display for Execution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "device={} passes={}",
            self.placement,
            self.passes()
        )?;
        if self.statistics_passes > 0 {
            write!(
                formatter,
                " (statistics={} apply={})",
                self.statistics_passes,
                u32::from(self.apply_pass)
            )?;
        }
        match self.blocks {
            Blocks::Cells(cells) => write!(formatter, " blocks={cells} cells"),
            Blocks::InMemory => formatter.write_str(" in memory"),
            Blocks::None => formatter.write_str(" no data"),
        }?;
        if self.holds_input {
            formatter.write_str(" input held")?;
        }
        Ok(())
    }
}

/// An optimized plan with how each of its nodes runs.
#[derive(Debug, Clone)]
pub struct PhysicalPlan {
    optimized: OptimizedPlan,
    executions: Vec<Execution>,
    options: PhysicalOptions,
}

impl PhysicalPlan {
    /// The physical plan of `optimized`, with the choices in `options`.
    pub fn new(optimized: OptimizedPlan, options: PhysicalOptions) -> Self {
        let mut block_sizes: Vec<u64> = Vec::with_capacity(optimized.nodes().len());
        let mut executions = Vec::with_capacity(optimized.nodes().len());
        let request = optimized.device_request();
        for node in optimized.nodes() {
            // Inputs come first, so the input's block size is known.
            let cells = match (node.operation(), node.input()) {
                (Operation::Scan { plan, .. }, _) => {
                    options.scan_block_cells(plan.root().schema().tiling.cell_block_size)
                }
                (_, Some(input)) => block_sizes[input],
                // Only a scan has no input (the arena's invariant, as `Plan::root`).
                (_, None) => unreachable!("optimized node without input"),
            };
            block_sizes.push(cells);
            let placement = place(
                request,
                options.devices,
                &operations(node),
                Device::has_kernel,
            );
            executions.push(execution(node, options, cells, placement));
        }
        Self {
            optimized,
            executions,
            options,
        }
    }

    /// The optimized plan it runs.
    pub fn optimized(&self) -> &OptimizedPlan {
        &self.optimized
    }

    /// The choices it was built with.
    pub fn options(&self) -> PhysicalOptions {
        self.options
    }

    /// How each node runs, one per [`OptimizedPlan::nodes`], in the same order.
    pub fn executions(&self) -> &[Execution] {
        &self.executions
    }

    /// The error a run of this plan raises before reading anything because a node
    /// cannot run where the plan asks (task 9.1): `KY-E504` when the device asked for
    /// was not found, else `KY-E501` for the first node (from the scan up) whose device
    /// has no kernel for it. `None` when every node can run.
    pub fn device_error(&self) -> Option<KytoError> {
        let placements = || self.executions.iter().map(Execution::placement);
        placements()
            .find(|placement| matches!(placement, Placement::NotFound(_)))
            .or_else(|| placements().find(|placement| placement.device().is_none()))
            .and_then(|placement| placement.error(self.options.devices))
    }

    /// The device the plan asks for (its last `on`, else the CPU).
    pub fn device_request(&self) -> DeviceRequest {
        self.optimized.device_request()
    }
}

/// How `node` runs with `options` on `placement`, streamed in blocks of `cells` cells
/// if it is.
fn execution(
    node: &OptimizedNode,
    options: PhysicalOptions,
    cells: u64,
    placement: Placement,
) -> Execution {
    let streamed = |statistics_passes: u32, apply_pass: bool| Execution {
        placement,
        statistics_passes,
        apply_pass,
        blocks: Blocks::Cells(cells),
        // Several statistics passes over one input: held once, if allowed (task 5.12).
        holds_input: options.hold_inputs && statistics_passes >= 2,
    };
    let held = |blocks: Blocks| Execution {
        placement,
        statistics_passes: 0,
        apply_pass: false,
        blocks,
        holds_input: false,
    };
    // A statistics pass first when the step needs all cells, then the stream.
    let mapped = |scope: CellScope| streamed(u32::from(scope == CellScope::CrossCell), true);
    match node.operation() {
        Operation::Scan { .. } => streamed(0, true),
        Operation::Fused { maps, .. } => mapped(
            maps.first()
                .map_or(CellScope::PerCell, LogicalNode::cell_scope),
        ),
        Operation::Node(plan) => match plan.as_ref() {
            Plan::Scan(_) => streamed(0, true),
            Plan::WithDefaultLayer(_) | Plan::OnDevice(_) => held(Blocks::None),
            Plan::Filter(filter) => match filter.params().predicate() {
                Predicate::Cells(_) => streamed(0, true),
                Predicate::Genes(_) => mapped(plan.cell_scope()),
            },
            Plan::SelectGenes(_) | Plan::TakeCells(_) | Plan::Sink(_) => streamed(0, true),
            Plan::MapElementwise(_) => mapped(plan.cell_scope()),
            Plan::CrossCell(cross_cell) => match cross_cell.params().op() {
                CrossCellOp::Scale { .. } => streamed(1, true),
                CrossCellOp::HighlyVariableGenes { flavor, .. } => match flavor {
                    HvgFlavor::Seurat | HvgFlavor::CellRanger => streamed(1, false),
                    HvgFlavor::SeuratV3 => streamed(2, false),
                },
                CrossCellOp::Pca { .. } => streamed(2 + PCA_POWER_ITERATIONS, false),
                // One pass gathers the layer's rows; the columns join the stream after.
                CrossCellOp::Scrublet { .. } => streamed(1, true),
            },
            Plan::WithColumns(with_columns) => {
                let metadata_only = match with_columns.params().columns() {
                    Columns::Cells(columns) => columns.iter().all(|(_, e)| e.is_metadata_only()),
                    Columns::Genes(columns) => columns.iter().all(|(_, e)| e.is_metadata_only()),
                };
                match (metadata_only, with_columns.params().columns()) {
                    (true, _) => held(Blocks::InMemory),
                    // Per cell: computed as the cell's block streams by.
                    (false, Columns::Cells(_)) => streamed(0, true),
                    // Per gene: sums over all cells, then a gene table column.
                    (false, Columns::Genes(_)) => streamed(1, false),
                }
            }
            Plan::Neighbors(_)
            | Plan::Cluster(_)
            | Plan::Umap(_)
            | Plan::WithEmbedding(_)
            | Plan::Metacells(_) => held(Blocks::InMemory),
        },
    }
}

/// One line per output, then one per node from the last (top) to the first (a scan),
/// each as the optimized plan prints it followed by how it runs in brackets.
impl fmt::Display for PhysicalPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.optimized.fmt_outputs(formatter)?;
        let nodes = self.optimized.nodes();
        for (index, (node, execution)) in nodes.iter().zip(&self.executions).enumerate().rev() {
            write!(formatter, "{} [{execution}]", NodeLine(index, node))?;
            if index > 0 {
                formatter.write_str("\n")?;
            }
        }
        Ok(())
    }
}

/// The physical plan of `outputs`, optimized with `optimize_options`.
pub fn physical_plan(
    outputs: &[Output],
    optimize_options: OptimizeOptions,
    options: PhysicalOptions,
) -> PhysicalPlan {
    PhysicalPlan::new(optimize(outputs, optimize_options), options)
}

impl Plan {
    /// This plan's physical plan with every optimizer rule, for a terminal that reads
    /// all of its output. The plan itself is unchanged.
    pub fn physical(self: &Arc<Self>, options: PhysicalOptions) -> PhysicalPlan {
        PhysicalPlan::new(self.optimize(), options)
    }
}
