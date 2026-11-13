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
//! | `with_default_layer` | 0 | 0 | none (resolved at build) |
//!
//! A pass counts one read of the node's input stream for one read of its output: a
//! node's work in its apply pass shares that pass with the per-cell nodes around it.
//! Blocks hold the store's cell block size (`C`, `schema.json` `tiling`) rows, or
//! fewer with [`PhysicalOptions::max_block_cells`] (task 5.5), and every gene read.
//! Does not: execute (5.x), choose blocks for a memory limit (`memory_limit.rs`), count the
//! passes over the store for the whole plan or estimate bytes and memory (4.6,
//! `estimate.rs`), pick
//! among several devices (9.1: [`Device`] has only the CPU), or print the logical
//! tree or JSON (4.7, `explain.rs`).

use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;

use crate::cross_cell::{CrossCellOp, HvgFlavor};
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

/// Where a node runs (R7: an argument, never a global setting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Device {
    /// The CPU (the streaming executor of 5.x). The only device until task 9.1.
    #[default]
    Cpu,
}

impl Device {
    /// The name `explain` shows (`cpu`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Cell blocks a stream reads at once by default (as `kyto_store::sink_h5ad`).
pub const DEFAULT_BLOCKS_IN_FLIGHT: NonZeroU64 = NonZeroU64::new(2).expect("2 is not zero");

/// The choices the physical plan is built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalOptions {
    /// The device every node runs on.
    pub device: Device,
    /// Cell blocks a stream reads and holds at once (memory grows with it; task 4.6's
    /// estimate counts it, 5.2's executor uses it).
    pub blocks_in_flight: NonZeroU64,
    /// At most this many cells per streamed block; `None` streams the store's cell
    /// blocks of `C` cells whole. Smaller blocks split each of the store's cell blocks
    /// (task 5.5: [`PhysicalPlan::within_memory_limit`] picks it for a memory limit).
    pub max_block_cells: Option<NonZeroU64>,
}

impl Default for PhysicalOptions {
    fn default() -> Self {
        Self {
            device: Device::default(),
            blocks_in_flight: DEFAULT_BLOCKS_IN_FLIGHT,
            max_block_cells: None,
        }
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
    device: Device,
    statistics_passes: u32,
    apply_pass: bool,
    blocks: Blocks,
}

impl Execution {
    /// The device it runs on.
    pub fn device(&self) -> Device {
        self.device
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
}

/// `device=cpu passes=2 (statistics=1 apply=1) blocks=4096 cells`.
impl fmt::Display for Execution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "device={} passes={}", self.device, self.passes())?;
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
        }
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
        for node in optimized.nodes() {
            // Inputs come first, so the input's block size is known.
            let cells = match (node.operation(), node.input()) {
                (Operation::Scan { plan, .. }, _) => {
                    let store_cells = plan.root().schema().tiling.cell_block_size;
                    options
                        .max_block_cells
                        .map_or(store_cells, |cells| store_cells.min(cells.get()))
                }
                (_, Some(input)) => block_sizes[input],
                // Only a scan has no input (the arena's invariant, as `Plan::root`).
                (_, None) => unreachable!("optimized node without input"),
            };
            block_sizes.push(cells);
            executions.push(execution(node, options.device, cells));
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
}

/// How `node` runs on `device`, streamed in blocks of `cells` cells if it is.
fn execution(node: &OptimizedNode, device: Device, cells: u64) -> Execution {
    let streamed = |statistics_passes: u32, apply_pass: bool| Execution {
        device,
        statistics_passes,
        apply_pass,
        blocks: Blocks::Cells(cells),
    };
    let held = |blocks: Blocks| Execution {
        device,
        statistics_passes: 0,
        apply_pass: false,
        blocks,
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
            Plan::WithDefaultLayer(_) => held(Blocks::None),
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
            Plan::Neighbors(_) | Plan::Cluster(_) | Plan::Umap(_) => held(Blocks::InMemory),
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
