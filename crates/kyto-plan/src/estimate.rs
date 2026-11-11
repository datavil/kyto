//! The cost estimate (task 4.6): what a [`PhysicalPlan`] would read and hold, from the
//! schema alone.
//!
//! Does: give a plan's total passes over the store, the bytes of matrix tiles those
//! passes read, its peak memory and its device ([`CostEstimate`]), from the store's
//! schema (sizes, dtypes, tiling), each node's output schema (upper bounds, task 4.1)
//! and the physical plan's passes and blocks (task 4.5). Reads nothing.
//!
//! **Passes over the store.** Each statistics pass of a streamed node streams its input
//! again from the scan (nothing between is kept); a node shared by two outputs makes
//! its passes once (the optimized plan is a DAG). Statistics passes of sibling nodes
//! (`highly_variable_genes` and `pca` on one layer) are not merged into one stream:
//! an upper bound, as is counting every streamed node's blocks at once below. One more, final pass delivers what
//! the outputs read from the stream: a layer, or a member written in a streamed node's
//! apply pass (a `with_cells` column computed from the matrix).
//!
//! **Bytes read** are the decoded bytes of the tiles each pass reads (format.md §4.7):
//! per layer the scan reads, `nnz x (4 + value bytes)` of `indices` and `data`,
//! `(C + 1) x 4` of `indptr` per tile and its `tile_offsets` (8 bytes per tile, plus 8),
//! times the passes. Empty tiles count as read (an upper bound: their number is only
//! known from `tile_offsets`). The compressed bytes on disk are fewer (zstd), and the
//! Parquet tables are not counted.
//!
//! **Peak memory** is the stream's working set plus what stays in memory, assuming all
//! of it is alive at once. Per block in flight (`min(blocks_in_flight, cell blocks)`),
//! with `e` the mean entries of a block (`nnz x rows / n_cells`, rows = `min(C,
//! n_cells)`), each streamed node holds:
//!
//! | Node | Bytes per block in flight |
//! | --- | --- |
//! | scan, per layer read | the larger of reading its tiles (`tiles` + `gene blocks x 2 x K x 4`: zarrs decodes whole inner chunks of `K` entries, [`ASSUMED_NNZ_CHUNK_SIZE`], of `indices` and `data`, every tile of a block at once at most) and joining them (`tiles` + `rows x 12 + 8` + the joined `indices` and `data`, which grow by doubling to `capacity(e)`, the last one twice while it moves: `capacity(e) x (4 + 1.5 x value bytes)`), with `tiles = gene blocks x (C + 1) x 4 + e x (4 + value bytes)` |
//! | a node writing layers in its pass (`normalize_total`, `log1p`, a fused pass, `scale`) | per layer written: `e x 8 + (rows + 1) x 8` (`float32` values; `scale` with centring is dense: `e = rows x n_genes`) |
//! | sink `zarr` | per layer, once: `S x (4 + value bytes)` ([`ASSUMED_NNZ_SHARD_SIZE`]) and one `indptr` shard, `gene blocks x (C + 1) x 4` |
//! | sink `h5ad` | per layer: `e x 8` (positions widened for HDF5) |
//!
//! Opening a layer, before the stream starts, decodes one 65 536-value chunk of its
//! `tile_offsets` and copies it (1 MiB for a moment); the blocks term is at least that.
//! What stays in memory: each scanned layer's `tile_offsets`; every column (8 bytes a
//! value; `Boolean` and 8-bit 1, 16-bit 2, 32-bit and `Enum` codes 4, `String` 16 as
//! Polars string views), embedding (`rows x width x dtype`) and graph (`nnz x (4 +
//! dtype) + (n_cells + 1) x 8`) a node writes; and the state of cross-cell nodes:
//! median `normalize_total` `n_cells x 8`, `scale` and `highly_variable_genes`
//! `n_genes x 16`, a `filter_genes` reading the matrix `n_genes x 8`, `pca` the streamed
//! Halko blocks `(2 x n_cells + n_genes) x (n_components + `[`PCA_OVERSAMPLES`]`) x 8`.
//!
//! The scan's terms are the reader's (`kyto_store::LayerReader`) as it allocates today,
//! checked within 25% on the fixtures (`tests/estimate_fixtures.rs`). The other rows
//! are promises to the executor (5.x) and the algorithms (6.x), which must check their
//! own row when they land.
//! Does not: read `tile_offsets` (exact per-block `nnz`) or the tables, count Polars
//! tables, threads' scratch or the output a terminal returns, or shrink blocks to a
//! memory limit (5.5 uses this estimate to do that).

use std::fmt;

use kyto_core::{ColumnDtype, FloatDtype, LayerDtype, LayerSchema};

use crate::cross_cell::CrossCellOp;
use crate::filter::Predicate;
use crate::map_elementwise::ElementwiseOp;
use crate::members::Member;
use crate::node::{CellScope, LogicalNode};
use crate::optimized_plan::{Operation, OptimizedNode};
use crate::physical_plan::{Blocks, Device, PhysicalPlan};
use crate::plan::Plan;
use crate::plan_schema::{PlanSchema, SparseMember};
use crate::sink::Sink;

/// Inner chunk size `K` of `indices` and `data` the estimate assumes: the store
/// writer's default (`kyto_store::DEFAULT_NNZ_CHUNK_SIZE`, provisional until 10.4).
/// `schema.json` does not record a store's own `K`.
pub const ASSUMED_NNZ_CHUNK_SIZE: u64 = 1 << 16;

/// Shard size `S` of `indices` and `data` the estimate assumes for a `zarr` sink: the
/// store writer's default (`kyto_store::DEFAULT_NNZ_SHARD_SIZE`).
pub const ASSUMED_NNZ_SHARD_SIZE: u64 = 1 << 20;

/// Bytes a layer's reader holds for a moment while it opens: one decoded chunk of
/// `tile_offsets` (format.md §4.4: 65 536 `uint64` values) and a copy of it. Opening
/// comes before streaming, so a plan needs the larger of this and its blocks.
const OPEN_LAYER_BYTES: u128 = 2 * 65_536 * 8;

/// Extra columns of the streamed randomized PCA's blocks beyond its components
/// (Halko et al., scikit-learn's default). Provisional; task 6.5 sets it.
pub const PCA_OVERSAMPLES: u64 = 10;

/// What a plan would read and hold, from its schema alone.
///
/// Sizes are bytes that saturate at `u64::MAX`, never wrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostEstimate {
    passes: u32,
    bytes_read: u64,
    block_memory: u64,
    resident_memory: u64,
    device: Device,
}

impl CostEstimate {
    /// Passes over the store's matrix (0 when the plan reads none).
    pub fn passes(&self) -> u32 {
        self.passes
    }

    /// Decoded bytes of the matrix tiles read, over all passes.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Peak memory: [`CostEstimate::block_memory`] plus
    /// [`CostEstimate::resident_memory`].
    pub fn peak_memory(&self) -> u64 {
        self.block_memory.saturating_add(self.resident_memory)
    }

    /// Bytes of the blocks in flight, in every streamed node at once (or, if more,
    /// what opening a layer takes for a moment).
    pub fn block_memory(&self) -> u64 {
        self.block_memory
    }

    /// Bytes held whatever the block size: statistics, in-memory members, columns.
    pub fn resident_memory(&self) -> u64 {
        self.resident_memory
    }

    /// The device the plan runs on.
    pub fn device(&self) -> Device {
        self.device
    }
}

/// `passes=1 bytes_read=18820524 peak_memory=60801241 device=cpu` (bytes).
impl fmt::Display for CostEstimate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "passes={} bytes_read={} peak_memory={} device={}",
            self.passes,
            self.bytes_read,
            self.peak_memory(),
            self.device
        )
    }
}

impl PhysicalPlan {
    /// What this plan would read and hold, without reading anything (the model in
    /// the `estimate` module doc).
    pub fn estimate(&self) -> CostEstimate {
        let nodes = self.optimized().nodes();
        let executions = self.executions();
        let schemas = node_schemas(nodes);
        let in_flight = u128::from(self.options().blocks_in_flight.get());

        let mut statistics_passes = 0_u32;
        let mut pass_bytes = 0_u128;
        let mut block = 0_u128;
        let mut opening = 0_u128;
        let mut resident = 0_u128;
        let mut streamed_writes = Vec::new();
        for (index, ((node, execution), schema)) in
            nodes.iter().zip(executions).zip(&schemas).enumerate()
        {
            statistics_passes = statistics_passes.saturating_add(execution.statistics_passes());
            let input = node.input().map(|input| &schemas[input]);
            let Blocks::Cells(cell_block_size) = execution.blocks() else {
                resident += node_resident(node, schema, input);
                continue;
            };
            if execution.apply_pass() && index > 0 {
                streamed_writes.extend(node.writes());
            }
            let sizes = BlockSizes::new(schema, input, cell_block_size);
            let (node_block, node_once) = match node.operation() {
                Operation::Scan { plan, reads } => {
                    let store = plan.root().schema();
                    let mut scan_block = 0;
                    for layer in scanned_layers(&store.layers, reads) {
                        let scan = ScanLayer::new(&sizes, layer);
                        scan_block += scan.block_bytes();
                        pass_bytes += scan.pass_bytes();
                        resident += scan.tile_offsets_bytes();
                        opening = OPEN_LAYER_BYTES;
                    }
                    (scan_block, 0)
                }
                _ => written_block_bytes(node, schema, input, &sizes),
            };
            block += node_block * in_flight.min(sizes.n_cell_blocks.max(1)) + node_once;
            resident += node_resident(node, schema, input);
        }

        let reads_stream = self.optimized().outputs().iter().any(|output| {
            output.reads().iter().any(|member| {
                matches!(member, Member::Layer(_)) || streamed_writes.contains(member)
            })
        });
        let streams = nodes
            .iter()
            .zip(executions)
            .any(|(_, execution)| matches!(execution.blocks(), Blocks::Cells(_)));
        let passes = statistics_passes.saturating_add(u32::from(reads_stream && streams));
        CostEstimate {
            passes,
            bytes_read: saturate(pass_bytes * u128::from(passes)),
            block_memory: saturate(block.max(opening)),
            resident_memory: saturate(resident),
            device: self.options().device,
        }
    }
}

impl Plan {
    /// This plan's [`CostEstimate`] on the CPU, for a terminal that reads all of its
    /// output ([`Plan::physical`] with the default options). Reads nothing.
    pub fn estimate(self: &std::sync::Arc<Self>) -> CostEstimate {
        self.physical(crate::physical_plan::PhysicalOptions::default())
            .estimate()
    }
}

/// The output schema of each optimized node: the logical node's, or for a fused pass
/// its input's with the maps' layers (`float32`, the input layer's `nnz`).
fn node_schemas(nodes: &[OptimizedNode]) -> Vec<PlanSchema> {
    let mut schemas: Vec<PlanSchema> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let schema = match (node.operation(), node.input()) {
            (Operation::Scan { plan, .. } | Operation::Node(plan), _) => {
                plan.output_schema().clone()
            }
            (Operation::Fused { maps, .. }, Some(input)) => {
                let mut schema = schemas[input].clone();
                for map in maps {
                    let member = SparseMember::Layer(map.layer().to_owned());
                    let nnz = schema.layer(map.layer()).map_or(0, |layer| layer.nnz);
                    let exact = schema.nnz_is_exact(&member);
                    let layer = LayerSchema {
                        name: map.output_layer().to_owned(),
                        dtype: LayerDtype::Float32,
                        nnz,
                    };
                    schema = schema.with_layer(layer, exact);
                }
                schema
            }
            // Only a scan has no input (the arena's invariant, as `Plan::root`).
            (Operation::Fused { .. }, None) => unreachable!("fused node without input"),
        };
        schemas.push(schema);
    }
    schemas
}

/// The store layers a scan reads, in schema order.
fn scanned_layers<'a>(
    layers: &'a [LayerSchema],
    reads: &'a crate::members::Members,
) -> impl Iterator<Item = &'a LayerSchema> {
    layers
        .iter()
        .filter(|layer| reads.contains(&Member::Layer(layer.name.clone())))
}

/// The block geometry of a streamed node, from its input's bounds (its own for a
/// scan) and the cell block size.
struct BlockSizes {
    /// Cells, an upper bound.
    n_cells: u128,
    /// The store's cell block size `C`.
    cell_block_size: u128,
    /// The store's gene blocks.
    n_gene_blocks: u128,
    /// Cell blocks of the stream.
    n_cell_blocks: u128,
    /// Rows of a full block: `min(C, n_cells)`.
    rows: u128,
}

impl BlockSizes {
    fn new(schema: &PlanSchema, input: Option<&PlanSchema>, cell_block_size: u64) -> Self {
        let sizes = input.unwrap_or(schema).atlas();
        let n_cells = u128::from(sizes.n_cells);
        let cell_block_size = u128::from(cell_block_size.max(1));
        let gene_block_size = u128::from(sizes.tiling.gene_block_size.max(1));
        let n_genes = u128::from(sizes.n_genes);
        Self {
            n_cells,
            cell_block_size,
            n_gene_blocks: n_genes.div_ceil(gene_block_size),
            n_cell_blocks: n_cells.div_ceil(cell_block_size),
            rows: cell_block_size.min(n_cells),
        }
    }

    /// Mean entries of a full block of a matrix with `nnz` entries.
    fn entries(&self, nnz: u64) -> u128 {
        if self.n_cells == 0 {
            return 0;
        }
        (u128::from(nnz) * self.rows).div_ceil(self.n_cells)
    }

    /// Bytes of one tile's `indptr`.
    fn tile_indptr_bytes(&self) -> u128 {
        (self.cell_block_size + 1) * 4
    }
}

/// One layer as the scan reads it (the reader's allocations, see the module doc).
struct ScanLayer {
    entries: u128,
    nnz: u128,
    value_bytes: u128,
    rows: u128,
    n_gene_blocks: u128,
    n_tiles: u128,
    tile_indptr_bytes: u128,
}

impl ScanLayer {
    fn new(sizes: &BlockSizes, layer: &LayerSchema) -> Self {
        Self {
            entries: sizes.entries(layer.nnz),
            nnz: u128::from(layer.nnz),
            value_bytes: layer_value_bytes(layer.dtype),
            rows: sizes.rows,
            n_gene_blocks: sizes.n_gene_blocks,
            n_tiles: sizes.n_cell_blocks * sizes.n_gene_blocks,
            tile_indptr_bytes: sizes.tile_indptr_bytes(),
        }
    }

    /// Bytes per block in flight: the larger of reading and joining its tiles.
    fn block_bytes(&self) -> u128 {
        if self.nnz == 0 {
            return 0;
        }
        let tiles =
            self.n_gene_blocks * self.tile_indptr_bytes + self.entries * (4 + self.value_bytes);
        // Every tile of the block may be decoding at once (one per thread), each holding
        // one decoded inner chunk of `indices` and one of `data`.
        let reading = tiles + self.n_gene_blocks * 2 * u128::from(ASSUMED_NNZ_CHUNK_SIZE) * 4;
        let capacity = vec_capacity(self.entries);
        // Row positions (`u32`), the joined `indptr` (`u64`), then `indices` at full
        // capacity and `data` moving from half to full capacity.
        let joining = tiles
            + self.rows * 4
            + (self.rows + 1) * 8
            + capacity * 4
            + capacity * self.value_bytes
            + capacity / 2 * self.value_bytes;
        reading.max(joining)
    }

    /// Decoded bytes one pass reads.
    fn pass_bytes(&self) -> u128 {
        let entries = self.nnz * (4 + self.value_bytes);
        let indptr = if self.nnz == 0 {
            0
        } else {
            self.n_tiles * self.tile_indptr_bytes
        };
        entries + indptr + self.tile_offsets_bytes()
    }

    /// Bytes of its `tile_offsets`, held while it is read.
    fn tile_offsets_bytes(&self) -> u128 {
        (self.n_tiles + 1) * 8
    }
}

/// The capacity a `Vec` reaches when `entries` items are pushed one at a time from
/// empty: doubling from 4 (Rust's growth for items of 2 to 1024 bytes).
fn vec_capacity(entries: u128) -> u128 {
    if entries == 0 {
        return 0;
    }
    entries.max(4).next_power_of_two().max(4)
}

/// A non-scan streamed node's bytes per block in flight (layers it writes in its
/// pass) and bytes it holds once (a sink's buffers).
fn written_block_bytes(
    node: &OptimizedNode,
    schema: &PlanSchema,
    input: Option<&PlanSchema>,
    sizes: &BlockSizes,
) -> (u128, u128) {
    let output_block = |nnz: u64| sizes.entries(nnz) * 8 + (sizes.rows + 1) * 8;
    let mut per_block = 0;
    for member in node.writes() {
        if let Member::Layer(name) = member
            && let Some(layer) = schema.layer(&name)
        {
            per_block += output_block(layer.nnz);
        }
    }
    let mut once = 0;
    if let Operation::Node(plan) = node.operation()
        && let Plan::Sink(sink) = plan.as_ref()
    {
        let layers = &input.unwrap_or(schema).atlas().layers;
        for layer in layers {
            let value_bytes = layer_value_bytes(layer.dtype);
            match sink.params() {
                Sink::Zarr { .. } => {
                    once += u128::from(ASSUMED_NNZ_SHARD_SIZE) * (4 + value_bytes)
                        + sizes.n_gene_blocks.max(1) * sizes.tile_indptr_bytes();
                }
                Sink::H5ad { .. } => per_block += sizes.entries(layer.nnz) * 8,
            }
        }
    }
    (per_block, once)
}

/// Bytes a node holds whatever the block size: the non-layer members it writes and
/// its statistics.
fn node_resident(node: &OptimizedNode, schema: &PlanSchema, input: Option<&PlanSchema>) -> u128 {
    if matches!(node.operation(), Operation::Scan { .. }) {
        // A scan's tables are Polars frames (not counted); its layers stream.
        return 0;
    }
    let atlas = schema.atlas();
    let n_cells = u128::from(atlas.n_cells);
    let n_genes = u128::from(atlas.n_genes);
    let mut bytes = 0;
    for member in node.writes() {
        bytes += match member {
            Member::Layer(_) => 0,
            Member::CellColumn(name) => schema
                .column(kyto_core::Axis::Cells, &name)
                .map_or(0, |column| n_cells * column_value_bytes(&column.dtype)),
            Member::GeneColumn(name) => schema
                .column(kyto_core::Axis::Genes, &name)
                .map_or(0, |column| n_genes * column_value_bytes(&column.dtype)),
            Member::Embedding(name) => schema.embedding(&name).map_or(0, |embedding| {
                let rows = match embedding.axis {
                    kyto_core::Axis::Cells => n_cells,
                    kyto_core::Axis::Genes => n_genes,
                };
                rows * u128::from(embedding.width) * float_bytes(embedding.dtype)
            }),
            Member::Graph(name) => schema.graph(&name).map_or(0, |graph| {
                let rows = match graph.axis {
                    kyto_core::Axis::Cells => n_cells,
                    kyto_core::Axis::Genes => n_genes,
                };
                u128::from(graph.nnz) * (4 + float_bytes(graph.dtype)) + (rows + 1) * 8
            }),
        };
    }
    bytes + statistics_bytes(node, input.unwrap_or(schema))
}

/// The state a cross-cell node keeps from its statistics passes, over its input.
fn statistics_bytes(node: &OptimizedNode, input: &PlanSchema) -> u128 {
    let n_cells = u128::from(input.atlas().n_cells);
    let n_genes = u128::from(input.atlas().n_genes);
    match node.operation() {
        Operation::Scan { .. } => 0,
        Operation::Fused { maps, .. } => match maps.first() {
            Some(map) if map.cell_scope() == CellScope::CrossCell => n_cells * 8,
            _ => 0,
        },
        Operation::Node(plan) => match plan.as_ref() {
            Plan::MapElementwise(map) => match map.params().op() {
                ElementwiseOp::NormalizeTotal { .. }
                    if plan.cell_scope() == CellScope::CrossCell =>
                {
                    n_cells * 8
                }
                _ => 0,
            },
            Plan::Filter(filter) => match filter.params().predicate() {
                Predicate::Genes(_) if plan.cell_scope() == CellScope::CrossCell => n_genes * 8,
                _ => 0,
            },
            Plan::CrossCell(cross_cell) => match cross_cell.params().op() {
                CrossCellOp::Scale { .. } | CrossCellOp::HighlyVariableGenes { .. } => n_genes * 16,
                CrossCellOp::Pca { n_components, .. } => {
                    (2 * n_cells + n_genes)
                        * (u128::from(*n_components) + u128::from(PCA_OVERSAMPLES))
                        * 8
                }
            },
            _ => 0,
        },
    }
}

/// Bytes of one stored value of a layer.
fn layer_value_bytes(dtype: LayerDtype) -> u128 {
    match dtype {
        LayerDtype::Float32 | LayerDtype::Uint32 => 4,
    }
}

/// Bytes of one value of a float dtype.
fn float_bytes(dtype: FloatDtype) -> u128 {
    match dtype {
        FloatDtype::Float32 => 4,
        FloatDtype::Float64 => 8,
    }
}

/// Bytes of one value of a column in memory (Polars): `String` as a 16-byte view.
fn column_value_bytes(dtype: &ColumnDtype) -> u128 {
    match dtype {
        ColumnDtype::Boolean | ColumnDtype::Int8 | ColumnDtype::UInt8 => 1,
        ColumnDtype::Int16 | ColumnDtype::UInt16 => 2,
        ColumnDtype::Int32
        | ColumnDtype::UInt32
        | ColumnDtype::Float32
        | ColumnDtype::Enum { .. } => 4,
        ColumnDtype::Int64 | ColumnDtype::UInt64 | ColumnDtype::Float64 => 8,
        ColumnDtype::String => 16,
    }
}

/// `value` as a `u64`, saturating.
fn saturate(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
