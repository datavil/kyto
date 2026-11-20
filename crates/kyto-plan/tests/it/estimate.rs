//! Task 4.6: the cost estimate of plans, from the schema alone.
//!
//! A store of 10 000 cells x 2 048 genes (tiles of 4 096 x 1 024: 3 cell blocks, 2 gene
//! blocks), one `uint32` layer of 2 000 000 entries. Checks the scan's bytes and memory
//! by hand, the passes over the store (statistics passes re-stream from the scan, a
//! shared sub-plan once, a final pass only for what is read from the stream), what
//! nodes keep in memory, blocks in flight and blocks split to fewer cells (task 5.5), the
//! executor's held block and pool (5.5), saturation and the text form. The fixture
//! check against measured memory is `crates/kyto-exec/tests/estimate_fixtures.rs`.
#![allow(clippy::unwrap_used)]

use std::num::NonZeroU64;
use std::sync::Arc;

use kyto_core::{AtlasSchema, CellExpr, GeneExpr, GeneIds};
use kyto_plan::{
    ASSUMED_CATEGORY_BYTES, ASSUMED_NNZ_CHUNK_SIZE, ASSUMED_NNZ_SHARD_SIZE, ASSUMED_STRING_BYTES,
    CostEstimate, Device, HELD_BLOCK_BYTES, HvgFlavor, Member, Metric, NeighborSearch, Neighbors,
    OptimizeOptions, Output, PCA_POWER_ITERATIONS, PhysicalOptions, Plan, Scrublet, Sink,
    physical_plan,
};

const N_CELLS: u64 = 10_000;
const N_GENES: u64 = 2_048;

fn schema_json(n_cells: u64, n_genes: u64, nnz: u64) -> String {
    format!(
        r#"{{
  "n_cells": {n_cells},
  "n_genes": {n_genes},
  "tiling": {{"cell_block_size": 4096, "gene_block_size": 1024}},
  "cells": {{"columns": [
    {{"name": "cell_id", "dtype": "String"}},
    {{"name": "score", "dtype": "Int32"}}
  ]}},
  "genes": {{"columns": [{{"name": "gene_id", "dtype": "String"}}]}},
  "layers": [{{"name": "counts", "dtype": "uint32", "nnz": {nnz}}}],
  "embeddings": [],
  "graphs": []
}}"#
    )
}

fn scan_of(n_cells: u64, n_genes: u64, nnz: u64) -> Arc<Plan> {
    let schema: AtlasSchema = serde_json::from_str(&schema_json(n_cells, n_genes, nnz)).unwrap();
    let genes: Vec<String> = (0..n_genes).map(|gene| format!("G{gene}")).collect();
    Plan::scan("estimate.kyto", schema, GeneIds::new(genes))
}

fn scan() -> Arc<Plan> {
    scan_of(N_CELLS, N_GENES, 2_000_000)
}

fn estimate(outputs: &[Output]) -> CostEstimate {
    physical_plan(outputs, OptimizeOptions::ALL, PhysicalOptions::default()).estimate()
}

fn estimate_all(plan: &Arc<Plan>) -> CostEstimate {
    estimate(&[Output::all(plan)])
}

/// The estimate with every statistics pass streamed from the store (no held inputs,
/// task 5.12).
fn estimate_streamed(outputs: &[Output]) -> CostEstimate {
    let options = PhysicalOptions {
        hold_inputs: false,
        ..PhysicalOptions::default()
    };
    physical_plan(outputs, OptimizeOptions::ALL, options).estimate()
}

/// Bytes one pass over `counts` reads: entries, the `indptr` of 6 tiles, `tile_offsets`.
const PASS_BYTES: u64 = 2_000_000 * 8 + 6 * 4_097 * 4 + 7 * 8;

/// The scan's block, per block in flight: a full block holds 819 200 entries on average
/// (2 000 000 x 4 096 / 10 000). Its 2 tiles are `2 x 4 097 x 4 + 819 200 x 8` bytes;
/// joining them adds the row positions (`4 096 x 4`), the `indptr` (`4 097 x 8`) and
/// `indices`/`data` grown to 1 048 576 entries (`x (4 + 1.5 x 4)`), more than reading
/// (tiles + `2 x 2 x 65 536 x 4`).
const SCAN_BLOCK: u64 = (2 * 4_097 * 4 + 819_200 * 8) + 4_096 * 4 + 4_097 * 8 + 1_048_576 * 10;

/// The scan's last block (task 5.21): 10 000 cells leave 1 808 rows for the third
/// block, 361 600 entries on average. Its tiles are `2 x 4 097 x 4 + 361 600 x 8`
/// bytes; joining them adds the row positions (`1 808 x 4`), the `indptr` (`1 809 x 8`)
/// and `indices`/`data` in a full block's pooled buffers (`1 048 576 x (4 + 4)`, none
/// growing), more than reading (tiles + `2 x 2 x 65 536 x 4`).
const LAST_READ: u64 = (2 * 4_097 * 4 + 361_600 * 8) + 1_808 * 4 + 1_809 * 8 + 1_048_576 * 8;

/// A joined block of `counts` (or one mapped from it) as the consumer holds it and the
/// pool keeps it: `indices` and values at the capacity pushing reaches, and `indptr`.
const HELD_BLOCK: u64 = 1_048_576 * 8 + 4_097 * 8;

/// The cell table in memory (task 5.10): `cell_id` at the assumed 40 bytes a `String`
/// value, `score` 4 bytes a value.
const CELL_TABLE: u64 = N_CELLS * 40 + N_CELLS * 4;

/// The gene table in memory: `gene_id` at 40 bytes a value.
const GENE_TABLE: u64 = N_GENES * 40;

#[test]
fn a_scan_read_whole() {
    let estimate = estimate_all(&scan());
    assert_eq!(estimate.passes(), 1);
    assert_eq!(estimate.bytes_read(), PASS_BYTES);
    // Two blocks in flight (the default), and `tile_offsets` (7 values) held.
    assert_eq!(estimate.block_memory(), 2 * SCAN_BLOCK);
    assert_eq!(estimate.resident_memory(), 7 * 8);
    assert_eq!(estimate.peak_memory(), 2 * SCAN_BLOCK + 56);
    assert_eq!(estimate.device(), Device::Cpu);
    assert_eq!(
        estimate.to_string(),
        format!(
            "passes=1 bytes_read={PASS_BYTES} peak_memory={} device=cpu",
            2 * SCAN_BLOCK + 56
        )
    );
}

#[test]
fn blocks_in_flight_scale_the_blocks_up_to_the_cell_blocks() {
    let plan = scan();
    let with = |blocks: u64| {
        let options = PhysicalOptions {
            blocks_in_flight: NonZeroU64::new(blocks).unwrap(),
            ..PhysicalOptions::default()
        };
        plan.physical(options).estimate()
    };
    // One block in flight over 3: while the last is read, the consumer holds the one
    // before, and the read takes the first's buffers back from the pool (task 5.18).
    assert_eq!(with(1).block_memory(), SCAN_BLOCK + HELD_BLOCK);
    assert_eq!(with(3).block_memory(), 3 * SCAN_BLOCK);
    // Only 3 cell blocks exist.
    assert_eq!(with(8).block_memory(), 3 * SCAN_BLOCK);
    assert_eq!(with(8).bytes_read(), PASS_BYTES);
}

/// Task 5.13: a run holds one block in flight per thread; under a memory limit the fit
/// starts there and halves them.
#[test]
fn a_run_holds_one_block_in_flight_per_thread() {
    use std::num::NonZeroUsize;

    use kyto_plan::{MemoryLimit, PhysicalPlan};

    let threads =
        |threads: usize| PhysicalOptions::for_threads(NonZeroUsize::new(threads).unwrap());
    assert_eq!(threads(1).blocks_in_flight.get(), 1);
    assert_eq!(threads(10).blocks_in_flight.get(), 10);
    assert_eq!(
        PhysicalOptions {
            blocks_in_flight: PhysicalOptions::default().blocks_in_flight,
            store_blocks_in_flight: None,
            ..threads(10)
        },
        PhysicalOptions::default()
    );
    // usize::MAX threads saturate (u64 is as wide on every target kyto builds for).
    assert_eq!(threads(usize::MAX).blocks_in_flight.get(), u64::MAX);
    let plan = scan();
    // 2 threads build the default's plan.
    assert_eq!(
        plan.physical(threads(2)).estimate(),
        plan.physical(PhysicalOptions::default()).estimate()
    );
    // A limit with room for two full blocks only: halved from 8 (8, 4, 2) to 2, the
    // blocks kept whole.
    let fits_two = 2 * SCAN_BLOCK + 56;
    let fitted = PhysicalPlan::within_memory_limit(
        plan.optimize(),
        threads(8),
        MemoryLimit::new(NonZeroU64::new(fits_two).unwrap()),
    )
    .unwrap();
    assert_eq!(fitted.options().blocks_in_flight.get(), 2);
    assert_eq!(fitted.estimate().peak_memory(), fits_two);
    // From 3 threads too (3, 2, 1: not halved to 1, which would lose a block).
    let fitted = PhysicalPlan::within_memory_limit(
        plan.optimize(),
        threads(3),
        MemoryLimit::new(NonZeroU64::new(fits_two).unwrap()),
    )
    .unwrap();
    assert_eq!(fitted.options().blocks_in_flight.get(), 2);
    assert_eq!(
        fitted.options().max_block_cells,
        Some(NonZeroU64::new(4_096).unwrap())
    );
    // A limit the threads' blocks fit leaves them as they are.
    let roomy = PhysicalPlan::within_memory_limit(
        plan.optimize(),
        threads(3),
        MemoryLimit::new(NonZeroU64::new(u64::MAX).unwrap()),
    )
    .unwrap();
    assert_eq!(roomy.options(), threads(3));
}

/// Task 5.14: on more threads than 2 a run's blocks shrink so the cells in flight stay
/// those of 2 store cell blocks; a memory limit's fit still starts from the store's.
#[test]
fn many_threads_share_two_cell_blocks() {
    use std::num::NonZeroUsize;

    use kyto_plan::{Blocks, MemoryLimit, PhysicalPlan};

    let threads =
        |threads: usize| PhysicalOptions::for_threads(NonZeroUsize::new(threads).unwrap());
    let scan_cells =
        |options: PhysicalOptions| match scan().physical(options).executions()[0].blocks() {
            Blocks::Cells(cells) => cells,
            blocks => panic!("a scan streams, got {blocks:?}"),
        };
    // ceil(2 x 4096 / threads), the store's 4096 on 1 and 2 threads.
    for (count, cells) in [(1, 4_096), (2, 4_096), (3, 2_731), (4, 2_048), (10, 820)] {
        assert_eq!(scan_cells(threads(count)), cells, "{count} threads");
    }
    // At least 1 cell, however many threads.
    assert_eq!(scan_cells(threads(100_000)), 1);
    assert_eq!(scan_cells(threads(usize::MAX)), 1);
    // A smaller max_block_cells wins; without the share the store's blocks stay.
    let capped = PhysicalOptions {
        max_block_cells: NonZeroU64::new(500),
        ..threads(4)
    };
    assert_eq!(scan_cells(capped), 500);
    let unshared = PhysicalOptions {
        store_blocks_in_flight: None,
        ..threads(10)
    };
    assert_eq!(scan_cells(unshared), 4_096);
    // A limit with room for 2 full blocks: the fit's candidates are the store's
    // blocks, as before the share (5.13's 2 blocks in flight of 4096 cells).
    let fits_two = 2 * SCAN_BLOCK + 56;
    let fitted = PhysicalPlan::within_memory_limit(
        scan().optimize(),
        threads(8),
        MemoryLimit::new(NonZeroU64::new(fits_two).unwrap()),
    )
    .unwrap();
    assert!(fitted.estimate().peak_memory() <= fits_two);
    let roomy = PhysicalPlan::within_memory_limit(
        scan().optimize(),
        threads(10),
        MemoryLimit::new(NonZeroU64::new(u64::MAX).unwrap()),
    )
    .unwrap();
    assert_eq!(roomy.options(), threads(10));
    assert_eq!(roomy.executions()[0].blocks(), Blocks::Cells(820));
    // The parts are cut from whole reads: every pass reads what 2 threads read
    // (each tile's indptr once), whatever the threads.
    let plan = scan();
    assert_eq!(
        plan.physical(threads(10)).estimate().bytes_read(),
        PASS_BYTES
    );
    assert_eq!(
        plan.physical(threads(3)).estimate().bytes_read(),
        PASS_BYTES
    );
}

#[test]
fn split_blocks_hold_fewer_cells() {
    let plan = scan();
    let with = |cells: u64| {
        let options = PhysicalOptions {
            blocks_in_flight: NonZeroU64::new(1).unwrap(),
            max_block_cells: NonZeroU64::new(cells),
            ..PhysicalOptions::default()
        };
        plan.physical(options).estimate()
    };
    // More than C cells is C.
    assert_eq!(with(10_000), with(4_096));
    // 1 024 cells: 204 800 entries a block (2 000 000 x 1 024 / 10 000), its 2 tiles
    // with their whole `indptr`, joined into 262 144 entries (more than reading them,
    // decoding a 65 536-entry chunk of `indices` and `data` per tile).
    let tiles = 2 * 4_097 * 4 + 204_800 * 8;
    let joining = tiles + 1_024 * 4 + 1_025 * 8 + 262_144 * 10;
    let reading = tiles + 2 * 2 * 65_536 * 4;
    assert!(joining > reading);
    let held = 262_144 * 8 + 1_025 * 8;
    let small = with(1_024);
    // 12 blocks (3 cell blocks of 4 parts): the consumer's block; the read takes the
    // pool's buffers (task 5.18: the stream allocates 2 blocks' buffers, no more).
    assert_eq!(small.block_memory(), joining + held);
    // Each block reads its tiles' whole `indptr`: 4 x the indptr bytes of whole blocks.
    assert_eq!(
        small.bytes_read(),
        2_000_000 * 8 + 4 * 6 * 4_097 * 4 + 7 * 8
    );
    assert_eq!(small.resident_memory(), with(4_096).resident_memory());
}

#[test]
fn a_tiny_store_needs_the_memory_to_open_its_layer() {
    // 10 entries: the blocks are smaller than one decoded `tile_offsets` chunk and
    // its copy (2 x 65 536 x 8 bytes), which opening the layer holds for a moment.
    let estimate = estimate_all(&scan_of(5, 4, 10));
    assert_eq!(estimate.block_memory(), 2 * 65_536 * 8);
}

#[test]
fn statistics_passes_stream_again_and_a_final_pass_reads_the_layers() {
    let normalized = scan()
        .normalize_total(None, Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap();
    let pca = normalized.pca(Some("log1p"), "X_pca", 50, 0, None).unwrap();
    let pca_passes = 2 + PCA_POWER_ITERATIONS;
    // Streamed: median 1 statistics pass; PCA's; then one pass delivers the layers.
    let all = estimate_streamed(&[Output::all(&pca)]);
    assert_eq!(all.passes(), 1 + pca_passes + 1);
    assert_eq!(all.passes_saved(), 0);
    assert_eq!(all.held_memory(), 0);
    assert_eq!(all.bytes_read(), u64::from(all.passes()) * PASS_BYTES);
    // Reading only the embedding needs no final pass.
    let embedding_output = [Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap()];
    assert_eq!(
        estimate_streamed(&embedding_output).passes(),
        1 + pca_passes
    );
    // Held (task 5.12, the default): PCA reads its input once and saves the rest.
    let held = estimate_all(&pca);
    assert_eq!(held.passes(), 1 + 1 + 1);
    assert_eq!(held.passes_saved(), pca_passes - 1);
    assert_eq!(held.bytes_read(), 3 * PASS_BYTES);
    assert_eq!(estimate(&embedding_output).passes(), 1 + 1);
    // The same without fusion: the passes do not depend on the rules.
    let unfused = physical_plan(
        &[Output::all(&pca)],
        OptimizeOptions::NONE,
        PhysicalOptions::default(),
    )
    .estimate();
    assert_eq!(unfused.passes(), held.passes());
}

#[test]
fn a_held_input_is_resident_and_saves_passes() {
    // PCA over `counts` (task 5.12): it holds the layer's entries (`uint32` values and
    // `u32` indices), each cell's `indptr` entry and position, and per block (3) the
    // held block's overhead, on top of everything streaming it holds.
    let pca = scan().pca(Some("counts"), "X_pca", 50, 0, None).unwrap();
    let output = [Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap()];
    let held = estimate(&output);
    let streamed = estimate_streamed(&output);
    let held_bytes = 2_000_000 * 8 + N_CELLS * 16 + 3 * HELD_BLOCK_BYTES;
    assert_eq!(held.held_memory(), held_bytes);
    assert_eq!(
        held.resident_memory(),
        streamed.resident_memory() + held_bytes
    );
    assert_eq!(held.block_memory(), streamed.block_memory());
    assert_eq!(held.passes(), 1);
    assert_eq!(held.passes_saved(), 1 + PCA_POWER_ITERATIONS);
    // A node with one statistics pass has none to save and holds nothing.
    let scaled = scan().scale(Some("counts"), "scaled", false, None).unwrap();
    let scale = estimate_all(&scaled);
    assert_eq!(scale.held_memory(), 0);
    assert_eq!(scale.passes_saved(), 0);
    assert_eq!(scale.passes(), 2);
}

#[test]
fn a_shared_sub_plan_makes_its_passes_once() {
    let normalized = scan()
        .normalize_total(None, Some("counts"), "normalized")
        .unwrap();
    let hvg = normalized
        .highly_variable_genes(Some("normalized"), HvgFlavor::SeuratV3, Some(100), None)
        .unwrap();
    let pca = normalized
        .pca(Some("normalized"), "X_pca", 10, 0, None)
        .unwrap();
    let outputs = [
        Output::members(&hvg, [Member::GeneColumn("highly_variable".into())]).unwrap(),
        Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap(),
    ];
    // Median 1 (once), seurat_v3 2, PCA 2 + its iterations; nothing read from the stream.
    assert_eq!(
        estimate_streamed(&outputs).passes(),
        1 + 2 + 2 + PCA_POWER_ITERATIONS
    );
    // Held (task 5.12): seurat_v3 and PCA read their input once each.
    let held = estimate(&outputs);
    assert_eq!(held.passes(), 1 + 1 + 1);
    assert_eq!(held.passes_saved(), 1 + 1 + PCA_POWER_ITERATIONS);
}

#[test]
fn metadata_only_plans_read_no_matrix() {
    let plan = scan()
        .with_cells([("twice".to_owned(), CellExpr::column("score"))])
        .unwrap();
    let estimate =
        estimate(&[Output::members(&plan, [Member::CellColumn("twice".into())]).unwrap()]);
    assert_eq!(estimate.passes(), 0);
    assert_eq!(estimate.bytes_read(), 0);
    // The scan reads no layer: no blocks, no `tile_offsets`. The executor reads the
    // cell table (task 5.10), half again its largest column (`cell_id`) for a moment,
    // and holds it and the column.
    assert_eq!(estimate.block_memory(), N_CELLS * 40 * 3 / 2);
    assert_eq!(estimate.resident_memory(), N_CELLS * 4 + CELL_TABLE);
}

#[test]
fn a_column_computed_in_the_stream_needs_a_final_pass() {
    let plan = scan()
        .with_cells([(
            "cd3".to_owned(),
            CellExpr::expression(Some("counts".to_owned()), "G1"),
        )])
        .unwrap();
    let estimate = estimate(&[Output::members(&plan, [Member::CellColumn("cd3".into())]).unwrap()]);
    assert_eq!(estimate.passes(), 1);
    assert_eq!(estimate.bytes_read(), PASS_BYTES);
}

#[test]
fn nodes_writing_layers_hold_an_output_block() {
    let base = estimate_all(&scan()).block_memory();
    assert_eq!(base, 2 * SCAN_BLOCK);
    let log1p = estimate_all(&scan().log1p(Some("counts"), "log1p").unwrap()).block_memory();
    // The output block (`float32` values and `u32` indices of a mean block and its
    // `indptr`) is held with the joined block, less than reading it; but while the last
    // block is read the consumer holds a block of both layers and the pool keeps one
    // buffer: of the 4 the first window allocated, the consumer holds 2 and the last
    // block's `log1p` takes one back (task 5.10). The last block holds 1 808 cells,
    // read at its own rows (task 5.21).
    const { assert!(HELD_BLOCK + 819_200 * 8 + 4_097 * 8 < SCAN_BLOCK) };
    assert_eq!(log1p, LAST_READ + 3 * HELD_BLOCK);
    // Centred `scale` is dense: every gene of every row, with the joined block.
    let scaled = estimate_all(&scan().scale(Some("counts"), "scaled", true, None).unwrap());
    let dense = 4_096 * N_GENES * 8 + 4_097 * 8;
    assert!(HELD_BLOCK + dense > SCAN_BLOCK);
    // While the last block (1 808 cells, task 5.21) is read: its tiles, the consumer's block of both layers (the
    // dense one at its capacity, 2^23 entries, is `dense`), the pool's other buffer (at
    // most `dense`), and the read's buffer taken from the pool, which may be the dense
    // one the consumer gave back last (task 5.18: `dense` instead of `HELD_BLOCK`).
    assert_eq!(
        scaled.block_memory(),
        LAST_READ - HELD_BLOCK + dense + (HELD_BLOCK + dense) + dense
    );
    // Its statistics: a mean and a variance per gene, and while its statistics pass
    // runs, the same per leaf of its reduction (task 5.6): 2 blocks in flight and the
    // tree's subtrees over 3 blocks (log2(3) + 1 = 2).
    assert_eq!(scaled.resident_memory() - 56, N_GENES * 16 * (1 + 2 + 2));
}

#[test]
fn a_layer_no_later_step_reads_is_not_carried() {
    // Task 5.16: `log1p` of counts, then a gene filter (it copies the layers a block
    // carries). An output reading only `log1p` drops `counts` after the map, so the
    // gene filter copies one layer; reading both, it copies both.
    let plan = scan()
        .log1p(Some("counts"), "log1p")
        .unwrap()
        .filter_genes(
            &GeneExpr::column("gene_id")
                .str(kyto_core::StrFunction::StartsWith("G1".to_owned()))
                .unwrap(),
        )
        .unwrap();
    let reading = |layers: &[&str]| {
        let members = layers
            .iter()
            .map(|layer| Member::Layer((*layer).to_owned()));
        estimate(&[Output::members(&plan, members).unwrap()]).block_memory()
    };
    let log1p_only = reading(&["log1p"]);
    let both = reading(&["counts", "log1p"]);
    assert!(log1p_only < both, "{log1p_only} vs {both}");
    // Per block in flight, at least the copy of `counts` the gene filter does not make.
    let filtered_entries = 819_200 * 1_111 / N_GENES;
    assert!(
        both - log1p_only >= 2 * filtered_entries,
        "{log1p_only} vs {both}"
    );
}

#[test]
fn in_memory_members_and_statistics_are_resident() {
    let pca = scan().pca(Some("counts"), "X_pca", 50, 0, None).unwrap();
    // Streamed, so nothing is held (task 5.12: `a_held_input_is_resident_and_saves_passes`).
    let pca_estimate =
        estimate_streamed(&[Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap()]);
    let embeddings = N_CELLS * 50 * 4 + N_GENES * 50 * 4;
    // Its Krylov basis and products (60 columns a block) with their copies, means and
    // the embedding's rows in f64 with their cells and in f32; per leaf of a pass (2 blocks in flight and 2
    // subtrees, as `scale` above) the sketch and the gene sums.
    let columns = (50 + 10).min(N_GENES);
    let krylov = N_GENES * (4 * columns * u64::from(PCA_POWER_ITERATIONS) + 1) * 8
        + N_CELLS * (50 + 1) * 8
        + N_CELLS * 50 * 4;
    let leaves = N_GENES * (columns * 8 + 16) * (2 + 2);
    assert_eq!(
        pca_estimate.resident_memory(),
        56 + embeddings + krylov + leaves
    );

    let neighbors = pca
        .neighbors(Neighbors {
            embedding: "X_pca".to_owned(),
            n_components: None,
            n_neighbors: 15,
            metric: Metric::Euclidean,
            search: NeighborSearch::Exact,
            seed: 0,
            key: None,
        })
        .unwrap();
    let graphs =
        estimate_streamed(&[
            Output::members(&neighbors, [Member::Graph("connectivities".into())]).unwrap(),
        ]);
    let distances = neighbors.output_schema().graph("distances").unwrap().nnz;
    let connectivities = neighbors
        .output_schema()
        .graph("connectivities")
        .unwrap()
        .nnz;
    let graph_bytes = (distances + connectivities) * 8 + 2 * (N_CELLS + 1) * 8;
    assert_eq!(
        graphs.resident_memory(),
        pca_estimate.resident_memory() + graph_bytes
    );
}

#[test]
fn sinks_hold_their_buffers() {
    let base = estimate_all(&scan()).block_memory();
    let zarr = scan()
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    // One shard of `S` entries and one `indptr` shard, once.
    assert_eq!(
        estimate_all(&zarr).block_memory() - base,
        (1 << 20) * 8 + 2 * 4_097 * 4
    );
    let h5ad = scan()
        .sink(Sink::H5ad {
            path: "out.h5ad".to_owned(),
        })
        .unwrap();
    // Positions widened to 8 bytes, per block in flight, held with the joined block:
    // less than reading it.
    const { assert!(HELD_BLOCK + 819_200 * 8 < SCAN_BLOCK) };
    assert_eq!(estimate_all(&h5ad).block_memory(), base);
}

#[test]
fn huge_sizes_saturate() {
    let plan = scan_of(u64::from(u32::MAX), 4, i64::MAX as u64)
        .scale(Some("counts"), "scaled", true, None)
        .unwrap();
    let estimate = estimate_all(&plan);
    // 2^63 entries of 8 bytes: more than `u64::MAX` bytes per pass.
    assert_eq!(estimate.bytes_read(), u64::MAX);
    // A block holds 2^63 x 4 096 / (2^32 - 1) entries: computed without overflow.
    assert!(estimate.peak_memory() > 1 << 46);
}

#[test]
fn an_absurd_doublet_ratio_saturates() {
    // `sim_doublet_ratio` 1e300: the doublet rows overflow `u128` and must saturate,
    // not wrap into a small estimate (bug-hunter, 6.12).
    let plan = scan()
        .scrublet(
            Some("counts"),
            Scrublet {
                sim_doublet_ratio: 1e300,
                ..Scrublet::default()
            },
        )
        .unwrap();
    assert_eq!(estimate_all(&plan).peak_memory(), u64::MAX);
}

#[test]
fn assumed_chunk_sizes_are_the_writer_defaults() {
    assert_eq!(ASSUMED_NNZ_CHUNK_SIZE, kyto_store::DEFAULT_NNZ_CHUNK_SIZE);
    assert_eq!(ASSUMED_NNZ_SHARD_SIZE, kyto_store::DEFAULT_NNZ_SHARD_SIZE);
}

#[test]
fn steps_read_the_tables_they_need() {
    let scan_only = estimate_all(&scan());
    let resident = |plan: &Arc<Plan>| estimate_all(plan).resident_memory() - 56;
    // A cell filter reads the cell table; a gene filter the gene table.
    let cells = scan()
        .filter_cells(
            &kyto_core::Expr::binary(
                CellExpr::column("score"),
                kyto_core::BinaryOp::Gt,
                kyto_core::Literal::Int(0),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(resident(&cells), CELL_TABLE);
    let genes = scan()
        .filter_genes(
            &GeneExpr::column("gene_id")
                .str(kyto_core::StrFunction::StartsWith("MT-".to_owned()))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(resident(&genes), GENE_TABLE);
    // `with_cells` reading one gene's values reads both; `take_cells` the cell table,
    // `select_genes` the gene table; a sink both.
    let gene_values = scan()
        .with_cells([(
            "g1".to_owned(),
            CellExpr::expression(Some("counts".to_owned()), "G1"),
        )])
        .unwrap();
    assert_eq!(
        resident(&gene_values),
        N_CELLS * 4 + CELL_TABLE + GENE_TABLE
    );
    let taken = scan().take_cells(["c1".to_owned()]).unwrap();
    assert_eq!(resident(&taken), CELL_TABLE);
    let selected = scan().select_genes(["G1".to_owned()]).unwrap();
    assert_eq!(resident(&selected), GENE_TABLE);
    let sink = scan()
        .sink(Sink::Zarr {
            path: "out.kyto".to_owned(),
        })
        .unwrap();
    assert_eq!(resident(&sink), CELL_TABLE + GENE_TABLE);
    // A map reads no table.
    let log1p = scan().log1p(Some("counts"), "log1p").unwrap();
    assert_eq!(resident(&log1p), scan_only.resident_memory() - 56);
}

#[test]
fn an_enum_column_holds_its_categories() {
    let json = schema_json(N_CELLS, N_GENES, 2_000_000).replace(
        r#"{"name": "gene_id", "dtype": "String"}"#,
        r#"{"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 30000, "ordered": false}"#,
    );
    let schema: AtlasSchema = serde_json::from_str(&json).unwrap();
    let genes: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let plan = Plan::scan("estimate.kyto", schema, GeneIds::new(genes))
        .filter_genes(
            &GeneExpr::column("symbol")
                .str(kyto_core::StrFunction::StartsWith("MT-".to_owned()))
                .unwrap(),
        )
        .unwrap();
    let estimate = estimate_all(&plan);
    // A 4-byte code per gene and 160 bytes per category, used or not.
    let symbol = N_GENES * 4 + 30_000 * 160;
    assert_eq!(estimate.resident_memory() - 56, GENE_TABLE + symbol);
    // A second Enum column: reading takes half again both, decoding at once.
    let json = json.replace(
        r#""ordered": false}"#,
        r#""ordered": false},
    {"name": "name", "dtype": "Enum", "n_categories": 30000, "ordered": false}"#,
    );
    let schema: AtlasSchema = serde_json::from_str(&json).unwrap();
    let genes: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let two = Plan::scan("estimate.kyto", schema, GeneIds::new(genes))
        .filter_genes(
            &GeneExpr::column("symbol")
                .str(kyto_core::StrFunction::StartsWith("MT-".to_owned()))
                .unwrap(),
        )
        .unwrap();
    // Reading only the gene column: no layer streams, the table read is the blocks
    // term.
    let output = Output::members(&two, [Member::GeneColumn("symbol".into())]).unwrap();
    let reading = physical_plan(&[output], OptimizeOptions::ALL, PhysicalOptions::default())
        .estimate()
        .block_memory();
    assert_eq!(reading, 2 * symbol * 3 / 2);
    assert_eq!((ASSUMED_STRING_BYTES, ASSUMED_CATEGORY_BYTES), (40, 160));
}

#[test]
fn a_filter_copies_the_blocks_it_rewrites() {
    let predicate = kyto_core::Expr::binary(
        CellExpr::column("score"),
        kyto_core::BinaryOp::Gt,
        kyto_core::Literal::Int(0),
    )
    .unwrap();
    // Before `log1p`: the filter copies each block of `counts` (its mean entries and
    // `indptr`) into a buffer from the pool and gives the joined one back; with the
    // `log1p` block that is more than reading the tiles.
    let plan = scan()
        .filter_cells(&predicate)
        .unwrap()
        .log1p(Some("counts"), "log1p")
        .unwrap();
    let copy = 819_200 * 8 + 4_097 * 8;
    let per_block = HELD_BLOCK + 2 * copy;
    assert!(per_block > SCAN_BLOCK);
    // The first window's two blocks at once: more than the last block's work with the
    // consumer's block of both layers. A block holds at most 2 buffers at once (the
    // joined block and its copy, then the copy and `log1p`, which takes the joined
    // block's buffers back, task 5.16): the first window allocated 4, the consumer
    // holds 2 and the last block takes the other 2 back, its read too (task 5.18).
    assert!(2 * per_block > per_block + 2 * HELD_BLOCK);
    assert_eq!(estimate_all(&plan).block_memory(), 2 * per_block);
    // A second filter copies into the buffers the first gave back: no more memory.
    let gene_filter = GeneExpr::column("gene_id")
        .str(kyto_core::StrFunction::StartsWith("MT-".to_owned()))
        .unwrap();
    let twice = scan()
        .filter_cells(&predicate)
        .unwrap()
        .filter_genes(&gene_filter)
        .unwrap()
        .log1p(Some("counts"), "log1p")
        .unwrap();
    assert_eq!(
        estimate_all(&twice).block_memory(),
        estimate_all(&plan).block_memory()
    );
    // After `log1p` a filter copies both layers: a block holds 4 buffers at once, the
    // first window allocated 8. While the last block is read (task 5.18: it takes 1
    // back, at its own 1 808 rows, task 5.21), the consumer holds 2 and the pool keeps
    // 5, more than while it computes.
    let after = plan.filter_genes(&gene_filter).unwrap();
    assert!(LAST_READ + 5 * HELD_BLOCK > (per_block + copy) + 2 * HELD_BLOCK);
    assert_eq!(
        estimate_all(&after).block_memory(),
        LAST_READ + 2 * HELD_BLOCK + 5 * HELD_BLOCK
    );
}
