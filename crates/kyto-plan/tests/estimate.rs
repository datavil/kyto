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

use kyto_core::{AtlasSchema, CellExpr, GeneIds};
use kyto_plan::{
    ASSUMED_NNZ_CHUNK_SIZE, ASSUMED_NNZ_SHARD_SIZE, CostEstimate, Device, HvgFlavor, Member,
    Metric, NeighborSearch, Neighbors, OptimizeOptions, Output, PCA_POWER_ITERATIONS,
    PhysicalOptions, Plan, Sink, physical_plan,
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

/// Bytes one pass over `counts` reads: entries, the `indptr` of 6 tiles, `tile_offsets`.
const PASS_BYTES: u64 = 2_000_000 * 8 + 6 * 4_097 * 4 + 7 * 8;

/// The scan's block, per block in flight: a full block holds 819 200 entries on average
/// (2 000 000 x 4 096 / 10 000). Its 2 tiles are `2 x 4 097 x 4 + 819 200 x 8` bytes;
/// joining them adds the row positions (`4 096 x 4`), the `indptr` (`4 097 x 8`) and
/// `indices`/`data` grown to 1 048 576 entries (`x (4 + 1.5 x 4)`), more than reading
/// (tiles + `2 x 2 x 65 536 x 4`).
const SCAN_BLOCK: u64 = (2 * 4_097 * 4 + 819_200 * 8) + 4_096 * 4 + 4_097 * 8 + 1_048_576 * 10;

/// A joined block of `counts` (or one mapped from it) as the consumer holds it and the
/// pool keeps it: `indices` and values at the capacity pushing reaches, and `indptr`.
const HELD_BLOCK: u64 = 1_048_576 * 8 + 4_097 * 8;

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
    // before and the pool keeps the first.
    assert_eq!(with(1).block_memory(), SCAN_BLOCK + 2 * HELD_BLOCK);
    assert_eq!(with(3).block_memory(), 3 * SCAN_BLOCK);
    // Only 3 cell blocks exist.
    assert_eq!(with(8).block_memory(), 3 * SCAN_BLOCK);
    assert_eq!(with(8).bytes_read(), PASS_BYTES);
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
    // 12 blocks (3 cell blocks of 4 parts): the consumer's block and the pool's one.
    assert_eq!(small.block_memory(), joining + 2 * held);
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
    let pca = normalized.pca(Some("log1p"), "X_pca", 50, 0).unwrap();
    let pca_passes = 2 + 2 * PCA_POWER_ITERATIONS;
    // Median: 1 statistics pass; PCA's; then one pass delivers the layers.
    let all = estimate_all(&pca);
    assert_eq!(all.passes(), 1 + pca_passes + 1);
    assert_eq!(all.bytes_read(), u64::from(all.passes()) * PASS_BYTES);
    // Reading only the embedding needs no final pass.
    let embedding =
        estimate(&[Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap()]);
    assert_eq!(embedding.passes(), 1 + pca_passes);
    // The same without fusion: the passes do not depend on the rules.
    let unfused = physical_plan(
        &[Output::all(&pca)],
        OptimizeOptions::NONE,
        PhysicalOptions::default(),
    )
    .estimate();
    assert_eq!(unfused.passes(), all.passes());
}

#[test]
fn a_shared_sub_plan_makes_its_passes_once() {
    let normalized = scan()
        .normalize_total(None, Some("counts"), "normalized")
        .unwrap();
    let hvg = normalized
        .highly_variable_genes(Some("normalized"), HvgFlavor::SeuratV3, Some(100), None)
        .unwrap();
    let pca = normalized.pca(Some("normalized"), "X_pca", 10, 0).unwrap();
    let outputs = [
        Output::members(&hvg, [Member::GeneColumn("highly_variable".into())]).unwrap(),
        Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap(),
    ];
    // Median 1 (once), seurat_v3 2, PCA 16; nothing read from the stream.
    assert_eq!(
        estimate(&outputs).passes(),
        1 + 2 + 2 + 2 * PCA_POWER_ITERATIONS
    );
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
    // The scan reads no layer: no blocks, no `tile_offsets`; the column is held.
    assert_eq!(estimate.block_memory(), 0);
    assert_eq!(estimate.resident_memory(), N_CELLS * 4);
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
    // block is read the consumer holds a block of both layers and the pool keeps the
    // buffers of the two layers of the block before.
    const { assert!(HELD_BLOCK + 819_200 * 8 + 4_097 * 8 < SCAN_BLOCK) };
    assert_eq!(log1p, SCAN_BLOCK + 4 * HELD_BLOCK);
    // Centred `scale` is dense: every gene of every row, with the joined block.
    let scaled = estimate_all(&scan().scale(Some("counts"), "scaled", true, None).unwrap());
    let dense = 4_096 * N_GENES * 8 + 4_097 * 8;
    assert!(HELD_BLOCK + dense > SCAN_BLOCK);
    // While the last block is computed: one block's work, the consumer's block of both
    // layers (the dense one at its capacity, 2^23 entries, is `dense`) and the pool's
    // 2 dense buffers.
    assert_eq!(
        scaled.block_memory(),
        (HELD_BLOCK + dense) + (HELD_BLOCK + dense) + 2 * dense
    );
    // Its statistics: a mean and a variance per gene, and while its statistics pass
    // runs, the same per leaf of its reduction (task 5.6): 2 blocks in flight and the
    // tree's subtrees over 3 blocks (log2(3) + 1 = 2).
    assert_eq!(scaled.resident_memory() - 56, N_GENES * 16 * (1 + 2 + 2));
}

#[test]
fn in_memory_members_and_statistics_are_resident() {
    let pca = scan().pca(Some("counts"), "X_pca", 50, 0).unwrap();
    let pca_estimate =
        estimate(&[Output::members(&pca, [Member::Embedding("X_pca".into())]).unwrap()]);
    let embeddings = N_CELLS * 50 * 4 + N_GENES * 50 * 4;
    let halko = (2 * N_CELLS + N_GENES) * (50 + 10) * 8;
    assert_eq!(pca_estimate.resident_memory(), 56 + embeddings + halko);

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
        estimate(&[Output::members(&neighbors, [Member::Graph("connectivities".into())]).unwrap()]);
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
fn assumed_chunk_sizes_are_the_writer_defaults() {
    assert_eq!(ASSUMED_NNZ_CHUNK_SIZE, kyto_store::DEFAULT_NNZ_CHUNK_SIZE);
    assert_eq!(ASSUMED_NNZ_SHARD_SIZE, kyto_store::DEFAULT_NNZ_SHARD_SIZE);
}
