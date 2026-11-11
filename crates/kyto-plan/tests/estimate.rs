//! Task 4.6: the cost estimate of plans, from the schema alone.
//!
//! A store of 10 000 cells x 2 048 genes (tiles of 4 096 x 1 024: 3 cell blocks, 2 gene
//! blocks), one `uint32` layer of 2 000 000 entries. Checks the scan's bytes and memory
//! by hand, the passes over the store (statistics passes re-stream from the scan, a
//! shared sub-plan once, a final pass only for what is read from the stream), what
//! nodes keep in memory, blocks in flight, saturation and the text form. The fixture
//! check against measured memory is `estimate_fixtures.rs`.
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
    assert_eq!(with(1).block_memory(), SCAN_BLOCK);
    assert_eq!(with(3).block_memory(), 3 * SCAN_BLOCK);
    // Only 3 cell blocks exist.
    assert_eq!(with(8).block_memory(), 3 * SCAN_BLOCK);
    assert_eq!(with(8).bytes_read(), PASS_BYTES);
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
    let log1p = estimate_all(&scan().log1p(Some("counts"), "log1p").unwrap()).block_memory();
    // `float32` values and `u32` indices of a mean block, and its `indptr`, x 2 in flight.
    assert_eq!(log1p - base, 2 * (819_200 * 8 + 4_097 * 8));
    // Centred `scale` is dense: every gene of every row.
    let scaled = estimate_all(&scan().scale(Some("counts"), "scaled", true, None).unwrap());
    assert_eq!(
        scaled.block_memory() - base,
        2 * (4_096 * N_GENES * 8 + 4_097 * 8)
    );
    // Its statistics: a mean and a variance per gene.
    assert_eq!(scaled.resident_memory() - 56, N_GENES * 16);
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
    // Positions widened to 8 bytes, per block in flight.
    assert_eq!(estimate_all(&h5ad).block_memory() - base, 2 * 819_200 * 8);
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
