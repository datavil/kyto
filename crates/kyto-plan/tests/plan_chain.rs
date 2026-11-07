//! Task 3.2 Done when: the original plan is unchanged after 10 chained calls (R2).
//! Also: the chain shares its inputs, validates layers, and long chains are safe.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{AtlasSchema, GeneIds};
use kyto_plan::Plan;

const SCHEMA: &str = r#"{
  "n_cells": 2700,
  "n_genes": 32738,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [{"name": "cell_id", "dtype": "String"}]},
  "genes": {"columns": [{"name": "gene_id", "dtype": "String"}]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 10},
    {"name": "log1p", "dtype": "float32", "nnz": 10}
  ],
  "embeddings": [],
  "graphs": []
}"#;

fn schema() -> AtlasSchema {
    serde_json::from_str(SCHEMA).unwrap()
}

#[test]
fn original_plan_unchanged_after_ten_chained_calls() {
    let original = Plan::scan("pbmc.kyto", schema(), GeneIds::default());
    let snapshot = original.to_string();
    let reference = Plan::scan("pbmc.kyto", schema(), GeneIds::default());

    let mut plans = vec![Arc::clone(&original)];
    for step in 0..10 {
        let layer = if step % 2 == 0 { "log1p" } else { "counts" };
        let next = plans.last().unwrap().with_default_layer(layer).unwrap();
        plans.push(next);
    }

    // The original: same node, same text, equal to a fresh scan, no default layer.
    assert_eq!(original.to_string(), snapshot);
    assert_eq!(*original, *reference);
    assert_eq!(original.default_layer(), None);
    assert_eq!(original.nodes().count(), 1);
    // Every plan in the chain is unchanged too, and each shares its input.
    for (depth, plan) in plans.iter().enumerate() {
        assert_eq!(plan.nodes().count(), depth + 1);
        if let Some(input) = plan.input() {
            assert!(Arc::ptr_eq(input, &plans[depth - 1]));
        }
    }
    let last = plans.last().unwrap();
    assert_eq!(last.default_layer(), Some("counts"));
    assert_eq!(plans[9].default_layer(), Some("log1p"));
    assert_eq!(last.schema(), original.schema());
    assert_eq!(last.root().source(), "pbmc.kyto");
    // One strong reference from `original`, one from `plans[0]`, one from `plans[1]`.
    assert_eq!(Arc::strong_count(&original), 3);
}

#[test]
fn branches_share_the_input_and_do_not_affect_each_other() {
    let base = Plan::scan("pbmc.kyto", schema(), GeneIds::default());
    let left = base.with_default_layer("counts").unwrap();
    let right = base.with_default_layer("log1p").unwrap();
    assert_eq!(left.default_layer(), Some("counts"));
    assert_eq!(right.default_layer(), Some("log1p"));
    assert!(Arc::ptr_eq(left.input().unwrap(), right.input().unwrap()));
    assert_ne!(*left, *right);
    assert_eq!(*left, *base.with_default_layer("counts").unwrap());
    assert_ne!(*left, *base);
}

#[test]
fn display_lists_nodes_from_top_to_scan() {
    let plan = Plan::scan("pbmc.kyto", schema(), GeneIds::default())
        .with_default_layer("log1p")
        .unwrap();
    assert_eq!(
        plan.to_string(),
        "WITH_DEFAULT_LAYER \"log1p\"\nSCAN \"pbmc.kyto\" (2700 cells x 32738 genes)"
    );
    assert_eq!(plan.name(), "WITH_DEFAULT_LAYER");
}

#[test]
fn unknown_layer_is_ky_e109_and_names_the_store() {
    let plan = Plan::scan("pbmc.kyto", schema(), GeneIds::default());
    let error = plan.with_default_layer("Counts").unwrap_err();
    assert_eq!(error.code(), "KY-E109");
    let text = error.to_string();
    assert!(text.contains("'Counts'"), "{text}");
    assert!(text.contains("layer"), "{text}");
    assert!(text.contains("pbmc.kyto"), "{text}");
}

#[test]
fn long_chains_drop_compare_and_print_without_recursion() {
    // Run on a small stack: a recursive drop or comparison would overflow it.
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut left = Plan::scan("pbmc.kyto", schema(), GeneIds::default());
            let mut right = Plan::scan("pbmc.kyto", schema(), GeneIds::default());
            for _ in 0..100_000 {
                left = left.with_default_layer("counts").unwrap();
                right = right.with_default_layer("counts").unwrap();
            }
            assert_eq!(*left, *right);
            assert_eq!(left.nodes().count(), 100_001);
            assert!(left.to_string().lines().count() == 100_001);
            drop(left);
            drop(right);
        })
        .unwrap()
        .join()
        .unwrap();
}
