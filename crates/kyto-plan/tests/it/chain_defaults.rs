//! Task 7.10: defaults that follow the chain. A matrix step without `layer` reads what
//! the latest step wrote (or the nearer default layer), a step without `into` names its
//! output after itself, and a default name the plan already has is `KY-E220`.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kyto_core::{AtlasSchema, GeneIds};
use kyto_plan::{
    ChainStep, Cluster, ClusterMethod, HvgFlavor, Metric, NeighborSearch, Neighbors, Plan,
};

const SCHEMA: &str = r#"{
  "n_cells": 20,
  "n_genes": 10,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [{"name": "cell_id", "dtype": "String"}]},
  "genes": {"columns": [{"name": "gene_id", "dtype": "String"}]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 10},
    {"name": "Log1p", "dtype": "float32", "nnz": 10}
  ],
  "embeddings": [],
  "graphs": []
}"#;

fn scan() -> Arc<Plan> {
    Plan::scan(
        "pbmc.kyto",
        serde_json::from_str::<AtlasSchema>(SCHEMA).unwrap(),
        GeneIds::default(),
    )
}

fn bare(plan: &Arc<Plan>, step: ChainStep) -> String {
    plan.into_or_default(step, None).unwrap()
}

#[test]
fn a_bare_step_reads_what_the_step_before_it_wrote() {
    let counts = scan().with_default_layer("counts").unwrap();
    assert_eq!(counts.chain_layer(), Some("counts"));
    let normalized = counts
        .normalize_total(Some(1e4), None, &bare(&counts, ChainStep::NormalizeTotal))
        .unwrap();
    assert_eq!(normalized.chain_layer(), Some("normalized"));
    // The default layer, which short names in expressions read, does not move.
    assert_eq!(normalized.default_layer(), Some("counts"));
    let hvg = normalized
        .highly_variable_genes(None, HvgFlavor::Seurat, Some(2), None)
        .unwrap();
    assert_eq!(hvg.chain_layer(), Some("normalized"));
    let pca = hvg
        .pca(None, &bare(&hvg, ChainStep::Pca), 2, 0, None)
        .unwrap();
    assert_eq!(
        pca.to_string().lines().next().unwrap(),
        r#"PCA "normalized" -> "X_pca" n_components=2 seed=0"#
    );
    // A nearer `with_default_layer` wins.
    let reset = normalized.with_default_layer("counts").unwrap();
    assert_eq!(reset.chain_layer(), Some("counts"));
    // An explicit layer is read as given.
    let explicit = normalized
        .scale(Some("counts"), "scaled", true, None)
        .unwrap();
    assert!(
        explicit
            .to_string()
            .starts_with(r#"SCALE "counts" -> "scaled""#)
    );
    assert_eq!(explicit.chain_layer(), Some("scaled"));
}

#[test]
fn a_writing_step_starts_the_chain_without_a_default_layer() {
    let plan = scan();
    assert_eq!(plan.chain_layer(), None);
    let error = plan.log1p(None, "x").unwrap_err();
    assert_eq!(error.code(), "KY-E211");
    let written = plan.log1p(Some("counts"), "x").unwrap();
    assert_eq!(written.chain_layer(), Some("x"));
    assert!(
        written
            .scale(None, "y", true, None)
            .unwrap()
            .to_string()
            .starts_with(r#"SCALE "x" -> "y""#)
    );
}

#[test]
fn default_names_are_the_steps_and_a_clash_is_ky_e220() {
    let plan = scan().with_default_layer("counts").unwrap();
    assert_eq!(bare(&plan, ChainStep::NormalizeTotal), "normalized");
    assert_eq!(bare(&plan, ChainStep::Scale), "scaled");
    assert_eq!(bare(&plan, ChainStep::Umap), "X_umap");
    assert_eq!(bare(&plan, ChainStep::Leiden), "leiden");
    assert_eq!(bare(&plan, ChainStep::Louvain), "louvain");
    // An explicit name passes through unchecked: the step checks it, and replaces.
    assert_eq!(
        plan.into_or_default(ChainStep::Log1p, Some("log1p"))
            .unwrap(),
        "log1p"
    );
    // The store's "Log1p" clashes with "log1p" ignoring case.
    let error = plan.into_or_default(ChainStep::Log1p, None).unwrap_err();
    assert_eq!(error.code(), "KY-E220");
    let message = error.to_string();
    assert!(
        message.contains("log1p() without into= writes the layer 'log1p'"),
        "{message}"
    );
    assert!(message.contains("'Log1p'"), "{message}");
    assert!(
        error.hint().starts_with("pass into= with a new name"),
        "{}",
        error.hint()
    );

    let pca = plan.pca(None, "X_pca", 2, 0, None).unwrap();
    assert_eq!(
        pca.into_or_default(ChainStep::Pca, None)
            .unwrap_err()
            .code(),
        "KY-E220"
    );
    // The loadings clash too.
    let loadings = plan.pca(None, "X", 2, 0, None).unwrap();
    let error = loadings.into_or_default(ChainStep::Pca, None);
    assert!(error.is_ok(), "X_loadings is not X_pca_loadings");
    let graph = pca
        .neighbors(Neighbors {
            embedding: pca.embedding_or_default("neighbors", None).unwrap(),
            n_components: None,
            n_neighbors: 3,
            metric: Metric::Euclidean,
            search: NeighborSearch::Exact,
            seed: 0,
            key: None,
        })
        .unwrap();
    let clustered = graph
        .cluster(Cluster {
            method: ClusterMethod::Leiden,
            graph: "connectivities".to_owned(),
            resolution: 1.0,
            seed: 0,
            into: bare(&graph, ChainStep::Leiden),
        })
        .unwrap();
    let error = clustered
        .into_or_default(ChainStep::Leiden, None)
        .unwrap_err();
    assert_eq!(error.code(), "KY-E220");
    assert!(error.to_string().contains("cell column 'leiden'"));
}

#[test]
fn neighbors_searches_the_latest_pca() {
    let plan = scan().with_default_layer("counts").unwrap();
    let error = plan.embedding_or_default("neighbors", None).unwrap_err();
    assert_eq!(error.code(), "KY-E215");
    let first = plan.pca(None, "first", 2, 0, None).unwrap();
    let second = first.pca(None, "second", 2, 0, None).unwrap();
    assert_eq!(first.chain_embedding(), Some("first"));
    assert_eq!(second.chain_embedding(), Some("second"));
    assert_eq!(
        second
            .embedding_or_default("neighbors", Some("first"))
            .unwrap(),
        "first"
    );
}
