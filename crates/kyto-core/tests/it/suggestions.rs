//! Task 8.2: every unknown column, gene, symbol, layer, embedding and `feature(...)`
//! name gets up to three close matches of its kind (Jaro-Winkler, best first) in its
//! `KY-E210` or `KY-E109` message, and none when nothing is close.
#![allow(clippy::unwrap_used)]

use kyto_core::{AtlasSchema, CellExpr, GeneExpr, GeneIds, RemovedGenes, Resolver};
use kyto_error::KytoError;

const SCHEMA: &str = r#"{
  "n_cells": 4,
  "n_genes": 7,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "n_genes", "dtype": "Int32"},
    {"name": "cluster", "dtype": "Enum", "n_categories": 3, "ordered": false}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "highly_variable", "dtype": "Boolean"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 6, "ordered": false}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 10},
    {"name": "log1p", "dtype": "float32", "nnz": 10}
  ],
  "embeddings": [
    {"name": "loadings", "axis": "genes", "width": 3, "dtype": "float32"},
    {"name": "pca", "axis": "cells", "width": 3, "dtype": "float32"}
  ],
  "graphs": []
}"#;

/// Genes in table order: `CD34` first (as on pbmc3k, chromosome 1), then the CD3
/// complex, a gene without a symbol and one whose ID is a plain name.
fn genes() -> GeneIds {
    let ids = [
        "ENSG01", "ENSG02", "ENSG03", "ENSG04", "ENSG05", "CD8A", "ENSG07",
    ];
    let symbols = [
        Some("CD34"),
        Some("CD3E"),
        Some("CD3D"),
        Some("CD3G"),
        None,
        Some("CD8A"),
        Some("MS4A1"),
    ];
    GeneIds::new(ids.map(str::to_owned)).with_symbols(symbols.map(|s| s.map(str::to_owned)))
}

fn error_of(expr: &CellExpr, removed: Option<&RemovedGenes>) -> KytoError {
    let schema: AtlasSchema = serde_json::from_str(SCHEMA).unwrap();
    let genes = genes();
    let mut resolver = Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto");
    if let Some(removed) = removed {
        resolver = resolver.with_removed_genes(removed);
    }
    resolver.resolve(expr).unwrap_err()
}

/// The code and suggestions of an error.
fn suggestions(error: &KytoError) -> (&'static str, Vec<String>) {
    let found = match error {
        KytoError::UnknownName { suggestions, .. }
        | KytoError::MissingMember { suggestions, .. } => suggestions.to_vec(),
        other => panic!("expected KY-E210 or KY-E109, got {other}"),
    };
    (error.code(), found)
}

fn of(expr: &CellExpr) -> (&'static str, Vec<String>) {
    suggestions(&error_of(expr, None))
}

fn feature(name: &str) -> (&'static str, Vec<String>) {
    of(&CellExpr::feature(name).unwrap())
}

#[test]
fn a_short_name_gets_columns_gene_ids_and_labelled_symbols() {
    let error = error_of(&CellExpr::feature("CD3").unwrap(), None);
    let symbols = ["CD3E", "CD3D", "CD3G"].map(|symbol| format!("{symbol} (gene symbol)"));
    assert_eq!(suggestions(&error), ("KY-E210", symbols.to_vec()));
    assert_eq!(
        error.to_string(),
        "KY-E210: unknown feature 'CD3' in feature('CD3'): it is not a column of the \
         cell table or a gene ID; did you mean 'CD3E (gene symbol)', 'CD3D (gene symbol)', \
         'CD3G (gene symbol)'?"
    );
    assert!(error.hint().contains("by='symbol'"), "{}", error.hint());
    assert_eq!(feature("clustr"), ("KY-E210", vec!["cluster".to_owned()]));
    // A symbol that is also a gene ID is offered once, as the ID.
    assert_eq!(feature("cd8a"), ("KY-E210", vec!["CD8A".to_owned()]));
}

#[test]
fn a_full_name_gets_names_under_its_prefix_or_the_closest_prefixes() {
    assert_eq!(
        feature("meta:clustr"),
        ("KY-E210", vec!["meta:cluster".to_owned()])
    );
    assert_eq!(
        feature("log1p:CD8"),
        ("KY-E210", vec!["log1p:CD8A".to_owned()])
    );
    assert_eq!(
        feature("logp1:CD8A"),
        ("KY-E210", vec!["log1p:CD8A".to_owned()])
    );
    assert_eq!(
        feature("mtea:cluster"),
        ("KY-E210", vec!["meta:cluster".to_owned()])
    );
    // An embedding component out of range has nothing to suggest.
    assert_eq!(feature("pca:7"), ("KY-E210", Vec::new()));
}

#[test]
fn columns_genes_and_symbols_get_their_own_kind() {
    assert_eq!(
        of(&CellExpr::column("n_gene")),
        ("KY-E210", vec!["n_genes".to_owned()])
    );
    assert_eq!(
        of(&CellExpr::expression(None, "cd8a")),
        ("KY-E210", vec!["CD8A".to_owned()])
    );
    assert_eq!(
        of(&CellExpr::expression_by_symbol(None, "CD3")),
        (
            "KY-E210",
            ["CD3E", "CD3D", "CD3G"].map(str::to_owned).to_vec()
        )
    );
    let schema: AtlasSchema = serde_json::from_str(SCHEMA).unwrap();
    let genes = genes();
    let error = Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto")
        .resolve(&GeneExpr::column("highly_variabel"))
        .unwrap_err();
    assert_eq!(
        suggestions(&error),
        ("KY-E210", vec!["highly_variable".to_owned()])
    );
}

#[test]
fn layers_and_embeddings_get_members_of_the_store() {
    let error = error_of(
        &CellExpr::expression(Some("count".to_owned()), "CD8A"),
        None,
    );
    assert_eq!(suggestions(&error), ("KY-E109", vec!["counts".to_owned()]));
    assert_eq!(
        error.to_string(),
        "KY-E109: no layer named 'count' in the store at 'pbmc.kyto'; did you mean 'counts'?"
    );
    assert_eq!(
        of(&CellExpr::embedding("pcaa", 0)),
        ("KY-E109", vec!["pca".to_owned()])
    );
}

#[test]
fn genes_removed_upstream_are_not_suggested() {
    let mut removed = RemovedGenes::new();
    removed.insert(["ENSG02".to_owned()], "select_genes");
    let error = error_of(&CellExpr::expression_by_symbol(None, "CD3"), Some(&removed));
    assert_eq!(
        suggestions(&error),
        // CD34 (a different number, so less close) takes the free place.
        (
            "KY-E210",
            ["CD3D", "CD3G", "CD34"].map(str::to_owned).to_vec()
        )
    );
}

#[test]
fn nothing_close_suggests_nothing_and_the_message_says_nothing() {
    let error = error_of(&CellExpr::feature("zzz").unwrap(), None);
    assert_eq!(suggestions(&error), ("KY-E210", Vec::new()));
    assert!(!error.to_string().contains("did you mean"), "{error}");
}
