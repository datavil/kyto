//! Task 3.9: gene identity. Genes are indexed by ID; `gene_expression(symbol,
//! by='symbol')` resolves to the ID of the one gene with that symbol when the plan is
//! built (R6), `KY-E204` listing the IDs when several genes carry it, `KY-E210` when
//! none does or the store has no usable `symbol` column.
#![allow(clippy::unwrap_used)]

use kyto_core::{AtlasSchema, BinaryOp, CellExpr, GeneIds, Literal, RemovedGenes, Resolver};
use kyto_error::KytoError;

/// A gene table with a `symbol` column (`symbol_dtype`) and two layers.
fn schema(symbol_dtype: Option<&str>) -> AtlasSchema {
    let symbol = symbol_dtype.map_or_else(String::new, |dtype| {
        format!(r#", {{"name": "symbol", {dtype}}}"#)
    });
    let text = format!(
        r#"{{
  "n_cells": 4,
  "n_genes": 5,
  "tiling": {{"cell_block_size": 4096, "gene_block_size": 1024}},
  "cells": {{"columns": [{{"name": "cell_id", "dtype": "String"}}]}},
  "genes": {{"columns": [{{"name": "gene_id", "dtype": "String"}}{symbol}]}},
  "layers": [
    {{"name": "counts", "dtype": "uint32", "nnz": 10}},
    {{"name": "log1p", "dtype": "float32", "nnz": 10}}
  ],
  "embeddings": [],
  "graphs": []
}}"#
    );
    serde_json::from_str(&text).unwrap()
}

const ENUM: &str = r#""dtype": "Enum", "n_categories": 3, "ordered": false"#;

/// Five genes: `TBCE` twice (as in 10x GRCh38), `CD3E` once, one gene without a
/// symbol, and a gene whose ID is another gene's symbol.
fn genes() -> GeneIds {
    let ids = ["ENSG01", "ENSG02", "ENSG03", "ENSG04", "CD3E"].map(str::to_owned);
    let symbols = [
        Some("TBCE"),
        Some("CD3E"),
        Some("TBCE"),
        None,
        Some("OTHER"),
    ]
    .map(|symbol| symbol.map(str::to_owned));
    GeneIds::new(ids).with_symbols(symbols)
}

fn resolve(expr: &CellExpr, genes: &GeneIds, symbol: Option<&str>) -> Result<String, KytoError> {
    let schema = schema(symbol);
    Resolver::new(&schema, genes, Some("log1p"), "pbmc.kyto")
        .resolve(expr)
        .map(|resolved| resolved.to_string())
}

fn by_symbol(symbol: &str) -> CellExpr {
    CellExpr::expression_by_symbol(None, symbol)
}

#[test]
fn a_unique_symbol_resolves_to_its_gene_id_once() {
    let resolved = resolve(&by_symbol("CD3E"), &genes(), Some(ENUM)).unwrap();
    // The symbol `CD3E` is gene ENSG02, not the gene whose ID is `CD3E`.
    assert_eq!(resolved, "gene_expression('ENSG02', layer='log1p')");
    let layered = CellExpr::expression_by_symbol(Some("counts".to_owned()), "CD3E");
    assert_eq!(
        layered.to_string(),
        "gene_expression('CD3E', layer='counts', by='symbol')"
    );
    assert_eq!(
        resolve(&layered, &genes(), Some(ENUM)).unwrap(),
        "gene_expression('ENSG02', layer='counts')"
    );
    // By ID, `CD3E` is the gene with that ID.
    let by_id = CellExpr::expression(None, "CD3E");
    assert_eq!(
        resolve(&by_id, &genes(), Some(ENUM)).unwrap(),
        "gene_expression('CD3E', layer='log1p')"
    );
}

#[test]
fn a_resolved_symbol_is_idempotent_and_inside_operations() {
    let expr = CellExpr::binary(by_symbol("CD3E"), BinaryOp::Gt, Literal::Int(1)).unwrap();
    assert_eq!(expr.to_string(), "gene_expression('CD3E', by='symbol') > 1");
    let schema = schema(Some(ENUM));
    let genes = genes();
    let resolver = Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto");
    let once = resolver.resolve(&expr).unwrap();
    assert_eq!(
        once.to_string(),
        "gene_expression('ENSG02', layer='log1p') > 1"
    );
    assert_eq!(resolver.resolve(&once).unwrap(), once);
}

#[test]
fn a_duplicate_symbol_is_ky_e204_listing_the_gene_ids_in_table_order() {
    let error = resolve(&by_symbol("TBCE"), &genes(), Some(ENUM)).unwrap_err();
    assert_eq!(error.code(), "KY-E204");
    assert_eq!(
        error,
        KytoError::AmbiguousSymbol {
            symbol: "TBCE".to_owned(),
            gene_ids: vec!["ENSG01".to_owned(), "ENSG03".to_owned()],
        }
    );
    let text = error.to_string();
    assert!(text.contains("symbol 'TBCE' matches more than one gene: 'ENSG01', 'ENSG03'"));
}

fn assert_unknown_symbol(error: &KytoError, symbol: &str, problem: &str) {
    let KytoError::UnknownName {
        kind,
        name,
        problem: got,
        ..
    } = error
    else {
        panic!("expected KY-E210, got {error}");
    };
    assert_eq!(kind, "gene symbol");
    assert_eq!(name, symbol);
    assert!(got.contains(problem), "{got}");
}

#[test]
fn an_unknown_symbol_is_ky_e210_and_matching_is_exact() {
    for symbol in ["cd3e", "CD3", "", "ENSG02", " CD3E"] {
        let error = resolve(&by_symbol(symbol), &genes(), Some(ENUM)).unwrap_err();
        assert_unknown_symbol(&error, symbol, "no gene of the store has this symbol");
    }
}

#[test]
fn a_store_without_symbols_says_why() {
    let no_symbols = GeneIds::new(["ENSG01", "CD3E"].map(str::to_owned));
    let error = resolve(&by_symbol("CD3E"), &no_symbols, None).unwrap_err();
    assert_unknown_symbol(&error, "CD3E", "the gene table has no 'symbol' column");
    let error = resolve(&by_symbol("CD3E"), &no_symbols, Some(r#""dtype": "Int64""#)).unwrap_err();
    assert_unknown_symbol(
        &error,
        "CD3E",
        "'symbol' column is Int64, not a string (Enum) column",
    );
}

#[test]
fn a_removed_gene_named_by_symbol_is_ky_e203() {
    let schema = schema(Some(ENUM));
    let genes = genes();
    let mut removed = RemovedGenes::new();
    removed.insert(["ENSG02".to_owned()], "select_genes");
    let error = Resolver::new(&schema, &genes, Some("log1p"), "pbmc.kyto")
        .with_removed_genes(&removed)
        .resolve(&by_symbol("CD3E"))
        .unwrap_err();
    assert_eq!(error.code(), "KY-E203");
}

#[test]
fn without_a_default_layer_a_symbol_read_is_ky_e211() {
    let schema = schema(Some(ENUM));
    let genes = genes();
    let error = Resolver::new(&schema, &genes, None, "pbmc.kyto")
        .resolve(&by_symbol("CD3E"))
        .unwrap_err();
    assert_eq!(error.code(), "KY-E211");
}

#[test]
fn symbols_are_held_per_gene_in_table_order() {
    let genes = genes();
    assert!(genes.has_symbols());
    assert_eq!(genes.with_symbol("TBCE"), ["ENSG01", "ENSG03"]);
    assert!(genes.with_symbol("missing").is_empty());
    assert_eq!(format!("{genes:?}"), "GeneIds(5 genes, 3 symbols)");
    // Too few symbols: the rest have none; too many: the extra ones are ignored.
    let short = GeneIds::new(["A", "B"].map(str::to_owned)).with_symbols([Some("X".to_owned())]);
    assert_eq!(short.with_symbol("X"), ["A"]);
    let long = GeneIds::new(["A"].map(str::to_owned))
        .with_symbols([Some("X".to_owned()), Some("X".to_owned())]);
    assert_eq!(long.with_symbol("X"), ["A"]);
    assert!(!GeneIds::new(["A".to_owned()]).has_symbols());
}
