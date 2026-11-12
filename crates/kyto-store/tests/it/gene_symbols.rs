//! Task 3.9: `read_symbols` reads only the gene `symbol` column (metadata, R1), one
//! symbol per gene in table order with nulls kept; `read_ids` refuses duplicate IDs
//! (the 3.5 open item: they would collapse silently in `GeneIds`).
#![allow(clippy::unwrap_used)]

use std::path::Path;

use kyto_store::{TableAxis, Tiling, read_ids, read_symbols, write_table};
use polars::prelude::{Column, DataFrame, IntoColumn, NamedFrom, ParquetWriter, Series};

const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 2,
};

fn genes(symbols: Column) -> DataFrame {
    let ids = Series::new("gene_id".into(), &["G1", "G2", "G3", "G4"]).into_column();
    DataFrame::new_infer_height(vec![ids, symbols]).unwrap()
}

fn write_foreign(store: &Path, mut frame: DataFrame) {
    let file = std::fs::File::create(store.join(TableAxis::Genes.file_name())).unwrap();
    ParquetWriter::new(file).finish(&mut frame).unwrap();
}

#[test]
fn symbols_are_read_in_table_order_with_nulls_and_repeats() {
    let store = tempfile::tempdir().unwrap();
    let symbols = Series::new(
        "symbol".into(),
        &[Some("TBCE"), None, Some("TBCE"), Some("CD3E")],
    )
    .into_column();
    // write_table stores the String column as an Enum (format.md §5.4).
    write_table(store.path(), TableAxis::Genes, &genes(symbols), TILING).unwrap();
    let read = read_symbols(store.path(), 4).unwrap();
    assert_eq!(
        read,
        [Some("TBCE"), None, Some("TBCE"), Some("CD3E")].map(|symbol| symbol.map(str::to_owned))
    );
}

#[test]
fn a_bad_symbol_column_is_ky_e113_or_e110() {
    let store = tempfile::tempdir().unwrap();
    let symbols = Series::new("symbol".into(), &["A", "B", "C", "D"]).into_column();
    write_table(store.path(), TableAxis::Genes, &genes(symbols), TILING).unwrap();
    let error = read_symbols(store.path(), 5).unwrap_err();
    assert_eq!(error.code(), "KY-E113");
    assert!(
        error
            .to_string()
            .contains("it has 4 rows, but schema.json says 5")
    );

    let store = tempfile::tempdir().unwrap();
    let numbers = Series::new("symbol".into(), &[1_i64, 2, 3, 4]).into_column();
    write_foreign(store.path(), genes(numbers));
    let error = read_symbols(store.path(), 4).unwrap_err();
    assert_eq!(error.code(), "KY-E113");
    assert!(
        error
            .to_string()
            .contains("the column 'symbol' is i64, not Enum"),
        "{error}"
    );

    let store = tempfile::tempdir().unwrap();
    let other = Series::new("name".into(), &["A", "B", "C", "D"]).into_column();
    write_foreign(store.path(), genes(other));
    assert_eq!(read_symbols(store.path(), 4).unwrap_err().code(), "KY-E110");
}

#[test]
fn duplicate_gene_ids_are_ky_e113() {
    let store = tempfile::tempdir().unwrap();
    let ids = Series::new("gene_id".into(), &["G1", "G2", "G1"]).into_column();
    write_foreign(
        store.path(),
        DataFrame::new_infer_height(vec![ids]).unwrap(),
    );
    let error = read_ids(store.path(), TableAxis::Genes, 3).unwrap_err();
    assert_eq!(error.code(), "KY-E113");
    assert!(
        error
            .to_string()
            .contains("the ID column 'gene_id' holds 'G1' more than once"),
        "{error}"
    );
}
