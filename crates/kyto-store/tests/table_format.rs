//! Task 2.3: a cell or gene table follows format.md §5. Writing duplicate IDs raises
//! `KY-E101` (the Done-when check); null and empty IDs, bad columns and bad stored
//! files raise coded errors (R10); the Parquet file uses the settings of §5.7 and
//! rewrites are byte-identical (§10.3).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::path::Path;

use kyto_error::KytoError;
use kyto_store::{TableAxis, Tiling, read_table, write_table};
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, IntoColumn, NamedFrom, ParquetReader,
    ParquetWriter, SerReader, Series,
};

const TILING: Tiling = Tiling {
    cell_block_size: 3,
    gene_block_size: 2,
};

fn strings(name: &str, values: &[Option<&str>]) -> Column {
    Series::new(name.into(), values).into_column()
}

fn ids(name: &str, values: &[&str]) -> Column {
    Series::new(name.into(), values).into_column()
}

fn frame(columns: Vec<Column>) -> DataFrame {
    DataFrame::new_infer_height(columns).unwrap()
}

fn write(axis: TableAxis, frame: &DataFrame) -> (tempfile::TempDir, Result<(), KytoError>) {
    let store = tempfile::tempdir().unwrap();
    let result = write_table(store.path(), axis, frame, TILING);
    (store, result)
}

fn write_error(axis: TableAxis, frame: &DataFrame) -> KytoError {
    let (store, result) = write(axis, frame);
    let error = result.unwrap_err();
    assert!(
        !store.path().join(axis.file_name()).exists(),
        "a rejected table must not create a file"
    );
    error
}

#[test]
fn duplicate_cell_ids_raise_ky_e101() {
    let table = frame(vec![ids(
        "cell_id",
        &[
            "b", "a", "c", "a", "b", "d", "e", "f", "g", "h", "e", "f", "g", "h", "d", "a",
        ],
    )]);
    let error = write_error(TableAxis::Cells, &table);
    assert_eq!(error.code(), "KY-E101");
    assert_eq!(error.python_class(), "DuplicateIdError");
    // Seven IDs repeat; the first five are listed in the order they repeat.
    assert_eq!(
        error.to_string(),
        "KY-E101: the cells table has 7 IDs that occur more than once, first: \
         'a', 'b', 'e', 'f', 'g'"
    );
    assert!(error.hint().contains("on_duplicate_ids='suffix'"));
}

#[test]
fn duplicate_gene_ids_raise_ky_e101() {
    let table = frame(vec![
        ids("gene_id", &["ENSG1", "ENSG2", "ENSG1"]),
        strings("symbol", &[Some("CD3E"), Some("CD3E"), None]),
    ]);
    let error = write_error(TableAxis::Genes, &table);
    assert_eq!(error.code(), "KY-E101");
    assert!(error.to_string().contains("the genes table has 1 IDs"));
    assert!(error.to_string().contains("'ENSG1'"));
}

#[test]
fn ids_compare_exact_bytes() {
    // Case, Unicode normal form and whitespace all make different IDs (format.md §5.2).
    let table = frame(vec![ids(
        "cell_id",
        &["a", "A", "a ", "\u{e9}", "e\u{301}"],
    )]);
    let (store, result) = write(TableAxis::Cells, &table);
    result.unwrap();
    assert_eq!(
        read_table(store.path(), TableAxis::Cells).unwrap().height(),
        5
    );
}

#[test]
fn null_and_empty_ids_raise_ky_e111() {
    let table = frame(vec![strings(
        "cell_id",
        &[Some("a"), None, Some("b"), None],
    )]);
    let error = write_error(TableAxis::Cells, &table);
    assert_eq!(error.code(), "KY-E111");
    assert_eq!(error.python_class(), "StoreInputError");
    assert_eq!(
        error.to_string(),
        "KY-E111: the cells table has 2 null IDs, first at rows '1', '3'"
    );

    let table = frame(vec![ids("gene_id", &["a", "", "b"])]);
    let error = write_error(TableAxis::Genes, &table);
    assert_eq!(error.code(), "KY-E111");
    assert!(error.to_string().contains("1 empty IDs, first at rows '1'"));
}

#[test]
fn id_column_must_be_first_and_string() {
    let missing = frame(vec![ids("barcode", &["a"])]);
    let error = write_error(TableAxis::Cells, &missing);
    assert_eq!(error.code(), "KY-E112");
    assert_eq!(
        error.to_string(),
        "KY-E112: column 'cell_id' of the cells table is missing"
    );

    let second = frame(vec![ids("barcode", &["a"]), ids("cell_id", &["a"])]);
    let error = write_error(TableAxis::Cells, &second);
    assert!(
        error
            .to_string()
            .contains("is column 1, not the first column")
    );

    let numeric = frame(vec![
        Series::new("gene_id".into(), &[1_i64, 2]).into_column(),
    ]);
    let error = write_error(TableAxis::Genes, &numeric);
    assert!(error.to_string().contains("has dtype i64, not String"));

    let error = write_error(TableAxis::Genes, &DataFrame::empty());
    assert!(
        error
            .to_string()
            .contains("'gene_id' of the genes table is missing")
    );
}

#[test]
fn bad_column_names_raise_ky_e112() {
    for (name, problem) in [
        ("log1p:CD3E", "has ':' in its name"),
        ("gene_id", "uses a reserved ID column name"),
        ("", "has an empty name"),
    ] {
        let table = frame(vec![ids("cell_id", &["a"]), ids(name, &["x"])]);
        let error = write_error(TableAxis::Cells, &table);
        assert_eq!(error.code(), "KY-E112", "{name}");
        assert!(error.to_string().contains(problem), "{error}");
    }
}

#[test]
fn dtypes_outside_v0_raise_ky_e112() {
    let categorical = ids("batch", &["x"])
        .cast(&DataType::from_categories(
            polars::prelude::Categories::global(),
        ))
        .unwrap();
    let binary = Series::new("weight".into(), &[1.0_f32])
        .cast(&DataType::Binary)
        .unwrap()
        .into_column();
    for column in [categorical, binary] {
        let table = frame(vec![ids("cell_id", &["a"]), column]);
        let error = write_error(TableAxis::Cells, &table);
        assert_eq!(error.code(), "KY-E112");
        assert!(
            error.to_string().contains("which v0 stores do not allow"),
            "{error}"
        );
        assert!(error.hint().contains("cast Categorical to String or Enum"));
    }
}

#[test]
fn existing_table_raises_ky_e103() {
    let table = frame(vec![ids("cell_id", &["a"])]);
    let (store, result) = write(TableAxis::Cells, &table);
    result.unwrap();
    let error = write_table(store.path(), TableAxis::Cells, &table, TILING).unwrap_err();
    assert_eq!(error.code(), "KY-E103");
    assert!(error.to_string().contains("table 'cells' already exists"));
}

#[test]
fn zero_block_size_raises_ky_e104() {
    let store = tempfile::tempdir().unwrap();
    let table = frame(vec![ids("gene_id", &["a"])]);
    let tiling = Tiling {
        cell_block_size: 4,
        gene_block_size: 0,
    };
    let error = write_table(store.path(), TableAxis::Genes, &table, tiling).unwrap_err();
    assert_eq!(error.code(), "KY-E104");
    assert!(error.to_string().contains("gene_block_size"));
}

#[test]
fn missing_file_raises_ky_e110() {
    let store = tempfile::tempdir().unwrap();
    let error = read_table(store.path(), TableAxis::Genes).unwrap_err();
    assert_eq!(error.code(), "KY-E110");
    assert!(error.to_string().contains("genes.parquet"));
}

/// Write `frame` with plain Polars, as a foreign or broken writer would.
fn write_foreign(store: &Path, axis: TableAxis, mut frame: DataFrame) {
    let file = std::fs::File::create(store.join(axis.file_name())).unwrap();
    ParquetWriter::new(file).finish(&mut frame).unwrap();
}

#[test]
fn stored_tables_that_break_section_5_raise_ky_e113() {
    let cases = [
        (
            frame(vec![ids("cell_id", &["a", "b"]), ids("batch", &["x", "y"])]),
            "column 'batch' of the cells table has dtype String; only the ID column may",
        ),
        (
            frame(vec![ids("cell_id", &["a", "a"])]),
            "the cells table has 1 IDs that occur more than once, first: 'a'",
        ),
        (
            frame(vec![ids("barcode", &["a"])]),
            "column 'cell_id' of the cells table is missing",
        ),
    ];
    for (table, problem) in cases {
        let store = tempfile::tempdir().unwrap();
        write_foreign(store.path(), TableAxis::Cells, table);
        let error = read_table(store.path(), TableAxis::Cells).unwrap_err();
        assert_eq!(error.code(), "KY-E113");
        assert_eq!(error.python_class(), "CorruptStoreError");
        assert!(
            error
                .to_string()
                .ends_with(&format!("is corrupt: {problem}")),
            "{error}"
        );
    }

    let store = tempfile::tempdir().unwrap();
    std::fs::write(store.path().join("cells.parquet"), b"not parquet").unwrap();
    let error = read_table(store.path(), TableAxis::Cells).unwrap_err();
    assert_eq!(error.code(), "KY-E110");
}

fn seven_genes() -> DataFrame {
    frame(vec![
        ids("gene_id", &["g0", "g1", "g2", "g3", "g4", "g5", "g6"]),
        strings(
            "symbol",
            &[
                Some("B"),
                Some("A"),
                None,
                Some("B"),
                Some("C"),
                Some(""),
                Some("A"),
            ],
        ),
        Series::new(
            "highly_variable".into(),
            &[true, false, true, true, false, false, true],
        )
        .into_column(),
    ])
}

#[test]
fn parquet_uses_zstd_statistics_and_block_row_groups() {
    let (store, result) = write(TableAxis::Genes, &seven_genes());
    result.unwrap();
    let file = std::fs::File::open(store.path().join("genes.parquet")).unwrap();
    let mut reader = ParquetReader::new(file);
    let metadata = reader.get_metadata().unwrap().clone();
    // Gene block size 2: row groups of 2, 2, 2 and 1 rows (format.md §5.7).
    let rows: Vec<usize> = metadata
        .row_groups
        .iter()
        .map(|group| group.num_rows())
        .collect();
    assert_eq!(rows, [2, 2, 2, 1]);
    for group in &metadata.row_groups {
        for column in group.parquet_columns() {
            assert_eq!(format!("{:?}", column.compression()), "Zstd");
            let statistics = column.statistics(&metadata.footer_buf);
            assert!(matches!(statistics, Some(Ok(_))), "statistics missing");
        }
    }
}

#[test]
fn rewrites_are_byte_identical() {
    let table = seven_genes();
    let (first, result) = write(TableAxis::Genes, &table);
    result.unwrap();
    let (second, result) = write(TableAxis::Genes, &table);
    result.unwrap();
    let bytes = |store: &Path| std::fs::read(store.join("genes.parquet")).unwrap();
    assert_eq!(bytes(first.path()), bytes(second.path()));
}

#[test]
fn plain_strings_become_enums_of_sorted_categories() {
    let (store, result) = write(TableAxis::Genes, &seven_genes());
    result.unwrap();
    let read = read_table(store.path(), TableAxis::Genes).unwrap();
    let symbol = read.column("symbol").unwrap();
    let DataType::Enum(categories, _) = symbol.dtype() else {
        panic!("symbol is {}, not Enum", symbol.dtype());
    };
    let categories: Vec<&str> = categories.categories().values_iter().collect();
    assert_eq!(categories, ["", "A", "B", "C"]);
    let values: Vec<Option<String>> = symbol
        .cast(&DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect();
    let expected = [
        Some("B"),
        Some("A"),
        None,
        Some("B"),
        Some("C"),
        Some(""),
        Some("A"),
    ];
    assert_eq!(values, expected.map(|value| value.map(str::to_owned)));
    // The ID column stays String.
    assert_eq!(read.column("gene_id").unwrap().dtype(), &DataType::String);
    // An Enum given by the caller keeps its categories, used or not, in its order.
    let frozen = FrozenCategories::new(["z", "y", "unused"]).unwrap();
    let louvain = ids("louvain", &["y", "z"])
        .cast(&DataType::from_frozen_categories(frozen.clone()))
        .unwrap();
    let table = frame(vec![ids("cell_id", &["a", "b"]), louvain]);
    let (store, result) = write(TableAxis::Cells, &table);
    result.unwrap();
    let read = read_table(store.path(), TableAxis::Cells).unwrap();
    let DataType::Enum(categories, _) = read.column("louvain").unwrap().dtype() else {
        panic!("louvain is not Enum");
    };
    let categories: Vec<&str> = categories.categories().values_iter().collect();
    assert_eq!(categories, ["z", "y", "unused"]);
}
