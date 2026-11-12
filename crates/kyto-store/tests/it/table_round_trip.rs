//! Task 2.3 Done-when check: a table round trip keeps dtypes and values exactly, for
//! every dtype format.md §5.3 allows, with nulls in each, NaN and -0.0 kept apart from
//! null (§5.5), Unicode strings, Enum categories (§5.4) and empty tables. Python
//! Polars reads the written files with the same dtypes.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::process::Command;

use kyto_store::{ColumnDtype, TableAxis, Tiling, read_table, table_columns, write_table};
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, IntoColumn, NamedFrom, Series,
};
use proptest::prelude::*;

const TILING: Tiling = Tiling {
    cell_block_size: 2,
    gene_block_size: 3,
};

fn enum_column(name: &str, categories: &[&str], values: &[Option<&str>]) -> Column {
    let frozen = FrozenCategories::new(categories.iter().copied()).unwrap();
    Series::new(name.into(), values)
        .into_column()
        .strict_cast(&DataType::from_frozen_categories(frozen))
        .unwrap()
}

/// One column of every dtype of format.md §5.3, each with a null and its extremes.
fn every_dtype() -> DataFrame {
    let columns = vec![
        Series::new("cell_id".into(), &["AAAC-1", "細胞-2", "🧬-3", "a\tb\n-4"]).into_column(),
        Series::new(
            "boolean".into(),
            &[Some(true), None, Some(false), Some(true)],
        )
        .into_column(),
        Series::new(
            "int8".into(),
            &[Some(i8::MIN), None, Some(0), Some(i8::MAX)],
        )
        .into_column(),
        Series::new(
            "int16".into(),
            &[Some(i16::MIN), None, Some(-1), Some(i16::MAX)],
        )
        .into_column(),
        Series::new(
            "int32".into(),
            &[Some(i32::MIN), None, Some(7), Some(i32::MAX)],
        )
        .into_column(),
        Series::new(
            "int64".into(),
            &[Some(i64::MIN), None, Some(-7), Some(i64::MAX)],
        )
        .into_column(),
        Series::new("uint8".into(), &[Some(0_u8), None, Some(1), Some(u8::MAX)]).into_column(),
        Series::new(
            "uint16".into(),
            &[Some(0_u16), None, Some(1), Some(u16::MAX)],
        )
        .into_column(),
        Series::new(
            "uint32".into(),
            &[Some(0_u32), None, Some(1), Some(u32::MAX)],
        )
        .into_column(),
        Series::new(
            "uint64".into(),
            &[Some(0_u64), None, Some(1), Some(u64::MAX)],
        )
        .into_column(),
        Series::new(
            "float32".into(),
            &[Some(f32::NAN), None, Some(-0.0), Some(f32::INFINITY)],
        )
        .into_column(),
        Series::new(
            "float64".into(),
            &[
                Some(f64::MIN_POSITIVE),
                Some(f64::NAN),
                None,
                Some(f64::NEG_INFINITY),
            ],
        )
        .into_column(),
        Series::new(
            "sample".into(),
            &[Some("Zürich"), None, Some(""), Some("東京")],
        )
        .into_column(),
        enum_column(
            "louvain",
            &["2", "10", "1", "unused"],
            &[Some("10"), Some("1"), None, Some("2")],
        ),
        enum_column("all_null", &[], &[None, None, None, None]),
    ];
    DataFrame::new_infer_height(columns).unwrap()
}

/// Every value as text, so NaN equals NaN and -0.0 differs from 0.0.
fn cells(column: &Column) -> Vec<String> {
    (0..column.len())
        .map(|row| format!("{:?}", column.get(row).unwrap()))
        .collect()
}

fn enum_categories(dtype: &DataType) -> Option<Vec<String>> {
    match dtype {
        DataType::Enum(categories, _) => Some(
            categories
                .categories()
                .values_iter()
                .map(str::to_owned)
                .collect(),
        ),
        _ => None,
    }
}

/// `read` equals `written` after plain strings became Enums of sorted categories.
fn assert_round_trip(written: &DataFrame, read: &DataFrame) {
    assert_eq!(read.height(), written.height());
    let names = |frame: &DataFrame| -> Vec<String> {
        frame
            .columns()
            .iter()
            .map(|column| column.name().to_string())
            .collect()
    };
    assert_eq!(names(read), names(written));
    for (position, (before, after)) in written.columns().iter().zip(read.columns()).enumerate() {
        let name = before.name();
        if position > 0 && before.dtype() == &DataType::String {
            let mut sorted: Vec<String> = before
                .str()
                .unwrap()
                .iter()
                .flatten()
                .map(str::to_owned)
                .collect();
            sorted.sort();
            sorted.dedup();
            assert_eq!(enum_categories(after.dtype()), Some(sorted), "{name}");
            let as_strings = after.cast(&DataType::String).unwrap();
            assert_eq!(cells(&as_strings), cells(before), "{name}");
            continue;
        }
        match enum_categories(before.dtype()) {
            Some(categories) => {
                assert_eq!(enum_categories(after.dtype()), Some(categories), "{name}");
            }
            None => assert_eq!(after.dtype(), before.dtype(), "{name}"),
        }
        assert_eq!(cells(after), cells(before), "{name}");
    }
}

fn round_trip(axis: TableAxis, frame: &DataFrame) -> (tempfile::TempDir, DataFrame) {
    let store = tempfile::tempdir().unwrap();
    write_table(store.path(), axis, frame, TILING).unwrap();
    let read = read_table(store.path(), axis).unwrap();
    assert_round_trip(frame, &read);
    (store, read)
}

#[test]
fn every_dtype_round_trips_exactly() {
    let written = every_dtype();
    let (_store, read) = round_trip(TableAxis::Cells, &written);
    // NaN stays NaN and is not null; null stays null (format.md §5.5).
    let float32 = read.column("float32").unwrap();
    assert_eq!(float32.null_count(), 1);
    assert!(float32.f32().unwrap().get(0).unwrap().is_nan());
    assert!(float32.f32().unwrap().get(2).unwrap().is_sign_negative());
}

#[test]
fn schema_columns_describe_the_stored_table() {
    let (_store, read) = round_trip(TableAxis::Cells, &every_dtype());
    let columns = table_columns(TableAxis::Cells, &read).unwrap();
    let json: Vec<String> = columns
        .iter()
        .map(|column| serde_json::to_string(column).unwrap())
        .collect();
    assert_eq!(json[0], r#"{"name":"cell_id","dtype":"String"}"#);
    assert_eq!(json[1], r#"{"name":"boolean","dtype":"Boolean"}"#);
    assert_eq!(
        json[12],
        r#"{"name":"sample","dtype":"Enum","n_categories":3,"ordered":false}"#
    );
    assert_eq!(
        json[14],
        r#"{"name":"all_null","dtype":"Enum","n_categories":0,"ordered":false}"#
    );
    let dtypes: Vec<&str> = columns.iter().map(|column| column.dtype.name()).collect();
    assert_eq!(
        dtypes,
        [
            "String", "Boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32",
            "UInt64", "Float32", "Float64", "Enum", "Enum", "Enum"
        ]
    );
    assert!(matches!(
        columns[13].dtype,
        ColumnDtype::Enum {
            n_categories: 4,
            ordered: false
        }
    ));
}

#[test]
fn empty_tables_round_trip() {
    // No rows: every column keeps its dtype; a plain string column becomes an Enum
    // with no categories.
    let empty = every_dtype().head(Some(0));
    for axis in [TableAxis::Cells, TableAxis::Genes] {
        let renamed = if axis == TableAxis::Genes {
            let mut frame = empty.clone();
            frame.rename("cell_id", "gene_id".into()).unwrap();
            frame
        } else {
            empty.clone()
        };
        let (_store, read) = round_trip(axis, &renamed);
        assert_eq!(read.height(), 0);
        assert_eq!(
            enum_categories(read.column("sample").unwrap().dtype()),
            Some(vec![])
        );
    }
    // Only the ID column.
    let ids_only = DataFrame::new_infer_height(vec![
        Series::new("gene_id".into(), &[] as &[&str]).into_column(),
    ])
    .unwrap();
    round_trip(TableAxis::Genes, &ids_only);
}

/// A Python interpreter with polars: `PYTHON`, else `python` on `PATH` (under
/// `uv run cargo test`, the project's `.venv`).
fn python() -> Command {
    Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
}

/// Read the table with Python Polars and print its schema and categories, one column
/// per line, in a form this test can compare.
const PYTHON_READ: &str = r#"
import sys
import polars as pl

frame = pl.read_parquet(sys.argv[1])
for name, dtype in frame.schema.items():
    if isinstance(dtype, pl.Enum):
        print(name, "Enum", list(dtype.categories), sep="|")
    else:
        print(name, dtype, sep="|")
print("rows", frame.height, sep="|")
print("cell_id", frame["cell_id"].to_list(), sep="|")
print("sample", frame["sample"].to_list(), sep="|")
print("float32", frame["float32"].is_nan().to_list(), frame["float32"].is_null().to_list(), sep="|")
"#;

#[test]
fn python_polars_reads_the_dtypes() {
    let (store, _read) = round_trip(TableAxis::Cells, &every_dtype());
    let path = store.path().join("cells.parquet");
    let output = python()
        .arg("-c")
        .arg(PYTHON_READ)
        .arg(&path)
        .env("PYTHONIOENCODING", "utf-8")
        .output()
        .expect("run python with polars (use `uv run cargo test`, or set PYTHON)");
    assert!(
        output.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)
        .unwrap()
        .replace("\r\n", "\n");
    let expected = "\
cell_id|String
boolean|Boolean
int8|Int8
int16|Int16
int32|Int32
int64|Int64
uint8|UInt8
uint16|UInt16
uint32|UInt32
uint64|UInt64
float32|Float32
float64|Float64
sample|Enum|['', 'Zürich', '東京']
louvain|Enum|['2', '10', '1', 'unused']
all_null|Enum|[]
rows|4
cell_id|['AAAC-1', '細胞-2', '🧬-3', 'a\\tb\\n-4']
sample|['Zürich', None, '', '東京']
float32|[True, None, False, False]|[False, True, False, False]
";
    assert_eq!(stdout, expected);
}

fn id_and_strings() -> impl Strategy<Value = (Vec<String>, Vec<Option<String>>)> {
    // Unique IDs from arbitrary Unicode; a string column with nulls and repeats.
    (0_usize..12).prop_flat_map(|n_rows| {
        (
            proptest::collection::btree_set("\\PC{1,6}", n_rows..=n_rows),
            proptest::collection::vec(proptest::option::of("[a-c\u{e9}\u{4e00}]{0,2}"), n_rows),
        )
            .prop_map(|(ids, values)| (ids.into_iter().collect(), values))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_string_tables_round_trip((ids, values) in id_and_strings()) {
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        let values: Vec<Option<&str>> = values.iter().map(Option::as_deref).collect();
        let frame = DataFrame::new_infer_height(vec![
            Series::new("gene_id".into(), &ids).into_column(),
            Series::new("symbol".into(), &values).into_column(),
        ])
        .unwrap();
        round_trip(TableAxis::Genes, &frame);
    }
}
