//! The `.h5ad` export (task 2.7): a store written with every v0 column dtype, nulls and
//! two layers over many cell blocks goes out through `sink_h5ad` and back in through
//! the 2.6 importer unchanged (layers bit for bit, tables cell for cell with float `NaN`
//! back as null, `ordered` flags); the anndata encodings and index dtypes are those anndata's h5py writer uses;
//! the target appears only on success; bad targets, layers, names and strings get coded
//! errors (R10). The parity test against anndata itself is `tests/test_sink_h5ad.py`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use hdf5_metno::File;
use hdf5_metno::types::VarLenUnicode;
use kyto_store::{
    AxisSelection, CellBlock, CellValues, ConvertOptions, LayerReader, LayerSpec,
    LayerWriteOptions, SinkH5adOptions, StoreWriter, TableAxis, TileGrid, Tiling, ValueDtype,
    convert, read_table, sink_h5ad, sink_h5ad_checked,
};
use polars::prelude::{
    Column, DataFrame, DataType, FrozenCategories, IntoColumn, NamedFrom, Series,
};
use zarrs::filesystem::FilesystemStore;

const N_CELLS: u64 = 23;
const N_GENES: u64 = 7;
const TILING: Tiling = Tiling {
    cell_block_size: 4,
    gene_block_size: 3,
};

fn enum_column(name: &str, categories: &[&str], values: &[Option<&str>]) -> Column {
    let frozen = FrozenCategories::new(categories.iter().copied()).unwrap();
    Series::new(name.into(), values)
        .strict_cast(&DataType::from_frozen_categories(frozen))
        .unwrap()
        .into_column()
}

fn cell_table(extra: Option<Column>) -> DataFrame {
    let rows: Vec<u64> = (0..N_CELLS).collect();
    let ids: Vec<String> = rows.iter().map(|row| format!("cell-{row}")).collect();
    let stage: Vec<Option<&str>> = rows
        .iter()
        .map(|row| [Some("late"), None, Some("early")][(row % 3) as usize])
        .collect();
    let mut columns = vec![
        Series::new("cell_id".into(), ids).into_column(),
        enum_column("stage", &["early", "mid", "late", ""], &stage),
        Series::new(
            "int16".into(),
            rows.iter().map(|row| *row as i16 - 5).collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "nullable_u8".into(),
            rows.iter()
                .map(|row| (!row.is_multiple_of(4)).then_some(*row as u8))
                .collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "nullable_i64".into(),
            rows.iter()
                .map(|row| (!row.is_multiple_of(5)).then_some(-(*row as i64)))
                .collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "flag".into(),
            rows.iter()
                .map(|row| row.is_multiple_of(2))
                .collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "nullable_flag".into(),
            rows.iter()
                .map(|row| (!row.is_multiple_of(3)).then_some(row.is_multiple_of(2)))
                .collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "float64".into(),
            rows.iter()
                .map(|row| {
                    if row.is_multiple_of(6) {
                        f64::NAN
                    } else {
                        *row as f64 / 3.0
                    }
                })
                .collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "float32".into(),
            rows.iter().map(|row| *row as f32 / 7.0).collect::<Vec<_>>(),
        )
        .into_column(),
        Series::new(
            "u64".into(),
            rows.iter().map(|row| u64::MAX - row).collect::<Vec<_>>(),
        )
        .into_column(),
    ];
    columns.extend(extra);
    DataFrame::new(N_CELLS as usize, columns).unwrap()
}

fn gene_table() -> DataFrame {
    let ids: Vec<String> = (0..N_GENES).map(|gene| format!("ENSG{gene:02}")).collect();
    let symbols: Vec<Option<&str>> = ["CD3E", "CD4", "MS4A1", "CD3E", "LYZ", "NKG7", "GNLY"]
        .into_iter()
        .map(Some)
        .collect();
    DataFrame::new(
        N_GENES as usize,
        vec![
            Series::new("gene_id".into(), ids).into_column(),
            enum_column(
                "symbol",
                &["CD3E", "CD4", "GNLY", "LYZ", "MS4A1", "NKG7"],
                &symbols,
            ),
        ],
    )
    .unwrap()
}

/// Row `row` of the test matrix: deterministic, some rows empty.
fn row_entries(row: u64) -> Vec<(u32, u32)> {
    (0..N_GENES)
        .filter(|gene| (row * 3 + gene * 5).is_multiple_of(4) && row % 7 != 3)
        .map(|gene| (gene as u32, (row * 10 + gene + 1) as u32))
        .collect()
}

/// Write the test store: `counts` (uint32) and `scaled` (float32), pushed in uneven
/// batches.
fn write_store(target: &Path, cells: &DataFrame) {
    let specs = [
        LayerSpec {
            name: "counts".to_owned(),
            dtype: ValueDtype::U32,
            options: LayerWriteOptions::default(),
        },
        LayerSpec {
            name: "scaled".to_owned(),
            dtype: ValueDtype::F32,
            options: LayerWriteOptions::default(),
        },
    ];
    let mut writer = StoreWriter::create(target, &gene_table(), &specs, TILING, false).unwrap();
    writer.mark_ordered(TableAxis::Cells, "stage");
    let mut start = 0;
    for batch in [5, 1, 9, 8] {
        let rows: Vec<Vec<(u32, u32)>> = (start..start + batch).map(row_entries).collect();
        let mut indptr = vec![0_u64];
        let (mut indices, mut counts, mut scaled) = (Vec::new(), Vec::new(), Vec::new());
        for row in &rows {
            for &(gene, value) in row {
                indices.push(gene);
                counts.push(value);
                scaled.push(value as f32 / 8.0 - 3.0);
            }
            indptr.push(indices.len() as u64);
        }
        writer
            .push(&[
                CellBlock::from_u32(indptr.clone(), indices.clone(), counts, N_GENES).unwrap(),
                CellBlock::from_f32(indptr, indices, scaled, N_GENES).unwrap(),
            ])
            .unwrap();
        start += batch;
    }
    assert_eq!(start, N_CELLS);
    writer.finish(cells, None).unwrap();
}

/// A layer read back whole: indptr, indices, values as little-endian bytes.
fn layer_back(store: &Path, layer: &str) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(store.join("schema.json")).unwrap()).unwrap();
    let tiling = Tiling {
        cell_block_size: schema["tiling"]["cell_block_size"].as_u64().unwrap(),
        gene_block_size: schema["tiling"]["gene_block_size"].as_u64().unwrap(),
    };
    let grid = TileGrid::new(
        schema["n_cells"].as_u64().unwrap(),
        schema["n_genes"].as_u64().unwrap(),
        tiling,
    )
    .unwrap();
    let storage = Arc::new(FilesystemStore::new(store).unwrap());
    let reader = LayerReader::open(storage, "store", layer, grid).unwrap();
    let (mut indptr, mut indices, mut data) = (vec![0_u64], Vec::new(), Vec::new());
    for selected in reader
        .cell_blocks(
            &AxisSelection::All,
            &AxisSelection::All,
            NonZeroUsize::new(3).unwrap(),
        )
        .unwrap()
    {
        let block = selected.unwrap().block;
        let base = indptr[indptr.len() - 1];
        indptr.extend(block.indptr()[1..].iter().map(|offset| offset + base));
        indices.extend_from_slice(block.indices());
        match block.data() {
            CellValues::F32(values) => {
                data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            }
            CellValues::U32(values) => {
                data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            }
        }
    }
    (indptr, indices, data)
}

fn schema_columns(store: &Path, key: &str) -> serde_json::Value {
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(store.join("schema.json")).unwrap()).unwrap();
    schema[key].clone()
}

fn options(x_layer: Option<&str>) -> SinkH5adOptions {
    SinkH5adOptions {
        x_layer: x_layer.map(str::to_owned),
        blocks_in_flight: NonZeroUsize::new(3).unwrap(),
        ..SinkH5adOptions::default()
    }
}

/// `back` (a store re-imported from `.h5ad`) equals `store` value for value, except that
/// a float `NaN` or null of `store` is null in `back`: the export writes float nulls as
/// `NaN` (anndata has no nullable float) and the importer reads `NaN` as null
/// (format.md §5.5, decided by the user, Nov 7, 2026, task 3.7).
fn assert_frames_equal(store: &DataFrame, back: &DataFrame) {
    let (left, right) = (store, back);
    assert_eq!(left.get_column_names(), right.get_column_names());
    for (a, b) in left.columns().iter().zip(right.columns()) {
        assert_eq!(a.dtype(), b.dtype(), "column {}", a.name());
        let name = a.name().to_string();
        let (a, b) = (a.as_materialized_series(), b.as_materialized_series());
        if a.dtype().is_float() {
            let a = a.cast(&DataType::Float64).unwrap();
            let b = b.cast(&DataType::Float64).unwrap();
            let pairs: Vec<_> = a
                .f64()
                .unwrap()
                .iter()
                .zip(b.f64().unwrap().iter())
                .collect();
            assert_eq!(a.len(), b.len(), "column {name}");
            for (x, y) in pairs {
                let missing = x.is_none_or(f64::is_nan);
                let same = if missing { y.is_none() } else { x == y };
                assert!(same, "column {name}: {x:?} became {y:?}");
            }
        } else {
            assert!(a.equals_missing(b), "column {name}");
        }
    }
}

#[test]
fn store_round_trips_through_h5ad_and_the_importer() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    write_store(&store, &cell_table(None));
    let h5ad = dir.path().join("out.h5ad");
    sink_h5ad(&store, &h5ad, &options(Some("counts"))).unwrap();
    // Only the file itself: no temporary left behind.
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 2, "{names:?}");

    let back = dir.path().join("back.kyto");
    let convert_options = ConvertOptions {
        tiling: TILING,
        ..ConvertOptions::default()
    };
    convert(&h5ad, &back, &convert_options).unwrap();
    for layer in ["counts", "scaled"] {
        assert_eq!(
            layer_back(&store, layer),
            layer_back(&back, layer),
            "{layer}"
        );
    }
    for axis in [TableAxis::Cells, TableAxis::Genes] {
        assert_frames_equal(
            &read_table(&store, axis).unwrap(),
            &read_table(&back, axis).unwrap(),
        );
    }
    for key in ["cells", "genes", "layers"] {
        assert_eq!(
            schema_columns(&store, key),
            schema_columns(&back, key),
            "{key}"
        );
    }
}

#[test]
fn encodings_match_anndata() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    write_store(&store, &cell_table(None));
    let h5ad = dir.path().join("out.h5ad");
    sink_h5ad(&store, &h5ad, &options(Some("scaled"))).unwrap();
    let file = File::open(&h5ad).unwrap();
    let text = |location: &hdf5_metno::Location, name: &str| -> String {
        location
            .attr(name)
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str()
            .to_owned()
    };
    let encoding = |member: &str| -> (String, String) {
        let location: hdf5_metno::Location = match file.group(member) {
            Ok(group) => (*group).clone(),
            Err(_) => (**file.dataset(member).unwrap()).clone(),
        };
        (
            text(&location, "encoding-type"),
            text(&location, "encoding-version"),
        )
    };
    let pair = |kind: &str, version: &str| (kind.to_owned(), version.to_owned());
    assert_eq!(encoding("/"), pair("anndata", "0.1.0"));
    for (member, kind, version) in [
        ("obs", "dataframe", "0.2.0"),
        ("obs/cell_id", "string-array", "0.2.0"),
        ("obs/stage", "categorical", "0.2.0"),
        ("obs/stage/codes", "array", "0.2.0"),
        ("obs/stage/categories", "string-array", "0.2.0"),
        ("obs/int16", "array", "0.2.0"),
        ("obs/nullable_u8", "nullable-integer", "0.1.0"),
        ("obs/nullable_u8/values", "array", "0.2.0"),
        ("obs/nullable_u8/mask", "array", "0.2.0"),
        ("obs/flag", "array", "0.2.0"),
        ("obs/nullable_flag", "nullable-boolean", "0.1.0"),
        ("var", "dataframe", "0.2.0"),
        ("X", "csr_matrix", "0.1.0"),
        ("layers", "dict", "0.1.0"),
        ("layers/counts", "csr_matrix", "0.1.0"),
        ("obsm", "dict", "0.1.0"),
        ("varm", "dict", "0.1.0"),
        ("obsp", "dict", "0.1.0"),
        ("varp", "dict", "0.1.0"),
        ("uns", "dict", "0.1.0"),
    ] {
        assert_eq!(encoding(member), pair(kind, version), "{member}");
    }
    let obs = file.group("obs").unwrap();
    assert_eq!(text(&obs, "_index"), "cell_id");
    let order: Vec<String> = obs
        .attr("column-order")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|name| name.as_str().to_owned())
        .collect();
    assert_eq!(
        order,
        [
            "stage",
            "int16",
            "nullable_u8",
            "nullable_i64",
            "flag",
            "nullable_flag",
            "float64",
            "float32",
            "u64"
        ]
    );
    let stage = file.group("obs/stage").unwrap();
    assert!(
        stage
            .attr("ordered")
            .unwrap()
            .read_scalar::<bool>()
            .unwrap()
    );
    let codes = file.dataset("obs/stage/codes").unwrap();
    assert_eq!(codes.dtype().unwrap().size(), 1, "int8 codes, as pandas");
    let codes: Vec<i8> = codes.read_raw().unwrap();
    assert_eq!(&codes[..4], &[2, -1, 0, 2]);
    let symbol = file.group("var/symbol").unwrap();
    assert!(
        !symbol
            .attr("ordered")
            .unwrap()
            .read_scalar::<bool>()
            .unwrap()
    );

    let x = file.group("X").unwrap();
    let shape: Vec<i64> = x.attr("shape").unwrap().read_raw().unwrap();
    assert_eq!(shape, [N_CELLS as i64, N_GENES as i64]);
    assert_eq!(
        file.dataset("X/indices").unwrap().dtype().unwrap().size(),
        4
    );
    assert_eq!(file.dataset("X/indptr").unwrap().dtype().unwrap().size(), 4);
    let data = file.dataset("X/data").unwrap();
    assert!(data.dtype().unwrap().is::<f32>());
    assert!(
        file.dataset("layers/counts/data")
            .unwrap()
            .dtype()
            .unwrap()
            .is::<u32>()
    );
    let indptr: Vec<i32> = file.dataset("X/indptr").unwrap().read_raw().unwrap();
    let (want_indptr, want_indices, _) = layer_back(&store, "scaled");
    assert_eq!(
        indptr,
        want_indptr.iter().map(|&v| v as i32).collect::<Vec<_>>()
    );
    let indices: Vec<i32> = file.dataset("X/indices").unwrap().read_raw().unwrap();
    assert_eq!(
        indices,
        want_indices.iter().map(|&v| v as i32).collect::<Vec<_>>()
    );
    // Masked entries hold 0 / false, as anndata writes them.
    let values: Vec<u8> = file
        .dataset("obs/nullable_u8/values")
        .unwrap()
        .read_raw()
        .unwrap();
    let mask: Vec<bool> = file
        .dataset("obs/nullable_u8/mask")
        .unwrap()
        .read_raw()
        .unwrap();
    assert_eq!(
        (values[0], mask[0], values[1], mask[1]),
        (0, true, 1, false)
    );
}

#[test]
fn without_x_every_layer_goes_to_layers() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    write_store(&store, &cell_table(None));
    let h5ad = dir.path().join("out.h5ad");
    sink_h5ad(&store, &h5ad, &options(None)).unwrap();
    let file = File::open(&h5ad).unwrap();
    assert!(file.group("X").is_err());
    let mut names = file.group("layers").unwrap().member_names().unwrap();
    names.sort();
    assert_eq!(names, ["counts", "scaled"]);
}

#[test]
fn empty_store_exports() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    let genes = DataFrame::new(
        0,
        vec![Series::new("gene_id".into(), Vec::<String>::new()).into_column()],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let writer = StoreWriter::create(&store, &genes, &specs, TILING, false).unwrap();
    let cells = DataFrame::new(
        0,
        vec![Series::new("cell_id".into(), Vec::<String>::new()).into_column()],
    )
    .unwrap();
    writer.finish(&cells, None).unwrap();
    let h5ad = dir.path().join("out.h5ad");
    sink_h5ad(&store, &h5ad, &options(Some("counts"))).unwrap();
    let file = File::open(&h5ad).unwrap();
    let indptr: Vec<i32> = file.dataset("X/indptr").unwrap().read_raw().unwrap();
    assert_eq!(indptr, [0]);
    assert!(file.group("obs").unwrap().attr("column-order").is_ok());
    drop(file);
    let back = dir.path().join("back.kyto");
    convert(&h5ad, &back, &ConvertOptions::default()).unwrap();
}

#[test]
fn bad_targets_and_layers_are_coded_and_leave_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    write_store(&store, &cell_table(None));
    let h5ad = dir.path().join("out.h5ad");

    let error = sink_h5ad(&store, &h5ad, &options(Some("count"))).unwrap_err();
    assert_eq!(error.code(), "KY-E109", "{error}");
    assert!(!h5ad.exists());

    std::fs::write(&h5ad, b"old").unwrap();
    let error = sink_h5ad(&store, &h5ad, &options(Some("counts"))).unwrap_err();
    assert_eq!(error.code(), "KY-E123", "{error}");
    assert_eq!(std::fs::read(&h5ad).unwrap(), b"old");
    let overwrite = SinkH5adOptions {
        overwrite: true,
        ..options(Some("counts"))
    };
    sink_h5ad(&store, &h5ad, &overwrite).unwrap();
    assert!(File::open(&h5ad).is_ok());

    let folder = dir.path().join("folder.h5ad");
    std::fs::create_dir(&folder).unwrap();
    assert_eq!(
        sink_h5ad(&store, &folder, &overwrite).unwrap_err().code(),
        "KY-E123"
    );

    let error = sink_h5ad(&store, &dir.path().join("slash.h5ad/"), &options(None)).unwrap_err();
    assert_eq!(error.code(), "KY-E110", "{error}");
    let error = sink_h5ad(&store, &store.join("inside.h5ad"), &options(None)).unwrap_err();
    assert_eq!(error.code(), "KY-E123", "{error}");
    assert!(error.to_string().contains("inside the store"), "{error}");
    // A long but legal name: the temporary name does not grow with it.
    let long = dir.path().join(format!("{}.h5ad", "a".repeat(245)));
    sink_h5ad(&store, &long, &options(None)).unwrap();
    std::fs::remove_file(&long).unwrap();

    let not_a_store = dir.path().join("nothing");
    std::fs::create_dir(&not_a_store).unwrap();
    let error = sink_h5ad(&not_a_store, &dir.path().join("x.h5ad"), &options(None)).unwrap_err();
    assert_eq!(error.code(), "KY-E116", "{error}");

    std::fs::write(store.join("schema.json"), b"{}").unwrap();
    let error = sink_h5ad(&store, &dir.path().join("y.h5ad"), &options(None)).unwrap_err();
    assert_eq!(error.code(), "KY-E122", "{error}");
    let entries = std::fs::read_dir(dir.path()).unwrap().count();
    assert_eq!(entries, 4, "in.kyto, out.h5ad, folder.h5ad, nothing");
}

/// `sink_h5ad_checked` runs its check once, when the temporary file is complete and
/// before the target appears; a failing check leaves the target as it was.
#[test]
fn the_commit_check_runs_last_and_stops_the_rename() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("in.kyto");
    write_store(&store, &cell_table(None));
    let out = dir.path().join("out");
    std::fs::create_dir(&out).unwrap();
    let h5ad = out.join("out.h5ad");
    let names = || -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    let calls = std::cell::Cell::new(0_u32);
    // What the check sees: no target yet, and the temporary file, complete (it opens
    // as HDF5 and holds `var`, which is written after the layers and `obs`).
    let stop = || {
        calls.set(calls.get() + 1);
        assert!(!h5ad.exists());
        let names = names();
        assert_eq!(names.len(), 1, "{names:?}");
        assert!(names[0].starts_with(".kyto-part-"), "{names:?}");
        assert!(
            File::open(out.join(&names[0]))
                .unwrap()
                .group("var")
                .is_ok()
        );
        Err(kyto_error::KytoError::Cancelled { done: 0, total: 0 })
    };
    let error = sink_h5ad_checked(&store, &h5ad, &options(Some("counts")), &stop).unwrap_err();
    assert_eq!(error.code(), "KY-E503", "{error}");
    assert_eq!(calls.get(), 1);
    assert!(names().is_empty(), "{:?}", names());

    // A check that passes: the file appears.
    let pass = || {
        calls.set(calls.get() + 1);
        assert!(!h5ad.exists());
        Ok(())
    };
    sink_h5ad_checked(&store, &h5ad, &options(Some("counts")), &pass).unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(names(), ["out.h5ad"]);

    // Stopped while overwriting: the old file is untouched.
    let before = std::fs::read(&h5ad).unwrap();
    let overwrite = SinkH5adOptions {
        overwrite: true,
        ..options(None)
    };
    let stop_overwrite = || Err(kyto_error::KytoError::Cancelled { done: 0, total: 0 });
    let error = sink_h5ad_checked(&store, &h5ad, &overwrite, &stop_overwrite).unwrap_err();
    assert_eq!(error.code(), "KY-E503", "{error}");
    assert_eq!(std::fs::read(&h5ad).unwrap(), before);
    assert_eq!(names(), ["out.h5ad"]);
}

#[test]
fn names_and_strings_h5ad_cannot_hold_are_e124() {
    for (column, needle) in [
        (
            Series::new("a/b".into(), vec![1_i32; N_CELLS as usize]).into_column(),
            "'/'",
        ),
        (
            Series::new("_index".into(), vec![1_i32; N_CELLS as usize]).into_column(),
            "_index",
        ),
        (
            Series::new(".".into(), vec![1_i32; N_CELLS as usize]).into_column(),
            "'.'",
        ),
        (
            enum_column("label", &["a\0b"], &vec![Some("a\0b"); N_CELLS as usize]),
            "NUL",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("in.kyto");
        write_store(&store, &cell_table(Some(column)));
        let h5ad = dir.path().join("out.h5ad");
        let error = sink_h5ad(&store, &h5ad, &options(Some("counts"))).unwrap_err();
        assert_eq!(error.code(), "KY-E124", "{error}");
        assert!(error.to_string().contains(needle), "{error}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "{error}");
    }
}
