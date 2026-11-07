//! 10x Genomics `.h5` input (Cell Ranger feature-barcode matrices, task 2.6).
//!
//! Does: read the v3 layout (Cell Ranger >= 3: group `matrix` with `barcodes`,
//! `features/{id,name,feature_type,genome,...}` and a genes x cells CSC matrix) and
//! the v2 layout (one group per genome with `barcodes`, `genes`, `gene_names`). The
//! matrix is stored by cell, so it streams one cell block at a time. Counts are
//! integers and become a `uint32` layer.
//! Does not: filter features. `scanpy.read_10x_h5` keeps only "Gene Expression"
//! features by default (`gex_only=True`) and indexes genes by symbol; kyto keeps every
//! feature, with the feature ID as `gene_id` and the symbol in `symbol`
//! (deviations.md). v2 files with several genomes are refused (pick one with
//! scanpy or h5py first).

use kyto_error::KytoError;
use polars::prelude::{Column, DataFrame, IntoColumn, NamedFrom, Series};

use crate::import_convert::{Import, ImportLayer};
use crate::import_h5::{H5File, length_1d, matrix_dtype, read_all, read_strings};
use crate::import_h5_matrix::{CsrRows, SparseParts};

/// Feature datasets that are not per-feature columns.
const SKIPPED_FEATURES: [&str; 3] = ["id", "name", "_all_tag_keys"];

/// Read a 10x `.h5` file; `layer` names its count matrix.
pub(crate) fn read_10x<'a>(file: &'a H5File, layer: &str) -> Result<Import<'a>, KytoError> {
    let accepted = "pass a 10x .h5 file from Cell Ranger (v2: one group per genome; v3: a group 'matrix' with 'features')";
    let (group, gene_columns) = if file.exists("matrix/features") {
        let mut columns = vec![
            ("gene_id".to_owned(), "matrix/features/id".to_owned()),
            ("symbol".to_owned(), "matrix/features/name".to_owned()),
        ];
        let features = file.group("matrix/features", accepted)?;
        let names = features
            .member_names()
            .map_err(|error| file.read_error("matrix/features", error))?;
        for name in names {
            if SKIPPED_FEATURES.contains(&name.as_str()) {
                continue;
            }
            let member = format!("matrix/features/{name}");
            if file.file.dataset(&member).is_ok() {
                columns.push((name, member));
            }
        }
        ("matrix".to_owned(), columns)
    } else {
        let names = file
            .file
            .member_names()
            .map_err(|error| file.read_error("/", error))?;
        let genomes: Vec<&String> = names
            .iter()
            .filter(|name| file.file.group(name).is_ok())
            .collect();
        let genome = match genomes.as_slice() {
            [genome] => (*genome).clone(),
            [] => return Err(file.unsupported("/", "it has no matrix group", accepted)),
            several => {
                return Err(file.unsupported(
                    "/",
                    &format!("it holds {} genomes ({}); kyto imports one", several.len(), names.join(", ")),
                    "write one genome to its own .h5 file (or an .h5ad with scanpy.read_10x_h5(genome=...)), then convert that",
                ));
            }
        };
        let columns = vec![
            ("gene_id".to_owned(), format!("{genome}/genes")),
            ("symbol".to_owned(), format!("{genome}/gene_names")),
        ];
        (genome, columns)
    };

    let shape_member = format!("{group}/shape");
    let shape_dataset = file.dataset(&shape_member, accepted)?;
    let shape = read_all::<u64>(file, &shape_member, &shape_dataset)?;
    let [n_genes, n_cells] = shape[..] else {
        return Err(file.unsupported(
            &shape_member,
            &format!("shape has {} entries, not 2", shape.len()),
            accepted,
        ));
    };

    let barcodes_member = format!("{group}/barcodes");
    let barcodes_dataset = file.dataset(&barcodes_member, accepted)?;
    let barcodes = read_strings(file, &barcodes_member, &barcodes_dataset)?;
    check_length(
        file,
        &barcodes_member,
        barcodes.len() as u64,
        n_cells,
        "cells",
    )?;
    let cells = DataFrame::new(
        barcodes.len(),
        vec![Series::new("cell_id".into(), barcodes).into_column()],
    )
    .map_err(|error| file.read_error(&barcodes_member, error))?;

    let mut columns: Vec<Column> = Vec::with_capacity(gene_columns.len());
    for (name, member) in gene_columns {
        let dataset = file.dataset(&member, accepted)?;
        length_1d(file, &member, &dataset)?;
        let values = read_strings(file, &member, &dataset)?;
        check_length(file, &member, values.len() as u64, n_genes, "features")?;
        columns.push(Series::new(name.as_str().into(), values).into_column());
    }
    let genes = DataFrame::new(n_genes as usize, columns)
        .map_err(|error| file.read_error(&group, error))?;

    let data_member = format!("{group}/data");
    let data = file.dataset(&data_member, accepted)?;
    let dtype = matrix_dtype(file, &data_member, &data)?;
    let parts = SparseParts::open(file, &group, n_cells, n_genes)?;
    Ok(Import {
        kind: "10x_h5",
        cells,
        genes,
        ordered: Vec::new(),
        layers: vec![ImportLayer {
            name: layer.to_owned(),
            source: Box::new(CsrRows::new(file, parts, dtype)),
        }],
    })
}

/// `KY-E121` unless `member` has `expected` entries (the matrix shape).
fn check_length(
    file: &H5File,
    member: &str,
    found: u64,
    expected: u64,
    what: &str,
) -> Result<(), KytoError> {
    if found == expected {
        return Ok(());
    }
    Err(KytoError::InvalidInput {
        path: file.label(member),
        location: "the whole dataset".to_owned(),
        problem: format!("it has {found} entries, but the matrix shape has {expected} {what}"),
        advice: "re-export the matrix with Cell Ranger".to_owned(),
    })
}
