//! Matrix Market + TSV input (10x `filtered_feature_bc_matrix` directories, task 2.6).
//!
//! Does: find `matrix.mtx`, `barcodes.tsv` and `features.tsv` (v3) or `genes.tsv`
//! (v2), each optionally gzipped, in a directory (or next to a given `.mtx` file), as
//! `scanpy.read_10x_mtx` does. The matrix is genes x cells, coordinate format,
//! entries in any order; it is scanned whole by [`TransposedRows`]. `integer` and
//! `pattern` matrices become `uint32` layers (pattern entries are 1), `real` ones
//! `float32`.
//! Does not: read `array`, `complex` or symmetric Matrix Market files, or filter
//! features (kyto keeps every feature, unlike scanpy's `gex_only=True`; the feature ID
//! is `gene_id`, deviations.md).
//!
//! [`TransposedRows`]: crate::import_transpose::TransposedRows

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use kyto_error::KytoError;
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

use crate::import_convert::{Import, ImportLayer};
use crate::import_h5::READ_CHUNK;
use crate::import_transpose::{EntryChunk, EntryScan, TransposedRows};
use crate::layer_open::ValueDtype;

/// What a 10x matrix directory holds.
const ACCEPTED: &str = "pass a directory (or its matrix.mtx) holding matrix.mtx, barcodes.tsv and features.tsv or genes.tsv, each optionally .gz, as Cell Ranger writes them";

/// The three files of a 10x matrix directory.
pub(crate) struct MtxFiles {
    pub matrix: PathBuf,
    pub barcodes: PathBuf,
    pub features: PathBuf,
}

/// The first of `names` that exists in `directory`, or `KY-E120`.
fn find(directory: &Path, names: &[&str]) -> Result<PathBuf, KytoError> {
    names
        .iter()
        .map(|name| directory.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| KytoError::UnsupportedInput {
            path: directory.display().to_string(),
            problem: format!("none of {} is there", names.join(", ")),
            accepted: ACCEPTED.to_owned(),
        })
}

impl MtxFiles {
    /// Find the files for `source`: a directory, or a `.mtx` / `.mtx.gz` file.
    pub(crate) fn find(source: &Path) -> Result<Self, KytoError> {
        let (directory, matrix) = if source.is_dir() {
            (
                source.to_path_buf(),
                find(source, &["matrix.mtx.gz", "matrix.mtx"])?,
            )
        } else {
            let directory = source.parent().unwrap_or(Path::new(".")).to_path_buf();
            (directory, source.to_path_buf())
        };
        Ok(Self {
            matrix,
            barcodes: find(&directory, &["barcodes.tsv.gz", "barcodes.tsv"])?,
            features: find(
                &directory,
                &[
                    "features.tsv.gz",
                    "features.tsv",
                    "genes.tsv.gz",
                    "genes.tsv",
                ],
            )?,
        })
    }
}

/// A buffered reader over a plain or gzipped file.
fn open_text(path: &Path) -> Result<Box<dyn BufRead>, KytoError> {
    let file = File::open(path).map_err(|error| KytoError::StoreIo {
        operation: "open input".to_owned(),
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let gzipped = path.extension().is_some_and(|extension| extension == "gz");
    let reader: Box<dyn Read> = if gzipped {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::with_capacity(1 << 20, reader)))
}

fn read_failed(path: &Path, error: impl std::fmt::Display) -> KytoError {
    KytoError::StoreIo {
        operation: "read input".to_owned(),
        path: path.display().to_string(),
        detail: error.to_string(),
    }
}

fn invalid(path: &Path, line: u64, problem: String, advice: &str) -> KytoError {
    KytoError::InvalidInput {
        path: path.display().to_string(),
        location: format!("line {line}"),
        problem,
        advice: advice.to_owned(),
    }
}

/// The tab-separated fields of every non-empty line.
fn read_tsv(path: &Path) -> Result<Vec<Vec<String>>, KytoError> {
    let mut rows = Vec::new();
    for line in open_text(path)?.lines() {
        let line = line.map_err(|error| read_failed(path, error))?;
        let line = line.strip_suffix('\r').unwrap_or(&line);
        if line.is_empty() {
            continue;
        }
        rows.push(line.split('\t').map(str::to_owned).collect());
    }
    Ok(rows)
}

/// The Matrix Market header: field kind and `rows cols entries`.
struct Header {
    dtype: ValueDtype,
    pattern: bool,
    integer: bool,
    n_rows: u64,
    n_cols: u64,
    n_entries: u64,
}

/// Read the banner, comments and size line; returns the header and the line count
/// read so far.
fn read_header(path: &Path, reader: &mut dyn BufRead) -> Result<(Header, u64), KytoError> {
    let unsupported = |problem: String| {
        KytoError::UnsupportedInput {
        path: path.display().to_string(),
        problem,
        accepted: "a Matrix Market 'matrix coordinate' file with field integer, real or pattern and symmetry general (as Cell Ranger and scipy.io.mmwrite write)".to_owned(),
    }
    };
    let mut line = String::new();
    let mut number = 0_u64;
    let mut next = |line: &mut String| -> Result<bool, KytoError> {
        line.clear();
        number += 1;
        let read = reader
            .read_line(line)
            .map_err(|error| read_failed(path, error))?;
        Ok(read > 0)
    };
    if !next(&mut line)? {
        return Err(unsupported("the file is empty".to_owned()));
    }
    let banner: Vec<String> = line.split_whitespace().map(str::to_lowercase).collect();
    let words: Vec<&str> = banner.iter().map(String::as_str).collect();
    let (pattern, integer, dtype) = match words.as_slice() {
        ["%%matrixmarket", "matrix", "coordinate", field, "general"] => match *field {
            "integer" | "unsigned-integer" => (false, true, ValueDtype::U32),
            "pattern" => (true, false, ValueDtype::U32),
            "real" | "double" => (false, false, ValueDtype::F32),
            other => return Err(unsupported(format!("field '{other}' is not supported"))),
        },
        ["%%matrixmarket", ..] => {
            return Err(unsupported(format!(
                "header '{}' is not supported",
                line.trim()
            )));
        }
        _ => return Err(unsupported("it has no %%MatrixMarket header".to_owned())),
    };
    loop {
        if !next(&mut line)? {
            return Err(unsupported("it has no size line".to_owned()));
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('%') {
            break;
        }
    }
    let sizes: Vec<Option<u64>> = line
        .split_whitespace()
        .map(|word| word.parse().ok())
        .collect();
    let [Some(n_rows), Some(n_cols), Some(n_entries)] = sizes[..] else {
        return Err(invalid(
            path,
            number,
            format!("size line '{}' is not three counts", line.trim()),
            "write 'rows columns entries' after the header",
        ));
    };
    Ok((
        Header {
            dtype,
            pattern,
            integer,
            n_rows,
            n_cols,
            n_entries,
        },
        number,
    ))
}

/// The entries of a genes x cells Matrix Market file, re-read from disk on every scan.
struct MtxEntries {
    path: PathBuf,
    header: Header,
}

const ENTRY_ADVICE: &str = "write one 'gene cell value' line per entry, 1-based, within the size line (Cell Ranger and scipy.io.mmwrite do)";

impl EntryScan for MtxEntries {
    fn scan(
        &mut self,
        visit: &mut dyn FnMut(EntryChunk<'_>) -> Result<(), KytoError>,
    ) -> Result<(), KytoError> {
        let path = self.path.as_path();
        let header = &self.header;
        let mut reader = open_text(path)?;
        let (_, mut number) = read_header(path, &mut *reader)?;
        let (mut cells, mut genes, mut values) = (Vec::new(), Vec::new(), Vec::new());
        let mut seen = 0_u64;
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|error| read_failed(path, error))?;
            if read == 0 {
                break;
            }
            number += 1;
            let mut words = line.split_ascii_whitespace();
            let Some(first) = words.next() else {
                continue;
            };
            if first.starts_with('%') {
                continue;
            }
            let bad = |problem: String| invalid(path, number, problem, ENTRY_ADVICE);
            seen += 1;
            if seen > header.n_entries {
                return Err(bad(format!(
                    "more entries than the {} of the size line",
                    header.n_entries
                )));
            }
            let position = |word: Option<&str>, limit: u64, axis: &str| -> Result<u32, KytoError> {
                match word.and_then(|word| word.parse::<u64>().ok()) {
                    Some(value) if (1..=limit).contains(&value) => Ok((value - 1) as u32),
                    _ => Err(bad(format!(
                        "{axis} '{}' is not a position from 1 to {limit}",
                        word.unwrap_or("")
                    ))),
                }
            };
            let gene = position(Some(first), header.n_rows, "gene")?;
            let cell = position(words.next(), header.n_cols, "cell")?;
            let value = if header.pattern {
                1.0
            } else {
                let word = words.next().unwrap_or("");
                let parsed = if header.integer {
                    word.parse::<i64>().ok().map(|value| value as f64)
                } else {
                    word.parse::<f64>().ok()
                };
                parsed.ok_or_else(|| {
                    bad(format!(
                        "value '{word}' is not a number of the header's field"
                    ))
                })?
            };
            cells.push(cell);
            genes.push(gene);
            values.push(value);
            if cells.len() == READ_CHUNK {
                visit(EntryChunk {
                    cells: &cells,
                    genes: &genes,
                    values: &values,
                })?;
                cells.clear();
                genes.clear();
                values.clear();
            }
        }
        if seen != header.n_entries {
            return Err(invalid(
                path,
                number,
                format!(
                    "the file ends after {seen} entries, but the size line gives {}",
                    header.n_entries
                ),
                ENTRY_ADVICE,
            ));
        }
        visit(EntryChunk {
            cells: &cells,
            genes: &genes,
            values: &values,
        })
    }
}

/// Read a 10x matrix directory; `layer` names its matrix.
pub(crate) fn read_mtx(
    files: &MtxFiles,
    layer: &str,
    budget_bytes: u64,
    block_rows: u64,
) -> Result<Import<'static>, KytoError> {
    let barcodes = read_tsv(&files.barcodes)?;
    let features = read_tsv(&files.features)?;
    let (header, _) = read_header(&files.matrix, &mut *open_text(&files.matrix)?)?;
    let mismatch = |what: &str, found: usize, path: &Path, expected: u64, axis: &str| {
        KytoError::InvalidInput {
            path: path.display().to_string(),
            location: "the whole file".to_owned(),
            problem: format!(
                "it lists {found} {what}, but the matrix has {expected} {axis} (10x matrices are genes x cells)"
            ),
            advice: "pass the matrix.mtx with the barcodes.tsv and features.tsv written with it"
                .to_owned(),
        }
    };
    if barcodes.len() as u64 != header.n_cols {
        return Err(mismatch(
            "barcodes",
            barcodes.len(),
            &files.barcodes,
            header.n_cols,
            "columns",
        ));
    }
    if features.len() as u64 != header.n_rows {
        return Err(mismatch(
            "features",
            features.len(),
            &files.features,
            header.n_rows,
            "rows",
        ));
    }
    let n_cells = header.n_cols;
    let n_genes = header.n_rows;
    let cell_ids: Vec<String> = barcodes
        .into_iter()
        .map(|mut fields| fields.swap_remove(0))
        .collect();
    let cells = DataFrame::new(
        cell_ids.len(),
        vec![Series::new("cell_id".into(), cell_ids).into_column()],
    )
    .map_err(|error| read_failed(&files.barcodes, error))?;
    let width = features.iter().map(Vec::len).max().unwrap_or(1);
    let names = ["gene_id", "symbol", "feature_type"];
    let mut columns = Vec::new();
    for (index, name) in names.iter().enumerate().take(width.min(3)) {
        let values: Vec<Option<&str>> = features
            .iter()
            .map(|fields| fields.get(index).map(String::as_str))
            .collect();
        columns.push(Series::new((*name).into(), values).into_column());
    }
    let genes = DataFrame::new(features.len(), columns)
        .map_err(|error| read_failed(&files.features, error))?;
    let dtype = header.dtype;
    let entries = MtxEntries {
        path: files.matrix.clone(),
        header,
    };
    // scanpy casts mtx values to float32 before building CSR, so repeats sum in float32.
    let source = TransposedRows::new(
        entries,
        dtype,
        n_cells,
        n_genes,
        files.matrix.display().to_string(),
        budget_bytes,
        block_rows,
    )?
    .with_f32_sums(true);
    Ok(Import {
        kind: "mtx",
        cells,
        genes,
        ordered: Vec::new(),
        layers: vec![ImportLayer {
            name: layer.to_owned(),
            source: Box::new(source),
        }],
    })
}
