//! Task 6.12: Scrublet against scanpy's `pp.scrublet`.
//!
//! The stored scanpy reference `scrublet` (`tests/fixtures/make_reference.py`:
//! `pp.scrublet(random_state=0)` on the fixtures' counts, and with `batch_key="batch"`
//! on synthetic) holds each cell's score and call and, per batch, the simulated
//! doublets' parents, their scores and the threshold. Done when:
//! - given scanpy's simulated doublets ([`Doublets::Given`]), every score (cells and
//!   doublets) is within [`SCORE_TOLERANCE`] of scanpy's, the same cells are scored,
//!   and the thresholds and calls are equal;
//! - with kyto's own seeded doublets, in the pipeline (a store converted from the
//!   fixture, the plan `scrublet` on its counts), the calls agree with scanpy's on at
//!   least [`CALL_AGREEMENT`] of the cells (measured and printed), and two runs (1 and
//!   all threads) give the same bits (R11). The synthetic fixture's per-batch run is
//!   held to [`SMALL_BATCH_AGREEMENT`] instead: its batches have about 165 cells, and
//!   scanpy's own calls with seeds 1 to 7 agree with its seed 0 on only 86% to 94% of
//!   them (the automatic threshold moves with the simulated doublets), so no seeded run
//!   can promise 95% there.
//!
//! Skipped without anndata (exit code 77), as the other parity tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{
    CellBlock, Doublets, ExecuteOptions, NodeStatistics, Pipeline, ScrubletResult, scrublet,
};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, Scrublet, physical_plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// How far kyto's scores may be from scanpy's given the same doublets (the Done-when).
const SCORE_TOLERANCE: f64 = 1e-5;
/// The fraction of cells whose calls must agree with kyto's own doublets.
const CALL_AGREEMENT: f64 = 0.95;
/// The same for the synthetic fixture's per-batch run: the lowest agreement of scanpy
/// with itself across seeds 0 to 7 (0.86, measured with scanpy 1.12.5).
const SMALL_BATCH_AGREEMENT: f64 = 0.85;

/// Writes `load_fixture(name)` as `.h5ad` and, as raw little-endian files, its counts
/// (CSR: `indptr` `<u8`, `indices` `<u4`, `data` `<f4`), its batches (one per line)
/// and the `scrublet` reference (floats `<f8`, ints `<u8`, booleans `u1`, a file per
/// array with `/` as `.`); prints the number of genes and the reference's batch labels
/// per run (`run:label,label`). Exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import numpy as np
from scipy.sparse import csr_matrix
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
adata = to_anndata(load_fixture(name))
adata.write_h5ad(h5ad)
counts = csr_matrix(adata.X)
counts.sort_indices()
np.asarray(counts.indptr, dtype='<u8').tofile(f'{out}/indptr')
np.asarray(counts.indices, dtype='<u4').tofile(f'{out}/indices')
np.asarray(counts.data, dtype='<f4').tofile(f'{out}/data')
if 'batch' in adata.obs.columns:
    open(f'{out}/batch', 'w').write('\\n'.join(adata.obs['batch'].astype(str).tolist()))
runs = {}
for key, array in load_reference(name, 'scrublet').items():
    parts = key.split('/')
    if parts[1] == 'batches':
        runs.setdefault(parts[0], set()).add(parts[2])
    path = f\"{out}/{key.replace('/', '.')}\"
    kind = array.dtype.kind
    if kind == 'b':
        np.asarray(array, dtype='u1').tofile(path)
    elif kind == 'f':
        np.asarray(array, dtype='<f8').tofile(path)
    else:
        np.asarray(array, dtype='<u8').tofile(path)
print(adata.n_vars, ' '.join(f\"{run}:{','.join(sorted(labels))}\" for run, labels in runs.items()))
";

struct Fixture {
    dir: PathBuf,
    h5ad: PathBuf,
    n_genes: u64,
    /// Per run (`all`, `by_batch`), its batch labels.
    runs: Vec<(String, Vec<String>)>,
}

fn write_fixture(name: &str, work: &Path) -> Option<Fixture> {
    let dir = work.join(format!("{name}-reference"));
    std::fs::create_dir_all(&dir).unwrap();
    let h5ad = work.join(format!("{name}.h5ad"));
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let output = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(&h5ad)
        .arg(&dir)
        .output()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if output.status.code() == Some(77) {
        return None;
    }
    assert!(
        output.status.success(),
        "writing fixture {name} failed: {output:?}"
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut words = text.split_whitespace();
    let n_genes = words.next().unwrap().parse().unwrap();
    let runs = words
        .map(|word| {
            let (run, labels) = word.split_once(':').unwrap();
            (
                run.to_owned(),
                labels.split(',').map(str::to_owned).collect(),
            )
        })
        .collect();
    Some(Fixture {
        dir,
        h5ad,
        n_genes,
        runs,
    })
}

fn read_bytes<const N: usize>(path: &Path) -> Vec<[u8; N]> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<N>()
        .0
        .to_vec()
}

fn read_f64(path: &Path) -> Vec<f64> {
    read_bytes::<8>(path)
        .into_iter()
        .map(f64::from_le_bytes)
        .collect()
}

fn read_u64(path: &Path) -> Vec<u64> {
    read_bytes::<8>(path)
        .into_iter()
        .map(u64::from_le_bytes)
        .collect()
}

fn read_bool(path: &Path) -> Vec<bool> {
    std::fs::read(path)
        .unwrap()
        .into_iter()
        .map(|byte| byte != 0)
        .collect()
}

fn counts(fixture: &Fixture) -> CellBlock {
    let indices = read_bytes::<4>(&fixture.dir.join("indices"))
        .into_iter()
        .map(u32::from_le_bytes)
        .collect();
    let data = read_bytes::<4>(&fixture.dir.join("data"))
        .into_iter()
        .map(f32::from_le_bytes)
        .collect();
    CellBlock::from_f32(
        read_u64(&fixture.dir.join("indptr")),
        indices,
        data,
        fixture.n_genes,
    )
    .unwrap()
}

fn batches(fixture: &Fixture) -> Vec<String> {
    std::fs::read_to_string(fixture.dir.join("batch"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

/// scanpy's run `run`: scores (NaN where not scored), scored, calls.
fn reference_cells(fixture: &Fixture, run: &str) -> (Vec<f64>, Vec<bool>, Vec<bool>) {
    let file = |array: &str| fixture.dir.join(format!("{run}.cells.{array}"));
    (
        read_f64(&file("doublet_score")),
        read_bool(&file("scored")),
        read_bool(&file("predicted_doublet")),
    )
}

/// The params of run `run`.
fn params(run: &str) -> Scrublet {
    Scrublet {
        batch_key: (run == "by_batch").then(|| "batch".to_owned()),
        ..Scrublet::default()
    }
}

#[test]
fn scores_match_scanpy_given_its_doublets() {
    let work = tempfile::tempdir().unwrap();
    for name in ["synthetic", "pbmc3k"] {
        let Some(fixture) = write_fixture(name, work.path()) else {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        };
        let counts = counts(&fixture);
        for (run, labels) in &fixture.runs {
            let batched = run == "by_batch";
            let batch_labels = batched.then(|| batches(&fixture));
            let file = |label: &str, array: &str| {
                fixture.dir.join(format!("{run}.batches.{label}.{array}"))
            };
            let mut given = BTreeMap::new();
            for label in labels {
                let flat = read_u64(&file(label, "doublet_parents"));
                let pairs = flat
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[first, second]| [first as u32, second as u32])
                    .collect();
                let key = if batched {
                    label.clone()
                } else {
                    String::new()
                };
                given.insert(key, pairs);
            }
            let result = scrublet(
                &counts,
                batch_labels.as_deref(),
                &params(run),
                &Doublets::Given(given),
            )
            .unwrap();
            let (scores, scored, predicted) = reference_cells(&fixture, run);
            let mut worst = 0.0_f64;
            for (cell, (&expected, &was_scored)) in scores.iter().zip(&scored).enumerate() {
                match result.scores[cell] {
                    Some(score) => {
                        assert!(was_scored, "{name} {run}: cell {cell} scored by kyto only");
                        worst = worst.max((score - expected).abs());
                    }
                    None => assert!(
                        !was_scored,
                        "{name} {run}: cell {cell} scored by scanpy only"
                    ),
                }
            }
            for batch in &result.batches {
                let label = if batched { batch.label.as_str() } else { "all" };
                let expected = read_f64(&file(label, "doublet_scores_sim"));
                assert_eq!(batch.simulated_scores.len(), expected.len());
                for (kyto, scanpy) in batch.simulated_scores.iter().zip(&expected) {
                    worst = worst.max((kyto - scanpy).abs());
                }
                let threshold = read_f64(&file(label, "threshold"))[0];
                match batch.threshold {
                    Some(found) => assert!(
                        (found - threshold).abs() <= 1e-12,
                        "{name} {run} {label}: threshold {found} vs scanpy {threshold}"
                    ),
                    None => assert!(threshold.is_nan(), "{name} {run} {label}: no threshold"),
                }
            }
            for (cell, &expected) in predicted.iter().enumerate() {
                assert_eq!(
                    result.predicted[cell].unwrap_or(false),
                    expected,
                    "{name} {run}: call of cell {cell}"
                );
            }
            println!(
                "scrublet {name} {run}: worst score difference {worst:e} over {} cells \
                 ({} batches)",
                scores.len(),
                result.batches.len()
            );
            assert!(worst <= SCORE_TOLERANCE, "{name} {run}: {worst}");
        }
    }
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// The plan's scores and calls by store table position, read from the stream's blocks
/// (the columns' operator), checked against the node's statistics.
fn run_pipeline(plan: &Arc<Plan>, params: &Scrublet, threads: usize) -> ScrubletResult {
    let members = [
        Member::CellColumn(params.score_into.clone()),
        Member::CellColumn(params.predicted_into.clone()),
    ];
    let output = Output::members(plan, members).unwrap();
    let physical = physical_plan(&[output], OptimizeOptions::ALL, PhysicalOptions::default());
    let mut stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let statistics = stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Scrublet(found) => Some((**found).clone()),
            _ => None,
        })
        .unwrap();
    let mut scores = Vec::new();
    let mut predicted = Vec::new();
    for block in stream.by_ref() {
        let block = block.unwrap();
        let score = block.columns.column(&params.score_into).unwrap();
        let call = block.columns.column(&params.predicted_into).unwrap();
        scores.extend(score.f64().unwrap().iter());
        predicted.extend(call.bool().unwrap().iter());
    }
    let result = statistics.result();
    // The stream's cells are the store's in order (no filter before the node).
    assert_eq!(statistics.cells().len(), scores.len());
    assert_eq!(scores, result.scores);
    assert_eq!(predicted, result.predicted);
    result.clone()
}

#[test]
fn seeded_calls_agree_with_scanpy_and_repeat() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let Some(fixture) = write_fixture(name, work.path()) else {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        };
        let store = work.path().join(format!("{name}.kyto"));
        let options = ConvertOptions {
            tiling: Tiling {
                cell_block_size: 256,
                gene_block_size: 4096,
            },
            ..ConvertOptions::default()
        };
        convert(&fixture.h5ad, &store, &options).unwrap();
        let base = scan(&store);
        for (run, _) in &fixture.runs {
            let params = params(run);
            let plan = base.scrublet(Some("counts"), params.clone()).unwrap();
            let first = run_pipeline(&plan, &params, 1);
            let again = run_pipeline(&plan, &params, all);
            assert_eq!(
                first, again,
                "{name} {run}: 1 and {all} threads differ (R11)"
            );
            let (_, scored, predicted) = reference_cells(&fixture, run);
            let agree = first
                .predicted
                .iter()
                .zip(&predicted)
                .filter(|(kyto, scanpy)| kyto.unwrap_or(false) == **scanpy)
                .count();
            let fraction = agree as f64 / predicted.len() as f64;
            let kyto_scored: Vec<bool> = first.scores.iter().map(Option::is_some).collect();
            assert_eq!(kyto_scored, scored, "{name} {run}: the cells scored");
            println!(
                "scrublet {name} {run}: seeded calls agree with scanpy on {agree} of {} cells \
                 ({fraction:.4}); kyto calls {}, scanpy {}",
                predicted.len(),
                first
                    .predicted
                    .iter()
                    .filter(|call| **call == Some(true))
                    .count(),
                predicted.iter().filter(|call| **call).count()
            );
            let bound = if name == "synthetic" && run == "by_batch" {
                SMALL_BATCH_AGREEMENT
            } else {
                CALL_AGREEMENT
            };
            assert!(fraction >= bound, "{name} {run}: {fraction}");
            // Another seed draws other doublets.
            let other = Scrublet {
                seed: 1,
                ..params.clone()
            };
            let moved = run_pipeline(
                &base.scrublet(Some("counts"), other.clone()).unwrap(),
                &other,
                all,
            );
            assert_ne!(moved.scores, first.scores, "{name} {run}: seed 1");
        }
    }
}
