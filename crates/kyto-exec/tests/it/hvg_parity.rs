//! Task 6.3: highly variable genes.
//!
//! Done when: the selected gene sets equal scanpy's on fixtures. The synthetic and
//! pbmc3k fixtures (written as `.h5ad` by `tests/fixtures`, converted here; skipped
//! without anndata, exit code 77) run each flavour as the stored scanpy reference
//! `hvg` (`tests/fixtures/make_reference.py`): `n_top_genes` of the manifest, on genes
//! seen in at least `min_cells_per_gene` cells (`select_genes`, pbmc3k), `seurat` and
//! `cell_ranger` on `normalize_total(1e4) -> log1p`, `seurat_v3` on the counts; with
//! `batch_key="batch"` too on synthetic (pbmc3k has no batches). Every gene column
//! kyto writes is compared with scanpy's: `highly_variable`, the ranks (identical
//! genes' ties aside, see `untie_ranks`) and the batch counts exactly, the statistics to 1e-6 (relative; absolute for the normalised
//! dispersions and variances, which are centred on 0), measured and printed.
//! Also the same bits at 1 and all threads and at blocks of 37 cells (R11).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{ExecuteOptions, Pipeline};
use kyto_plan::{
    HvgFlavor, Member, OptimizeOptions, Output, PhysicalOptions, Plan, hvg_columns, physical_plan,
};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};
use polars::prelude::{DataFrame, DataType};

/// Writes `load_fixture(name)` as `.h5ad` and the `hvg` reference's arrays as raw
/// little-endian files (`f64` floats, `i64` ints, `u8` booleans, gene IDs one per
/// line), then prints `n_top_genes min_cells_per_gene target_sum`; exits 77 without
/// anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import numpy as np
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
to_anndata(load_fixture(name)).write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
for key, array in load_reference(name, 'hvg').items():
    path = f\"{out}/{key.replace('/', '.')}\"
    kind = array.dtype.kind
    if kind == 'U':
        open(path, 'w').write('\\n'.join(array.tolist()))
    elif kind == 'b':
        np.asarray(array, dtype='u1').tofile(path)
    elif kind == 'f':
        np.asarray(array, dtype='<f8').tofile(path)
    else:
        np.asarray(array, dtype='<i8').tofile(path)
parameters = manifest[name + '/hvg']['parameters']
target = manifest[name + '/normalize_log1p']['parameters']['target_sum']
from fixtures.make_reference import _filtered_counts
adata = _filtered_counts(name)
if 'batch' in adata.obs.columns:
    # Per gene, the widest tie of normalised variance inside one batch among the
    # ranks below n_top_genes: numpy orders those ties unstably.
    import scanpy as sc
    span = np.zeros(adata.n_vars, dtype='<i8')
    labels = adata.obs['batch'].astype(str)
    for label in sorted(labels.unique()):
        part = adata[(labels == label).to_numpy()].copy()
        result = sc.pp.highly_variable_genes(
            part, flavor='seurat_v3', n_top_genes=parameters['n_top_genes'], inplace=False)
        norm = result['variances_norm'].to_numpy()
        rank = np.argsort(np.argsort(-norm))
        values, counts = np.unique(norm, return_counts=True)
        width = dict(zip(values.tolist(), (counts - 1).tolist()))
        for gene in range(adata.n_vars):
            if rank[gene] < parameters['n_top_genes']:
                span[gene] = max(span[gene], width[norm[gene]])
    span.tofile(f'{out}/seurat_v3_by_batch.tie_span')
print(parameters['n_top_genes'], parameters['min_cells_per_gene'], target)
";

/// Write fixture `name` to `h5ad` and its reference to `out`; its `n_top_genes`,
/// `min_cells_per_gene` and `target_sum`, or `None` without anndata.
fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> Option<(u64, u64, f64)> {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests");
    let output = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .current_dir(tests)
        .args(["-c", WRITE_FIXTURE, name])
        .arg(h5ad)
        .arg(out)
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
    Some((
        words.next().unwrap().parse().unwrap(),
        words.next().unwrap().parse().unwrap(),
        words.next().unwrap().parse().unwrap(),
    ))
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

/// kyto's `seurat_v3` ranks with ties resolved as scanpy's: genes of exactly equal
/// normalised variance (scanpy's `variances_norm`, identical genes) share their ranks
/// in an order numpy's unstable `argsort` picks (and that differs across platforms,
/// memory/fixture_references.md); kyto ranks them in gene order. Within each such
/// group the multisets of ranks must be equal; scanpy's ranks are then taken.
fn untie_ranks(kyto: &[f64], scanpy: &[f64], variances_norm: &[f64]) -> Vec<f64> {
    let mut groups: std::collections::BTreeMap<u64, Vec<usize>> = std::collections::BTreeMap::new();
    for (gene, value) in variances_norm.iter().enumerate() {
        groups.entry(value.to_bits()).or_default().push(gene);
    }
    let mut untied = kyto.to_vec();
    for genes in groups.values().filter(|genes| genes.len() > 1) {
        let sorted = |ranks: &[f64]| -> Vec<u64> {
            let mut bits: Vec<u64> = genes.iter().map(|&gene| ranks[gene].to_bits()).collect();
            bits.sort_unstable();
            bits
        };
        if sorted(kyto) == sorted(scanpy) {
            for &gene in genes {
                untied[gene] = scanpy[gene];
            }
        }
    }
    untied
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

/// The HVG plan of `flavor` over `base` (counts, the reference's genes).
fn hvg_plan(
    base: &Arc<Plan>,
    flavor: HvgFlavor,
    n_top_genes: u64,
    batch_key: Option<&str>,
    target_sum: f64,
) -> Arc<Plan> {
    let layer = if flavor == HvgFlavor::SeuratV3 {
        Arc::clone(base)
    } else {
        base.normalize_total(Some(target_sum), Some("counts"), "normalized")
            .unwrap()
            .log1p(Some("normalized"), "log1p")
            .unwrap()
    };
    let input = if flavor == HvgFlavor::SeuratV3 {
        "counts"
    } else {
        "log1p"
    };
    layer
        .highly_variable_genes(Some(input), flavor, Some(n_top_genes), batch_key)
        .unwrap()
}

/// The gene table after running `plan` (its HVG columns among them).
fn run(
    plan: &Arc<Plan>,
    columns: &[String],
    threads: usize,
    block_cells: Option<u64>,
) -> DataFrame {
    let output = Output::members(
        plan,
        columns.iter().map(|name| Member::GeneColumn(name.clone())),
    )
    .unwrap();
    let options = PhysicalOptions {
        max_block_cells: block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    let physical = physical_plan(std::slice::from_ref(&output), OptimizeOptions::ALL, options);
    let mut stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))
        .unwrap();
    let genes = stream
        .genes()
        .expect("HVG writes gene columns")
        .table()
        .clone();
    for block in stream.by_ref() {
        block.unwrap();
    }
    genes
}

/// `kyto`'s column `name` as `f64` (NaN for null).
fn floats(genes: &DataFrame, name: &str) -> Vec<f64> {
    let column = genes
        .column(name)
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    column
        .f64()
        .unwrap()
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect()
}

/// Compare kyto's column with scanpy's: the worst difference of a statistic, relative
/// (absolute for the normalised columns, centred on 0: `absolute`), and the first
/// gene out of tolerance (1e-6; `exact` columns must be equal, NaN equal to NaN).
fn compare(kyto: &[f64], scanpy: &[f64], exact: bool, absolute: bool) -> (f64, Option<String>) {
    assert_eq!(kyto.len(), scanpy.len(), "length");
    let mut worst = 0.0_f64;
    let mut failure = None;
    for (gene, (&kyto, &scanpy)) in kyto.iter().zip(scanpy).enumerate() {
        if kyto.is_nan() && scanpy.is_nan() || kyto == scanpy {
            continue;
        }
        let scale = if absolute {
            1.0
        } else {
            scanpy.abs().max(f64::MIN_POSITIVE)
        };
        let error = (kyto - scanpy).abs() / scale;
        if (exact || error > 1e-6) && failure.is_none() {
            failure = Some(format!("gene {gene}: kyto {kyto} vs scanpy {scanpy}"));
        }
        worst = worst.max(error);
    }
    (worst, failure)
}

#[test]
fn highly_variable_genes_match_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let mut failures = Vec::new();
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let reference = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&reference).unwrap();
        let Some((n_top_genes, min_cells, target_sum)) = write_fixture(name, &h5ad, &reference)
        else {
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
        convert(&h5ad, &store, &options).unwrap();
        let gene_ids: Vec<String> = std::fs::read_to_string(reference.join("genes.gene_id"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        let mut base = scan(&store);
        if min_cells > 0 {
            base = base.select_genes(gene_ids.clone()).unwrap();
        }
        let batched = name == "synthetic";
        for flavor in [
            HvgFlavor::Seurat,
            HvgFlavor::CellRanger,
            HvgFlavor::SeuratV3,
        ] {
            for batch_key in [None, Some("batch")] {
                if batch_key.is_some() && !batched {
                    continue;
                }
                let label = match batch_key {
                    None => flavor.name().to_owned(),
                    Some(key) => format!("{}_by_{key}", flavor.name()),
                };
                let plan = hvg_plan(&base, flavor, n_top_genes, batch_key, target_sum);
                let columns: Vec<String> = hvg_columns(flavor, batch_key.is_some())
                    .into_iter()
                    .map(|column| column.name)
                    .collect();
                let genes = run(&plan, &columns, 1, None);
                let ids = genes.column("gene_id").unwrap().str().unwrap().clone();
                assert!(
                    ids.iter()
                        .map(Option::unwrap)
                        .eq(gene_ids.iter().map(String::as_str)),
                    "{name} {label}: genes"
                );
                let mut report = Vec::new();
                for column in &columns {
                    let file = reference.join(format!("{label}.{column}"));
                    let scanpy: Vec<f64> = match column.as_str() {
                        "highly_variable" | "highly_variable_intersection" => {
                            read_bytes::<1>(&file)
                                .into_iter()
                                .map(|[byte]| f64::from(byte))
                                .collect()
                        }
                        "highly_variable_nbatches" => read_bytes::<8>(&file)
                            .into_iter()
                            .map(|bytes| i64::from_le_bytes(bytes) as f64)
                            .collect(),
                        _ => read_f64(&file),
                    };
                    let exact = matches!(
                        column.as_str(),
                        "highly_variable"
                            | "highly_variable_intersection"
                            | "highly_variable_nbatches"
                            | "highly_variable_rank"
                    );
                    let mut kyto = floats(&genes, column);
                    if column == "highly_variable_rank" {
                        let variances_norm =
                            read_f64(&reference.join(format!("{label}.variances_norm")));
                        kyto = untie_ranks(&kyto, &scanpy, &variances_norm);
                        if batch_key.is_some() {
                            // Ties inside one batch (equal normalised variance there, not
                            // overall) reach the median rank: within the widest such tie
                            // the gene is in, numpy's order is taken.
                            let span =
                                read_bytes::<8>(&reference.join("seurat_v3_by_batch.tie_span"));
                            for (gene, bytes) in span.into_iter().enumerate() {
                                let span = i64::from_le_bytes(bytes) as f64;
                                if (kyto[gene] - scanpy[gene]).abs() <= span {
                                    kyto[gene] = scanpy[gene];
                                }
                            }
                        }
                    }
                    let absolute = column.ends_with("_norm");
                    let (worst, failure) = compare(&kyto, &scanpy, exact, absolute);
                    if let Some(failure) = failure {
                        failures.push(format!("{name} {label} {column}: {failure}"));
                    }
                    if !exact {
                        report.push(format!("{column} {worst:.1e}"));
                    }
                }
                let flagged = floats(&genes, "highly_variable")
                    .iter()
                    .filter(|&&flag| flag == 1.0)
                    .count();
                eprintln!(
                    "{name} {label}: {flagged} genes flagged; worst difference: {}",
                    report.join(", ")
                );
                // R11: the same bits on all threads and at smaller blocks (small blocks
                // on synthetic only, pbmc3k's 2700 cells in 37-cell blocks is slow).
                let reruns: &[(usize, Option<u64>)] = if batched {
                    &[(all, None), (all, Some(37))]
                } else {
                    &[(all, None)]
                };
                for &(threads, block_cells) in reruns {
                    let other = run(&plan, &columns, threads, block_cells);
                    for column in &columns {
                        let bits = |frame: &DataFrame| -> Vec<u64> {
                            floats(frame, column)
                                .iter()
                                .map(|value| value.to_bits())
                                .collect()
                        };
                        assert_eq!(
                            bits(&genes),
                            bits(&other),
                            "{name} {label} {column}: {threads} threads, blocks {block_cells:?}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "differences from scanpy:\n{}",
        failures.join("\n")
    );
}
