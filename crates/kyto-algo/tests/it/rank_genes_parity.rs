//! Task 6.10: differential expression.
//!
//! Done when: matches `scanpy.tl.rank_genes_groups` to 1e-6. The synthetic and pbmc3k
//! fixtures (written as `.h5ad` by `tests/fixtures`, converted here; skipped without
//! anndata, exit code 77) run `normalize_total(1e4) -> log1p` on the genes seen in at
//! least `min_cells_per_gene` cells, then `rank_genes_groups`:
//! - against the stored 1.5 reference (synthetic, `cell_type`, t-test and Wilcoxon
//!   against the rest; numpy's `float32` `log1p`, 1 ulp off kyto's correctly rounded
//!   one on some values, deviations.md);
//! - against scanpy run live on kyto's input (scanpy's steps with a correctly rounded
//!   `log1p`) for every variant the helper lists: both tests, against the rest and a
//!   reference, a `groups` subset (its rest holds the other groups), `tie_correct`,
//!   Bonferroni, `n_genes` and `rankby_abs`; on pbmc3k a `group` column of total-count
//!   quartiles (categories in reverse order) with some cells null (in no group, but in
//!   every rest), and Wilcoxon over several passes (`max_rank_values` below the
//!   layer's values).
//!
//! R11 reruns cover every synthetic variant and pbmc3k's two main ones (debug runs
//! take seconds).
//!
//! Every (group, gene) row is compared: statistic and log fold change (scanpy keeps
//! `float32`) to 1e-6 relative or 1e-6 absolute, p-values and adjusted p-values to 1e-6
//! relative; each group's genes in the same order wherever scanpy's statistics are not
//! tied (scanpy's order of ties is its sort's, deviations.md). Also the same frame at 1
//! and all threads and at blocks of 37 cells (R11), and Wilcoxon in one pass or many.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_algo::{
    DifferentialTest, PValueCorrection, RANK_GENES_COLUMNS, RankGenesGroups, rank_genes_groups,
};
use kyto_core::{GeneIds, Tiling};
use kyto_exec::ExecuteOptions;
use kyto_plan::{PhysicalOptions, Plan};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};
use polars::prelude::DataFrame;

/// Writes `load_fixture(name)` as `.h5ad` (pbmc3k with a `group` column), the filtered
/// gene IDs, and scanpy's `rank_genes_groups_df` for each variant as text
/// (`<variant>.tsv`: group, gene, score, logfoldchanges, pvals, pvals_adj; synthetic
/// also `reference.<method>.tsv` from the stored reference); prints
/// `min_cells_per_gene target_sum` then one line per variant:
/// `name method groupby groups reference tie_correct corr_method n_genes rankby_abs
/// max_rank_values` (`-` for none). Exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import numpy as np
import pandas as pd
import scanpy as sc
from fixtures import make_reference
from fixtures.make_reference import _filtered_counts
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
adata = to_anndata(load_fixture(name))
def quartile_groups(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    edges = np.quantile(totals, [0.25, 0.5, 0.75])
    labels = np.array(['q0', 'q1', 'q2', 'q3'], dtype=object)[np.searchsorted(edges, totals)]
    labels[::97] = None
    # Categories not sorted: groups come in category order, as scanpy's.
    return pd.Categorical(labels, categories=['q3', 'q2', 'q1', 'q0'])
if name == 'pbmc3k':
    adata.obs['group'] = quartile_groups(adata.X)
adata.write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
target = manifest[name + '/normalize_log1p']['parameters']['target_sum']
min_cells = manifest[name + '/hvg']['parameters']['min_cells_per_gene']
open(f'{out}/filtered', 'w').write('\\n'.join(_filtered_counts(name).var_names.tolist()))
def write(frame, path):
    with open(path, 'w') as file:
        for row in frame.itertuples(index=False):
            file.write(f'{row.group}\\t{row.names}\\t{float(row.scores)!r}\\t'
                       f'{float(row.logfoldchanges)!r}\\t{float(row.pvals)!r}\\t{float(row.pvals_adj)!r}\\n')
if name == 'synthetic':
    reference = load_reference(name, 'rank_genes')
    for method in ('t-test', 'wilcoxon'):
        frame = pd.DataFrame({column: reference[f'{method}/{column}'] for column in
                              ('group', 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj')})
        write(frame, f'{out}/reference.{method}.tsv')
# kyto's input: numpy's float32 log1p is 1 ulp off on some values, kyto's is correctly
# rounded (deviations.md, task 6.2); scanpy on that input.
log1p = sc.pp.log1p
def rounded_log1p(adata, **kwargs):
    adata.X.data = np.log1p(adata.X.data.astype(np.float64)).astype(np.float32)
sc.pp.log1p = rounded_log1p
try:
    exact = make_reference._log_normalized(name)
finally:
    sc.pp.log1p = log1p
if name == 'synthetic':
    groupby, first, second, third = 'cell_type', 'type_0', 'type_1', 'type_2'
    variants = [
        ('t-test', 't-test', '-', '-', False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon', 'wilcoxon', '-', '-', False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon.tie', 'wilcoxon', '-', '-', True, 'benjamini-hochberg', '-', False, '-'),
        ('t-test.reference', 't-test', f'{third},{first}', second, False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon.reference', 'wilcoxon', f'{third},{first}', second, False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon.reference.tie', 'wilcoxon', '-', second, True, 'bonferroni', '-', False, '-'),
        ('wilcoxon.subset', 'wilcoxon', f'type_3,{first}', '-', False, 'benjamini-hochberg', '-', False, '-'),
        ('t-test.subset', 't-test', f'type_3,{first}', '-', False, 'bonferroni', '25', True, '-'),
        ('wilcoxon.passes', 'wilcoxon', '-', '-', False, 'benjamini-hochberg', '-', False, '2000'),
    ]
else:
    exact.obs['group'] = adata.obs['group'].array
    groupby = 'group'
    variants = [
        ('t-test', 't-test', '-', '-', False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon', 'wilcoxon', '-', '-', False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon.passes', 'wilcoxon', '-', '-', True, 'benjamini-hochberg', '-', False, '1000000'),
        ('t-test.reference', 't-test', '-', 'q0', False, 'benjamini-hochberg', '-', False, '-'),
        ('wilcoxon.reference', 'wilcoxon', 'q3,q1', 'q0', False, 'benjamini-hochberg', '-', False, '-'),
    ]
print(min_cells, target)
for variant, method, groups, reference, tie, corr, n_genes, rankby_abs, passes in variants:
    data = exact.copy()
    sc.tl.rank_genes_groups(
        data, groupby=groupby, method=method,
        groups='all' if groups == '-' else groups.split(','),
        reference='rest' if reference == '-' else reference,
        tie_correct=tie, corr_method=corr,
        n_genes=data.n_vars if n_genes == '-' else int(n_genes), rankby_abs=rankby_abs,
    )
    write(sc.get.rank_genes_groups_df(data, group=None), f'{out}/{variant}.tsv')
    print(variant, method, groupby, groups, reference, tie, corr, n_genes, rankby_abs, passes)
";

/// One scanpy run the helper wrote, and the parameters for kyto's.
struct Variant {
    name: String,
    parameters: RankGenesGroups,
}

/// Write fixture `name` to `h5ad` and scanpy's frames to `out`; its
/// `min_cells_per_gene`, `target_sum` and variants, or `None` without anndata.
fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> Option<(u64, f64, Vec<Variant>)> {
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
        "writing fixture {name} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut lines = text.lines();
    let mut first = lines.next().unwrap().split_whitespace();
    let min_cells = first.next().unwrap().parse().unwrap();
    let target_sum = first.next().unwrap().parse().unwrap();
    let variants = lines
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let [
                name,
                method,
                groupby,
                groups,
                reference,
                tie,
                correction,
                n_genes,
                rankby_abs,
                passes,
            ] = words[..]
            else {
                panic!("variant line {line}")
            };
            let method = match method {
                "t-test" => DifferentialTest::TTest,
                "wilcoxon" => DifferentialTest::Wilcoxon,
                other => panic!("method {other}"),
            };
            let mut parameters = RankGenesGroups::new(groupby, method);
            parameters.layer = Some("log1p".to_owned());
            parameters.groups =
                (groups != "-").then(|| groups.split(',').map(str::to_owned).collect());
            parameters.reference = (reference != "-").then(|| reference.to_owned());
            parameters.tie_correct = tie == "True";
            parameters.correction = match correction {
                "benjamini-hochberg" => PValueCorrection::BenjaminiHochberg,
                "bonferroni" => PValueCorrection::Bonferroni,
                other => panic!("correction {other}"),
            };
            parameters.n_genes = (n_genes != "-").then(|| n_genes.parse().unwrap());
            parameters.rankby_abs = rankby_abs == "True";
            if passes != "-" {
                parameters.max_rank_values = passes.parse().unwrap();
            }
            Variant {
                name: name.to_owned(),
                parameters,
            }
        })
        .collect();
    Some((min_cells, target_sum, variants))
}

/// One row of scanpy's `rank_genes_groups_df`.
#[derive(Debug, Clone)]
struct Row {
    group: String,
    gene: String,
    values: [f64; 4],
}

fn read_scanpy(path: &Path) -> Vec<Row> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            Row {
                group: fields[0].to_owned(),
                gene: fields[1].to_owned(),
                values: std::array::from_fn(|index| fields[2 + index].parse().unwrap()),
            }
        })
        .collect()
}

fn read_kyto(frame: &DataFrame) -> Vec<Row> {
    let names: Vec<&str> = frame
        .get_column_names()
        .iter()
        .map(|name| name.as_str())
        .collect();
    assert_eq!(names, RANK_GENES_COLUMNS);
    let text = |name: &str| -> Vec<String> {
        frame
            .column(name)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|value| value.unwrap().to_owned())
            .collect()
    };
    let float = |name: &str| -> Vec<f64> {
        frame
            .column(name)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(Option::unwrap)
            .collect()
    };
    let groups = text("group");
    let genes = text("gene_id");
    let columns: [Vec<f64>; 4] = std::array::from_fn(|index| float(RANK_GENES_COLUMNS[2 + index]));
    (0..frame.height())
        .map(|row| Row {
            group: groups[row].clone(),
            gene: genes[row].clone(),
            values: std::array::from_fn(|index| columns[index][row]),
        })
        .collect()
}

/// The worst differences of kyto's rows from scanpy's: statistic and log fold change
/// (as scanpy's `float32`), p-value and adjusted p-value; asserts the rows are the same
/// (group, gene) pairs with each group's genes in scanpy's order where scanpy's
/// statistics are not tied.
fn compare(label: &str, kyto: &[Row], scanpy: &[Row]) -> [f64; 4] {
    assert_eq!(kyto.len(), scanpy.len(), "{label}: rows");
    let positions: HashMap<(&str, &str), usize> = scanpy
        .iter()
        .enumerate()
        .map(|(position, row)| ((row.group.as_str(), row.gene.as_str()), position))
        .collect();
    let mut worst = [0.0_f64; 4];
    for (position, row) in kyto.iter().enumerate() {
        let other = positions
            .get(&(row.group.as_str(), row.gene.as_str()))
            .map(|position| &scanpy[*position])
            .unwrap_or_else(|| panic!("{label}: {} {} not in scanpy's", row.group, row.gene));
        assert_eq!(row.group, scanpy[position].group, "{label}: group order");
        if row.gene != scanpy[position].gene {
            assert_eq!(
                other.values[0] as f32, scanpy[position].values[0] as f32,
                "{label}: {} gene {position} is {}, scanpy's {} (not a tie)",
                row.group, row.gene, scanpy[position].gene
            );
        }
        for (index, worst) in worst.iter_mut().enumerate() {
            let (kyto, scanpy) = (row.values[index], other.values[index]);
            let kyto = if index < 2 {
                f64::from(kyto as f32)
            } else {
                kyto
            };
            let difference = if index < 2 {
                (kyto - scanpy).abs() / scanpy.abs().max(1.0)
            } else {
                (kyto - scanpy).abs() / scanpy.abs().max(f64::MIN_POSITIVE)
            };
            let difference = if kyto == scanpy { 0.0 } else { difference };
            *worst = worst.max(difference);
        }
    }
    worst
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

fn run(
    plan: &Arc<Plan>,
    parameters: &RankGenesGroups,
    threads: usize,
    block_cells: Option<u64>,
) -> DataFrame {
    let physical = PhysicalOptions {
        max_block_cells: block_cells.and_then(NonZeroU64::new),
        ..PhysicalOptions::default()
    };
    rank_genes_groups(
        plan,
        parameters,
        physical,
        &ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()),
    )
    .unwrap()
}

#[test]
fn rank_genes_groups_matches_scanpy_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let out = work.path().join(format!("{name}-scanpy"));
        std::fs::create_dir(&out).unwrap();
        let Some((min_cells, target_sum, variants)) = write_fixture(name, &h5ad, &out) else {
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
        let mut base = scan(&store);
        if min_cells > 0 {
            let filtered = std::fs::read_to_string(out.join("filtered")).unwrap();
            base = base
                .select_genes(filtered.lines().map(str::to_owned))
                .unwrap();
        }
        let plan = base
            .normalize_total(Some(target_sum), Some("counts"), "normalized")
            .unwrap()
            .log1p(Some("normalized"), "log1p")
            .unwrap();
        let tolerance = 1e-6;
        if name == "synthetic" {
            for (method, test) in [
                ("t-test", DifferentialTest::TTest),
                ("wilcoxon", DifferentialTest::Wilcoxon),
            ] {
                let mut parameters = RankGenesGroups::new("cell_type", test);
                parameters.layer = Some("log1p".to_owned());
                let kyto = read_kyto(&run(&plan, &parameters, 1, None));
                let scanpy = read_scanpy(&out.join(format!("reference.{method}.tsv")));
                let worst = compare(method, &kyto, &scanpy);
                eprintln!(
                    "{name} {method}, stored reference: worst statistic {:.1e}, log fold \
                     change {:.1e}, p {:.1e}, adjusted p {:.1e}",
                    worst[0], worst[1], worst[2], worst[3]
                );
                assert!(
                    worst.iter().all(|worst| *worst <= tolerance),
                    "{method}: {worst:?}"
                );
            }
        }
        for variant in &variants {
            let label = format!("{name} {}", variant.name);
            let frame = run(&plan, &variant.parameters, 1, None);
            let kyto = read_kyto(&frame);
            let scanpy = read_scanpy(&out.join(format!("{}.tsv", variant.name)));
            let worst = compare(&label, &kyto, &scanpy);
            eprintln!(
                "{label}, scanpy on kyto's log1p ({} rows): worst statistic {:.1e}, log \
                 fold change {:.1e}, p {:.1e}, adjusted p {:.1e}",
                kyto.len(),
                worst[0],
                worst[1],
                worst[2],
                worst[3]
            );
            assert!(
                worst.iter().all(|worst| *worst <= tolerance),
                "{label}: {worst:?}"
            );
            // R11 on every synthetic variant, on pbmc3k's first two (a debug run there
            // takes seconds).
            let repeats =
                if name == "synthetic" || matches!(variant.name.as_str(), "t-test" | "wilcoxon") {
                    &[(all, None), (all, Some(37))][..]
                } else {
                    &[]
                };
            for &(threads, block_cells) in repeats {
                let again = run(&plan, &variant.parameters, threads, block_cells);
                assert!(
                    again.equals_missing(&frame),
                    "{label}: R11 at {threads} threads, {block_cells:?} cells"
                );
            }
            if variant.name.ends_with("passes") {
                let mut one_pass = variant.parameters.clone();
                one_pass.max_rank_values = kyto_algo::DEFAULT_MAX_RANK_VALUES;
                assert!(
                    run(&plan, &one_pass, 1, None).equals_missing(&frame),
                    "{label}: one pass and several give the same frame"
                );
            }
        }
    }
}
