//! Task 6.6: nearest neighbours against scanpy, and HNSW recall.
//!
//! The stored scanpy reference `neighbors` (`tests/fixtures/make_reference.py`:
//! `pp.neighbors(n_neighbors=15, n_pcs, metric="euclidean")` on the reference `pca`
//! embedding; fewer than 4096 cells, so scanpy's exact path: scikit-learn's brute-force
//! `KNeighborsTransformer`) is compared with kyto's exact search on the same stored
//! `X_pca` (`pca.npz`, the embedding that reference searched). Rows must hold the same
//! neighbours with distances within `float32` precision, except where scanpy's own
//! handling explains the difference (deviations.md): a tie at the row's last
//! neighbour (scanpy keeps whichever scikit-learn's heap gives; kyto the earlier
//! cell), and scanpy keeping a cell as its own neighbour when a duplicate of it comes
//! first (kyto excludes the cell by position). Measured: pbmc3k every row equal,
//! distances to 1.2e-7 relative; synthetic 490 of 500 rows equal, the other 10 its
//! duplicate cells (7 ties, 3 self loops). HNSW recall 0.99992 on pbmc3k, 1.0 on
//! synthetic; on kyto's own PCA, 0.99989 overlap with scanpy. The whole pipeline (kyto's own PCA, then `neighbors`) is compared
//! as a neighbour overlap, since kyto's PCA differs from scanpy's in the last digits.
//! Approximate search (HNSW) is compared with the exact graph; its recall is printed
//! (`--nocapture`) and must stay at or above [`RECALL_BOUND`].

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use kyto_core::{GeneIds, Tiling};
use kyto_exec::{
    ExecuteOptions, NeighborGraph, NeighborStatistics, NodeStatistics, Pipeline, nearest_neighbors,
};
use kyto_plan::{
    Member, Metric, NeighborSearch, Neighbors, OptimizeOptions, Output, PhysicalOptions, Plan,
    physical_plan,
};
use kyto_store::{ConvertOptions, TableAxis, convert, open_atlas_schema, read_ids};

/// The recall HNSW keeps on the fixtures (share of the exact neighbours it finds).
const RECALL_BOUND: f64 = 0.99;

/// Writes `load_fixture(name)` as `.h5ad`, the filtered and HVG gene IDs, the
/// reference `X_pca` as `<f4` (`embedding`) and the reference `distances` graph
/// (`indptr`, `indices` as `<u8`, `data` as `<f8`); prints `min_cells_per_gene
/// target_sum n_pcs n_neighbors width`. Exits 77 without anndata.
const WRITE_FIXTURE: &str = "\
import sys
try:
    from fixtures.interop import to_anndata
except ImportError:
    sys.exit(77)
import json
import numpy as np
from fixtures.make_reference import _filtered_counts
from fixtures.reference import load_reference
from fixtures.registry import load_fixture
name, h5ad, out = sys.argv[1:4]
to_anndata(load_fixture(name)).write_h5ad(h5ad)
manifest = json.load(open('fixtures/reference/manifest.json'))['references']
parameters = manifest[name + '/neighbors']['parameters']
target = manifest[name + '/normalize_log1p']['parameters']['target_sum']
pca = load_reference(name, 'pca')
neighbors = load_reference(name, 'neighbors')
open(f'{out}/filtered', 'w').write('\\n'.join(_filtered_counts(name).var_names.tolist()))
open(f'{out}/hvg', 'w').write('\\n'.join(pca['genes/gene_id'].tolist()))
np.asarray(pca['embedding'], dtype='<f4').tofile(f'{out}/embedding')
for key in ['indptr', 'indices']:
    np.asarray(neighbors['distances/' + key], dtype='<u8').tofile(f'{out}/{key}')
np.asarray(neighbors['distances/data'], dtype='<f8').tofile(f'{out}/data')
print(parameters['min_cells_per_gene'], target, parameters['n_pcs'],
      parameters['n_neighbors'], pca['embedding'].shape[1])
";

/// What `WRITE_FIXTURE` prints.
struct Fixture {
    min_cells: u64,
    target_sum: f64,
    n_pcs: u64,
    n_neighbors: u64,
    width: usize,
}

fn write_fixture(name: &str, h5ad: &Path, out: &Path) -> Option<Fixture> {
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
    let words: Vec<&str> = text.split_whitespace().collect();
    Some(Fixture {
        min_cells: words[0].parse().unwrap(),
        target_sum: words[1].parse().unwrap(),
        n_pcs: words[2].parse().unwrap(),
        n_neighbors: words[3].parse().unwrap(),
        width: words[4].parse().unwrap(),
    })
}

fn read<const N: usize, T>(path: &Path, from: fn([u8; N]) -> T) -> Vec<T> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<N>()
        .0
        .iter()
        .map(|bytes| from(*bytes))
        .collect()
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

/// A graph as rows of (neighbour, distance).
type Rows = Vec<Vec<(u64, f64)>>;

fn reference_rows(out: &Path) -> Rows {
    let indptr = read(&out.join("indptr"), u64::from_le_bytes);
    let indices = read(&out.join("indices"), u64::from_le_bytes);
    let data = read(&out.join("data"), f64::from_le_bytes);
    indptr
        .windows(2)
        .map(|bounds| {
            (bounds[0] as usize..bounds[1] as usize)
                .map(|entry| (indices[entry], data[entry]))
                .collect()
        })
        .collect()
}

fn kyto_rows(graph: &NeighborGraph) -> Rows {
    graph
        .indptr()
        .windows(2)
        .map(|bounds| {
            (bounds[0] as usize..bounds[1] as usize)
                .map(|entry| {
                    (
                        u64::from(graph.indices()[entry]),
                        f64::from(graph.distances()[entry]),
                    )
                })
                .collect()
        })
        .collect()
}

fn params(n_pcs: u64, n_neighbors: u64, search: NeighborSearch) -> Neighbors {
    Neighbors {
        embedding: "X_pca".to_owned(),
        n_components: Some(n_pcs),
        n_neighbors,
        metric: Metric::Euclidean,
        search,
        seed: 0,
        key: None,
    }
}

/// Two distances equal within `float32` precision.
fn same_distance(left: f64, right: f64) -> bool {
    (left - right).abs() <= 2e-7 * left.abs().max(right.abs()) + 1e-12
}

/// How kyto's rows differ from scanpy's: (rows equal, rows explained by a tie at the
/// last neighbour, rows explained by scanpy keeping the cell itself), the largest
/// relative distance difference on shared neighbours; panics on any other difference.
fn compare_exact(name: &str, kyto: &Rows, scanpy: &Rows) -> (usize, usize, usize, f64) {
    assert_eq!(kyto.len(), scanpy.len(), "{name}: rows");
    let (mut equal, mut ties, mut selves) = (0, 0, 0);
    let mut worst: f64 = 0.0;
    for (row, (ours, theirs)) in kyto.iter().zip(scanpy).enumerate() {
        let row_id = row as u64;
        let last = ours.iter().map(|entry| entry.1).fold(0.0, f64::max);
        let keeps_self = theirs.iter().any(|entry| entry.0 == row_id);
        let theirs: Vec<(u64, f64)> = theirs
            .iter()
            .copied()
            .filter(|entry| entry.0 != row_id)
            .collect();
        let our_ids: BTreeSet<u64> = ours.iter().map(|entry| entry.0).collect();
        let their_ids: BTreeSet<u64> = theirs.iter().map(|entry| entry.0).collect();
        for &(id, distance) in &theirs {
            if let Some(&(_, ours)) = ours.iter().find(|entry| entry.0 == id) {
                worst = worst.max((ours - distance).abs() / distance.max(1e-30));
                assert!(
                    same_distance(ours, distance),
                    "{name}: row {row} -> {id}: {ours} vs {distance}"
                );
            }
        }
        if our_ids == their_ids && !keeps_self {
            equal += 1;
            continue;
        }
        // Every neighbour only scanpy has ties kyto's last one.
        for &(id, distance) in theirs.iter().filter(|entry| !our_ids.contains(&entry.0)) {
            assert!(
                same_distance(distance, last),
                "{name}: row {row}: scanpy's {id} at {distance}, kyto's last at {last}"
            );
        }
        // Every neighbour only kyto has ties its last one, but one duplicate (distance
        // 0) where scanpy kept the cell itself.
        let mut extra_zero = 0;
        for &(id, distance) in ours.iter().filter(|entry| !their_ids.contains(&entry.0)) {
            if !same_distance(distance, last) {
                assert!(
                    keeps_self && distance == 0.0,
                    "{name}: row {row}: kyto's {id} at {distance}, last at {last}"
                );
                extra_zero += 1;
            }
        }
        assert!(extra_zero <= 1, "{name}: row {row}");
        if keeps_self {
            selves += 1;
        } else {
            ties += 1;
        }
    }
    (equal, ties, selves, worst)
}

fn recall(approximate: &Rows, exact: &Rows) -> f64 {
    let mut found = 0_usize;
    let mut total = 0_usize;
    for (ours, exact) in approximate.iter().zip(exact) {
        let exact_ids: BTreeSet<u64> = exact.iter().map(|entry| entry.0).collect();
        found += ours
            .iter()
            .filter(|entry| exact_ids.contains(&entry.0))
            .count();
        total += exact.len();
    }
    found as f64 / total as f64
}

fn in_pool<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

#[test]
fn exact_matches_scanpy_and_hnsw_recall_on_fixtures() {
    let work = tempfile::tempdir().unwrap();
    let all = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    for name in ["synthetic", "pbmc3k"] {
        let h5ad = work.path().join(format!("{name}.h5ad"));
        let out = work.path().join(format!("{name}-reference"));
        std::fs::create_dir(&out).unwrap();
        let Some(fixture) = write_fixture(name, &h5ad, &out) else {
            eprintln!("skipped: anndata is not installed on this platform");
            return;
        };
        let embedding = read(&out.join("embedding"), f32::from_le_bytes);
        let scanpy = reference_rows(&out);
        let exact_params = params(fixture.n_pcs, fixture.n_neighbors, NeighborSearch::Exact);
        let exact = in_pool(1, || {
            nearest_neighbors(&embedding, fixture.width, &exact_params)
        })
        .unwrap();
        // R11: the same graph on every thread count.
        let exact_all = in_pool(all, || {
            nearest_neighbors(&embedding, fixture.width, &exact_params)
        })
        .unwrap();
        assert_eq!(exact, exact_all, "{name}: exact at 1 and {all} threads");
        let exact_rows = kyto_rows(&exact);
        let (equal, ties, selves, worst) = compare_exact(name, &exact_rows, &scanpy);
        eprintln!(
            "{name}: exact vs scanpy: {equal} of {} rows equal, {ties} differ by a tie at \
             the last neighbour, {selves} by scanpy keeping the cell itself; distances \
             within {worst:.1e} relative",
            scanpy.len()
        );
        if name == "pbmc3k" {
            assert_eq!(equal, scanpy.len(), "{name}: every row equal");
        }

        let approximate_params = params(
            fixture.n_pcs,
            fixture.n_neighbors,
            NeighborSearch::Approximate,
        );
        let approximate = in_pool(1, || {
            nearest_neighbors(&embedding, fixture.width, &approximate_params)
        })
        .unwrap();
        let approximate_all = in_pool(all, || {
            nearest_neighbors(&embedding, fixture.width, &approximate_params)
        })
        .unwrap();
        assert_eq!(
            approximate, approximate_all,
            "{name}: HNSW at 1 and {all} threads"
        );
        let approximate_rows = kyto_rows(&approximate);
        let found = recall(&approximate_rows, &exact_rows);
        let found_scanpy = recall(&approximate_rows, &scanpy);
        eprintln!(
            "{name}: HNSW recall {found:.5} of kyto's exact neighbours, {found_scanpy:.5} \
             of scanpy's (bound {RECALL_BOUND})"
        );
        assert!(found >= RECALL_BOUND, "{name}: HNSW recall {found}");
        // Every distance HNSW writes is exact.
        for (ours, exact) in approximate_rows.iter().zip(&exact_rows) {
            for &(id, distance) in ours {
                if let Some(&(_, expected)) = exact.iter().find(|entry| entry.0 == id) {
                    assert_eq!(distance, expected);
                }
            }
        }
        // Another seed builds another index, not another result beyond the bound.
        let reseeded = nearest_neighbors(
            &embedding,
            fixture.width,
            &Neighbors {
                seed: 7,
                ..approximate_params.clone()
            },
        )
        .unwrap();
        assert!(recall(&kyto_rows(&reseeded), &exact_rows) >= RECALL_BOUND);
    }
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

#[test]
fn pipeline_neighbors_on_kytos_pca_match_scanpy() {
    let work = tempfile::tempdir().unwrap();
    let name = "pbmc3k";
    let h5ad = work.path().join(format!("{name}.h5ad"));
    let out = work.path().join(format!("{name}-reference"));
    std::fs::create_dir(&out).unwrap();
    let Some(fixture) = write_fixture(name, &h5ad, &out) else {
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
    if fixture.min_cells > 0 {
        base = base
            .select_genes(read_lines(&out.join("filtered")))
            .unwrap();
    }
    let plan = base
        .normalize_total(Some(fixture.target_sum), Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
        .select_genes(read_lines(&out.join("hvg")))
        .unwrap()
        .pca(Some("log1p"), "X_pca", fixture.n_pcs, 0)
        .unwrap()
        .neighbors(params(
            fixture.n_pcs,
            fixture.n_neighbors,
            NeighborSearch::Exact,
        ))
        .unwrap();
    let output = Output::members(&plan, [Member::Graph("distances".to_owned())]).unwrap();
    let physical = physical_plan(
        std::slice::from_ref(&output),
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let neighbors: NeighborStatistics = stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Neighbors(neighbors) => Some((**neighbors).clone()),
            _ => None,
        })
        .expect("neighbors' statistics");
    let scanpy = reference_rows(&out);
    assert_eq!(neighbors.cells().len(), scanpy.len());
    assert!(neighbors.cells().windows(2).all(|pair| pair[0] < pair[1]));
    let graph = neighbors.distances();
    assert_eq!(graph.n_neighbors(), fixture.n_neighbors as usize);
    let overlap = recall(&kyto_rows(graph), &scanpy);
    eprintln!("{name}: neighbours on kyto's PCA, overlap with scanpy's {overlap:.5}");
    assert!(overlap >= 0.995, "overlap {overlap}");
}
