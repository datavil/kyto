//! Task 6.5: `pca` in the executor, against the exact SVD of the centred matrix, on
//! stores written for the test (scanpy parity is `pca_parity.rs`).
//!
//! Checked: with fewer genes than the Krylov basis can hold, the result is the exact
//! one (variances, ratios, loadings with scikit-learn's signs, embedding); with more,
//! the leading components of data with a spectral gap match it closely; `uint32`
//! counts are read as numbers; cell filters before the node select the embedding's rows;
//! components beyond the data's rank have zero variance and orthonormal loadings; too
//! few cells, too many components for the cells reaching the node, and a value that is
//! not finite are `KY-E215` (nothing panics). Task 6.14: `pca` over a gene set gives the
//! bits of a gene filter before it, with every gene kept and zero loadings outside the
//! set; an empty set, or one with fewer genes than components, is `KY-E215`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use faer::Mat;
use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Literal, Tiling};
use kyto_error::KytoError;
use kyto_exec::{ExecuteOptions, NodeStatistics, PcaStatistics, Pipeline};
use kyto_plan::{Member, OptimizeOptions, Output, PhysicalOptions, Plan, physical_plan};
use kyto_store::{
    CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype, open_atlas_schema,
    read_ids,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

/// A small deterministic generator (memory/test_seed_isolation.md).
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// `n_cells x n_genes` dense values of rank `rank` plus `noise`, negative values set to 0
/// (sparse, as log data): scores times loadings with decaying scales.
pub(crate) fn data(
    n_cells: usize,
    n_genes: usize,
    rank: usize,
    noise: f64,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut random = Lcg(seed);
    let scores: Vec<Vec<f64>> = (0..n_cells)
        .map(|_| (0..rank).map(|_| random.next() * 2.0 - 1.0).collect())
        .collect();
    let loadings: Vec<Vec<f64>> = (0..rank)
        .map(|component| {
            let scale = 4.0 / (1.0 + component as f64);
            (0..n_genes)
                .map(|_| (random.next() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect();
    scores
        .iter()
        .map(|score| {
            (0..n_genes)
                .map(|gene| {
                    let value: f64 = score
                        .iter()
                        .zip(&loadings)
                        .map(|(score, loading)| score * loading[gene])
                        .sum::<f64>()
                        + noise * (random.next() * 2.0 - 1.0);
                    value.max(0.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Write `rows` as layer `values` (`float32`, or `uint32` rounded when `counts`) of a
/// store with cell column `score` = the cell's position and the boolean gene column
/// `selected` (false for every third gene, null for gene 1).
pub(crate) fn write_store(target: &Path, rows: &[Vec<f32>], counts: bool) {
    let n_genes = rows[0].len();
    let gene_ids: Vec<String> = (0..n_genes).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..rows.len()).map(|cell| format!("c{cell}")).collect();
    let scores: Vec<f64> = (0..rows.len()).map(|cell| cell as f64).collect();
    let selected: Vec<Option<bool>> = (0..n_genes)
        .map(|gene| (gene != 1).then_some(gene % 3 != 0))
        .collect();
    let genes = DataFrame::new(
        n_genes,
        vec![
            Series::new("gene_id".into(), gene_ids).into_column(),
            Series::new("selected".into(), selected).into_column(),
        ],
    )
    .unwrap();
    let cells = DataFrame::new(
        rows.len(),
        vec![
            Series::new("cell_id".into(), cell_ids).into_column(),
            Series::new("score".into(), scores).into_column(),
        ],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "values".to_owned(),
        dtype: if counts {
            ValueDtype::U32
        } else {
            ValueDtype::F32
        },
        options: LayerWriteOptions::default(),
    }];
    let tiling = Tiling {
        cell_block_size: 32,
        gene_block_size: 64,
    };
    let mut writer = StoreWriter::create(target, &genes, &specs, tiling, false).unwrap();
    let mut indptr = vec![0_u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for row in rows {
        for (gene, &value) in row.iter().enumerate() {
            if value != 0.0 {
                indices.push(gene as u32);
                data.push(value);
            }
        }
        indptr.push(indices.len() as u64);
    }
    let block = if counts {
        let data = data.iter().map(|&value| value as u32).collect();
        CellBlock::from_u32(indptr, indices, data, n_genes as u64).unwrap()
    } else {
        CellBlock::from_f32(indptr, indices, data, n_genes as u64).unwrap()
    };
    writer.push(&[block]).unwrap();
    writer.finish(&cells, None).unwrap();
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

fn run(plan: &Arc<Plan>, threads: usize) -> Result<PcaStatistics, KytoError> {
    let output = Output::members(plan, [Member::Embedding("X_pca".to_owned())]).unwrap();
    let physical = physical_plan(
        std::slice::from_ref(&output),
        OptimizeOptions::ALL,
        PhysicalOptions::default(),
    );
    let stream = Pipeline::new(&physical, 0)
        .unwrap()
        .run(ExecuteOptions::new(NonZeroUsize::new(threads).unwrap()))?;
    Ok(stream
        .statistics()
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Pca(pca) => Some((**pca).clone()),
            _ => None,
        })
        .expect("pca's statistics"))
}

/// The exact PCA of `rows` (as `f64`): variances, ratios, loadings (genes x k,
/// row-major, scikit-learn's signs) and embedding (cells x k).
struct Exact {
    variance: Vec<f64>,
    ratio: Vec<f64>,
    loadings: Vec<f64>,
    embedding: Vec<f64>,
}

fn exact(rows: &[Vec<f32>], k: usize) -> Exact {
    let (n, genes) = (rows.len(), rows[0].len());
    let means: Vec<f64> = (0..genes)
        .map(|gene| rows.iter().map(|row| f64::from(row[gene])).sum::<f64>() / n as f64)
        .collect();
    let centred = Mat::from_fn(n, genes, |cell, gene| {
        f64::from(rows[cell][gene]) - means[gene]
    });
    let svd = centred.thin_svd().unwrap();
    let singular = svd.S().column_vector();
    let total: f64 = (0..genes)
        .map(|gene| {
            (0..n)
                .map(|cell| centred[(cell, gene)].powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let variance: Vec<f64> = (0..k)
        .map(|component| singular[component].powi(2) / (n - 1) as f64)
        .collect();
    let mut loadings = vec![0.0; genes * k];
    for component in 0..k {
        let v = svd.V().col(component);
        let mut largest = 0.0_f64;
        let mut sign = 1.0;
        for gene in 0..genes {
            if v[gene].abs() > largest {
                largest = v[gene].abs();
                sign = v[gene].signum();
            }
        }
        for gene in 0..genes {
            loadings[gene * k + component] = sign * v[gene];
        }
    }
    let mut embedding = vec![0.0; n * k];
    for cell in 0..n {
        for component in 0..k {
            embedding[cell * k + component] = (0..genes)
                .map(|gene| centred[(cell, gene)] * loadings[gene * k + component])
                .sum();
        }
    }
    Exact {
        ratio: variance.iter().map(|value| value / total).collect(),
        variance,
        loadings,
        embedding,
    }
}

fn largest_difference(kyto: impl IntoIterator<Item = f64>, exact: &[f64]) -> f64 {
    let scale = exact.iter().map(|value| value.abs()).fold(0.0, f64::max);
    kyto.into_iter()
        .zip(exact)
        .map(|(kyto, exact)| (kyto - exact).abs())
        .fold(0.0, f64::max)
        / scale
}

fn relative(kyto: &[f64], exact: &[f64]) -> f64 {
    kyto.iter()
        .zip(exact)
        .map(|(kyto, exact)| (kyto - exact).abs() / exact.abs())
        .fold(0.0, f64::max)
}

/// Compare `pca` with the exact PCA of `rows`: variances to `tolerance` relative,
/// loadings and embedding to `tolerance` of their largest value (`float32`
/// rounding at least).
fn assert_exact(pca: &PcaStatistics, rows: &[Vec<f32>], k: usize, tolerance: f64, label: &str) {
    let exact = exact(rows, k);
    let variance = relative(pca.variance(), &exact.variance);
    let ratio = relative(pca.variance_ratio(), &exact.ratio);
    let loadings = largest_difference(
        pca.loadings().iter().map(|&v| f64::from(v)),
        &exact.loadings,
    );
    let embedding = largest_difference(
        pca.embedding().iter().map(|&v| f64::from(v)),
        &exact.embedding,
    );
    eprintln!(
        "{label}: variance {variance:.1e}, ratio {ratio:.1e}, loadings {loadings:.1e}, \
         embedding {embedding:.1e}"
    );
    assert!(
        variance <= tolerance && ratio <= tolerance,
        "{label}: {variance} {ratio}"
    );
    assert!(
        loadings <= tolerance.max(1e-6),
        "{label}: loadings {loadings}"
    );
    assert!(
        embedding <= tolerance.max(1e-6),
        "{label}: embedding {embedding}"
    );
}

#[test]
fn a_basis_spanning_every_gene_gives_the_exact_pca() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("small.kyto");
    let rows = data(300, 40, 6, 0.5, 1);
    write_store(&store, &rows, false);
    let pca = run(
        &scan(&store)
            .pca(Some("values"), "X_pca", 5, 0, None)
            .unwrap(),
        4,
    )
    .unwrap();
    assert_eq!(pca.cells(), (0..300).collect::<Vec<u64>>());
    assert_exact(&pca, &rows, 5, 1e-10, "40 genes");
    // Every gene's mean.
    let first: f64 = rows.iter().map(|row| f64::from(row[0])).sum::<f64>() / 300.0;
    assert!((pca.means()[0] - first).abs() < 1e-12);
}

#[test]
fn a_sketched_basis_finds_the_leading_components() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("wide.kyto");
    // 600 genes: the basis holds at most 13 x 9 = 117 directions.
    let rows = data(400, 600, 4, 0.2, 2);
    write_store(&store, &rows, false);
    let pca = run(
        &scan(&store)
            .pca(Some("values"), "X_pca", 3, 0, None)
            .unwrap(),
        2,
    )
    .unwrap();
    assert_exact(&pca, &rows, 3, 1e-8, "600 genes");
}

#[test]
fn counts_are_read_as_numbers() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("counts.kyto");
    let rows: Vec<Vec<f32>> = data(200, 30, 3, 0.5, 3)
        .into_iter()
        .map(|row| row.into_iter().map(|value| (value * 3.0).round()).collect())
        .collect();
    write_store(&store, &rows, true);
    let pca = run(
        &scan(&store)
            .pca(Some("values"), "X_pca", 4, 0, None)
            .unwrap(),
        1,
    )
    .unwrap();
    assert_exact(&pca, &rows, 4, 1e-10, "uint32 counts");
}

#[test]
fn a_cell_filter_before_pca_selects_the_rows() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("filtered.kyto");
    let rows = data(300, 40, 5, 0.5, 4);
    write_store(&store, &rows, false);
    let keep = Expr::binary(CellExpr::column("score"), BinaryOp::Lt, Literal::Int(150)).unwrap();
    let plan = scan(&store)
        .filter_cells(&keep)
        .unwrap()
        .take_cells((0..150).step_by(2).map(|cell| format!("c{cell}")))
        .unwrap()
        .pca(Some("values"), "X_pca", 3, 0, None)
        .unwrap();
    let pca = run(&plan, 3).unwrap();
    let kept: Vec<u64> = (0..150).step_by(2).collect();
    assert_eq!(pca.cells(), kept);
    let kept_rows: Vec<Vec<f32>> = kept
        .iter()
        .map(|&cell| rows[cell as usize].clone())
        .collect();
    assert_exact(&pca, &kept_rows, 3, 1e-10, "filtered");
}

#[test]
fn components_beyond_the_rank_have_zero_variance() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("rank.kyto");
    // Rank 2 exactly: no noise, no negative values (shifted up).
    let rows: Vec<Vec<f32>> = (0..50)
        .map(|cell| {
            let (a, b) = (cell as f32 / 7.0, (cell % 5) as f32);
            (0..20)
                .map(|gene| 1.0 + a * (gene % 3) as f32 + b * (gene % 4) as f32)
                .collect()
        })
        .collect();
    write_store(&store, &rows, false);
    let pca = run(
        &scan(&store)
            .pca(Some("values"), "X_pca", 5, 0, None)
            .unwrap(),
        2,
    )
    .unwrap();
    let variance = pca.variance();
    assert!(variance[0] > 0.1 && variance[1] > 0.1, "{variance:?}");
    let total = variance[0] / pca.variance_ratio()[0];
    assert!(
        variance[2..].iter().all(|&value| value <= 1e-10 * total),
        "{variance:?}"
    );
    let ratios: f64 = pca.variance_ratio().iter().sum();
    assert!((ratios - 1.0).abs() < 1e-9, "{ratios}");
    // Orthonormal loadings, the zero-variance ones included.
    let loadings = pca.loadings();
    for left in 0..5 {
        for right in 0..5 {
            let dot: f64 = (0..20)
                .map(|gene| {
                    f64::from(loadings[gene * 5 + left]) * f64::from(loadings[gene * 5 + right])
                })
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-6, "{left} {right}: {dot}");
        }
    }
    assert!(
        pca.embedding()
            .iter()
            .skip(2)
            .step_by(5)
            .all(|value| value.abs() < 1e-3)
    );
    assert_eq!(
        run(
            &scan(&store)
                .pca(Some("values"), "X_pca", 5, 0, None)
                .unwrap(),
            1
        )
        .unwrap(),
        pca
    );
}

#[test]
fn too_few_cells_too_many_components_and_non_finite_values_are_refused() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("refused.kyto");
    let mut rows = data(40, 10, 3, 0.5, 5);
    write_store(&store, &rows, false);
    let below = |cells: i64| {
        scan(&store)
            .filter_cells(
                &Expr::binary(CellExpr::column("score"), BinaryOp::Lt, Literal::Int(cells))
                    .unwrap(),
            )
            .unwrap()
    };
    for (cells, components, problem) in [
        (0, 2, "0 cells reach pca"),
        (1, 1, "1 cell reaches pca"),
        (
            3,
            4,
            "3 cells and 10 genes reach pca, so it has at most 3 components",
        ),
    ] {
        let plan = below(cells)
            .pca(Some("values"), "X_pca", components, 0, None)
            .unwrap();
        let error = run(&plan, 2).unwrap_err();
        assert_eq!(error.code(), "KY-E215", "{error}");
        assert!(error.to_string().contains(problem), "{error}");
    }
    // As many components as cells is fine.
    assert_eq!(
        run(
            &below(3).pca(Some("values"), "X_pca", 3, 0, None).unwrap(),
            1
        )
        .unwrap()
        .variance()
        .len(),
        3
    );
    // A NaN.
    rows[7][3] = f32::NAN;
    let store = work.path().join("nan.kyto");
    write_store(&store, &rows, false);
    let error = run(
        &scan(&store)
            .pca(Some("values"), "X_pca", 2, 0, None)
            .unwrap(),
        2,
    )
    .unwrap_err();
    assert_eq!(error.code(), "KY-E215", "{error}");
    assert!(
        error
            .to_string()
            .contains("gene 3 holds a value that is not finite"),
        "{error}"
    );
}

#[test]
fn large_means_are_centred_row_by_row() {
    // Bug-hunter: `X^T X Q - n mu mu^T Q` lost 1.7e-2 of the variance at means of 1e7.
    let work = tempfile::tempdir().unwrap();
    for offset in [1e6_f32, 1e7] {
        let store = work.path().join(format!("offset{offset}.kyto"));
        let rows: Vec<Vec<f32>> = data(200, 12, 3, 1.0, 6)
            .into_iter()
            .map(|row| row.into_iter().map(|value| value * 8.0 + offset).collect())
            .collect();
        write_store(&store, &rows, false);
        let pca = run(
            &scan(&store)
                .pca(Some("values"), "X_pca", 3, 0, None)
                .unwrap(),
            2,
        )
        .unwrap();
        assert_exact(&pca, &rows, 3, 1e-6, &format!("offset {offset}"));
    }
}

#[test]
fn equal_cells_have_no_variance() {
    // Bug-hunter: identical cells whose mean is not exact gave ratios of inf.
    let work = tempfile::tempdir().unwrap();
    for row in [[0.1_f32, 0.7, 3.3, 0.1, 0.1, 5.5], [2.0; 6]] {
        let store = work.path().join(format!("constant{}.kyto", row[0]));
        write_store(&store, &vec![row.to_vec(); 7], false);
        let pca = run(
            &scan(&store)
                .pca(Some("values"), "X_pca", 3, 0, None)
                .unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(pca.variance(), [0.0; 3], "{row:?}");
        assert!(
            pca.variance_ratio().iter().all(|ratio| ratio.is_nan()),
            "{row:?}"
        );
        assert!(
            pca.embedding().iter().all(|value| value.abs() < 1e-6),
            "{row:?}"
        );
    }
}

/// Whether gene `gene` of a `write_store` store is in its `selected` set.
fn is_selected(gene: usize) -> bool {
    gene != 1 && !gene.is_multiple_of(3)
}

#[test]
fn a_gene_set_gives_the_bits_of_a_gene_filter_and_keeps_every_gene() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("gene_set.kyto");
    // More genes in the set (199) than the Krylov basis holds (15 columns x 10), so the
    // result is sketched: any difference in the order of the sums would show.
    let rows = data(260, 300, 6, 0.5, 9);
    write_store(&store, &rows, false);
    let selected = GeneExpr::column("selected");
    let keep = Expr::binary(CellExpr::column("score"), BinaryOp::Lt, Literal::Int(240)).unwrap();
    let base = scan(&store).filter_cells(&keep).unwrap();
    let masked = base
        .pca(Some("values"), "X_pca", 5, 3, Some(&selected))
        .unwrap();
    let filtered = base
        .filter_genes(&selected)
        .unwrap()
        .pca(Some("values"), "X_pca", 5, 3, None)
        .unwrap();
    let expected = run(&filtered, 2).unwrap();
    let kept: Vec<u64> = (0..300)
        .filter(|&gene| is_selected(gene))
        .map(|gene| gene as u64)
        .collect();
    assert_eq!(expected.genes(), kept);
    for threads in [1, 4] {
        let pca = run(&masked, threads).unwrap();
        assert_eq!(pca.cells(), expected.cells());
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let bits64 = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(pca.embedding()), bits(expected.embedding()));
        assert_eq!(bits64(pca.variance()), bits64(expected.variance()));
        assert_eq!(
            bits64(pca.variance_ratio()),
            bits64(expected.variance_ratio())
        );
        assert_eq!(bits64(pca.means()), bits64(expected.means()));
        // Every gene keeps a loadings row: the filter's rows in the set, zeros outside.
        assert_eq!(pca.genes(), (0..300).collect::<Vec<u64>>());
        assert_eq!(pca.loadings().len(), 300 * 5);
        let mut next = expected.loadings().chunks(5);
        for (gene, row) in pca.loadings().chunks(5).enumerate() {
            if is_selected(gene) {
                assert_eq!(bits(row), bits(next.next().unwrap()), "gene {gene}");
            } else {
                assert_eq!(row, [0.0; 5], "gene {gene}");
            }
        }
        assert!(next.next().is_none());
    }
    // And against the exact PCA of the set's columns (a spectral gap: close, not exact).
    let subset: Vec<Vec<f32>> = rows[..240]
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(gene, _)| is_selected(*gene))
                .map(|(_, &value)| value)
                .collect()
        })
        .collect();
    let exact = exact(&subset, 5);
    assert!(relative(expected.variance(), &exact.variance) < 1e-6);
}

#[test]
fn a_gene_set_is_taken_over_the_genes_present_at_the_node() {
    // A selection before the node reorders and drops genes: the set follows the genes
    // of the stream, and the loadings rows are theirs.
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("gene_set_order.kyto");
    let rows = data(120, 24, 4, 0.5, 10);
    write_store(&store, &rows, false);
    let order: Vec<usize> = (2..24).rev().collect();
    let selection = scan(&store)
        .select_genes(order.iter().map(|gene| format!("G{gene}")))
        .unwrap();
    let selected = GeneExpr::column("selected");
    let pca = run(
        &selection
            .pca(Some("values"), "X_pca", 3, 0, Some(&selected))
            .unwrap(),
        2,
    )
    .unwrap();
    assert_eq!(
        pca.genes(),
        order.iter().map(|&gene| gene as u64).collect::<Vec<_>>()
    );
    let in_set: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&gene| is_selected(gene))
        .collect();
    let subset: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| in_set.iter().map(|&gene| row[gene]).collect())
        .collect();
    // 14 genes: the basis spans them, so the result is exact.
    let exact = exact(&subset, 3);
    assert!(relative(pca.variance(), &exact.variance) < 1e-10);
    let mut expected = exact.loadings.chunks(3);
    for (&gene, row) in order.iter().zip(pca.loadings().chunks(3)) {
        if is_selected(gene) {
            let exact = expected.next().unwrap();
            for (kyto, exact) in row.iter().zip(exact) {
                assert!((f64::from(*kyto) - exact).abs() < 1e-6, "gene {gene}");
            }
        } else {
            assert_eq!(row, [0.0; 3], "gene {gene}");
        }
    }
}

#[test]
fn a_gene_set_with_too_few_genes_is_refused() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("gene_set_small.kyto");
    write_store(&store, &data(40, 12, 3, 0.5, 11), false);
    // 7 of the 12 genes are in the set: 8 components pass the plan's bound, not the run.
    let few = scan(&store)
        .pca(
            Some("values"),
            "X_pca",
            8,
            0,
            Some(&GeneExpr::column("selected")),
        )
        .unwrap();
    let error = run(&few, 2).unwrap_err();
    assert_eq!(error.code(), "KY-E215");
    let message = error.to_string();
    assert!(
        message.contains("40 cells and 7 genes (those of its gene set, `genes`) reach pca"),
        "{message}"
    );
    // No gene in the set at all.
    let none = Expr::binary(
        GeneExpr::column("gene_id"),
        BinaryOp::Eq,
        Literal::String("none".to_owned()),
    )
    .unwrap();
    let empty = scan(&store)
        .pca(Some("values"), "X_pca", 1, 0, Some(&none))
        .unwrap();
    let error = run(&empty, 1).unwrap_err();
    assert_eq!(error.code(), "KY-E215");
    assert!(
        error
            .to_string()
            .contains("the predicate is true for none of the genes reaching pca"),
        "{error}"
    );
}
