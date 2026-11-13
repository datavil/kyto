//! A test-only reference evaluator for task 4.4's "optimized results equal unoptimized
//! results": no executor exists before 5.x, so this runs a logical plan, and an
//! optimized plan, over a small seeded dataset in memory, and the tests compare what
//! each output reads.
//!
//! Every node's semantics live in one function used by both runs, so a difference can
//! only come from the rewrites (moved filters, left-out nodes, fused passes, shared
//! sub-plans, a scan reading less). The values follow the node's scope, not the real
//! algorithm: `normalize_total`, `log1p`, `scale` and the filters are computed for real;
//! `highly_variable_genes`, `pca`, `neighbors` and the clusterings are stand-ins that
//! depend on every cell (statistics over cells, pairwise distances, ranks), as the real
//! ones do, so moving a filter past one changes the result. A member missing from a
//! run's data (a scan that read too little) is an error. The dataset has the
//! synthetic fixture's edge cases (`tests/fixtures/synthetic.py`): batches, an all-zero
//! cell and an all-zero gene, nulls, a stored derived column with lineage.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kyto_core::{AtlasSchema, Axis, Expr, ExprAxis, ExprNode, GeneIds, Source};
use kyto_plan::{
    Columns, CrossCellOp, ElementwiseOp, MapElementwise, Member, Members, Operation,
    OptimizeOptions, OptimizedPlan, Output, Plan, Predicate, hvg_columns, loadings_name, optimize,
};
use polars::prelude::*;

pub const N_CELLS: usize = 24;
pub const N_GENES: usize = 6;

/// Cell `c3` and gene `G4` are all zero.
const ZERO_CELL: usize = 3;
const ZERO_GENE: usize = 4;

/// The store: every member's values.
#[derive(Debug, Clone)]
pub struct Data {
    pub cells: DataFrame,
    pub genes: DataFrame,
    pub layers: BTreeMap<String, Vec<Vec<f64>>>,
    pub embeddings: BTreeMap<String, (Axis, Vec<Vec<f64>>)>,
    pub graphs: BTreeMap<String, Vec<Vec<f64>>>,
    /// Stored derived cell columns and the layer each was computed from.
    pub lineage: BTreeMap<String, String>,
}

/// A small deterministic generator (no shared RNG, memory/test_seed_isolation.md).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

pub fn cell_ids() -> Vec<String> {
    (0..N_CELLS).map(|cell| format!("c{cell}")).collect()
}

pub fn gene_id_list() -> Vec<String> {
    (0..N_GENES).map(|gene| format!("G{gene}")).collect()
}

/// The seeded dataset.
pub fn dataset() -> Data {
    let mut random = Lcg(42);
    let counts: Vec<Vec<f64>> = (0..N_CELLS)
        .map(|cell| {
            (0..N_GENES)
                .map(|gene| {
                    let draw = random.next() % 8;
                    if cell == ZERO_CELL || gene == ZERO_GENE || draw < 4 {
                        0.0
                    } else {
                        // Batch b2 sequenced deeper.
                        (draw - 3 + 2 * u64::from(cell % 3 == 2)) as f64
                    }
                })
                .collect()
        })
        .collect();
    let log1p: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| row.iter().map(|value| value.ln_1p()).collect())
        .collect();
    let totals: Vec<f32> = counts
        .iter()
        .map(|row| row.iter().sum::<f64>() as f32)
        .collect();
    let batch: Vec<String> = (0..N_CELLS).map(|cell| format!("b{}", cell % 3)).collect();
    let score: Vec<Option<i32>> = (0..N_CELLS)
        .map(|cell| (cell % 7 != 5).then(|| (cell as i32 * 7) % 11 - 3))
        .collect();
    let cells = DataFrame::new(
        N_CELLS,
        vec![
            Column::new("cell_id".into(), cell_ids()),
            Column::new("batch".into(), batch),
            Column::new("score".into(), score),
            Column::new("total_counts".into(), totals),
        ],
    )
    .unwrap();
    let means: Vec<f64> = (0..N_GENES)
        .map(|gene| counts.iter().map(|row| row[gene]).sum::<f64>() / N_CELLS as f64)
        .collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![
            Column::new("gene_id".into(), gene_id_list()),
            Column::new(
                "symbol".into(),
                (0..N_GENES)
                    .map(|gene| format!("S{gene}"))
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "highly_variable".into(),
                (0..N_GENES)
                    .map(|gene| (gene != 2).then_some(gene % 2 == 0))
                    .collect::<Vec<_>>(),
            ),
            Column::new("mean".into(), means),
        ],
    )
    .unwrap();
    let x_pca: Vec<Vec<f64>> = (0..N_CELLS)
        .map(|_| (0..3).map(|_| random.next() as f64 / 1e9 - 2.0).collect())
        .collect();
    let pcs: Vec<Vec<f64>> = (0..N_GENES)
        .map(|_| (0..2).map(|_| random.next() as f64 / 1e9).collect())
        .collect();
    let connectivities: Vec<Vec<f64>> = (0..N_CELLS)
        .map(|left| {
            (0..N_CELLS)
                .map(|right| f64::from(u8::from((left + right) % 4 == 0)))
                .collect()
        })
        .collect();
    Data {
        cells,
        genes,
        layers: BTreeMap::from([("counts".to_owned(), counts), ("log1p".to_owned(), log1p)]),
        embeddings: BTreeMap::from([
            ("X_pca".to_owned(), (Axis::Cells, x_pca)),
            ("PCs".to_owned(), (Axis::Genes, pcs)),
        ]),
        graphs: BTreeMap::from([("connectivities".to_owned(), connectivities)]),
        lineage: BTreeMap::from([("total_counts".to_owned(), "counts".to_owned())]),
    }
}

const SCHEMA: &str = r#"{
  "n_cells": 24,
  "n_genes": 6,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {"columns": [
    {"name": "cell_id", "dtype": "String"},
    {"name": "batch", "dtype": "Enum", "n_categories": 3, "ordered": false},
    {"name": "score", "dtype": "Int32"},
    {"name": "total_counts", "dtype": "Float32", "lineage": {"layer": "counts", "node": "n2"}}
  ]},
  "genes": {"columns": [
    {"name": "gene_id", "dtype": "String"},
    {"name": "symbol", "dtype": "Enum", "n_categories": 6, "ordered": false},
    {"name": "highly_variable", "dtype": "Boolean"},
    {"name": "mean", "dtype": "Float64"}
  ]},
  "layers": [
    {"name": "counts", "dtype": "uint32", "nnz": 60},
    {"name": "log1p", "dtype": "float32", "nnz": 60}
  ],
  "embeddings": [
    {"name": "X_pca", "axis": "cells", "width": 3, "dtype": "float32"},
    {"name": "PCs", "axis": "genes", "width": 2, "dtype": "float32"}
  ],
  "graphs": [
    {"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 150}
  ]
}"#;

/// The scan of the dataset's store.
pub fn scan() -> Arc<Plan> {
    let schema: AtlasSchema = serde_json::from_str(SCHEMA).unwrap();
    let symbols = (0..N_GENES).map(|gene| Some(format!("S{gene}")));
    Plan::scan(
        "synthetic.kyto",
        schema,
        GeneIds::new(gene_id_list()).with_symbols(symbols),
    )
}

// --- Running plans ----------------------------------------------------------------

/// The data of `plan`, run node by node as built.
pub fn run_logical(plan: &Arc<Plan>, store: &Data) -> Result<Data, String> {
    let mut chain: Vec<&Plan> = plan.nodes().collect();
    chain.reverse();
    let mut data = store.clone();
    for node in chain.into_iter().skip(1) {
        apply(node, &mut data)?;
    }
    Ok(data)
}

/// The data of each output of `plan`, run node by node over the arena.
pub fn run_optimized(plan: &OptimizedPlan, store: &Data) -> Vec<Result<Data, String>> {
    let mut results: Vec<Result<Data, String>> = Vec::with_capacity(plan.nodes().len());
    for node in plan.nodes() {
        let result = match node.operation() {
            Operation::Scan { reads, .. } => Ok(load(store, reads)),
            operation => {
                let input = node.input().expect("only a scan has no input");
                results[input].clone().and_then(|mut data| {
                    match operation {
                        Operation::Node(logical) => apply(logical, &mut data)?,
                        Operation::Fused { maps, writes } => apply_fused(maps, writes, &mut data)?,
                        Operation::Scan { .. } => unreachable!(),
                    }
                    Ok(data)
                })
            }
        };
        results.push(result);
    }
    plan.outputs()
        .iter()
        .map(|output| results[output.node()].clone())
        .collect()
}

/// The store with only `reads` (and the ID columns).
fn load(store: &Data, reads: &Members) -> Data {
    let columns = |frame: &DataFrame, id: &str, keep: &dyn Fn(&str) -> bool| {
        let names: Vec<PlSmallStr> = frame
            .get_column_names()
            .into_iter()
            .filter(|name| name.as_str() == id || keep(name.as_str()))
            .cloned()
            .collect();
        frame.select(names).unwrap()
    };
    let has = |member: Member| reads.contains(&member);
    Data {
        cells: columns(&store.cells, "cell_id", &|name| {
            has(Member::CellColumn(name.to_owned()))
        }),
        genes: columns(&store.genes, "gene_id", &|name| {
            has(Member::GeneColumn(name.to_owned()))
        }),
        layers: filtered(&store.layers, |name| has(Member::Layer(name.to_owned()))),
        embeddings: filtered(&store.embeddings, |name| {
            has(Member::Embedding(name.to_owned()))
        }),
        graphs: filtered(&store.graphs, |name| has(Member::Graph(name.to_owned()))),
        lineage: store.lineage.clone(),
    }
}

fn filtered<V: Clone>(
    map: &BTreeMap<String, V>,
    keep: impl Fn(&str) -> bool,
) -> BTreeMap<String, V> {
    map.iter()
        .filter(|(name, _)| keep(name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// What a terminal reading `reads` sees of `data`, as text (exact: floats print their
/// shortest round-trip form): the IDs in order, then each member.
pub fn view(data: &Data, reads: &Members) -> BTreeMap<String, String> {
    let mut view = BTreeMap::new();
    view.insert(
        "cell_id".to_owned(),
        format!("{:?}", ids(&data.cells, "cell_id")),
    );
    view.insert(
        "gene_id".to_owned(),
        format!("{:?}", ids(&data.genes, "gene_id")),
    );
    for member in reads {
        let text = match member {
            Member::Layer(name) => data.layers.get(name).map(|rows| format!("{rows:?}")),
            Member::CellColumn(name) => column_text(&data.cells, name),
            Member::GeneColumn(name) => column_text(&data.genes, name),
            Member::Embedding(name) => data.embeddings.get(name).map(|value| format!("{value:?}")),
            Member::Graph(name) => data.graphs.get(name).map(|rows| format!("{rows:?}")),
        };
        view.insert(
            member.to_string(),
            text.unwrap_or_else(|| "MISSING".to_owned()),
        );
    }
    view
}

fn column_text(frame: &DataFrame, name: &str) -> Option<String> {
    let column = frame.column(name).ok()?;
    let values: Vec<String> = column
        .as_materialized_series()
        .iter()
        .map(|value| format!("{value:?}"))
        .collect();
    Some(format!("{:?} {values:?}", column.dtype()))
}

pub fn ids(frame: &DataFrame, name: &str) -> Vec<String> {
    let values = frame.column(name).unwrap().str().unwrap();
    (0..values.len())
        .map(|row| values.get(row).unwrap().to_owned())
        .collect()
}

/// Checks that every output of the optimized plan of `outputs` (with `options`) sees
/// what the logical plan's run sees. Returns a message on the first difference.
pub fn check_equal(
    outputs: &[Output],
    optimized: &OptimizedPlan,
    options: OptimizeOptions,
) -> Result<(), String> {
    let projection = options.projection_pushdown;
    let store = dataset();
    let runs = run_optimized(optimized, &store);
    for (position, (output, run)) in outputs.iter().zip(runs).enumerate() {
        match (run_logical(output.plan(), &store), run) {
            (Ok(logical), Ok(optimized_data)) => {
                let (left, right) = (
                    view(&logical, output.reads()),
                    view(&optimized_data, output.reads()),
                );
                if left != right {
                    let key = left
                        .keys()
                        .find(|key| left.get(*key) != right.get(*key))
                        .unwrap();
                    return Err(format!(
                        "output {position}: {key} differs\nlogical:   {}\noptimized: {}\nplan:\n{optimized}",
                        left[key], right[key]
                    ));
                }
            }
            (Ok(_), Err(error)) => {
                return Err(format!(
                    "output {position}: only the optimized run fails: {error}\n{optimized}"
                ));
            }
            // Leaving out an unused node may skip its error (deviations.md), and only
            // that: the same plan without projection pushdown must fail too.
            (Err(_), Ok(_)) if projection => {
                let without = OptimizeOptions {
                    projection_pushdown: false,
                    ..options
                };
                let rerun = run_optimized(&optimize(outputs, without), &store);
                if rerun[position].is_ok() {
                    return Err(format!(
                        "output {position}: a rule other than projection pushdown removed an error\n{optimized}"
                    ));
                }
            }
            (Err(error), Ok(_)) => {
                return Err(format!(
                    "output {position}: only the logical run fails: {error}\n{optimized}"
                ));
            }
            (Err(_), Err(_)) => {}
        }
    }
    Ok(())
}

// --- Node semantics ---------------------------------------------------------------

fn apply(plan: &Plan, data: &mut Data) -> Result<(), String> {
    match plan {
        Plan::Scan(_) => Err("a scan is not applied".to_owned()),
        Plan::WithDefaultLayer(_) | Plan::Sink(_) => Ok(()),
        Plan::Filter(node) => match node.params().predicate() {
            Predicate::Cells(predicate) => {
                let rows = mask(&evaluate(predicate, data)?)?;
                select_cells(data, &rows);
                Ok(())
            }
            Predicate::Genes(predicate) => {
                let rows = mask(&evaluate(predicate, data)?)?;
                select_genes(data, &rows);
                Ok(())
            }
        },
        Plan::SelectGenes(node) => {
            let rows = positions(&ids(&data.genes, "gene_id"), node.params().gene_ids())?;
            select_genes(data, &rows);
            Ok(())
        }
        Plan::TakeCells(node) => {
            let rows = positions(&ids(&data.cells, "cell_id"), node.params().cell_ids())?;
            select_cells(data, &rows);
            Ok(())
        }
        Plan::MapElementwise(node) => {
            let map = node.params();
            let input = layer(data, map.layer())?;
            let median = median_total(map, input);
            let rows = input.iter().map(|row| map_row(map, row, median)).collect();
            data.layers.insert(map.output_layer().to_owned(), rows);
            Ok(())
        }
        Plan::CrossCell(node) => cross_cell(node.params().op(), data),
        Plan::WithColumns(node) => match node.params().columns() {
            Columns::Cells(columns) => {
                let values = columns
                    .iter()
                    .map(|(name, expr)| Ok(evaluate(expr, data)?.with_name(name.as_str().into())))
                    .collect::<Result<Vec<_>, String>>()?;
                for value in values {
                    data.cells.with_column(value.into_column()).unwrap();
                }
                Ok(())
            }
            Columns::Genes(columns) => {
                let values = columns
                    .iter()
                    .map(|(name, expr)| Ok(evaluate(expr, data)?.with_name(name.as_str().into())))
                    .collect::<Result<Vec<_>, String>>()?;
                for value in values {
                    data.genes.with_column(value.into_column()).unwrap();
                }
                Ok(())
            }
        },
        Plan::Neighbors(node) => {
            let params = node.params();
            let (_, rows) = data
                .embeddings
                .get(&params.embedding)
                .ok_or_else(|| format!("missing embedding {}", params.embedding))?;
            let width = params.n_components.map_or(usize::MAX, |n| n as usize);
            let distances: Vec<Vec<f64>> = rows
                .iter()
                .map(|left| {
                    rows.iter()
                        .map(|right| {
                            left.iter()
                                .zip(right)
                                .take(width)
                                .map(|(a, b)| (a - b).abs())
                                .sum()
                        })
                        .collect()
                })
                .collect();
            let connectivities = distances
                .iter()
                .map(|row| row.iter().map(|distance| 1.0 / (1.0 + distance)).collect())
                .collect();
            let (distance_name, connectivity_name) = params.graph_names();
            data.graphs.insert(distance_name, distances);
            data.graphs.insert(connectivity_name, connectivities);
            Ok(())
        }
        Plan::Cluster(node) => {
            let params = node.params();
            let graph = data
                .graphs
                .get(&params.graph)
                .ok_or_else(|| format!("missing graph {}", params.graph))?;
            let sums: Vec<f64> = graph
                .iter()
                .map(|row| row.iter().sum::<f64>() * params.resolution)
                .collect();
            let labels: Vec<String> = sums
                .iter()
                .enumerate()
                .map(|(cell, sum)| {
                    let rank = sums
                        .iter()
                        .enumerate()
                        .filter(|(other, value)| *value < sum || (*value == sum && *other < cell))
                        .count();
                    format!("c{}", rank % 3)
                })
                .collect();
            data.cells
                .with_column(Column::new(params.into.as_str().into(), labels))
                .unwrap();
            Ok(())
        }
        Plan::Umap(node) => {
            // A stand-in layout: each cell's graph row sum, then the seed, per dimension.
            let params = node.params();
            let graph = data
                .graphs
                .get(&params.graph)
                .ok_or_else(|| format!("missing graph {}", params.graph))?;
            let rows = graph
                .iter()
                .map(|row| {
                    let sum = row.iter().sum::<f64>();
                    (0..params.n_components)
                        .map(|axis| sum * (axis + 1) as f64 + params.seed as f64)
                        .collect()
                })
                .collect();
            data.embeddings
                .insert(params.into.clone(), (Axis::Cells, rows));
            Ok(())
        }
    }
}

/// A fused pass: the median of the first map's input (if it needs one), then each
/// cell's row through every map in turn; only `writes` are kept.
fn apply_fused(
    maps: &[MapElementwise],
    writes: &BTreeSet<String>,
    data: &mut Data,
) -> Result<(), String> {
    let first = maps.first().ok_or("an empty fused pass")?;
    let median = median_total(first, layer(data, first.layer())?);
    let n_cells = data.cells.height();
    let mut kept: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    for cell in 0..n_cells {
        let mut pass: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for map in maps {
            let input = match pass.get(map.layer()) {
                Some(row) => row.clone(),
                None => layer(data, map.layer())?[cell].clone(),
            };
            let median = if std::ptr::eq(map, first) {
                median
            } else {
                None
            };
            pass.insert(map.output_layer(), map_row(map, &input, median));
        }
        for name in writes {
            let row = pass
                .remove(name.as_str())
                .ok_or("a kept layer no map writes")?;
            kept.entry(name.clone()).or_default().push(row);
        }
    }
    for name in writes {
        data.layers
            .insert(name.clone(), kept.remove(name).unwrap_or_default());
    }
    Ok(())
}

fn layer<'a>(data: &'a Data, name: &str) -> Result<&'a Vec<Vec<f64>>, String> {
    data.layers
        .get(name)
        .ok_or_else(|| format!("missing layer {name}"))
}

/// The median cell total of `input` over the cells with a non-zero total (scanpy's
/// `normalize_total(target_sum=None)`, as the executor computes it, 6.2), when `map`
/// scales to it.
fn median_total(map: &MapElementwise, input: &[Vec<f64>]) -> Option<f64> {
    if !matches!(map.op(), ElementwiseOp::NormalizeTotal { target_sum: None }) {
        return None;
    }
    let mut totals: Vec<f64> = input
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .filter(|total| *total > 0.0)
        .collect();
    totals.sort_by(f64::total_cmp);
    let n = totals.len();
    Some(match n {
        0 => 0.0,
        _ if n % 2 == 1 => totals[n / 2],
        _ => (totals[n / 2 - 1] + totals[n / 2]) / 2.0,
    })
}

/// One cell's row through `map`, each value rounded to the `float32` a map writes (its
/// logical dtype, as the executor stores it; 5.8 compares the two).
fn map_row(map: &MapElementwise, row: &[f64], median: Option<f64>) -> Vec<f64> {
    map_row_f64(map, row, median)
        .into_iter()
        .map(|value| f64::from(value as f32))
        .collect()
}

fn map_row_f64(map: &MapElementwise, row: &[f64], median: Option<f64>) -> Vec<f64> {
    match map.op() {
        ElementwiseOp::Log1p => row.iter().map(|value| value.ln_1p()).collect(),
        ElementwiseOp::NormalizeTotal { target_sum } => {
            let target = target_sum.or(median).unwrap_or(0.0);
            let total: f64 = row.iter().sum();
            row.iter()
                .map(|value| {
                    if total == 0.0 {
                        0.0
                    } else {
                        value * target / total
                    }
                })
                .collect()
        }
    }
}

fn cross_cell(op: &CrossCellOp, data: &mut Data) -> Result<(), String> {
    match op {
        CrossCellOp::Scale {
            layer: name,
            into,
            zero_center,
            max_value,
        } => {
            let input = layer(data, name)?;
            let (means, variances) = gene_moments(input, data.genes.height());
            let rows = input
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(gene, value)| {
                            let std = variances[gene].sqrt();
                            let centred = if *zero_center {
                                value - means[gene]
                            } else {
                                *value
                            };
                            let scaled = if std == 0.0 { 0.0 } else { centred / std };
                            max_value.map_or(scaled, |max| scaled.min(max))
                        })
                        .collect()
                })
                .collect();
            data.layers.insert(into.clone(), rows);
        }
        CrossCellOp::HighlyVariableGenes {
            layer: name,
            flavor,
            n_top_genes,
            batch_key,
        } => {
            let (means, variances) = gene_moments(layer(data, name)?, data.genes.height());
            let n_genes = means.len();
            let n_top = n_top_genes.map_or(n_genes / 2, |n| n as usize);
            let rank: Vec<usize> = (0..n_genes)
                .map(|gene| {
                    (0..n_genes)
                        .filter(|&other| {
                            variances[other] > variances[gene]
                                || (variances[other] == variances[gene] && other < gene)
                        })
                        .count()
                })
                .collect();
            let n_batches = match batch_key {
                Some(key) => {
                    let column = data.cells.column(key).map_err(|error| error.to_string())?;
                    let values: BTreeSet<String> = column
                        .as_materialized_series()
                        .iter()
                        .filter(|value| !value.is_null())
                        .map(|value| format!("{value}"))
                        .collect();
                    values.len() as i64
                }
                None => 0,
            };
            for column in hvg_columns(*flavor, batch_key.is_some()) {
                let name = column.name.as_str();
                let values: Column = match name {
                    "highly_variable" | "highly_variable_intersection" => Column::new(
                        name.into(),
                        rank.iter().map(|rank| *rank < n_top).collect::<Vec<_>>(),
                    ),
                    "highly_variable_nbatches" => {
                        Column::new(name.into(), vec![n_batches; n_genes])
                    }
                    "highly_variable_rank" => Column::new(
                        name.into(),
                        rank.iter().map(|rank| *rank as f64).collect::<Vec<_>>(),
                    ),
                    "means" => Column::new(name.into(), means.clone()),
                    "dispersions" | "variances" => Column::new(name.into(), variances.clone()),
                    _ => Column::new(
                        name.into(),
                        means
                            .iter()
                            .zip(&variances)
                            .map(|(mean, variance)| variance / (mean + 1.0))
                            .collect::<Vec<_>>(),
                    ),
                };
                data.genes.with_column(values).unwrap();
            }
        }
        CrossCellOp::Pca {
            layer: name,
            into,
            n_components,
            ..
        } => {
            let input = layer(data, name)?;
            let (means, _) = gene_moments(input, data.genes.height());
            let width = *n_components as usize;
            let n_genes = means.len();
            let embedding = input
                .iter()
                .map(|row| {
                    (0..width)
                        .map(|component| match n_genes {
                            0 => 0.0,
                            _ => {
                                (row[component % n_genes] - means[component % n_genes])
                                    * (component + 1) as f64
                            }
                        })
                        .collect()
                })
                .collect();
            let loadings = means
                .iter()
                .map(|mean| {
                    (0..width)
                        .map(|component| mean * (component + 1) as f64)
                        .collect()
                })
                .collect();
            data.embeddings
                .insert(into.clone(), (Axis::Cells, embedding));
            data.embeddings
                .insert(loadings_name(into), (Axis::Genes, loadings));
        }
    }
    Ok(())
}

/// Each gene's mean and (population) variance over the cells.
fn gene_moments(input: &[Vec<f64>], n_genes: usize) -> (Vec<f64>, Vec<f64>) {
    let n_cells = input.len() as f64;
    let means: Vec<f64> = (0..n_genes)
        .map(|gene| input.iter().map(|row| row[gene]).sum::<f64>() / n_cells)
        .collect();
    let variances = (0..n_genes)
        .map(|gene| {
            input
                .iter()
                .map(|row| (row[gene] - means[gene]).powi(2))
                .sum::<f64>()
                / n_cells
        })
        .collect();
    (means, variances)
}

fn positions(have: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    wanted
        .iter()
        .map(|id| {
            have.iter()
                .position(|candidate| candidate == id)
                .ok_or_else(|| format!("no row {id}"))
        })
        .collect()
}

fn mask(values: &Series) -> Result<Vec<usize>, String> {
    let values = values.bool().map_err(|error| error.to_string())?;
    Ok((0..values.len())
        .filter(|row| values.get(*row) == Some(true))
        .collect())
}

fn take(frame: &DataFrame, rows: &[usize]) -> DataFrame {
    let indices: Vec<IdxSize> = rows.iter().map(|row| *row as IdxSize).collect();
    frame
        .take(&IdxCa::from_vec("index".into(), indices))
        .unwrap()
}

fn select_cells(data: &mut Data, rows: &[usize]) {
    data.cells = take(&data.cells, rows);
    let pick = |matrix: &Vec<Vec<f64>>| {
        rows.iter()
            .map(|row| matrix[*row].clone())
            .collect::<Vec<_>>()
    };
    for matrix in data.layers.values_mut() {
        *matrix = pick(matrix);
    }
    for (axis, matrix) in data.embeddings.values_mut() {
        if *axis == Axis::Cells {
            *matrix = pick(matrix);
        }
    }
    for matrix in data.graphs.values_mut() {
        *matrix = pick(matrix)
            .into_iter()
            .map(|row| rows.iter().map(|column| row[*column]).collect())
            .collect();
    }
}

fn select_genes(data: &mut Data, columns: &[usize]) {
    data.genes = take(&data.genes, columns);
    for matrix in data.layers.values_mut() {
        for row in matrix.iter_mut() {
            *row = columns.iter().map(|column| row[*column]).collect();
        }
    }
    for (axis, matrix) in data.embeddings.values_mut() {
        if *axis == Axis::Genes {
            *matrix = columns
                .iter()
                .map(|column| matrix[*column].clone())
                .collect();
        }
    }
}

/// The values of a resolved expression over its table's rows. Non-column leaves become
/// stand-in columns of their values.
fn evaluate<A: ExprAxis>(expr: &Expr<A>, data: &Data) -> Result<Series, String> {
    let mut frame = match A::AXIS {
        Axis::Cells => data.cells.clone(),
        Axis::Genes => data.genes.clone(),
    };
    let mut leaves: BTreeMap<String, String> = BTreeMap::new();
    let mut stack = vec![expr.node().as_ref()];
    while let Some(node) = stack.pop() {
        match node {
            ExprNode::Source(Source::Column { name }) => {
                if frame.column(name).is_err() {
                    return Err(format!("missing column {name}"));
                }
            }
            ExprNode::Source(source) => {
                let key = format!("{source:?}");
                if !leaves.contains_key(&key) {
                    let name = format!("__leaf{}", leaves.len());
                    let values = leaf_values(source, A::AXIS, data)?;
                    frame
                        .with_column(Column::new(name.as_str().into(), values))
                        .unwrap();
                    leaves.insert(key, name);
                }
            }
            ExprNode::Literal(_) => {}
            ExprNode::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            ExprNode::Not(input) | ExprNode::Str { input, .. } | ExprNode::Null { input, .. } => {
                stack.push(input);
            }
        }
    }
    let lowered = expr
        .node()
        .to_polars_with(&|source| match source {
            Source::Column { name } => Some(col(name.as_str())),
            other => leaves
                .get(&format!("{other:?}"))
                .map(|name| col(name.as_str())),
        })
        .ok_or("expression does not lower")?;
    let result = frame
        .lazy()
        .with_column(lowered.alias("__value"))
        .collect()
        .map_err(|error| error.to_string())?;
    Ok(result
        .column("__value")
        .unwrap()
        .as_materialized_series()
        .clone())
}

fn leaf_values(source: &Source, axis: Axis, data: &Data) -> Result<Vec<f64>, String> {
    let gene_ids = ids(&data.genes, "gene_id");
    let cell_ids = ids(&data.cells, "cell_id");
    match (source, axis) {
        (
            Source::Expression {
                layer: Some(name),
                gene_id,
            },
            Axis::Cells,
        ) => {
            let gene = positions(&gene_ids, std::slice::from_ref(gene_id))?[0];
            Ok(layer(data, name)?.iter().map(|row| row[gene]).collect())
        }
        (
            Source::Profile {
                layer: Some(name),
                cell_id,
            },
            Axis::Genes,
        ) => {
            let cell = positions(&cell_ids, std::slice::from_ref(cell_id))?[0];
            Ok(layer(data, name)?[cell].clone())
        }
        (Source::Embedding { name, component }, _) => {
            let (_, rows) = data
                .embeddings
                .get(name)
                .ok_or_else(|| format!("missing embedding {name}"))?;
            rows.iter()
                .map(|row| {
                    row.get(*component as usize)
                        .copied()
                        .ok_or_else(|| format!("no component {component}"))
                })
                .collect()
        }
        (Source::Recompute { name }, Axis::Cells) => {
            let lineage = data
                .lineage
                .get(name)
                .ok_or_else(|| format!("no lineage for {name}"))?;
            Ok(layer(data, lineage)?
                .iter()
                .map(|row| row.iter().sum())
                .collect())
        }
        (other, _) => Err(format!("unexpected leaf {other:?}")),
    }
}
