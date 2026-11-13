//! Task 5.11: embeddings, graphs and recomputed columns in the executor.
//!
//! On a small store (120 cells in three groups, 30 genes, blocks of 16 cells):
//! - a sink writes the `pca` embedding and loadings, the `neighbors` graphs and the
//!   `umap` layout of the plan bit for bit, to `.kyto` and `.h5ad` (`obsm`, `varm`,
//!   `obsp`), and a store read back sinks to the same members (round trip);
//! - after a cell filter the rows are those of the kept cells and a graph keeps the
//!   edges between them, re-indexed (anndata's subset);
//! - `neighbors`, `leiden` and `umap` read members from the store, and run past a cell
//!   filter or `take_cells` (the rows, or the subgraph, of the cells reaching them);
//! - embedding components are expression leaves of cell and gene steps;
//! - a column a sink computed records lineage and recomputes to the same values on the
//!   store it wrote, and to the metric over the genes present after a gene filter;
//!   without a recipe it is `KY-E303`.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, GeneExpr, GeneIds, Literal, QcMetric, Tiling};
use kyto_exec::{CellGraph, ExecuteOptions, NodeStatistics, Pipeline, Statistics, cluster};
use kyto_plan::{
    Cluster, ClusterMethod, Metric, NeighborSearch, Neighbors, OptimizeOptions, Output,
    PhysicalOptions, Plan, Sink, Umap, loadings_name, physical_plan,
};
use kyto_store::{
    CellBlock, FloatValues, LayerSpec, LayerWriteOptions, StoreWriter, TableAxis, ValueDtype,
    open_atlas_schema, read_embedding, read_graph, read_ids, read_table,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

const N_CELLS: usize = 120;
const N_GENES: usize = 30;
const TILING: Tiling = Tiling {
    cell_block_size: 16,
    gene_block_size: 8,
};

/// Three groups of 40 cells, each expressing its own ten genes (plus a little noise).
fn write_store(target: &Path) {
    let gene_ids: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene}")).collect();
    let cell_ids: Vec<String> = (0..N_CELLS).map(|cell| format!("c{cell}")).collect();
    let groups: Vec<&str> = (0..N_CELLS)
        .map(|cell| ["a", "b", "c"][cell / 40])
        .collect();
    let genes = DataFrame::new(
        N_GENES,
        vec![Series::new("gene_id".into(), gene_ids).into_column()],
    )
    .unwrap();
    let cells = DataFrame::new(
        N_CELLS,
        vec![
            Series::new("cell_id".into(), cell_ids).into_column(),
            Series::new("group".into(), groups).into_column(),
        ],
    )
    .unwrap();
    let specs = [LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LayerWriteOptions::default(),
    }];
    let mut writer = StoreWriter::create(target, &genes, &specs, TILING, false).unwrap();
    let mut indptr = vec![0_u64];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for cell in 0..N_CELLS {
        let group = cell / 40;
        for gene in 0..N_GENES {
            let count = if gene / 10 == group {
                5 + (cell * 7 + gene * 3) % 11
            } else {
                (cell * 13 + gene * 5) % 9 / 7
            };
            if count > 0 {
                indices.push(gene as u32);
                data.push(count as u32);
            }
        }
        indptr.push(indices.len() as u64);
    }
    writer
        .push(&[CellBlock::from_u32(indptr, indices, data, N_GENES as u64).unwrap()])
        .unwrap();
    writer.finish(&cells, None).unwrap();
}

fn scan(store: &Path) -> Arc<Plan> {
    let schema = open_atlas_schema(store).unwrap();
    let gene_ids = GeneIds::new(read_ids(store, TableAxis::Genes, schema.n_genes).unwrap());
    Plan::scan(store.display().to_string(), schema, gene_ids)
}

fn neighbors_params(embedding: &str) -> Neighbors {
    Neighbors {
        embedding: embedding.to_owned(),
        n_components: None,
        n_neighbors: 6,
        metric: Metric::Euclidean,
        search: NeighborSearch::Exact,
        seed: 0,
        key: None,
    }
}

fn leiden_params() -> Cluster {
    Cluster {
        method: ClusterMethod::Leiden,
        graph: "connectivities".to_owned(),
        resolution: 1.0,
        seed: 3,
        into: "leiden".to_owned(),
    }
}

fn umap_params() -> Umap {
    Umap {
        n_epochs: Some(20),
        seed: 5,
        ..Umap::default()
    }
}

/// normalize/log1p -> pca(4) -> neighbors -> umap -> leiden.
fn analysis(store: &Path) -> Arc<Plan> {
    scan(store)
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap()
        .pca(Some("log1p"), "X_pca", 4, 0)
        .unwrap()
        .neighbors(neighbors_params("X_pca"))
        .unwrap()
        .umap(umap_params())
        .unwrap()
        .cluster(leiden_params())
        .unwrap()
}

fn options() -> ExecuteOptions {
    ExecuteOptions::new(NonZeroUsize::new(2).unwrap())
}

fn physical(plan: &Arc<Plan>) -> kyto_plan::PhysicalPlan {
    let options = PhysicalOptions {
        max_block_cells: NonZeroU64::new(16),
        ..PhysicalOptions::default()
    };
    physical_plan(&[Output::all(plan)], OptimizeOptions::ALL, options)
}

/// Every node's statistics of a run of `plan`.
fn statistics(plan: &Arc<Plan>) -> Statistics {
    let mut stream = Pipeline::new(&physical(plan), 0)
        .unwrap()
        .run(options())
        .unwrap();
    while let Some(block) = stream.next() {
        stream.recycle(block.unwrap());
    }
    stream.statistics().clone()
}

fn sink(plan: &Arc<Plan>, target: Sink) -> Result<u64, kyto_error::KytoError> {
    Pipeline::new(&physical(&plan.sink(target)?), 0)?.sink(options(), false)
}

fn zarr(path: &Path) -> Sink {
    Sink::Zarr {
        path: path.display().to_string(),
    }
}

fn f32s(values: &FloatValues) -> &[f32] {
    match values {
        FloatValues::F32(values) => values,
        FloatValues::F64(_) => panic!("expected float32"),
    }
}

/// The graph of `(indptr, indices, weights)` between the rows `kept` (in order).
fn induced(
    indptr: &[u64],
    indices: &[u32],
    weights: &[f32],
    kept: &[usize],
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut new_of = vec![None; indptr.len() - 1];
    for (new, old) in kept.iter().enumerate() {
        new_of[*old] = Some(new as u32);
    }
    let (mut out_indptr, mut out_indices, mut out_weights) = (vec![0], Vec::new(), Vec::new());
    for &row in kept {
        for entry in indptr[row] as usize..indptr[row + 1] as usize {
            if let Some(new) = new_of[indices[entry] as usize] {
                out_indices.push(new);
                out_weights.push(weights[entry]);
            }
        }
        out_indptr.push(out_indices.len() as u64);
    }
    (out_indptr, out_indices, out_weights)
}

#[test]
fn a_sink_writes_the_plans_embeddings_and_graphs_and_they_round_trip() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = analysis(&store);
    let found = statistics(&plan);
    let (mut pca, mut neighbors, mut umap, mut clustering) = (None, None, None, None);
    for (_, statistics) in found.iter() {
        match statistics {
            NodeStatistics::Pca(found) => pca = Some(found.clone()),
            NodeStatistics::Neighbors(found) => neighbors = Some(found.clone()),
            NodeStatistics::Umap(found) => umap = Some(found.clone()),
            NodeStatistics::Cluster(found) => clustering = Some(found.clone()),
            _ => {}
        }
    }
    let (pca, neighbors, umap, clustering) = (
        pca.unwrap(),
        neighbors.unwrap(),
        umap.unwrap(),
        clustering.unwrap(),
    );
    assert_eq!(pca.genes(), (0..N_GENES as u64).collect::<Vec<_>>());

    let out = work.path().join("out.kyto");
    assert_eq!(sink(&plan, zarr(&out)).unwrap(), N_CELLS as u64);
    let schema = open_atlas_schema(&out).unwrap();
    let names: Vec<&str> = schema.embeddings.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["X_pca", "X_pca_loadings", "X_umap"]);
    let graphs: Vec<&str> = schema.graphs.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(graphs, ["connectivities", "distances"]);
    assert_eq!(
        f32s(&read_embedding(&out, "X_pca").unwrap().values),
        pca.embedding()
    );
    assert_eq!(
        f32s(
            &read_embedding(&out, &loadings_name("X_pca"))
                .unwrap()
                .values
        ),
        pca.loadings()
    );
    assert_eq!(
        f32s(&read_embedding(&out, "X_umap").unwrap().values),
        umap.layout().embedding()
    );
    let distances = read_graph(&out, "distances").unwrap();
    assert_eq!(distances.indptr, neighbors.distances().indptr());
    assert_eq!(distances.indices, neighbors.distances().indices());
    assert_eq!(f32s(&distances.weights), neighbors.distances().distances());
    let connectivities = read_graph(&out, "connectivities").unwrap();
    assert_eq!(connectivities.indices, neighbors.connectivities().indices());
    assert_eq!(
        f32s(&connectivities.weights),
        neighbors.connectivities().weights()
    );
    let labels = read_table(&out, TableAxis::Cells).unwrap();
    let labels: Vec<String> = labels
        .column("leiden")
        .unwrap()
        .cast(&polars::prelude::DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|label| label.unwrap().to_owned())
        .collect();
    let expected: Vec<String> = clustering
        .clustering()
        .labels()
        .iter()
        .map(u32::to_string)
        .collect();
    assert_eq!(labels, expected);

    // Round trip: the written store sinks again to the same members.
    let again = work.path().join("again.kyto");
    sink(&scan(&out), zarr(&again)).unwrap();
    assert_eq!(
        open_atlas_schema(&again).unwrap().embeddings,
        schema.embeddings
    );
    assert_eq!(open_atlas_schema(&again).unwrap().graphs, schema.graphs);
    for name in ["X_pca", "X_pca_loadings", "X_umap"] {
        assert_eq!(
            read_embedding(&again, name).unwrap(),
            read_embedding(&out, name).unwrap()
        );
    }
    for name in ["distances", "connectivities"] {
        assert_eq!(
            read_graph(&again, name).unwrap(),
            read_graph(&out, name).unwrap()
        );
    }

    // `.h5ad`: obsm, varm and obsp as anndata writes them.
    let h5ad = work.path().join("out.h5ad");
    sink(
        &scan(&out),
        Sink::H5ad {
            path: h5ad.display().to_string(),
        },
    )
    .unwrap();
    let file = hdf5_metno::File::open(&h5ad).unwrap();
    let x_pca = file.dataset("obsm/X_pca").unwrap();
    assert_eq!(x_pca.shape(), [N_CELLS, 4]);
    assert_eq!(x_pca.read_raw::<f32>().unwrap(), pca.embedding());
    let loadings = file.dataset("varm/X_pca_loadings").unwrap();
    assert_eq!(loadings.shape(), [N_GENES, 4]);
    assert_eq!(file.dataset("obsm/X_umap").unwrap().shape(), [N_CELLS, 2]);
    let obsp = file.group("obsp/connectivities").unwrap();
    let indptr: Vec<u64> = obsp
        .dataset("indptr")
        .unwrap()
        .read_raw::<i32>()
        .unwrap()
        .iter()
        .map(|&offset| offset as u64)
        .collect();
    assert_eq!(indptr, neighbors.connectivities().indptr());
    assert_eq!(
        obsp.dataset("data").unwrap().read_raw::<f32>().unwrap(),
        neighbors.connectivities().weights()
    );
    assert!(file.group("obsp/distances").is_ok());
}

#[test]
fn a_cell_filter_before_the_sink_keeps_the_rows_and_edges_of_the_kept_cells() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let base = scan(&store)
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap()
        .pca(Some("log1p"), "X_pca", 3, 0)
        .unwrap()
        .neighbors(neighbors_params("X_pca"))
        .unwrap();
    let found = statistics(&base);
    let (mut pca, mut neighbors) = (None, None);
    for (_, statistics) in found.iter() {
        match statistics {
            NodeStatistics::Pca(found) => pca = Some(found.clone()),
            NodeStatistics::Neighbors(found) => neighbors = Some(found.clone()),
            _ => {}
        }
    }
    let (pca, neighbors) = (pca.unwrap(), neighbors.unwrap());
    // Keep the cells of groups `a` and `c`, and a few of `b` (every third).
    let keep = Expr::binary(
        Expr::binary(
            CellExpr::column("group"),
            BinaryOp::NotEq,
            Literal::String("b".into()),
        )
        .unwrap(),
        BinaryOp::Or,
        Expr::binary(
            CellExpr::column("cell_id"),
            BinaryOp::Eq,
            Literal::String("c42".into()),
        )
        .unwrap(),
    )
    .unwrap();
    let kept: Vec<usize> = (0..N_CELLS)
        .filter(|cell| cell / 40 != 1 || *cell == 42)
        .collect();
    let out = work.path().join("out.kyto");
    let written = sink(&base.filter_cells(&keep).unwrap(), zarr(&out)).unwrap();
    assert_eq!(written, kept.len() as u64);

    let rows: Vec<f32> = kept
        .iter()
        .flat_map(|cell| pca.embedding()[cell * 3..cell * 3 + 3].iter().copied())
        .collect();
    assert_eq!(f32s(&read_embedding(&out, "X_pca").unwrap().values), rows);
    let graph = neighbors.connectivities();
    let (indptr, indices, weights) =
        induced(graph.indptr(), graph.indices(), graph.weights(), &kept);
    let connectivities = read_graph(&out, "connectivities").unwrap();
    assert_eq!(connectivities.indptr, indptr);
    assert_eq!(connectivities.indices, indices);
    assert_eq!(f32s(&connectivities.weights), weights);
    // c42's neighbours were all in group b: no edge is left in its row.
    let row = kept.iter().position(|cell| *cell == 42).unwrap();
    assert!(connectivities.indptr[row + 1] - connectivities.indptr[row] <= 6);
    assert_eq!(schema_n_cells(&out), kept.len() as u64);
}

fn schema_n_cells(store: &Path) -> u64 {
    open_atlas_schema(store).unwrap().n_cells
}

#[test]
fn neighbors_leiden_and_umap_read_members_of_the_store_and_run_past_filters() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let plan = analysis(&store);
    let out = work.path().join("out.kyto");
    sink(&plan, zarr(&out)).unwrap();
    let original = statistics(&plan);
    let original_neighbors = original
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Neighbors(found) => Some(found.clone()),
            _ => None,
        })
        .unwrap();

    // The stored X_pca gives the same graph; leiden on it the same labels.
    let stored = scan(&out)
        .neighbors(neighbors_params("X_pca"))
        .unwrap()
        .cluster(leiden_params())
        .unwrap();
    let found = statistics(&stored);
    let neighbors = found
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Neighbors(found) => Some(found.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(neighbors.distances(), original_neighbors.distances());
    assert_eq!(
        neighbors.connectivities(),
        original_neighbors.connectivities()
    );

    // Leiden on the stored connectivities of the cells `take_cells` keeps: the subgraph.
    let taken: Vec<usize> = (0..N_CELLS).filter(|cell| cell % 4 != 1).collect();
    let plan = scan(&out)
        .take_cells(taken.iter().map(|cell| format!("c{cell}")))
        .unwrap()
        .cluster(leiden_params())
        .unwrap();
    let found = statistics(&plan);
    let clustering = found
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Cluster(found) => Some(found.clone()),
            _ => None,
        })
        .unwrap();
    let graph = original_neighbors.connectivities();
    let (indptr, indices, weights) =
        induced(graph.indptr(), graph.indices(), graph.weights(), &taken);
    let expected = cluster(
        CellGraph::new(&indptr, &indices, &weights).unwrap(),
        &leiden_params(),
    )
    .unwrap();
    assert_eq!(
        clustering.cells(),
        taken.iter().map(|cell| *cell as u64).collect::<Vec<_>>()
    );
    assert_eq!(clustering.clustering(), &expected);

    // umap on the stored graph, and neighbors past a filter on the stored embedding.
    let plan = scan(&out)
        .take_cells(taken.iter().map(|cell| format!("c{cell}")))
        .unwrap()
        .neighbors(Neighbors {
            key: Some("again".to_owned()),
            ..neighbors_params("X_pca")
        })
        .unwrap()
        .umap(Umap {
            graph: "again_connectivities".to_owned(),
            ..umap_params()
        })
        .unwrap()
        .umap(Umap {
            graph: "connectivities".to_owned(),
            into: "X_umap_stored".to_owned(),
            ..umap_params()
        })
        .unwrap();
    let found = statistics(&plan);
    let mut layouts = 0;
    for (_, statistics) in found.iter() {
        match statistics {
            NodeStatistics::Neighbors(found) => assert_eq!(found.cells().len(), taken.len()),
            NodeStatistics::Umap(found) => {
                assert_eq!(found.cells().len(), taken.len());
                assert!(
                    found
                        .layout()
                        .embedding()
                        .iter()
                        .all(|value| value.is_finite())
                );
                layouts += 1;
            }
            _ => {}
        }
    }
    assert_eq!(layouts, 2);
}

#[test]
fn embedding_components_are_cell_and_gene_leaves() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let pca = scan(&store)
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "log1p")
        .unwrap()
        .pca(Some("log1p"), "X_pca", 3, 0)
        .unwrap();
    let found = statistics(&pca);
    let pca_found = found
        .iter()
        .find_map(|(_, statistics)| match statistics {
            NodeStatistics::Pca(found) => Some(found.clone()),
            _ => None,
        })
        .unwrap();
    // In the plan: a cell column from the embedding, a filter on it, a gene column from
    // the loadings.
    let positive = Expr::binary(
        CellExpr::embedding("X_pca", 0),
        BinaryOp::Gt,
        Literal::Float(0.0),
    )
    .unwrap();
    let plan = pca
        .with_cells([("pc2".to_owned(), CellExpr::embedding("X_pca", 1))])
        .unwrap()
        .filter_cells(&positive)
        .unwrap()
        .with_genes([(
            "load1".to_owned(),
            GeneExpr::embedding(loadings_name("X_pca"), 0),
        )])
        .unwrap();
    let out = work.path().join("out.kyto");
    sink(&plan, zarr(&out)).unwrap();
    let kept: Vec<usize> = (0..N_CELLS)
        .filter(|cell| pca_found.embedding()[cell * 3] > 0.0)
        .collect();
    assert!(!kept.is_empty() && kept.len() < N_CELLS);
    let cells = read_table(&out, TableAxis::Cells).unwrap();
    let pc2: Vec<f32> = cells
        .column("pc2")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let expected: Vec<f32> = kept
        .iter()
        .map(|cell| pca_found.embedding()[cell * 3 + 1])
        .collect();
    assert_eq!(pc2, expected);
    let genes = read_table(&out, TableAxis::Genes).unwrap();
    let load1: Vec<f32> = genes
        .column("load1")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let expected: Vec<f32> = (0..N_GENES)
        .map(|gene| pca_found.loadings()[gene * 3])
        .collect();
    assert_eq!(load1, expected);

    // From the store: the written embedding's rows are the kept cells'.
    let again = work.path().join("again.kyto");
    let plan = scan(&out)
        .with_cells([("first".to_owned(), CellExpr::embedding("X_pca", 0))])
        .unwrap();
    sink(&plan, zarr(&again)).unwrap();
    let first: Vec<f32> = read_table(&again, TableAxis::Cells)
        .unwrap()
        .column("first")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let expected: Vec<f32> = kept
        .iter()
        .map(|cell| pca_found.embedding()[cell * 3])
        .collect();
    assert_eq!(first, expected);
}

fn u32_column(store: &Path, name: &str) -> Vec<u32> {
    read_table(store, TableAxis::Cells)
        .unwrap()
        .column(name)
        .unwrap()
        .u32()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

#[test]
fn a_recomputed_column_equals_the_column_the_plan_wrote() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let n_genes = || CellExpr::qc(Some("counts".to_owned()), QcMetric::GenesDetected).unwrap();
    let written = work.path().join("qc.kyto");
    sink(
        &scan(&store)
            .with_cells([("n_genes".to_owned(), n_genes())])
            .unwrap(),
        zarr(&written),
    )
    .unwrap();
    let schema = open_atlas_schema(&written).unwrap();
    let column = schema
        .cells
        .columns
        .iter()
        .find(|c| c.name == "n_genes")
        .unwrap();
    let lineage = column
        .lineage
        .clone()
        .expect("a computed QC column records lineage");
    assert_eq!(
        (lineage.layer.as_str(), lineage.node.as_str()),
        ("counts", "1")
    );

    // On the store it wrote, the recipe gives the stored values exactly.
    let recompute = || CellExpr::column("n_genes").recompute().unwrap();
    let again = work.path().join("again.kyto");
    sink(
        &scan(&written)
            .with_cells([("again".to_owned(), recompute())])
            .unwrap(),
        zarr(&again),
    )
    .unwrap();
    assert_eq!(u32_column(&again, "again"), u32_column(&again, "n_genes"));
    // A filter on the recomputed column runs too.
    let busy = Expr::binary(recompute(), BinaryOp::Gt, Literal::Int(9)).unwrap();
    let filtered = work.path().join("busy.kyto");
    let kept = sink(
        &scan(&written).filter_cells(&busy).unwrap(),
        zarr(&filtered),
    )
    .unwrap();
    let expected = u32_column(&written, "n_genes")
        .iter()
        .filter(|n| **n > 9)
        .count();
    assert_eq!(kept, expected as u64);

    // After a gene selection it is the metric over the genes present.
    let selected = work.path().join("selected.kyto");
    sink(
        &scan(&written)
            .select_genes((0..15).map(|gene| format!("G{gene}")))
            .unwrap()
            .with_cells([
                ("recomputed".to_owned(), recompute()),
                ("direct".to_owned(), n_genes()),
            ])
            .unwrap(),
        zarr(&selected),
    )
    .unwrap();
    let recomputed = u32_column(&selected, "recomputed");
    assert_eq!(recomputed, u32_column(&selected, "direct"));
    assert_ne!(recomputed, u32_column(&written, "n_genes"));

    // Lineage is not recorded when a later node changes the genes or rewrites the layer.
    let lineage_of = |plan: Arc<Plan>, name: &str| {
        let target = work.path().join(format!("{name}.kyto"));
        sink(&plan, zarr(&target)).unwrap();
        let schema = open_atlas_schema(&target).unwrap();
        schema
            .cells
            .columns
            .iter()
            .find(|column| column.name == "n_genes")
            .unwrap()
            .lineage
            .clone()
    };
    let with_qc = scan(&store)
        .with_cells([("n_genes".to_owned(), n_genes())])
        .unwrap();
    let genes_changed = with_qc
        .select_genes((0..20).map(|gene| format!("G{gene}")))
        .unwrap();
    assert_eq!(lineage_of(genes_changed, "genes_changed"), None);
    let rewritten = with_qc.log1p(Some("counts"), "counts").unwrap();
    assert_eq!(lineage_of(rewritten, "rewritten"), None);

    // A store that carries the column over keeps its lineage but not its recipe.
    let carried = work.path().join("carried.kyto");
    sink(&scan(&written), zarr(&carried)).unwrap();
    let plan = scan(&carried)
        .with_cells([("again".to_owned(), recompute())])
        .unwrap()
        .sink(zarr(&work.path().join("never.kyto")))
        .unwrap();
    let error = Pipeline::new(&physical(&plan), 0).unwrap_err();
    assert_eq!(error.code(), "KY-E303");
    assert!(error.to_string().contains("does not compute it"), "{error}");
    assert!(!work.path().join("never.kyto").exists());
}

/// A QC gene set reads gene columns: lineage is recorded only while those reach the sink
/// as the QC read them, so a recompute on the new store gives the stored values.
#[test]
fn lineage_needs_the_gene_set_columns_unchanged_at_the_sink() {
    let work = tempfile::tempdir().unwrap();
    let store = work.path().join("atlas.kyto");
    write_store(&store);
    let flag = |gene: &str| {
        Expr::binary(
            GeneExpr::column("gene_id"),
            BinaryOp::NotEq,
            Literal::String(gene.to_owned()),
        )
        .unwrap()
    };
    let in_set = || {
        CellExpr::qc(
            Some("counts".to_owned()),
            QcMetric::TotalCountsIn {
                genes: GeneExpr::column("flag"),
            },
        )
        .unwrap()
    };
    let with_qc = scan(&store)
        .with_genes([("flag".to_owned(), flag("G1"))])
        .unwrap()
        .with_cells([("tin".to_owned(), in_set())])
        .unwrap();
    let lineage_and_table = |plan: &Arc<Plan>, name: &str| {
        let target = work.path().join(format!("{name}.kyto"));
        sink(plan, zarr(&target)).unwrap();
        let schema = open_atlas_schema(&target).unwrap();
        let lineage = schema
            .cells
            .columns
            .iter()
            .find(|column| column.name == "tin")
            .unwrap()
            .lineage
            .clone();
        (lineage, target)
    };

    // The gene set's column reaches the sink unchanged: lineage, and the recipe gives
    // the stored values.
    let (lineage, written) = lineage_and_table(&with_qc, "kept");
    assert!(lineage.is_some());
    let again = work.path().join("again.kyto");
    sink(
        &scan(&written)
            .with_cells([(
                "again".to_owned(),
                CellExpr::column("tin").recompute().unwrap(),
            )])
            .unwrap(),
        zarr(&again),
    )
    .unwrap();
    let table = read_table(&again, TableAxis::Cells).unwrap();
    let stored = table
        .column("tin")
        .unwrap()
        .as_materialized_series()
        .clone();
    let recomputed = table
        .column("again")
        .unwrap()
        .as_materialized_series()
        .clone();
    assert!(recomputed.equals_missing(&stored.cast(recomputed.dtype()).unwrap()));

    // A later node redefines the gene set's column: no lineage (the recipe would read
    // the new definition).
    let redefined = with_qc
        .with_genes([("flag".to_owned(), flag("G15"))])
        .unwrap();
    assert_eq!(lineage_and_table(&redefined, "redefined").0, None);
}
