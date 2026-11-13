//! Embeddings and graphs in a store (format.md §6, §7; task 5.11): the store writer
//! writes them, `read_embedding`/`read_graph` read them back exactly, `schema.json`
//! lists them sorted by name, the `.h5ad` export writes `obsm`, `varm` and `obsp` as
//! anndata does, and bad input or a store that disagrees with its schema gets coded
//! errors (R10).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::path::Path;

use hdf5_metno::File;
use kyto_core::{Axis, FloatDtype};
use kyto_store::{
    Embedding, FloatValues, Graph, SinkH5adOptions, StoreWriter, Tiling, read_atlas_schema,
    read_embedding, read_graph, sink_h5ad,
};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

fn table(id_column: &str, prefix: &str, n: u64) -> DataFrame {
    let ids: Vec<String> = (0..n).map(|row| format!("{prefix}{row}")).collect();
    DataFrame::new(
        n as usize,
        vec![Series::new(id_column.into(), ids).into_column()],
    )
    .unwrap()
}

/// Five cells, three genes, no layers; tiny blocks so embeddings span several chunks.
const N_CELLS: u64 = 5;
const N_GENES: u64 = 3;

fn tiling() -> Tiling {
    Tiling {
        cell_block_size: 2,
        gene_block_size: 2,
    }
}

fn cell_embedding() -> Embedding {
    Embedding {
        name: "X_pca".to_owned(),
        axis: Axis::Cells,
        width: 3,
        values: FloatValues::F32((0..15).map(|value| value as f32 * 0.5 - 2.0).collect()),
    }
}

fn gene_embedding() -> Embedding {
    Embedding {
        name: "PCs".to_owned(),
        axis: Axis::Genes,
        width: 2,
        values: FloatValues::F64(vec![1.0, -0.0, f64::NAN, 1e300, 0.25, -3.5]),
    }
}

/// Row 1 empty, a weight of 0 stored, a self loop.
fn graph() -> Graph {
    Graph {
        name: "connectivities".to_owned(),
        indptr: vec![0, 2, 2, 3, 5, 6],
        indices: vec![1, 4, 2, 0, 3, 2],
        weights: FloatValues::F32(vec![0.5, 1.0, 0.0, 0.25, 0.75, 1.0]),
    }
}

fn write(
    target: &Path,
    embeddings: Vec<Embedding>,
    graphs: Vec<Graph>,
) -> Result<(), kyto_error::KytoError> {
    let mut writer = StoreWriter::create(
        target,
        &table("gene_id", "g", N_GENES),
        &[],
        tiling(),
        false,
    )?;
    for embedding in embeddings {
        writer.add_embedding(embedding);
    }
    for graph in graphs {
        writer.add_graph(graph);
    }
    writer.set_plan(serde_json::json!({"format": "kyto-plan", "version": 1, "nodes": []}));
    writer.finish(&table("cell_id", "c", N_CELLS), None)
}

fn same_bits(left: &FloatValues, right: &FloatValues) -> bool {
    match (left, right) {
        (FloatValues::F32(left), FloatValues::F32(right)) => left
            .iter()
            .map(|value| value.to_bits())
            .eq(right.iter().map(|value| value.to_bits())),
        (FloatValues::F64(left), FloatValues::F64(right)) => left
            .iter()
            .map(|value| value.to_bits())
            .eq(right.iter().map(|value| value.to_bits())),
        _ => false,
    }
}

#[test]
fn embeddings_and_graphs_round_trip_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("a.kyto");
    write(
        &store,
        vec![cell_embedding(), gene_embedding()],
        vec![graph()],
    )
    .unwrap();

    let schema = read_atlas_schema(&store).unwrap();
    let names: Vec<&str> = schema.embeddings.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["PCs", "X_pca"], "sorted by name (byte order)");
    assert_eq!(schema.embeddings[0].dtype, FloatDtype::Float64);
    assert_eq!(schema.embeddings[1].width, 3);
    assert_eq!(schema.graphs.len(), 1);
    assert_eq!(schema.graphs[0].nnz, 6);

    for expected in [cell_embedding(), gene_embedding()] {
        let read = read_embedding(&store, &expected.name).unwrap();
        assert_eq!((read.axis, read.width), (expected.axis, expected.width));
        assert!(
            same_bits(&read.values, &expected.values),
            "{}",
            expected.name
        );
    }
    assert_eq!(read_graph(&store, "connectivities").unwrap(), graph());

    // Attributes and chunking as format.md §6, §7.
    let array: serde_json::Value =
        serde_json::from_slice(&std::fs::read(store.join("embeddings/X_pca/zarr.json")).unwrap())
            .unwrap();
    assert_eq!(array["attributes"]["kyto"]["axis"], "cells");
    assert_eq!(array["shape"], serde_json::json!([5, 3]));
    assert_eq!(
        array["chunk_grid"]["configuration"]["chunk_shape"],
        serde_json::json!([2, 3])
    );
    assert_eq!(
        array["dimension_names"],
        serde_json::json!(["cell", "component"])
    );
    let group: serde_json::Value = serde_json::from_slice(
        &std::fs::read(store.join("graphs/connectivities/zarr.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        group["attributes"]["kyto"],
        serde_json::json!({"kind": "csr", "axis": "cells"})
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(store.join("provenance.json")).unwrap()).unwrap();
    assert_eq!(provenance["plan"]["format"], "kyto-plan");
}

#[test]
fn the_h5ad_export_writes_obsm_varm_and_obsp() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("a.kyto");
    write(
        &store,
        vec![cell_embedding(), gene_embedding()],
        vec![graph()],
    )
    .unwrap();
    let target = dir.path().join("a.h5ad");
    sink_h5ad(&store, &target, &SinkH5adOptions::default()).unwrap();

    let file = File::open(&target).unwrap();
    let encoding = |member: &str| -> String {
        let attribute = file.dataset(member).unwrap().attr("encoding-type").unwrap();
        attribute
            .read_scalar::<hdf5_metno::types::VarLenUnicode>()
            .unwrap()
            .to_string()
    };
    let x_pca = file.dataset("obsm/X_pca").unwrap();
    assert_eq!(x_pca.shape(), [5, 3]);
    assert_eq!(
        x_pca.read_raw::<f32>().unwrap(),
        cell_embedding().values.to_f32()
    );
    assert_eq!(encoding("obsm/X_pca"), "array");
    let pcs = file.dataset("varm/PCs").unwrap();
    assert_eq!(pcs.shape(), [3, 2]);
    let FloatValues::F64(expected) = gene_embedding().values else {
        unreachable!()
    };
    let read = pcs.read_raw::<f64>().unwrap();
    assert!(
        read.iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    let graph = graph();
    let obsp = file.group("obsp/connectivities").unwrap();
    assert_eq!(
        obsp.dataset("indptr").unwrap().read_raw::<i32>().unwrap(),
        [0, 2, 2, 3, 5, 6]
    );
    assert_eq!(
        obsp.dataset("indices").unwrap().read_raw::<i32>().unwrap(),
        graph
            .indices
            .iter()
            .map(|&index| index as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        obsp.dataset("data").unwrap().read_raw::<f32>().unwrap(),
        graph.weights.to_f32()
    );
    let shape = obsp.attr("shape").unwrap().read_raw::<i64>().unwrap();
    assert_eq!(shape, [5, 5]);
}

#[test]
fn bad_members_get_coded_errors() {
    let dir = tempfile::tempdir().unwrap();
    let code = |result: Result<(), kyto_error::KytoError>| result.unwrap_err().code();

    let mut short = cell_embedding();
    short.values = FloatValues::F32(vec![0.0; 12]);
    assert_eq!(
        code(write(&dir.path().join("1"), vec![short], vec![])),
        "KY-E104"
    );
    let mut wide = gene_embedding();
    wide.axis = Axis::Cells;
    assert_eq!(
        code(write(&dir.path().join("2"), vec![wide], vec![])),
        "KY-E104"
    );

    let mut unsorted = graph();
    unsorted.indices.swap(0, 1);
    let error = write(&dir.path().join("3"), vec![], vec![unsorted]).unwrap_err();
    assert_eq!(error.code(), "KY-E104");
    assert!(error.to_string().contains("strictly increasing"), "{error}");
    let mut beyond = graph();
    beyond.indices[1] = 5;
    assert_eq!(
        code(write(&dir.path().join("4"), vec![], vec![beyond])),
        "KY-E104"
    );

    let mut twin = cell_embedding();
    twin.name = "x_PCA".to_owned();
    assert_eq!(
        code(write(
            &dir.path().join("5"),
            vec![cell_embedding(), twin],
            vec![]
        )),
        "KY-E103"
    );
    let mut bad_name = graph();
    bad_name.name = "a/b".to_owned();
    assert_eq!(
        code(write(&dir.path().join("6"), vec![], vec![bad_name])),
        "KY-E102"
    );
    // Nothing was left behind by the failed writes.
    assert!(!dir.path().join("1").exists() && !dir.path().join("6").exists());

    // Reads: a missing member, and a store whose schema disagrees with the arrays.
    let store = dir.path().join("ok.kyto");
    write(&store, vec![cell_embedding()], vec![graph()]).unwrap();
    let error = read_embedding(&store, "X_pcb").unwrap_err();
    assert_eq!(error.code(), "KY-E109");
    assert!(
        error.to_string().contains("X_pca"),
        "suggests the close name: {error}"
    );
    assert_eq!(
        read_graph(&store, "distances").unwrap_err().code(),
        "KY-E109"
    );
    let schema_path = store.join("schema.json");
    let text = std::fs::read_to_string(&schema_path).unwrap();
    std::fs::write(&schema_path, text.replace("\"width\": 3", "\"width\": 4")).unwrap();
    let error = read_embedding(&store, "X_pca").unwrap_err();
    assert_eq!(error.code(), "KY-E126");
    assert!(error.to_string().contains("shape"), "{error}");
    std::fs::write(&schema_path, text.replace("\"nnz\": 6", "\"nnz\": 5")).unwrap();
    assert_eq!(
        read_graph(&store, "connectivities").unwrap_err().code(),
        "KY-E126"
    );
}
