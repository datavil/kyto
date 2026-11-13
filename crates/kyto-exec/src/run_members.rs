//! Embeddings and graphs in a run (task 5.11): where each one comes from at a step of
//! the stream, and its rows as the run holds them.
//!
//! Does: name the source of an embedding ([`EmbeddingSource`]: the scanned store, a
//! `pca`'s embedding or loadings, a `umap` layout) or a graph ([`GraphSource`]: the
//! store, a `neighbors` node's distances or connectivities), as `Pipeline::new`
//! resolves them in plan order; give their rows keyed by store position ([`Rows`],
//! [`KeyedGraph`]), so a step reads the row of each cell (or gene) it holds, a sink
//! writes the rows of the cells it keeps, and a graph is re-indexed to the cells a
//! filter kept (the subgraph between them, as anndata subsets `obsp`); read the stored
//! ones once per run ([`StoredMembers`]); and stream the cells a step sees
//! ([`collect_cells`]) when a cell filter or `take_cells` lies between it and the
//! member it reads.
//! Does not: decide which source a name has (the pipeline walks the plan), or compute
//! anything new (the stages do).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use kyto_store::{FloatValues, read_embedding, read_graph};
use zarrs::storage::ReadableStorageTraits;

use crate::block_stream::{StreamSource, Windows};
use crate::cell_block_pool::CellBlockPool;
use crate::statistics::{NodeStatistics, Statistics};

/// Where an embedding comes from, at a step of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingSource {
    /// The scanned store's embedding of that name.
    Stored { name: String, axis: Axis },
    /// The cell embedding of the `pca` node at this index of the optimized plan.
    Pca { node: usize },
    /// The gene loadings of the `pca` node at this index.
    Loadings { node: usize },
    /// The layout of the `umap` node at this index.
    Umap { node: usize },
}

impl EmbeddingSource {
    /// The axis its rows follow.
    pub(crate) fn axis(&self) -> Axis {
        match self {
            Self::Stored { axis, .. } => *axis,
            Self::Loadings { .. } => Axis::Genes,
            Self::Pca { .. } | Self::Umap { .. } => Axis::Cells,
        }
    }
}

/// Where a graph comes from, at a step of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphSource {
    /// The scanned store's graph of that name.
    Stored { name: String },
    /// The distances (or connectivities) of the `neighbors` node at this index.
    Neighbors { node: usize, distances: bool },
}

/// An embedding's rows: row `r` is store position `keys[r]` (a cell, or a gene for
/// gene embeddings), `width` values a row.
#[derive(Debug, Clone)]
pub(crate) struct Rows {
    pub(crate) keys: Arc<Vec<u64>>,
    pub(crate) width: usize,
    pub(crate) values: FloatValues,
    /// The row of each store position, `u32::MAX` for none.
    row_of: Vec<u32>,
}

impl Rows {
    pub(crate) fn new(keys: Arc<Vec<u64>>, width: usize, values: FloatValues) -> Self {
        let row_of = row_index(&keys);
        Self {
            keys,
            width,
            values,
            row_of,
        }
    }

    /// The row of store position `key`, if it has one.
    pub(crate) fn row(&self, key: u64) -> Option<usize> {
        let row = *self.row_of.get(usize::try_from(key).ok()?)?;
        (row != u32::MAX).then_some(row as usize)
    }

    /// The rows of store positions `keys`, in that order; `KY-E126` naming `name` for a
    /// position without a row (a step kept a cell or gene the producer never saw).
    pub(crate) fn take(
        &self,
        keys: &[u64],
        kind: &str,
        name: &str,
    ) -> Result<FloatValues, KytoError> {
        let rows = keys
            .iter()
            .map(|&key| {
                self.row(key).ok_or_else(|| KytoError::CorruptMember {
                    kind: kind.to_owned(),
                    name: name.to_owned(),
                    problem: format!("store position {key} has no row in it"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let width = self.width;
        Ok(self
            .values
            .take(rows.iter().flat_map(|row| row * width..(row + 1) * width)))
    }
}

/// A cell graph whose row `r` (and column `r`) is the cell at store position
/// `cells[r]`.
#[derive(Debug, Clone)]
pub(crate) struct KeyedGraph {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) indptr: Vec<u64>,
    pub(crate) indices: Vec<u32>,
    pub(crate) weights: FloatValues,
}

impl KeyedGraph {
    /// The graph between the cells at store positions `kept` (in that order: row `i` is
    /// `kept[i]`): each kept cell's edges to other kept cells, re-indexed, as anndata
    /// subsets `obsp`. `KY-E126` naming `name` for a kept cell without a row.
    pub(crate) fn induced(&self, kept: &[u64], name: &str) -> Result<Self, KytoError> {
        let row_of = row_index(&self.cells);
        let new_of = row_index(kept);
        let lookup = |index: &[u32], key: u64| {
            usize::try_from(key)
                .ok()
                .and_then(|key| index.get(key))
                .copied()
                .filter(|row| *row != u32::MAX)
        };
        let mut indptr = Vec::with_capacity(kept.len() + 1);
        indptr.push(0_u64);
        let mut indices = Vec::new();
        let mut taken = Vec::new();
        let mut row_entries: Vec<(u32, usize)> = Vec::new();
        for &cell in kept {
            let row = lookup(&row_of, cell).ok_or_else(|| KytoError::CorruptMember {
                kind: "graph".to_owned(),
                name: name.to_owned(),
                problem: format!("the cell at store position {cell} has no row in it"),
            })? as usize;
            row_entries.clear();
            for entry in self.indptr[row] as usize..self.indptr[row + 1] as usize {
                let neighbour = self.cells[self.indices[entry] as usize];
                if let Some(new) = lookup(&new_of, neighbour) {
                    row_entries.push((new, entry));
                }
            }
            row_entries.sort_unstable();
            for (new, entry) in &row_entries {
                indices.push(*new);
                taken.push(*entry);
            }
            indptr.push(indices.len() as u64);
        }
        Ok(Self {
            cells: Arc::new(kept.to_vec()),
            indptr,
            indices,
            weights: self.weights.take(taken.into_iter()),
        })
    }
}

/// The position of each key in `keys`, indexed by key (`u32::MAX` for none).
fn row_index(keys: &[u64]) -> Vec<u32> {
    let size = keys.iter().max().map_or(0, |max| *max as usize + 1);
    let mut index = vec![u32::MAX; size];
    for (row, key) in keys.iter().enumerate() {
        index[*key as usize] = row as u32;
    }
    index
}

/// The scanned store's embeddings and graphs a run reads, each read once.
#[derive(Debug)]
pub(crate) struct StoredMembers {
    store: String,
    embeddings: BTreeMap<String, Arc<Rows>>,
    graphs: BTreeMap<String, Arc<KeyedGraph>>,
}

impl StoredMembers {
    pub(crate) fn new(store: &str) -> Self {
        Self {
            store: store.to_owned(),
            embeddings: BTreeMap::new(),
            graphs: BTreeMap::new(),
        }
    }

    fn embedding(&mut self, name: &str) -> Result<Arc<Rows>, KytoError> {
        if let Some(rows) = self.embeddings.get(name) {
            return Ok(Arc::clone(rows));
        }
        let embedding = read_embedding(Path::new(&self.store), name)?;
        let keys = Arc::new((0..embedding.n_rows()).collect());
        let width = usize::try_from(embedding.width).unwrap_or(usize::MAX);
        let rows = Arc::new(Rows::new(keys, width, embedding.values));
        self.embeddings.insert(name.to_owned(), Arc::clone(&rows));
        Ok(rows)
    }

    fn graph(&mut self, name: &str) -> Result<Arc<KeyedGraph>, KytoError> {
        if let Some(graph) = self.graphs.get(name) {
            return Ok(Arc::clone(graph));
        }
        let graph = read_graph(Path::new(&self.store), name)?;
        let keyed = Arc::new(KeyedGraph {
            cells: Arc::new((0..graph.n_cells()).collect()),
            indptr: graph.indptr,
            indices: graph.indices,
            weights: graph.weights,
        });
        self.graphs.insert(name.to_owned(), Arc::clone(&keyed));
        Ok(keyed)
    }

    /// The rows of the embedding from `source`, once its producer has run.
    pub(crate) fn rows(
        &mut self,
        source: &EmbeddingSource,
        statistics: &Statistics,
    ) -> Result<Arc<Rows>, KytoError> {
        let rows = match (source, statistics_of(source_node(source), statistics)) {
            (EmbeddingSource::Stored { name, .. }, _) => return self.embedding(name),
            (EmbeddingSource::Pca { .. }, Some(NodeStatistics::Pca(pca))) => Rows::new(
                Arc::clone(&pca.cells),
                pca.n_components(),
                FloatValues::F32(pca.embedding().to_vec()),
            ),
            (EmbeddingSource::Loadings { .. }, Some(NodeStatistics::Pca(pca))) => Rows::new(
                Arc::clone(&pca.genes),
                pca.n_components(),
                FloatValues::F32(pca.loadings().to_vec()),
            ),
            (EmbeddingSource::Umap { .. }, Some(NodeStatistics::Umap(umap))) => Rows::new(
                Arc::clone(&umap.cells),
                umap.layout().n_components(),
                FloatValues::F32(umap.layout().embedding().to_vec()),
            ),
            _ => unreachable!("an embedding's producer runs before the steps that read it"),
        };
        Ok(Arc::new(rows))
    }

    /// The graph from `source`, once its producer has run.
    pub(crate) fn keyed_graph(
        &mut self,
        source: &GraphSource,
        statistics: &Statistics,
    ) -> Result<Arc<KeyedGraph>, KytoError> {
        match source {
            GraphSource::Stored { name } => self.graph(name),
            GraphSource::Neighbors { node, distances } => {
                let Some(NodeStatistics::Neighbors(found)) = statistics.get(*node) else {
                    unreachable!("a graph's neighbors node runs before the steps that read it")
                };
                let (indptr, indices, weights) = if *distances {
                    let graph = found.distances();
                    (graph.indptr(), graph.indices(), graph.distances())
                } else {
                    let graph = found.connectivities();
                    (graph.indptr(), graph.indices(), graph.weights())
                };
                Ok(Arc::new(KeyedGraph {
                    cells: Arc::clone(&found.cells),
                    indptr: indptr.to_vec(),
                    indices: indices.to_vec(),
                    weights: FloatValues::F32(weights.to_vec()),
                }))
            }
        }
    }
}

fn source_node(source: &EmbeddingSource) -> Option<usize> {
    match source {
        EmbeddingSource::Stored { .. } => None,
        EmbeddingSource::Pca { node }
        | EmbeddingSource::Loadings { node }
        | EmbeddingSource::Umap { node } => Some(*node),
    }
}

fn statistics_of(node: Option<usize>, statistics: &Statistics) -> Option<&NodeStatistics> {
    node.and_then(|node| statistics.get(node))
}

/// The store positions of the cells `source`'s stream carries, in stream order: one
/// pass through the steps so far (task 5.11), reading layers only if a step needs
/// them.
///
/// # Errors
/// Any error of the stream.
pub(crate) fn collect_cells<S>(
    source: &StreamSource<S>,
    thread_pool: &rayon::ThreadPool,
) -> Result<Vec<u64>, KytoError>
where
    S: ?Sized + ReadableStorageTraits + Send + Sync + 'static,
{
    let pool = CellBlockPool::new(source.window.get());
    thread_pool.install(|| {
        let mut cells = Vec::new();
        let mut windows = Windows::open(source, &pool)?;
        while let Some(items) = windows.next_window() {
            for item in items {
                let block = item?;
                cells.extend_from_slice(&block.cells);
                for (_, layer) in block.layers {
                    pool.recycle(layer);
                }
            }
        }
        Ok(cells)
    })
}
