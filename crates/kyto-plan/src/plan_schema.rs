//! The output schema of a plan node: what the plan holds after that node (task 4.1).
//!
//! Does: hold the members and columns a node outputs as an `AtlasSchema`, whose sizes
//! (`n_cells`, `n_genes`, every `nnz`) are upper bounds, next to the sizes known
//! exactly; give the small edits nodes make (add or replace a column, layer,
//! embedding or graph; mark an axis as changed).
//! Does not: check names (the plan builders do, task 4.2 adds the rest) or know how
//! a node computes its output.
//!
//! Like Polars' `IR::schema()`, a node's schema follows from its input's schema and its
//! own parameters alone, without reading data. Polars schemas have no row count; kyto's
//! keep the bounds because the memory limit (5.5) and the cost estimate (4.6) need them.

use std::collections::BTreeSet;

use kyto_core::{
    AtlasSchema, Axis, ColumnSchema, EmbeddingSchema, GraphSchema, LayerSchema, TableSchema,
};
use serde::{Deserialize, Serialize};

/// A sparse member whose `nnz` the schema records.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SparseMember {
    /// A layer, by name.
    Layer(String),
    /// A graph, by name.
    Graph(String),
}

/// What a plan holds after one node: members, columns and sizes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanSchema {
    /// Members and columns, exact; `n_cells`, `n_genes` and each `nnz` are upper
    /// bounds (exact unless listed below), the sizes to plan memory with.
    atlas: AtlasSchema,
    /// The number of cells, when known without running the plan.
    n_cells: Option<u64>,
    /// The number of genes, when known without running the plan.
    n_genes: Option<u64>,
    /// The sparse members whose `nnz` in `atlas` is only an upper bound.
    bounded_nnz: BTreeSet<SparseMember>,
}

impl PlanSchema {
    /// The schema of a scan: the store's, every size exact.
    pub fn of_store(atlas: AtlasSchema) -> Self {
        Self {
            n_cells: Some(atlas.n_cells),
            n_genes: Some(atlas.n_genes),
            atlas,
            bounded_nnz: BTreeSet::new(),
        }
    }

    /// Members and columns; sizes are upper bounds (see [`PlanSchema::n_cells`]).
    pub fn atlas(&self) -> &AtlasSchema {
        &self.atlas
    }

    /// The number of cells, or `None` when only the bound `atlas().n_cells` is known
    /// (after a filter on cells).
    pub fn n_cells(&self) -> Option<u64> {
        self.n_cells
    }

    /// The number of genes, or `None` when only the bound `atlas().n_genes` is known
    /// (after a filter on genes).
    pub fn n_genes(&self) -> Option<u64> {
        self.n_genes
    }

    /// The exact length of `axis`, if known.
    pub fn axis_len(&self, axis: Axis) -> Option<u64> {
        match axis {
            Axis::Cells => self.n_cells,
            Axis::Genes => self.n_genes,
        }
    }

    /// The upper bound of the length of `axis`.
    pub fn axis_len_bound(&self, axis: Axis) -> u64 {
        match axis {
            Axis::Cells => self.atlas.n_cells,
            Axis::Genes => self.atlas.n_genes,
        }
    }

    /// The exact `nnz` of `member`, or `None` if it is only bounded or not a member.
    pub fn nnz(&self, member: &SparseMember) -> Option<u64> {
        if self.bounded_nnz.contains(member) {
            return None;
        }
        match member {
            SparseMember::Layer(name) => self.layer(name).map(|layer| layer.nnz),
            SparseMember::Graph(name) => self.graph(name).map(|graph| graph.nnz),
        }
    }

    /// Whether the `nnz` of `member` is exact.
    pub(crate) fn nnz_is_exact(&self, member: &SparseMember) -> bool {
        !self.bounded_nnz.contains(member)
    }

    /// The table of `axis`.
    pub fn table(&self, axis: Axis) -> &TableSchema {
        match axis {
            Axis::Cells => &self.atlas.cells,
            Axis::Genes => &self.atlas.genes,
        }
    }

    /// The column `name` of the table of `axis`.
    pub fn column(&self, axis: Axis, name: &str) -> Option<&ColumnSchema> {
        self.table(axis)
            .columns
            .iter()
            .find(|column| column.name == name)
    }

    /// The layer `name`.
    pub fn layer(&self, name: &str) -> Option<&LayerSchema> {
        self.atlas.layers.iter().find(|layer| layer.name == name)
    }

    /// The embedding `name`.
    pub fn embedding(&self, name: &str) -> Option<&EmbeddingSchema> {
        self.atlas
            .embeddings
            .iter()
            .find(|embedding| embedding.name == name)
    }

    /// The graph `name`.
    pub fn graph(&self, name: &str) -> Option<&GraphSchema> {
        self.atlas.graphs.iter().find(|graph| graph.name == name)
    }

    /// This schema after a node that keeps some of `axis` (a filter or a selection):
    /// `exact` is the new length if known, `bound` its upper bound. Every layer's
    /// `nnz`, and that of every graph over `axis`, becomes a bound.
    pub(crate) fn with_axis_changed(mut self, axis: Axis, exact: Option<u64>, bound: u64) -> Self {
        match axis {
            Axis::Cells => {
                self.n_cells = exact;
                self.atlas.n_cells = bound;
            }
            Axis::Genes => {
                self.n_genes = exact;
                self.atlas.n_genes = bound;
            }
        }
        for layer in &self.atlas.layers {
            self.bounded_nnz
                .insert(SparseMember::Layer(layer.name.clone()));
        }
        for graph in self.atlas.graphs.iter().filter(|graph| graph.axis == axis) {
            self.bounded_nnz
                .insert(SparseMember::Graph(graph.name.clone()));
        }
        self
    }

    /// This schema with `column` in the table of `axis`: it replaces a column of the
    /// same name in place, or is appended (Polars' `with_columns` order).
    pub(crate) fn with_column(mut self, axis: Axis, column: ColumnSchema) -> Self {
        let table = match axis {
            Axis::Cells => &mut self.atlas.cells,
            Axis::Genes => &mut self.atlas.genes,
        };
        match table
            .columns
            .iter_mut()
            .find(|stored| stored.name == column.name)
        {
            Some(stored) => *stored = column,
            None => table.columns.push(column),
        }
        self
    }

    /// This schema with `layer`, replacing a layer of the same name in place or
    /// appended; `exact` says whether its `nnz` is exact.
    pub(crate) fn with_layer(mut self, layer: LayerSchema, exact: bool) -> Self {
        let member = SparseMember::Layer(layer.name.clone());
        upsert(&mut self.atlas.layers, layer, |stored, new| {
            stored.name == new.name
        });
        self.mark(member, exact);
        self
    }

    /// This schema with `embedding`, replacing one of the same name in place or
    /// appended.
    pub(crate) fn with_embedding(mut self, embedding: EmbeddingSchema) -> Self {
        upsert(&mut self.atlas.embeddings, embedding, |stored, new| {
            stored.name == new.name
        });
        self
    }

    /// This schema with `graph`, replacing one of the same name in place or appended;
    /// `exact` says whether its `nnz` is exact.
    pub(crate) fn with_graph(mut self, graph: GraphSchema, exact: bool) -> Self {
        let member = SparseMember::Graph(graph.name.clone());
        upsert(&mut self.atlas.graphs, graph, |stored, new| {
            stored.name == new.name
        });
        self.mark(member, exact);
        self
    }

    fn mark(&mut self, member: SparseMember, exact: bool) {
        if exact {
            self.bounded_nnz.remove(&member);
        } else {
            self.bounded_nnz.insert(member);
        }
    }
}

/// Replace the first item of `items` that `same` matches with `item`, or append it.
fn upsert<T>(items: &mut Vec<T>, item: T, same: impl Fn(&T, &T) -> bool) {
    match items.iter().position(|stored| same(stored, &item)) {
        Some(position) => items[position] = item,
        None => items.push(item),
    }
}
