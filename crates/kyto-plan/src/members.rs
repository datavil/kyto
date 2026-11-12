//! What each plan node reads and writes, by member: what the optimizer (task 4.4) must
//! know to move a node or leave it out.
//!
//! Does: name the members of an atlas ([`Member`]: a layer, a cell or gene column, an
//! embedding, a graph), list every member of a schema, and say which members a logical
//! node reads (its input layer, the leaves of its expressions, the lineage layer of a
//! recomputed column) and which it writes (`into`, the HVG columns, the PCA loadings,
//! the graph pair). A sink reads everything it writes out.
//! Does not: say anything about axes (a filter or a selection reads no member but
//! changes every member's rows; the rules ask [`crate::LogicalNode::changes_axis`]),
//! or check that a member exists (the builders did, R6).

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use kyto_core::{AtlasSchema, Axis, Expr, ExprAxis, ExprNode, Source};
use serde::{Deserialize, Serialize};

use crate::cross_cell::{CrossCellOp, hvg_columns, loadings_name};
use crate::filter::Predicate;
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;
use crate::with_columns::Columns;

const CELL_ID: &str = "cell_id";
const GENE_ID: &str = "gene_id";

/// One member of an atlas, by name: what a terminal reads and a node reads or writes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "name")]
pub enum Member {
    /// A layer.
    Layer(String),
    /// A column of the cell table.
    CellColumn(String),
    /// A column of the gene table.
    GeneColumn(String),
    /// An embedding (one group for both axes, format.md §2.1).
    Embedding(String),
    /// A graph.
    Graph(String),
}

impl Member {
    /// The column `name` of the table of `axis`.
    pub fn column(axis: Axis, name: impl Into<String>) -> Self {
        match axis {
            Axis::Cells => Self::CellColumn(name.into()),
            Axis::Genes => Self::GeneColumn(name.into()),
        }
    }
}

/// `layer:counts`, `cells:batch`, `genes:highly_variable`, `embedding:X_pca`,
/// `graph:connectivities`.
impl fmt::Display for Member {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, name) = match self {
            Self::Layer(name) => ("layer", name),
            Self::CellColumn(name) => ("cells", name),
            Self::GeneColumn(name) => ("genes", name),
            Self::Embedding(name) => ("embedding", name),
            Self::Graph(name) => ("graph", name),
        };
        write!(formatter, "{kind}:{name}")
    }
}

/// A set of members, in a fixed order.
pub type Members = BTreeSet<Member>;

/// Every member of `schema`: its layers, the columns of both tables (not the ID
/// columns, which always come along), its embeddings and graphs.
pub fn all_members(schema: &AtlasSchema) -> Members {
    let layers = schema
        .layers
        .iter()
        .map(|layer| Member::Layer(layer.name.clone()));
    let cells = schema
        .cells
        .columns
        .iter()
        .filter(|column| column.name != CELL_ID)
        .map(|column| Member::CellColumn(column.name.clone()));
    let genes = schema
        .genes
        .columns
        .iter()
        .filter(|column| column.name != GENE_ID)
        .map(|column| Member::GeneColumn(column.name.clone()));
    let embeddings = schema
        .embeddings
        .iter()
        .map(|embedding| Member::Embedding(embedding.name.clone()));
    let graphs = schema
        .graphs
        .iter()
        .map(|graph| Member::Graph(graph.name.clone()));
    layers
        .chain(cells)
        .chain(genes)
        .chain(embeddings)
        .chain(graphs)
        .collect()
}

/// Whether `member` is one of `schema`.
pub(crate) fn has_member(schema: &AtlasSchema, member: &Member) -> bool {
    match member {
        Member::Layer(name) => schema.layers.iter().any(|layer| layer.name == *name),
        Member::CellColumn(name) => schema
            .cells
            .columns
            .iter()
            .any(|column| column.name == *name),
        Member::GeneColumn(name) => schema
            .genes
            .columns
            .iter()
            .any(|column| column.name == *name),
        Member::Embedding(name) => schema
            .embeddings
            .iter()
            .any(|embedding| embedding.name == *name),
        Member::Graph(name) => schema.graphs.iter().any(|graph| graph.name == *name),
    }
}

/// The members a node reads from its input. A scan reads none (it is the input).
pub(crate) fn node_reads(plan: &Plan) -> Members {
    let mut reads = Members::new();
    let Some(input) = plan.input() else {
        return reads;
    };
    let input = input.output_schema();
    match plan {
        Plan::Scan(_) | Plan::WithDefaultLayer(_) | Plan::SelectGenes(_) | Plan::TakeCells(_) => {}
        Plan::Filter(node) => match node.params().predicate() {
            Predicate::Cells(predicate) => expr_reads(predicate, input, &mut reads),
            Predicate::Genes(predicate) => expr_reads(predicate, input, &mut reads),
        },
        Plan::MapElementwise(node) => {
            reads.insert(Member::Layer(node.params().layer().to_owned()));
        }
        Plan::CrossCell(node) => match node.params().op() {
            CrossCellOp::Scale { layer, .. } | CrossCellOp::Pca { layer, .. } => {
                reads.insert(Member::Layer(layer.clone()));
            }
            CrossCellOp::HighlyVariableGenes {
                layer, batch_key, ..
            } => {
                reads.insert(Member::Layer(layer.clone()));
                if let Some(batch_key) = batch_key {
                    reads.insert(Member::CellColumn(batch_key.clone()));
                }
            }
        },
        Plan::WithColumns(node) => match node.params().columns() {
            Columns::Cells(columns) => {
                for (_, expr) in columns {
                    expr_reads(expr, input, &mut reads);
                }
            }
            Columns::Genes(columns) => {
                for (_, expr) in columns {
                    expr_reads(expr, input, &mut reads);
                }
            }
        },
        Plan::Neighbors(node) => {
            reads.insert(Member::Embedding(node.params().embedding.clone()));
        }
        Plan::Cluster(node) => {
            reads.insert(Member::Graph(node.params().graph.clone()));
        }
        Plan::Sink(_) => reads = all_members(plan.schema()),
    }
    reads
}

/// The members a node writes (adds or replaces). A scan writes every member of its
/// store.
pub(crate) fn node_writes(plan: &Plan) -> Members {
    let mut writes = Members::new();
    match plan {
        Plan::Scan(node) => writes = all_members(node.params().schema()),
        Plan::WithDefaultLayer(_)
        | Plan::Filter(_)
        | Plan::SelectGenes(_)
        | Plan::TakeCells(_)
        | Plan::Sink(_) => {}
        Plan::MapElementwise(node) => {
            writes.insert(Member::Layer(node.params().output_layer().to_owned()));
        }
        Plan::CrossCell(node) => match node.params().op() {
            CrossCellOp::Scale { into, .. } => {
                writes.insert(Member::Layer(into.clone()));
            }
            CrossCellOp::HighlyVariableGenes {
                flavor, batch_key, ..
            } => {
                for column in hvg_columns(*flavor, batch_key.is_some()) {
                    writes.insert(Member::GeneColumn(column.name));
                }
            }
            CrossCellOp::Pca { into, .. } => {
                writes.insert(Member::Embedding(into.clone()));
                writes.insert(Member::Embedding(loadings_name(into)));
            }
        },
        Plan::WithColumns(node) => match node.params().columns() {
            Columns::Cells(columns) => {
                writes.extend(
                    columns
                        .iter()
                        .map(|(name, _)| Member::CellColumn(name.clone())),
                );
            }
            Columns::Genes(columns) => {
                writes.extend(
                    columns
                        .iter()
                        .map(|(name, _)| Member::GeneColumn(name.clone())),
                );
            }
        },
        Plan::Neighbors(node) => {
            let (distances, connectivities) = node.params().graph_names();
            writes.insert(Member::Graph(distances));
            writes.insert(Member::Graph(connectivities));
        }
        Plan::Cluster(node) => {
            writes.insert(Member::CellColumn(node.params().into.clone()));
        }
    }
    writes
}

/// Whether a filter's predicate depends on the genes present: it reads one gene's
/// values (an error once a filter on data removed that gene, `KY-E203`) or recomputes
/// a derived column (`.recompute()`, a sum over the genes present).
pub(crate) fn reads_genes(plan: &Plan) -> bool {
    let Plan::Filter(node) = plan else {
        return false;
    };
    let root = match node.params().predicate() {
        Predicate::Cells(predicate) => predicate.node(),
        Predicate::Genes(predicate) => predicate.node(),
    };
    let mut found = false;
    for_each_leaf(root, &mut |source| {
        found |= matches!(
            source,
            Source::Recompute { .. }
                | Source::Expression { .. }
                | Source::GeneSymbol { .. }
                | Source::Qc { .. }
        );
    });
    found
}

/// The members the resolved expression `expr` reads, added to `reads`. A recomputed
/// column reads the column (for its lineage) and the layer it is computed from.
fn expr_reads<A: ExprAxis>(expr: &Expr<A>, input: &PlanSchema, reads: &mut Members) {
    for_each_leaf(expr.node(), &mut |source| match source {
        Source::Column { name } => {
            reads.insert(Member::column(A::AXIS, name.clone()));
        }
        Source::Expression {
            layer: Some(layer), ..
        }
        | Source::Profile {
            layer: Some(layer), ..
        } => {
            reads.insert(Member::Layer(layer.clone()));
        }
        Source::Embedding { name, .. } => {
            reads.insert(Member::Embedding(name.clone()));
        }
        // A QC metric reads its layer and, for a gene set, the gene columns of its
        // predicate (task 6.1).
        Source::Qc { layer, metric } => {
            if let Some(layer) = layer {
                reads.insert(Member::Layer(layer.clone()));
            }
            if let Some(genes) = metric.genes() {
                expr_reads(genes, input, reads);
            }
        }
        Source::Recompute { name } => {
            reads.insert(Member::column(A::AXIS, name.clone()));
            if let Some(lineage) = input
                .column(A::AXIS, name)
                .and_then(|column| column.lineage.as_ref())
            {
                reads.insert(Member::Layer(lineage.layer.clone()));
            }
        }
        // A resolved expression names its layer and holds no symbol or feature leaf
        // (R6); nothing else to read.
        Source::Expression { layer: None, .. }
        | Source::Profile { layer: None, .. }
        | Source::GeneSymbol { .. }
        | Source::Feature { .. } => {}
    });
}

/// Calls `visit` on each leaf of the tree under `root`, with a loop (trees are up to
/// 1000 levels deep) that visits a shared subtree once.
pub(crate) fn for_each_leaf(root: &ExprNode, visit: &mut dyn FnMut(&Source)) {
    let mut seen: HashSet<*const ExprNode> = HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(node)) {
            continue;
        }
        match node {
            ExprNode::Source(source) => visit(source),
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
}
