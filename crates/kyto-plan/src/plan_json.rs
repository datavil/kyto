//! The plan JSON (task 4.3): a versioned text form of a plan, read back into an equal
//! plan.
//!
//! Does: write a plan as `{"format": "kyto-plan", "version": 1, "nodes": [...]}`, its
//! nodes from the scan up, each with its stable [`NodeId`] (`id`, the position from
//! the scan), the ID of its input (`input`, `null` for the scan) and its step: the plan
//! method that builds it (`method`) and that method's arguments as the plan stored them,
//! names resolved in full (R6). The scan carries the snapshot the plan was built on:
//! the store location, its `schema.json` and its gene IDs and symbols, so reading the
//! text back needs no store (and reads none). Expressions are [`ExprJson`] lists.
//! Read it back by replaying each step through its builder on the plan read so far:
//! every check of 4.2 runs again, so text that was edited into an invalid plan is
//! refused with the builder's own error, and a plan written by [`Plan::to_json`]
//! comes back equal, writing the same text again.
//! Does not: open or check the store (the executor does when the plan runs), or carry
//! state derived from the steps (output schemas, removed genes: rebuilt by replay).
//!
//! Versions: [`PLAN_JSON_VERSION`] is the one version written and read; another is
//! `KY-E218`, checked first, as soon as the text parses as a JSON object. Any other
//! problem with the text is `KY-E217`, naming the line and column or the node.
//!
//! What is checked: every step, by its builder; the scan's snapshot only as far as the
//! store reader checks a `schema.json` (no modalities, positive block sizes), plus
//! unique names and gene IDs matching the schema. The rest of the snapshot is trusted,
//! as a scan trusts the store it reads; the executor checks the store when it runs.
//!
//! `input` makes each node's edge explicit, so a plan with joins (none planned, 4.1
//! Open (9)) can be written in the same shape; this version accepts only chains:
//! node `i` has ID `i` and input `i - 1`.

use std::sync::Arc;

use kyto_core::{AtlasSchema, ExprJson, GeneIds};
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::cluster::{Cluster, ClusterMethod};
use crate::cross_cell::{CrossCellOp, HvgFlavor};
use crate::filter::Predicate;
use crate::map_elementwise::ElementwiseOp;
use crate::neighbors::{Metric, NeighborSearch, Neighbors};
use crate::node::NodeId;
use crate::plan::Plan;
use crate::scrublet::Scrublet;
use crate::sink::Sink;
use crate::umap::{Umap, UmapInit};
use crate::with_columns::Columns;

/// The `format` value of every plan JSON.
pub const PLAN_JSON_FORMAT: &str = "kyto-plan";

/// The plan JSON version this kyto writes and reads.
pub const PLAN_JSON_VERSION: u64 = 1;

/// The whole text.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanJson {
    format: String,
    version: u64,
    nodes: Vec<NodeJson>,
}

/// What is read first: the version and format, of any JSON type, whatever else the
/// text holds (so another version's shape cannot hide that it is another version).
#[derive(Deserialize)]
struct Header {
    version: Option<serde_json::Value>,
    format: Option<serde_json::Value>,
}

/// One node: its ID, its input's ID and its step.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeJson {
    id: NodeId,
    input: Option<NodeId>,
    step: Step,
}

/// A step: the plan method that builds the node, and its arguments as stored.
#[derive(Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum Step {
    Scan {
        source: String,
        schema: AtlasSchema,
        gene_ids: Vec<String>,
        /// One per gene (`null` for none), or `null` when the store has no symbols.
        symbols: Option<Vec<Option<String>>>,
    },
    WithDefaultLayer {
        layer: String,
    },
    FilterCells {
        predicate: ExprJson,
    },
    FilterGenes {
        predicate: ExprJson,
    },
    SelectGenes {
        gene_ids: Vec<String>,
    },
    TakeCells {
        cell_ids: Vec<String>,
    },
    NormalizeTotal {
        layer: String,
        into: String,
        target_sum: Option<f64>,
    },
    Log1p {
        layer: String,
        into: String,
    },
    Scale {
        layer: String,
        into: String,
        zero_center: bool,
        max_value: Option<f64>,
    },
    HighlyVariableGenes {
        layer: String,
        flavor: HvgFlavor,
        n_top_genes: Option<u64>,
        batch_key: Option<String>,
    },
    Pca {
        layer: String,
        into: String,
        n_components: u64,
        seed: u64,
    },
    Scrublet {
        layer: String,
        batch_key: Option<String>,
        sim_doublet_ratio: f64,
        expected_doublet_rate: f64,
        stdev_doublet_rate: f64,
        n_prin_comps: u64,
        n_neighbors: Option<u64>,
        threshold: Option<f64>,
        seed: u64,
        score_into: String,
        predicted_into: String,
    },
    WithCells {
        columns: Vec<ColumnJson>,
    },
    WithGenes {
        columns: Vec<ColumnJson>,
    },
    Neighbors {
        embedding: String,
        n_components: Option<u64>,
        n_neighbors: u64,
        metric: Metric,
        search: NeighborSearch,
        seed: u64,
        key: Option<String>,
    },
    Leiden {
        graph: String,
        resolution: f64,
        seed: u64,
        into: String,
    },
    Louvain {
        graph: String,
        resolution: f64,
        seed: u64,
        into: String,
    },
    Umap {
        graph: String,
        n_components: u64,
        min_dist: f64,
        spread: f64,
        n_epochs: Option<u64>,
        alpha: f64,
        gamma: f64,
        negative_sample_rate: u64,
        init: UmapInit,
        seed: u64,
        into: String,
    },
    SinkZarr {
        path: String,
    },
    SinkH5ad {
        path: String,
    },
}

/// A named column of `with_cells` / `with_genes`.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnJson {
    name: String,
    expr: ExprJson,
}

impl Step {
    /// The step of `node` (its parameters only).
    fn of(node: &Plan) -> Self {
        match node {
            Plan::Scan(scan) => {
                let scan = scan.params();
                let gene_ids = scan.gene_ids();
                Self::Scan {
                    source: scan.source().to_owned(),
                    schema: AtlasSchema::clone(scan.schema()),
                    gene_ids: gene_ids.ordered().to_vec(),
                    symbols: gene_ids.symbols().map(|symbols| {
                        symbols
                            .into_iter()
                            .map(|symbol| symbol.map(str::to_owned))
                            .collect()
                    }),
                }
            }
            Plan::WithDefaultLayer(node) => Self::WithDefaultLayer {
                layer: node.params().layer().to_owned(),
            },
            Plan::Filter(node) => match node.params().predicate() {
                Predicate::Cells(predicate) => Self::FilterCells {
                    predicate: ExprJson::from_expr(predicate),
                },
                Predicate::Genes(predicate) => Self::FilterGenes {
                    predicate: ExprJson::from_expr(predicate),
                },
            },
            Plan::SelectGenes(node) => Self::SelectGenes {
                gene_ids: node.params().gene_ids().to_vec(),
            },
            Plan::TakeCells(node) => Self::TakeCells {
                cell_ids: node.params().cell_ids().to_vec(),
            },
            Plan::MapElementwise(node) => {
                let params = node.params();
                let layer = params.layer().to_owned();
                let into = params.output_layer().to_owned();
                match params.op() {
                    ElementwiseOp::NormalizeTotal { target_sum } => Self::NormalizeTotal {
                        layer,
                        into,
                        target_sum,
                    },
                    ElementwiseOp::Log1p => Self::Log1p { layer, into },
                }
            }
            Plan::CrossCell(node) => match node.params().op().clone() {
                CrossCellOp::Scale {
                    layer,
                    into,
                    zero_center,
                    max_value,
                } => Self::Scale {
                    layer,
                    into,
                    zero_center,
                    max_value,
                },
                CrossCellOp::HighlyVariableGenes {
                    layer,
                    flavor,
                    n_top_genes,
                    batch_key,
                } => Self::HighlyVariableGenes {
                    layer,
                    flavor,
                    n_top_genes,
                    batch_key,
                },
                CrossCellOp::Pca {
                    layer,
                    into,
                    n_components,
                    seed,
                } => Self::Pca {
                    layer,
                    into,
                    n_components,
                    seed,
                },
                CrossCellOp::Scrublet { layer, params } => {
                    let Scrublet {
                        batch_key,
                        sim_doublet_ratio,
                        expected_doublet_rate,
                        stdev_doublet_rate,
                        n_prin_comps,
                        n_neighbors,
                        threshold,
                        seed,
                        score_into,
                        predicted_into,
                    } = params;
                    Self::Scrublet {
                        layer,
                        batch_key,
                        sim_doublet_ratio,
                        expected_doublet_rate,
                        stdev_doublet_rate,
                        n_prin_comps,
                        n_neighbors,
                        threshold,
                        seed,
                        score_into,
                        predicted_into,
                    }
                }
            },
            Plan::WithColumns(node) => match node.params().columns() {
                Columns::Cells(columns) => Self::WithCells {
                    columns: columns_json(columns),
                },
                Columns::Genes(columns) => Self::WithGenes {
                    columns: columns_json(columns),
                },
            },
            Plan::Neighbors(node) => {
                let Neighbors {
                    embedding,
                    n_components,
                    n_neighbors,
                    metric,
                    search,
                    seed,
                    key,
                } = node.params().clone();
                Self::Neighbors {
                    embedding,
                    n_components,
                    n_neighbors,
                    metric,
                    search,
                    seed,
                    key,
                }
            }
            Plan::Cluster(node) => {
                let Cluster {
                    method,
                    graph,
                    resolution,
                    seed,
                    into,
                } = node.params().clone();
                match method {
                    ClusterMethod::Leiden => Self::Leiden {
                        graph,
                        resolution,
                        seed,
                        into,
                    },
                    ClusterMethod::Louvain => Self::Louvain {
                        graph,
                        resolution,
                        seed,
                        into,
                    },
                }
            }
            Plan::Umap(node) => {
                let Umap {
                    graph,
                    n_components,
                    min_dist,
                    spread,
                    n_epochs,
                    alpha,
                    gamma,
                    negative_sample_rate,
                    init,
                    seed,
                    into,
                } = node.params().clone();
                Self::Umap {
                    graph,
                    n_components,
                    min_dist,
                    spread,
                    n_epochs,
                    alpha,
                    gamma,
                    negative_sample_rate,
                    init,
                    seed,
                    into,
                }
            }
            Plan::Sink(node) => match node.params().clone() {
                Sink::Zarr { path } => Self::SinkZarr { path },
                Sink::H5ad { path } => Self::SinkH5ad { path },
            },
        }
    }

    /// The plan method, as `method` in the text.
    fn method(&self) -> &'static str {
        match self {
            Self::Scan { .. } => "scan",
            Self::WithDefaultLayer { .. } => "with_default_layer",
            Self::FilterCells { .. } => "filter_cells",
            Self::FilterGenes { .. } => "filter_genes",
            Self::SelectGenes { .. } => "select_genes",
            Self::TakeCells { .. } => "take_cells",
            Self::NormalizeTotal { .. } => "normalize_total",
            Self::Log1p { .. } => "log1p",
            Self::Scale { .. } => "scale",
            Self::HighlyVariableGenes { .. } => "highly_variable_genes",
            Self::Pca { .. } => "pca",
            Self::Scrublet { .. } => "scrublet",
            Self::WithCells { .. } => "with_cells",
            Self::WithGenes { .. } => "with_genes",
            Self::Neighbors { .. } => "neighbors",
            Self::Leiden { .. } => "leiden",
            Self::Louvain { .. } => "louvain",
            Self::Umap { .. } => "umap",
            Self::SinkZarr { .. } => "sink_zarr",
            Self::SinkH5ad { .. } => "sink_h5ad",
        }
    }

    /// `input` extended by this step, through its builder (every check runs again).
    /// `node` names the node in `KY-E217` messages. An error is located within the
    /// step (`/predicate` for its expression, empty for the builder's own errors).
    fn apply(self, input: &Arc<Plan>, node: &str) -> Result<Arc<Plan>, Located> {
        let built = match self {
            Self::Scan { .. } => {
                return Err(Located::new(
                    "/method",
                    invalid(node, "only the first node is a scan"),
                ));
            }
            Self::WithDefaultLayer { layer } => input.with_default_layer(&layer),
            Self::FilterCells { predicate: text } => input.filter_cells(&predicate(&text, node)?),
            Self::FilterGenes { predicate: text } => input.filter_genes(&predicate(&text, node)?),
            Self::SelectGenes { gene_ids } => input.select_genes(gene_ids),
            Self::TakeCells { cell_ids } => input.take_cells(cell_ids),
            Self::NormalizeTotal {
                layer,
                into,
                target_sum,
            } => input.normalize_total(target_sum, Some(&layer), &into),
            Self::Log1p { layer, into } => input.log1p(Some(&layer), &into),
            Self::Scale {
                layer,
                into,
                zero_center,
                max_value,
            } => input.scale(Some(&layer), &into, zero_center, max_value),
            Self::HighlyVariableGenes {
                layer,
                flavor,
                n_top_genes,
                batch_key,
            } => {
                input.highly_variable_genes(Some(&layer), flavor, n_top_genes, batch_key.as_deref())
            }
            Self::Pca {
                layer,
                into,
                n_components,
                seed,
            } => input.pca(Some(&layer), &into, n_components, seed),
            Self::Scrublet {
                layer,
                batch_key,
                sim_doublet_ratio,
                expected_doublet_rate,
                stdev_doublet_rate,
                n_prin_comps,
                n_neighbors,
                threshold,
                seed,
                score_into,
                predicted_into,
            } => input.scrublet(
                Some(&layer),
                Scrublet {
                    batch_key,
                    sim_doublet_ratio,
                    expected_doublet_rate,
                    stdev_doublet_rate,
                    n_prin_comps,
                    n_neighbors,
                    threshold,
                    seed,
                    score_into,
                    predicted_into,
                },
            ),
            Self::WithCells { columns } => {
                input.with_cells(columns_from_json::<kyto_core::CellAxis>(columns, node)?)
            }
            Self::WithGenes { columns } => {
                input.with_genes(columns_from_json::<kyto_core::GeneAxis>(columns, node)?)
            }
            Self::Neighbors {
                embedding,
                n_components,
                n_neighbors,
                metric,
                search,
                seed,
                key,
            } => input.neighbors(Neighbors {
                embedding,
                n_components,
                n_neighbors,
                metric,
                search,
                seed,
                key,
            }),
            Self::Leiden {
                graph,
                resolution,
                seed,
                into,
            } => input.cluster(Cluster {
                method: ClusterMethod::Leiden,
                graph,
                resolution,
                seed,
                into,
            }),
            Self::Louvain {
                graph,
                resolution,
                seed,
                into,
            } => input.cluster(Cluster {
                method: ClusterMethod::Louvain,
                graph,
                resolution,
                seed,
                into,
            }),
            Self::Umap {
                graph,
                n_components,
                min_dist,
                spread,
                n_epochs,
                alpha,
                gamma,
                negative_sample_rate,
                init,
                seed,
                into,
            } => input.umap(Umap {
                graph,
                n_components,
                min_dist,
                spread,
                n_epochs,
                alpha,
                gamma,
                negative_sample_rate,
                init,
                seed,
                into,
            }),
            Self::SinkZarr { path } => input.sink(Sink::Zarr { path }),
            Self::SinkH5ad { path } => input.sink(Sink::H5ad { path }),
        };
        Located::at("", built)
    }
}

/// The expression of a filter step, an error located at `/predicate`.
fn predicate<A: kyto_core::ExprAxis>(
    predicate: &ExprJson,
    node: &str,
) -> Result<kyto_core::Expr<A>, Located> {
    let context = format!("{node}, predicate");
    predicate
        .to_expr(&context)
        .map_err(|error| Located::in_expression("/predicate", &context, error))
}

fn columns_json<A: kyto_core::ExprAxis>(
    columns: &[(String, kyto_core::Expr<A>)],
) -> Vec<ColumnJson> {
    columns
        .iter()
        .map(|(name, expr)| ColumnJson {
            name: name.clone(),
            expr: ExprJson::from_expr(expr),
        })
        .collect()
}

fn columns_from_json<A: kyto_core::ExprAxis>(
    columns: Vec<ColumnJson>,
    node: &str,
) -> Result<Vec<(String, kyto_core::Expr<A>)>, Located> {
    columns
        .into_iter()
        .enumerate()
        .map(|(index, column)| {
            let context = format!("{node}, column '{}'", column.name);
            let expr = column.expr.to_expr(&context).map_err(|error| {
                Located::in_expression(&format!("/columns/{index}/expr"), &context, error)
            })?;
            Ok((column.name, expr))
        })
        .collect()
}

/// An error reading a plan JSON and where it is: a JSON Pointer (RFC 6901) into the
/// text, the empty pointer for the whole text (task 8.4's diagnostics carry it;
/// `from_json` drops it).
pub(crate) struct Located {
    pub(crate) pointer: String,
    /// Boxed: a `Result` carrying it stays small.
    pub(crate) error: Box<KytoError>,
}

impl Located {
    fn new(pointer: impl Into<String>, error: KytoError) -> Self {
        Self {
            pointer: pointer.into(),
            error: Box::new(error),
        }
    }

    /// `result` with its error located at `pointer`.
    fn at<T>(pointer: impl Into<String>, result: Result<T, KytoError>) -> Result<T, Self> {
        result.map_err(|error| Self::new(pointer, error))
    }

    /// An error reading the expression at `pointer` (written in messages as
    /// `context`), located at its entry when the error names one: `ExprJson` writes
    /// `{context} entry 3`, and `{context} entry 3, genes entry 1` for an entry of a QC
    /// metric's gene set (`/3/genes/1`).
    fn in_expression(pointer: &str, context: &str, error: KytoError) -> Self {
        let mut pointer = pointer.to_owned();
        if let KytoError::InvalidPlanJson { location, .. } = &error
            && let Some(mut rest) = location.strip_prefix(context)
        {
            let mut entries = String::new();
            let mut first = true;
            loop {
                let marker = if first { " entry " } else { ", genes entry " };
                let Some(after) = rest.strip_prefix(marker) else {
                    break;
                };
                let digits =
                    after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
                if digits == 0 {
                    break;
                }
                if !first {
                    entries.push_str("/genes");
                }
                entries.push('/');
                entries.push_str(&after[..digits]);
                rest = &after[digits..];
                first = false;
            }
            if rest.is_empty() {
                pointer.push_str(&entries);
            }
        }
        Self::new(pointer, error)
    }

    /// `KY-E217` for a `serde_json` error, for the whole text.
    pub(crate) fn syntax(error: &serde_json::Error) -> Self {
        Self::new("", syntax(error))
    }

    /// This error, located under `prefix`.
    fn under(mut self, prefix: &str) -> Self {
        self.pointer.insert_str(0, prefix);
        self
    }
}

/// `KY-E217` at `location`.
fn invalid(location: impl Into<String>, problem: impl Into<String>) -> KytoError {
    KytoError::InvalidPlanJson {
        location: location.into(),
        problem: problem.into(),
    }
}

/// `KY-E217` for a `serde_json` error: its line and column, and its reason.
fn syntax(error: &serde_json::Error) -> KytoError {
    let mut problem = error.to_string();
    // serde_json appends " at line L column C"; the location says it once.
    if let Some(cut) = problem.rfind(" at line ") {
        problem.truncate(cut);
    }
    invalid(
        format!("line {} column {}", error.line(), error.column()),
        problem,
    )
}

/// The scan of a `Step::Scan`, its gene IDs and symbols checked against its schema.
/// Errors are located within the step.
fn scan(
    source: String,
    schema: AtlasSchema,
    gene_ids: Vec<String>,
    symbols: Option<Vec<Option<String>>>,
) -> Result<Arc<Plan>, Located> {
    let location = "node 0 (scan)";
    if let Some((pointer, problem)) = schema_problem(&schema) {
        return Err(Located::new(
            format!("/schema{pointer}"),
            invalid(location, problem),
        ));
    }
    let n_genes = gene_ids.len() as u64;
    if n_genes != schema.n_genes {
        return Err(Located::new(
            "/gene_ids",
            invalid(
                location,
                format!(
                    "it lists {n_genes} gene IDs, and its schema has n_genes = {}",
                    schema.n_genes
                ),
            ),
        ));
    }
    if let Some(symbols) = &symbols
        && symbols.len() != gene_ids.len()
    {
        return Err(Located::new(
            "/symbols",
            invalid(
                location,
                format!(
                    "it lists {} symbols for {} genes; write one per gene (null for none)",
                    symbols.len(),
                    gene_ids.len()
                ),
            ),
        ));
    }
    let mut ids = GeneIds::new(gene_ids);
    if ids.len() as u64 != n_genes {
        return Err(Located::new(
            "/gene_ids",
            invalid(location, "its gene IDs are not unique"),
        ));
    }
    if let Some(symbols) = symbols {
        ids = ids.with_symbols(symbols);
    }
    Ok(Plan::scan(source, schema, ids))
}

/// What makes `schema` one no store has: the checks the store reader makes beyond
/// serde (format v0 stores no modalities, block sizes are positive), and names that
/// repeat within a member group or a table (the writers refuse them; a plan built on
/// them would resolve a name to whichever came first). With the problem, where it is
/// within the schema (a JSON Pointer).
fn schema_problem(schema: &AtlasSchema) -> Option<(String, String)> {
    if !schema.modalities.is_empty() {
        return Some((
            "/modalities".to_owned(),
            "its schema lists modalities, which format 0.1 stores do not have".to_owned(),
        ));
    }
    for (name, size) in [
        ("cell_block_size", schema.tiling.cell_block_size),
        ("gene_block_size", schema.tiling.gene_block_size),
    ] {
        if size == 0 {
            return Some((
                format!("/tiling/{name}"),
                format!("its schema has {name} = 0; block sizes are positive"),
            ));
        }
    }
    let groups: [(&str, &str, Vec<&str>); 5] = [
        (
            "layer",
            "/layers",
            schema
                .layers
                .iter()
                .map(|layer| layer.name.as_str())
                .collect(),
        ),
        (
            "embedding",
            "/embeddings",
            schema
                .embeddings
                .iter()
                .map(|embedding| embedding.name.as_str())
                .collect(),
        ),
        (
            "graph",
            "/graphs",
            schema
                .graphs
                .iter()
                .map(|graph| graph.name.as_str())
                .collect(),
        ),
        (
            "cell column",
            "/cells/columns",
            schema
                .cells
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .collect(),
        ),
        (
            "gene column",
            "/genes/columns",
            schema
                .genes
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .collect(),
        ),
    ];
    for (kind, pointer, names) in groups {
        let mut seen = std::collections::HashSet::new();
        if let Some((index, name)) = names
            .into_iter()
            .enumerate()
            .find(|(_, name)| !seen.insert(*name))
        {
            return Some((
                format!("{pointer}/{index}/name"),
                format!("its schema lists the {kind} '{name}' twice"),
            ));
        }
    }
    None
}

impl Plan {
    /// The plan JSON of this plan: one line of text, the same for equal plans (see the
    /// module documentation for the format). [`Plan::from_json`] reads it back.
    pub fn to_json(&self) -> String {
        let mut nodes: Vec<&Self> = self.nodes().collect();
        nodes.reverse();
        let nodes = nodes
            .into_iter()
            .map(|node| NodeJson {
                id: node.id(),
                input: node.input().map(|input| input.id()),
                step: Step::of(node),
            })
            .collect();
        let json = PlanJson {
            format: PLAN_JSON_FORMAT.to_owned(),
            version: PLAN_JSON_VERSION,
            nodes,
        };
        // Every key is a string and every float written is finite (builders refuse
        // others; float literals use `ExprJson`'s text form), so this cannot fail.
        serde_json::to_string(&json)
            .unwrap_or_else(|error| unreachable!("a plan always serializes to JSON: {error}"))
    }

    /// The plan that `text` (written by [`Plan::to_json`]) describes, rebuilt step by
    /// step through the plan's builders, so every build-time check runs again. Reads no
    /// store.
    ///
    /// # Errors
    /// `KY-E218` if the text's `version` is not [`PLAN_JSON_VERSION`]; `KY-E217` if it
    /// is not JSON, is not a plan JSON, breaks the format (unknown or missing keys,
    /// wrong types, an unknown `method`, IDs or inputs out of order, a scan anywhere
    /// but first, gene IDs that do not match the scan's schema, a malformed
    /// expression); and any error a step's builder raises for its arguments (such as
    /// `KY-E109`, `KY-E205` or `KY-E215`), as when the plan was built.
    pub fn from_json(text: &str) -> Result<Arc<Self>, KytoError> {
        read_plan_json(text).map_err(|located| *located.error)
    }
}

/// The plan `text` describes ([`Plan::from_json`]), or the first error and where it is.
pub(crate) fn read_plan_json(text: &str) -> Result<Arc<Plan>, Located> {
    check_header(text)?;
    let json: PlanJson =
        serde_json::from_str(text).map_err(|error| Located::new("", syntax(&error)))?;
    replay(json)
}

/// `KY-E218` for another version, `KY-E217` for a missing or malformed `version` or
/// `format`, checked before anything else in the text; `KY-E217` for text that is not
/// a JSON object.
pub(crate) fn check_header(text: &str) -> Result<(), Located> {
    let header: Header =
        serde_json::from_str(text).map_err(|error| Located::new("", syntax(&error)))?;
    match header.version.as_ref().map(serde_json::Value::as_u64) {
        Some(Some(PLAN_JSON_VERSION)) => {}
        Some(Some(found)) => {
            return Err(Located::new(
                "/version",
                KytoError::PlanJsonVersion {
                    found,
                    supported: PLAN_JSON_VERSION,
                },
            ));
        }
        Some(None) => {
            return Err(Located::new(
                "/version",
                invalid(
                    "key 'version'",
                    format!(
                        "it is {}, and a plan JSON version is a non-negative integer",
                        header.version.unwrap_or_default()
                    ),
                ),
            ));
        }
        None => {
            return Err(Located::new(
                "/version",
                invalid("key 'version'", "it is missing"),
            ));
        }
    }
    if header.format.as_ref().and_then(serde_json::Value::as_str) != Some(PLAN_JSON_FORMAT) {
        let found = header
            .format
            .map_or_else(|| "missing".to_owned(), |format| format!("{format}"));
        return Err(Located::new(
            "/format",
            invalid(
                "key 'format'",
                format!("it is {found}, and a plan JSON has \"{PLAN_JSON_FORMAT}\""),
            ),
        ));
    }
    Ok(())
}

/// The plan of `json`, each step replayed through its builder.
fn replay(json: PlanJson) -> Result<Arc<Plan>, Located> {
    let mut plan: Option<Arc<Plan>> = None;
    for (position, node) in json.nodes.into_iter().enumerate() {
        let location = format!("node {position} ({})", node.step.method());
        let pointer = format!("/nodes/{position}");
        let id = position as u64;
        if node.id != NodeId(id) {
            return Err(Located::new(
                format!("{pointer}/id"),
                invalid(
                    location,
                    format!(
                        "its id is {}; nodes are listed from the scan up, node i has id i",
                        node.id
                    ),
                ),
            ));
        }
        let expected_input = id.checked_sub(1).map(NodeId);
        if node.input != expected_input {
            let expected = expected_input.map_or_else(|| "null".to_owned(), |id| id.to_string());
            let found = node
                .input
                .map_or_else(|| "null".to_owned(), |id| id.to_string());
            return Err(Located::new(
                format!("{pointer}/input"),
                invalid(
                    location,
                    format!(
                        "its input is {found}, and this version reads chains only: the input of node i is node i - 1 ({expected})"
                    ),
                ),
            ));
        }
        let step_pointer = format!("{pointer}/step");
        plan = Some(match (plan, node.step) {
            (
                None,
                Step::Scan {
                    source,
                    schema,
                    gene_ids,
                    symbols,
                },
            ) => scan(source, schema, gene_ids, symbols)
                .map_err(|located| located.under(&step_pointer))?,
            (None, _) => {
                return Err(Located::new(
                    format!("{step_pointer}/method"),
                    invalid(location, "the first node of a plan is a scan"),
                ));
            }
            (Some(input), step) => step
                .apply(&input, &location)
                .map_err(|located| located.under(&step_pointer))?,
        });
    }
    plan.ok_or_else(|| {
        Located::new(
            "/nodes",
            invalid("key 'nodes'", "a plan has at least one node, its scan"),
        )
    })
}
