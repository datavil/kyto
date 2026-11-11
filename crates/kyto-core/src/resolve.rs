//! The feature resolver (task 3.5): turns the names in an expression into full,
//! checked names against one plan's schema, once, when the plan is built (R6).
//!
//! Does: resolve `feature(...)` names to a table column, a gene of a layer or an
//! embedding component, by kind against the schema, with no priority order: a name
//! that matches two features is `KY-E201` (docs/feature_names.md has the grammar);
//! write the default layer into every gene and cell read that left it out (so the
//! stored expression names its layer in full); check every column, gene, layer and
//! embedding name, and the cell ID of `cell_profile` for being non-empty; set each
//! column's dtype kind from the schema and check the operations on it again
//! (`KY-E205`), including ordering, which an unordered categorical column does not
//! have (pandas raises the same); refuse a gene read of a gene removed earlier in the
//! plan (`KY-E203`, task 3.7, R9); refuse a stored derived cell column (one whose
//! schema records lineage) read after a step that changed the gene axis (`KY-E301`,
//! task 3.8), and likewise a stored derived gene column after a step that changed the
//! cell axis, or any stored derived column after a step that wrote its lineage layer
//! again (task 4.1), unless the read is `.recompute()`d, which needs that lineage
//! (`KY-E302`). It also turns a gene named by symbol (`by='symbol'`, task 3.9) into
//! its gene ID: a symbol must match exactly one gene not removed upstream (`KY-E204`
//! lists the IDs otherwise).
//! Does not: suggest close matches for unknown names or symbols (task 8.2), or check
//! that a cell ID exists (the plan holds no cell IDs).

use std::collections::BTreeMap;

use kyto_error::KytoError;

use crate::expr::{Expr, ExprAxis, ExprNode, Operand, source_code};
use crate::expr_dtype::ExprDtype;
use crate::expr_source::Source;
use crate::feature_name::FeatureName;
use crate::gene_ids::GeneIds;
use crate::removed_genes::RemovedGenes;
use crate::schema::{AtlasSchema, Axis, EmbeddingSchema, TableSchema};
use crate::schema_column::{ColumnDtype, ColumnSchema};
use crate::{BinaryOp, CellAxis};

/// The gene table column that holds gene symbols (format.md §5.2, task 3.9).
pub const SYMBOL_COLUMN: &str = "symbol";

/// The prefix of a cell table column in a full feature name: `meta:cluster`.
pub const META_PREFIX: &str = "meta";

/// Resolves expressions against one plan: its schema, its store's gene IDs, the genes
/// removed upstream and its default layer. Holds borrowed state only and changes
/// nothing.
#[derive(Debug, Clone, Copy)]
pub struct Resolver<'a> {
    schema: &'a AtlasSchema,
    gene_ids: &'a GeneIds,
    removed_genes: Option<&'a RemovedGenes>,
    gene_axis_changed_by: Option<&'a str>,
    cell_axis_changed_by: Option<&'a str>,
    rewritten_layers: Option<&'a BTreeMap<String, String>>,
    default_layer: Option<&'a str>,
    store: &'a str,
}

/// One feature a `feature(...)` name can mean.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Candidate {
    /// A column of the cell table.
    Column { name: String, dtype: ExprDtype },
    /// A gene of a layer.
    Gene { layer: String, gene_id: String },
    /// A component of a cell embedding.
    Embedding { name: String, component: u64 },
}

impl Candidate {
    /// The full name, as docs/feature_names.md writes it.
    fn full_name(&self) -> String {
        match self {
            Self::Column { name, .. } => format!("{META_PREFIX}:{name}"),
            Self::Gene { layer, gene_id } => format!("{layer}:{gene_id}"),
            Self::Embedding { name, component } => format!("{name}:{component}"),
        }
    }

    /// The kind, added to the full name where two candidates are written alike.
    fn kind(&self) -> &'static str {
        match self {
            Self::Column { .. } => "cell column",
            Self::Gene { .. } => "gene",
            Self::Embedding { .. } => "embedding component",
        }
    }

    /// The source this candidate reads, and its dtype kind.
    fn into_source(self) -> (Source, ExprDtype) {
        match self {
            Self::Column { name, dtype } => (Source::Column { name }, dtype),
            Self::Gene { layer, gene_id } => (
                Source::Expression {
                    layer: Some(layer),
                    gene_id,
                },
                ExprDtype::Numeric,
            ),
            Self::Embedding { name, component } => {
                (Source::Embedding { name, component }, ExprDtype::Numeric)
            }
        }
    }
}

/// `cell` or `gene`.
fn axis_word(axis: Axis) -> &'static str {
    match axis {
        Axis::Cells => "cell",
        Axis::Genes => "gene",
    }
}

/// `component` as an embedding component in a full name: decimal digits without a
/// sign or leading zeros (`0`, `12`), or `None`.
fn parse_component(component: &str) -> Option<u64> {
    let canonical = !component.is_empty()
        && component.bytes().all(|byte| byte.is_ascii_digit())
        && (component == "0" || !component.starts_with('0'));
    if canonical {
        component.parse().ok()
    } else {
        None
    }
}

/// `has 50 components, 0 to 49`.
fn component_range(embedding: &EmbeddingSchema) -> String {
    match embedding.width {
        0 => "has no components".to_owned(),
        1 => "has 1 component, 0".to_owned(),
        width => format!("has {width} components, 0 to {}", width - 1),
    }
}

impl<'a> Resolver<'a> {
    /// A resolver for a plan over the store at `store` with schema `schema`, whose
    /// genes are `gene_ids`, and whose default layer is `default_layer`. The caller
    /// guarantees the default layer is a layer of `schema`.
    pub fn new(
        schema: &'a AtlasSchema,
        gene_ids: &'a GeneIds,
        default_layer: Option<&'a str>,
        store: &'a str,
    ) -> Self {
        Self {
            schema,
            gene_ids,
            removed_genes: None,
            gene_axis_changed_by: None,
            cell_axis_changed_by: None,
            rewritten_layers: None,
            default_layer,
            store,
        }
    }

    /// This resolver, refusing reads of the genes in `removed_genes` (`KY-E203`). The
    /// genes stay in `gene_ids`: a removed gene is a known name that can no longer be
    /// read, not an unknown one (`KY-E210`).
    pub fn with_removed_genes(self, removed_genes: &'a RemovedGenes) -> Self {
        Self {
            removed_genes: Some(removed_genes),
            ..self
        }
    }

    /// This resolver, refusing reads of stored derived cell columns (columns whose
    /// schema records lineage) as stale (`KY-E301`): `changed_by` (a plan node, as
    /// messages name it: `filter_genes`) changed the gene axis after they were computed.
    /// Every stored column was computed before the plan, so any such node makes them
    /// stale, whichever genes it keeps. A `.recompute()`d read is not refused.
    pub fn with_gene_axis_changed_by(self, changed_by: &'a str) -> Self {
        Self {
            gene_axis_changed_by: Some(changed_by),
            ..self
        }
    }

    /// This resolver, refusing reads of stored derived gene columns (per-gene
    /// statistics over cells, such as `n_cells` or HVG flags, whose schema records
    /// lineage) as stale (`KY-E301`): `changed_by` (`filter_cells`) changed the cell
    /// axis after they were computed. The mirror of
    /// [`Resolver::with_gene_axis_changed_by`] (task 4.1).
    pub fn with_cell_axis_changed_by(self, changed_by: &'a str) -> Self {
        Self {
            cell_axis_changed_by: Some(changed_by),
            ..self
        }
    }

    /// This resolver, refusing reads of stored derived columns of either table whose
    /// lineage names a layer in `rewritten_layers` as stale (`KY-E301`): the plan node
    /// mapped to it (`log1p`, as messages name it) wrote that layer again after they
    /// were computed (task 4.1).
    pub fn with_rewritten_layers(self, rewritten_layers: &'a BTreeMap<String, String>) -> Self {
        Self {
            rewritten_layers: Some(rewritten_layers),
            ..self
        }
    }

    /// `KY-E301` if `column`, read on axis `axis`, is a stored derived column (its
    /// schema records lineage) and, upstream, the other axis changed or the layer it
    /// was computed from was written again.
    fn check_not_stale(&self, axis: Axis, column: &ColumnSchema) -> Result<(), KytoError> {
        let Some(lineage) = &column.lineage else {
            return Ok(());
        };
        let other_axis_changed_by = match axis {
            Axis::Cells => self.gene_axis_changed_by,
            Axis::Genes => self.cell_axis_changed_by,
        };
        let layer_rewritten_by = self
            .rewritten_layers
            .and_then(|layers| layers.get(&lineage.layer))
            .map(String::as_str);
        match other_axis_changed_by.or(layer_rewritten_by) {
            Some(changed_by) => Err(KytoError::StaleColumn {
                column: column.name.clone(),
                changed_by: changed_by.to_owned(),
            }),
            None => Ok(()),
        }
    }

    /// `KY-E203` if `gene_id` was removed earlier in the plan.
    fn check_not_removed(&self, gene_id: &str) -> Result<(), KytoError> {
        match self
            .removed_genes
            .and_then(|removed| removed.removed_by(gene_id))
        {
            Some(removed_by) => Err(KytoError::RemovedGene {
                gene: gene_id.to_owned(),
                removed_by: removed_by.to_owned(),
            }),
            None => Ok(()),
        }
    }

    /// `expr` with every name resolved and checked: `feature(...)` replaced by the
    /// column, gene or embedding component it names, every gene and cell read given
    /// its layer, every column given its dtype kind. A new expression; `expr` is
    /// unchanged. Resolving a resolved expression gives it back unchanged.
    ///
    /// # Errors
    /// - `KY-E201` if a `feature(...)` name matches more than one feature.
    /// - `KY-E210` if a column, gene, `feature(...)` name or embedding component
    ///   matches nothing, or a cell ID is empty.
    /// - `KY-E211` if a gene or cell is read from the default layer and there is none.
    /// - `KY-E109` if a layer or embedding named in full is not in the store.
    /// - `KY-E205` if a resolved column's kind does not fit an operation on it.
    /// - `KY-E203` if a gene read (`gene_expression`, or a `feature(...)` name that
    ///   resolves to one gene) names a gene removed earlier in the plan.
    /// - `KY-E301` if a cell column read (`cell_column`, or a `feature(...)` name that
    ///   resolves to one) names a stored derived column after the gene axis changed.
    /// - `KY-E302` if a `.recompute()`d column records no lineage.
    pub fn resolve<A: ExprAxis>(&self, expr: &Expr<A>) -> Result<Expr<A>, KytoError> {
        self.node(expr.node())
    }

    /// The resolved expression of `node`, rebuilt bottom-up so each operation checks
    /// its operands' resolved kinds.
    #[recursive::recursive]
    fn node<A: ExprAxis>(&self, node: &ExprNode) -> Result<Expr<A>, KytoError> {
        match node {
            ExprNode::Source(source) => self.source(source),
            ExprNode::Literal(literal) => Ok(Expr::from_node(
                ExprNode::Literal(literal.clone()),
                literal.dtype(),
            )),
            ExprNode::Binary { op, left, right } => {
                let left = self.operand(left)?;
                let right = self.operand(right)?;
                let ordering = matches!(
                    op,
                    BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
                );
                let unordered = [&left, &right]
                    .into_iter()
                    .find_map(|operand| match operand {
                        Operand::Expr(expr) if ordering && self.is_unordered_enum(expr) => {
                            Some(expr.to_string())
                        }
                        _ => None,
                    });
                let expr = Expr::binary(left, *op, right)?;
                match unordered {
                    Some(operand) => Err(KytoError::ExpressionOperand {
                        operation: op.symbol().to_owned(),
                        accepted: "numeric operands, string operands or an ordered \
                                   categorical column"
                            .to_owned(),
                        got: "an unordered categorical column".to_owned(),
                        operand,
                    }),
                    None => Ok(expr),
                }
            }
            ExprNode::Not(input) => self.node::<A>(input)?.not(),
            ExprNode::Str { function, input } => self.node::<A>(input)?.str(function.clone()),
            ExprNode::Null { test, input } => self.node::<A>(input)?.null_test(*test),
        }
    }

    /// A child of a binary operation as an operand: literals stay literals.
    fn operand<A: ExprAxis>(&self, node: &ExprNode) -> Result<Operand<A>, KytoError> {
        match node {
            ExprNode::Literal(literal) => Ok(Operand::Literal(literal.clone())),
            _ => self.node(node).map(Operand::Expr),
        }
    }

    /// Whether `expr` is a resolved column stored as an unordered `Enum`.
    fn is_unordered_enum<A: ExprAxis>(&self, expr: &Expr<A>) -> bool {
        let ExprNode::Source(Source::Column { name } | Source::Recompute { name }) =
            expr.node().as_ref()
        else {
            return false;
        };
        self.column(A::AXIS, name)
            .is_some_and(|column| matches!(column.dtype, ColumnDtype::Enum { ordered: false, .. }))
    }

    fn table(&self, axis: Axis) -> &'a TableSchema {
        match axis {
            Axis::Cells => &self.schema.cells,
            Axis::Genes => &self.schema.genes,
        }
    }

    fn column(&self, axis: Axis, name: &str) -> Option<&'a ColumnSchema> {
        self.table(axis)
            .columns
            .iter()
            .find(|column| column.name == name)
    }

    /// The ID of the one gene whose symbol is `symbol` (task 3.9): `KY-E204` listing
    /// the IDs when several genes carry it, `KY-E210` (built by `unknown`) when none
    /// does or the store has no usable `symbol` column.
    fn gene_by_symbol(
        &self,
        symbol: &str,
        unknown: &impl Fn(String, &str, String) -> KytoError,
    ) -> Result<String, KytoError> {
        if !self.gene_ids.has_symbols() {
            let problem = match self.column(Axis::Genes, SYMBOL_COLUMN) {
                Some(column) => format!(
                    "the gene table's '{SYMBOL_COLUMN}' column is {}, not a string (Enum) column, so genes cannot be looked up by symbol",
                    column.dtype.name()
                ),
                None => format!(
                    "the gene table has no '{SYMBOL_COLUMN}' column, so genes cannot be looked up by symbol; select the gene by ID with by='id'"
                ),
            };
            return Err(unknown("gene symbol".to_owned(), symbol, problem));
        }
        let carriers = self.gene_ids.with_symbol(symbol);
        // Genes removed upstream are not candidates (task 4.1): after
        // `select_genes`, a symbol shared with a removed gene names the kept one, as
        // anndata's `var_names` hold only the kept genes. If every carrier was removed,
        // the first one is returned so the caller raises `KY-E203` for it.
        let kept: Vec<&str> = carriers
            .iter()
            .copied()
            .filter(|gene_id| {
                self.removed_genes
                    .is_none_or(|removed| removed.removed_by(gene_id).is_none())
            })
            .collect();
        match (kept.as_slice(), carriers.first()) {
            ([], None) => Err(unknown(
                "gene symbol".to_owned(),
                symbol,
                format!("no gene of the store has this {SYMBOL_COLUMN}"),
            )),
            ([], Some(removed)) => Ok((*removed).to_owned()),
            ([gene_id], _) => Ok((*gene_id).to_owned()),
            (gene_ids, _) => Err(KytoError::AmbiguousSymbol {
                symbol: symbol.to_owned(),
                gene_ids: gene_ids.iter().map(|&id| id.to_owned()).collect(),
            }),
        }
    }

    fn has_layer(&self, name: &str) -> bool {
        self.schema.layers.iter().any(|layer| layer.name == name)
    }

    fn layer_names(&self) -> Vec<String> {
        self.schema
            .layers
            .iter()
            .map(|layer| layer.name.clone())
            .collect()
    }

    fn embedding(&self, name: &str) -> Option<&'a EmbeddingSchema> {
        self.schema
            .embeddings
            .iter()
            .find(|embedding| embedding.name == name)
    }

    /// The layer to read: `layer` if given (checked), else the default layer.
    fn layer(&self, layer: Option<&str>, expression: &str) -> Result<String, KytoError> {
        match layer {
            Some(layer) if self.has_layer(layer) => Ok(layer.to_owned()),
            Some(layer) => Err(KytoError::MissingMember {
                kind: "layer".to_owned(),
                name: layer.to_owned(),
                store: self.store.to_owned(),
            }),
            None => {
                self.default_layer
                    .map(str::to_owned)
                    .ok_or_else(|| KytoError::NoDefaultLayer {
                        expression: expression.to_owned(),
                        layers: self.layer_names(),
                    })
            }
        }
    }

    /// The resolved leaf reading `source` on axis `A`.
    fn source<A: ExprAxis>(&self, source: &Source) -> Result<Expr<A>, KytoError> {
        let code = || source_code::<A>(source);
        let unknown = |kind: String, name: &str, problem: String| KytoError::UnknownName {
            kind,
            name: name.to_owned(),
            expression: code(),
            problem,
        };
        let (resolved, dtype) = match source {
            Source::Column { name } => {
                let word = axis_word(A::AXIS);
                let column = self.column(A::AXIS, name).ok_or_else(|| {
                    unknown(
                        format!("{word} column"),
                        name,
                        format!("the {word} table has no column with this name"),
                    )
                })?;
                self.check_not_stale(A::AXIS, column)?;
                (source.clone(), ExprDtype::of_column(column.dtype))
            }
            Source::Recompute { name } => {
                let word = axis_word(A::AXIS);
                let column = self.column(A::AXIS, name).ok_or_else(|| {
                    unknown(
                        format!("{word} column"),
                        name,
                        format!("the {word} table has no column with this name"),
                    )
                })?;
                if column.lineage.is_none() {
                    return Err(KytoError::NoLineage {
                        expression: code(),
                        kind: format!("{word} column"),
                        column: name.clone(),
                    });
                }
                (source.clone(), ExprDtype::of_column(column.dtype))
            }
            Source::Expression { layer, gene_id } => {
                if !self.gene_ids.contains(gene_id) {
                    return Err(unknown(
                        "gene".to_owned(),
                        gene_id,
                        "no gene of the store has this gene_id".to_owned(),
                    ));
                }
                self.check_not_removed(gene_id)?;
                let layer = self.layer(layer.as_deref(), &code())?;
                (
                    Source::Expression {
                        layer: Some(layer),
                        gene_id: gene_id.clone(),
                    },
                    ExprDtype::Numeric,
                )
            }
            Source::GeneSymbol { layer, symbol } => {
                let gene_id = self.gene_by_symbol(symbol, &unknown)?;
                self.check_not_removed(&gene_id)?;
                let layer = self.layer(layer.as_deref(), &code())?;
                (
                    Source::Expression {
                        layer: Some(layer),
                        gene_id,
                    },
                    ExprDtype::Numeric,
                )
            }
            Source::Profile { layer, cell_id } => {
                if cell_id.is_empty() {
                    return Err(unknown(
                        "cell".to_owned(),
                        cell_id,
                        "cell IDs are never empty".to_owned(),
                    ));
                }
                let layer = self.layer(layer.as_deref(), &code())?;
                (
                    Source::Profile {
                        layer: Some(layer),
                        cell_id: cell_id.clone(),
                    },
                    ExprDtype::Numeric,
                )
            }
            Source::Embedding { name, component } => {
                let embedding = self
                    .embedding(name)
                    .ok_or_else(|| KytoError::MissingMember {
                        kind: "embedding".to_owned(),
                        name: name.clone(),
                        store: self.store.to_owned(),
                    })?;
                if embedding.axis != A::AXIS {
                    let other = match embedding.axis {
                        Axis::Cells => "cell_embedding",
                        Axis::Genes => "gene_embedding",
                    };
                    return Err(unknown(
                        format!("{} embedding", axis_word(A::AXIS)),
                        name,
                        format!(
                            "'{name}' is a {} embedding; read it with {other}('{name}', {component})",
                            axis_word(embedding.axis)
                        ),
                    ));
                }
                if *component >= embedding.width {
                    return Err(unknown(
                        "embedding component".to_owned(),
                        &component.to_string(),
                        format!("'{name}' {}", component_range(embedding)),
                    ));
                }
                (source.clone(), ExprDtype::Numeric)
            }
            Source::Feature { name } => self.feature::<A>(name, &code())?,
        };
        Ok(Expr::from_node(ExprNode::Source(resolved), dtype))
    }

    /// The source a `feature(...)` name resolves to, and its dtype kind.
    fn feature<A: ExprAxis>(
        &self,
        name: &FeatureName,
        expression: &str,
    ) -> Result<(Source, ExprDtype), KytoError> {
        let unknown = |problem: String| KytoError::UnknownName {
            kind: "feature".to_owned(),
            name: name.to_string(),
            expression: expression.to_owned(),
            problem,
        };
        if A::AXIS != CellAxis::AXIS {
            return Err(unknown("feature(...) names cell features only".to_owned()));
        }
        let mut candidates = Vec::new();
        // A short gene name without a default layer could be read from every layer.
        let mut needs_default_layer = false;
        match name {
            FeatureName::Short { name } => {
                self.push_column(&mut candidates, name);
                if self.gene_ids.contains(name) {
                    match self.default_layer {
                        Some(layer) => candidates.push(Candidate::Gene {
                            layer: layer.to_owned(),
                            gene_id: name.clone(),
                        }),
                        None => {
                            needs_default_layer = true;
                            candidates.extend(self.schema.layers.iter().map(|layer| {
                                Candidate::Gene {
                                    layer: layer.name.clone(),
                                    gene_id: name.clone(),
                                }
                            }));
                        }
                    }
                }
            }
            FeatureName::Full { prefix, name } => {
                if prefix == META_PREFIX {
                    self.push_column(&mut candidates, name);
                }
                if self.has_layer(prefix) && self.gene_ids.contains(name) {
                    candidates.push(Candidate::Gene {
                        layer: prefix.clone(),
                        gene_id: name.clone(),
                    });
                }
                if let Some(embedding) = self.cell_embedding(prefix)
                    && let Some(component) = parse_component(name)
                    && component < embedding.width
                {
                    candidates.push(Candidate::Embedding {
                        name: prefix.clone(),
                        component,
                    });
                }
            }
        }
        let has_column = candidates
            .iter()
            .any(|candidate| matches!(candidate, Candidate::Column { .. }));
        if needs_default_layer && !has_column {
            return Err(KytoError::NoDefaultLayer {
                expression: expression.to_owned(),
                layers: self.layer_names(),
            });
        }
        match candidates.len() {
            0 => Err(unknown(self.no_feature_problem(name))),
            1 => {
                let candidate = candidates.remove(0);
                match &candidate {
                    Candidate::Gene { gene_id, .. } => self.check_not_removed(gene_id)?,
                    Candidate::Column { name, .. } => {
                        if let Some(column) = self.column(Axis::Cells, name) {
                            self.check_not_stale(Axis::Cells, column)?;
                        }
                    }
                    Candidate::Embedding { .. } => {}
                }
                Ok(candidate.into_source())
            }
            _ => Err(KytoError::AmbiguousName {
                name: name.to_string(),
                candidates: Self::candidate_names(&candidates),
            }),
        }
    }

    fn push_column(&self, candidates: &mut Vec<Candidate>, name: &str) {
        if let Some(column) = self.column(Axis::Cells, name) {
            candidates.push(Candidate::Column {
                name: name.to_owned(),
                dtype: ExprDtype::of_column(column.dtype),
            });
        }
    }

    fn cell_embedding(&self, name: &str) -> Option<&'a EmbeddingSchema> {
        self.embedding(name)
            .filter(|embedding| embedding.axis == Axis::Cells)
    }

    /// The full names of `candidates`; a name two candidates share gets its kind.
    fn candidate_names(candidates: &[Candidate]) -> Vec<String> {
        let names: Vec<String> = candidates.iter().map(Candidate::full_name).collect();
        names
            .iter()
            .zip(candidates)
            .map(|(name, candidate)| {
                if names.iter().filter(|other| *other == name).count() > 1 {
                    format!("{name} ({})", candidate.kind())
                } else {
                    name.clone()
                }
            })
            .collect()
    }

    /// Why a `feature(...)` name matches nothing.
    fn no_feature_problem(&self, name: &FeatureName) -> String {
        match name {
            FeatureName::Short { name } => {
                if self.cell_embedding(name).is_some() {
                    format!("'{name}' is an embedding; name one component, such as '{name}:0'")
                } else {
                    "it is not a column of the cell table or a gene ID".to_owned()
                }
            }
            FeatureName::Full { prefix, name } => {
                let mut reasons = Vec::new();
                if prefix == META_PREFIX {
                    reasons.push(format!("the cell table has no column '{name}'"));
                }
                if self.has_layer(prefix) {
                    reasons.push(format!("layer '{prefix}' has no gene '{name}'"));
                }
                if let Some(embedding) = self.cell_embedding(prefix) {
                    reasons.push(match parse_component(name) {
                        Some(_) => format!("embedding '{prefix}' {}", component_range(embedding)),
                        None => format!(
                            "'{name}' is not a component of embedding '{prefix}' \
                             (components are written 0, 1, 2, ...)"
                        ),
                    });
                }
                if reasons.is_empty() {
                    reasons.push(format!(
                        "'{prefix}' is not '{META_PREFIX}', a layer or a cell embedding"
                    ));
                }
                reasons.join("; ")
            }
        }
    }
}
