//! The JSON form of an expression, for the plan JSON (task 4.3).
//!
//! Does: write an expression as a flat list of entries, leaves and operations, each
//! operation naming its operands by their position in the list, the root last
//! ([`ExprJson`]); and build an expression back from such a list through the same
//! checked constructors Python and Rust code use, so a list read from text is checked
//! like a hand-built expression (operand kinds `KY-E205`, depth `KY-E207`, feature
//! names `KY-E208`), and structural problems (an operand not written before its
//! operation, a source on the wrong axis) are `KY-E217`.
//! Does not: resolve names. An expression built back is unresolved: column reads have
//! dtype `Unknown` until a plan resolves them again (the plan JSON replays each step,
//! which resolves its expressions, so a resolved expression comes back equal).
//!
//! Why a flat list and not nested objects: an expression may be 1000 levels deep
//! ([`crate::MAX_EXPR_DEPTH`]), deeper than `serde_json` reads by default and than a
//! recursive reader should risk on a small thread stack; and a subtree shared through
//! `Arc` by several operations (`a & a`) is written once. Sharing does not survive
//! resolution (the resolver rebuilds a tree, task 3.5), so plan expressions are written
//! expanded; and a short text sharing entries could expand into a huge tree when a plan
//! resolves it, so [`ExprJson::to_expr`] refuses one that expands beyond
//! [`MAX_EXPANDED_NODES`].

use std::collections::HashMap;
use std::sync::Arc;

use kyto_error::KytoError;
use serde::de::{self, Unexpected};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::expr::{BinaryOp, Expr, ExprAxis, ExprNode, Literal, NullTest, Operand, StrFunction};
use crate::expr_dtype::ExprDtype;
use crate::expr_source::Source;
use crate::feature_name::FeatureName;
use crate::schema::Axis;

/// The most nodes an expression read from JSON may have once every shared entry is
/// counted at each use (`KY-E217` beyond). Far beyond any written expression.
pub const MAX_EXPANDED_NODES: u64 = 1_000_000;

/// An expression as JSON: its entries, each operation after its operands, the root
/// last. Serializes as a JSON array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExprJson(Vec<ExprEntry>);

/// One entry of an [`ExprJson`]: a leaf, a literal, or an operation on earlier entries
/// (named by their 0-based position). Tagged by `op`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExprEntry {
    /// `cell_column(name)` / `gene_column(name)`.
    Column {
        /// The column name.
        name: String,
    },
    /// `cell_column(name).recompute()` / `gene_column(name).recompute()`.
    Recompute {
        /// The column name.
        name: String,
    },
    /// `gene_expression(gene_id, layer=layer)` (cell axis only).
    GeneExpression {
        /// The layer; `null` is the plan's default layer (resolved plans name it).
        layer: Option<String>,
        /// The gene ID.
        gene_id: String,
    },
    /// `gene_expression(symbol, by='symbol', layer=layer)`, unresolved (cell axis only).
    GeneSymbol {
        /// The layer; `null` is the plan's default layer.
        layer: Option<String>,
        /// The gene symbol.
        symbol: String,
    },
    /// `cell_profile(cell_id, layer=layer)` (gene axis only).
    CellProfile {
        /// The layer; `null` is the plan's default layer (resolved plans name it).
        layer: Option<String>,
        /// The cell ID.
        cell_id: String,
    },
    /// `cell_embedding(name, component)` / `gene_embedding(name, component)`.
    Embedding {
        /// The embedding name.
        name: String,
        /// The component, 0-based.
        component: u64,
    },
    /// `feature(name)`, unresolved (cell axis only).
    Feature {
        /// The name as written (`CD3E`, `log1p:CD3E`).
        name: String,
    },
    /// A boolean literal.
    Boolean {
        /// The value.
        value: bool,
    },
    /// An integer literal.
    Int {
        /// The value.
        value: i64,
    },
    /// A float literal. Non-finite values are written `"NaN"`, `"inf"` and `"-inf"`
    /// (JSON numbers cannot hold them).
    Float {
        /// The value.
        #[serde(with = "float_text")]
        value: f64,
    },
    /// A string literal.
    String {
        /// The value.
        value: String,
    },
    /// `left <operator> right`.
    Binary {
        /// The Python operator (`==`, `+`, `&`, ...).
        operator: Operator,
        /// The entry of the left operand.
        left: usize,
        /// The entry of the right operand.
        right: usize,
    },
    /// `~input`.
    Not {
        /// The entry of the operand.
        input: usize,
    },
    /// `input.str.starts_with(prefix)`.
    StartsWith {
        /// The prefix.
        prefix: String,
        /// The entry of the operand.
        input: usize,
    },
    /// `input.str.ends_with(suffix)`.
    EndsWith {
        /// The suffix.
        suffix: String,
        /// The entry of the operand.
        input: usize,
    },
    /// `input.is_null()`.
    IsNull {
        /// The entry of the operand.
        input: usize,
    },
    /// `input.is_not_null()`.
    IsNotNull {
        /// The entry of the operand.
        input: usize,
    },
}

/// A binary operator, written as its Python symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Operator {
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "!=")]
    NotEq,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    LtEq,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    GtEq,
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Sub,
    #[serde(rename = "*")]
    Mul,
    #[serde(rename = "/")]
    Div,
    #[serde(rename = "&")]
    And,
    #[serde(rename = "|")]
    Or,
}

impl From<BinaryOp> for Operator {
    fn from(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Eq => Self::Eq,
            BinaryOp::NotEq => Self::NotEq,
            BinaryOp::Lt => Self::Lt,
            BinaryOp::LtEq => Self::LtEq,
            BinaryOp::Gt => Self::Gt,
            BinaryOp::GtEq => Self::GtEq,
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
            BinaryOp::And => Self::And,
            BinaryOp::Or => Self::Or,
        }
    }
}

impl From<Operator> for BinaryOp {
    fn from(op: Operator) -> Self {
        match op {
            Operator::Eq => Self::Eq,
            Operator::NotEq => Self::NotEq,
            Operator::Lt => Self::Lt,
            Operator::LtEq => Self::LtEq,
            Operator::Gt => Self::Gt,
            Operator::GtEq => Self::GtEq,
            Operator::Add => Self::Add,
            Operator::Sub => Self::Sub,
            Operator::Mul => Self::Mul,
            Operator::Div => Self::Div,
            Operator::And => Self::And,
            Operator::Or => Self::Or,
        }
    }
}

impl ExprJson {
    /// The entries of `expr`, each node once (a subtree shared through `Arc` is one
    /// entry), operands before their operation, the root last. Walks the tree with a
    /// loop, not recursion.
    pub fn from_expr<A: ExprAxis>(expr: &Expr<A>) -> Self {
        let mut entries = Vec::new();
        let mut written: HashMap<*const ExprNode, usize> = HashMap::new();
        let mut stack: Vec<(&Arc<ExprNode>, bool)> = vec![(expr.node(), false)];
        while let Some((node, operands_written)) = stack.pop() {
            let key = Arc::as_ptr(node);
            if written.contains_key(&key) {
                continue;
            }
            if !operands_written {
                stack.push((node, true));
                // Pushed in reverse, so the left operand gets the lower position.
                for operand in operands(node).into_iter().rev() {
                    stack.push((operand, false));
                }
                continue;
            }
            let position = |operand: &Arc<ExprNode>| written[&Arc::as_ptr(operand)];
            let entry = entry_of(node, position);
            written.insert(key, entries.len());
            entries.push(entry);
        }
        Self(entries)
    }

    /// The entries, operands before their operation, the root last.
    pub fn entries(&self) -> &[ExprEntry] {
        &self.0
    }

    /// The expression these entries write, on axis `A`, built with the checked
    /// constructors. Unresolved: resolve it against a plan. `context` says where the
    /// expression sits, for `KY-E217` messages (`node 2 (filter_cells), predicate`).
    ///
    /// # Errors
    /// `KY-E217` for no entries, an operand not written before its operation, a source
    /// of the other axis, or more than [`MAX_EXPANDED_NODES`] nodes once expanded; `KY-E205`, `KY-E207` or `KY-E208` as the constructors
    /// raise them.
    pub fn to_expr<A: ExprAxis>(&self, context: &str) -> Result<Expr<A>, KytoError> {
        let mut built: Vec<Expr<A>> = Vec::with_capacity(self.0.len());
        // Nodes of each entry's tree, a shared operand counted at each use.
        let mut expanded: Vec<u64> = Vec::with_capacity(self.0.len());
        for (position, entry) in self.0.iter().enumerate() {
            let expr = build_entry(entry, &built, context, position)?;
            let size = operand_positions(entry)
                .into_iter()
                .fold(1_u64, |size, operand| {
                    size.saturating_add(expanded[operand])
                });
            if size > MAX_EXPANDED_NODES {
                return Err(KytoError::InvalidPlanJson {
                    location: format!("{context} entry {position}"),
                    problem: format!(
                        "it expands to {size} nodes once shared entries are counted at each use, more than the {MAX_EXPANDED_NODES} an expression may have"
                    ),
                });
            }
            built.push(expr);
            expanded.push(size);
        }
        built.pop().ok_or_else(|| KytoError::InvalidPlanJson {
            location: context.to_owned(),
            problem: "the expression has no entries".to_owned(),
        })
    }
}

/// The positions of the operands of `entry` (all written before it: `build_entry`
/// checked).
fn operand_positions(entry: &ExprEntry) -> Vec<usize> {
    match entry {
        ExprEntry::Binary { left, right, .. } => vec![*left, *right],
        ExprEntry::Not { input }
        | ExprEntry::StartsWith { input, .. }
        | ExprEntry::EndsWith { input, .. }
        | ExprEntry::IsNull { input }
        | ExprEntry::IsNotNull { input } => vec![*input],
        _ => Vec::new(),
    }
}

/// The operands of `node`, in order.
fn operands(node: &Arc<ExprNode>) -> Vec<&Arc<ExprNode>> {
    match node.as_ref() {
        ExprNode::Source(_) | ExprNode::Literal(_) => Vec::new(),
        ExprNode::Binary { left, right, .. } => vec![left, right],
        ExprNode::Not(input) | ExprNode::Str { input, .. } | ExprNode::Null { input, .. } => {
            vec![input]
        }
    }
}

/// The entry of `node`, its operands at `position(operand)`.
fn entry_of(node: &ExprNode, position: impl Fn(&Arc<ExprNode>) -> usize) -> ExprEntry {
    match node {
        ExprNode::Source(source) => match source {
            Source::Column { name } => ExprEntry::Column { name: name.clone() },
            Source::Recompute { name } => ExprEntry::Recompute { name: name.clone() },
            Source::Expression { layer, gene_id } => ExprEntry::GeneExpression {
                layer: layer.clone(),
                gene_id: gene_id.clone(),
            },
            Source::GeneSymbol { layer, symbol } => ExprEntry::GeneSymbol {
                layer: layer.clone(),
                symbol: symbol.clone(),
            },
            Source::Profile { layer, cell_id } => ExprEntry::CellProfile {
                layer: layer.clone(),
                cell_id: cell_id.clone(),
            },
            Source::Embedding { name, component } => ExprEntry::Embedding {
                name: name.clone(),
                component: *component,
            },
            Source::Feature { name } => ExprEntry::Feature {
                name: name.to_string(),
            },
        },
        ExprNode::Literal(literal) => match literal {
            Literal::Boolean(value) => ExprEntry::Boolean { value: *value },
            Literal::Int(value) => ExprEntry::Int { value: *value },
            Literal::Float(value) => ExprEntry::Float { value: *value },
            Literal::String(value) => ExprEntry::String {
                value: value.clone(),
            },
        },
        ExprNode::Binary { op, left, right } => ExprEntry::Binary {
            operator: (*op).into(),
            left: position(left),
            right: position(right),
        },
        ExprNode::Not(input) => ExprEntry::Not {
            input: position(input),
        },
        ExprNode::Str { function, input } => match function {
            StrFunction::StartsWith(prefix) => ExprEntry::StartsWith {
                prefix: prefix.clone(),
                input: position(input),
            },
            StrFunction::EndsWith(suffix) => ExprEntry::EndsWith {
                suffix: suffix.clone(),
                input: position(input),
            },
        },
        ExprNode::Null { test, input } => match test {
            NullTest::IsNull => ExprEntry::IsNull {
                input: position(input),
            },
            NullTest::IsNotNull => ExprEntry::IsNotNull {
                input: position(input),
            },
        },
    }
}

/// The expression of `entry`, the `position`-th, given the ones `built` before it.
fn build_entry<A: ExprAxis>(
    entry: &ExprEntry,
    built: &[Expr<A>],
    context: &str,
    position: usize,
) -> Result<Expr<A>, KytoError> {
    let invalid = |problem: String| KytoError::InvalidPlanJson {
        location: format!("{context} entry {position}"),
        problem,
    };
    let operand = |index: usize| -> Result<Expr<A>, KytoError> {
        built.get(index).cloned().ok_or_else(|| {
            invalid(format!(
                "it reads entry {index}, which is not written before it"
            ))
        })
    };
    let only_on = |axis: Axis, what: &str| -> Result<(), KytoError> {
        if A::AXIS == axis {
            return Ok(());
        }
        let (this, other) = match A::AXIS {
            Axis::Cells => ("cell", "gene"),
            Axis::Genes => ("gene", "cell"),
        };
        Err(invalid(format!(
            "'{what}' makes a {other} expression, and this is a {this} expression"
        )))
    };
    let source =
        |source: Source, dtype: ExprDtype| Ok(Expr::from_node(ExprNode::Source(source), dtype));
    let literal = |literal: Literal| {
        let dtype = literal.dtype();
        Ok(Expr::from_node(ExprNode::Literal(literal), dtype))
    };
    match entry {
        ExprEntry::Column { name } => Ok(Expr::column(name.clone())),
        ExprEntry::Recompute { name } => Expr::column(name.clone()).recompute(),
        ExprEntry::GeneExpression { layer, gene_id } => {
            only_on(Axis::Cells, "gene_expression")?;
            source(
                Source::Expression {
                    layer: layer.clone(),
                    gene_id: gene_id.clone(),
                },
                ExprDtype::Numeric,
            )
        }
        ExprEntry::GeneSymbol { layer, symbol } => {
            only_on(Axis::Cells, "gene_symbol")?;
            source(
                Source::GeneSymbol {
                    layer: layer.clone(),
                    symbol: symbol.clone(),
                },
                ExprDtype::Numeric,
            )
        }
        ExprEntry::CellProfile { layer, cell_id } => {
            only_on(Axis::Genes, "cell_profile")?;
            source(
                Source::Profile {
                    layer: layer.clone(),
                    cell_id: cell_id.clone(),
                },
                ExprDtype::Numeric,
            )
        }
        ExprEntry::Embedding { name, component } => Ok(Expr::embedding(name.clone(), *component)),
        ExprEntry::Feature { name } => {
            only_on(Axis::Cells, "feature")?;
            source(
                Source::Feature {
                    name: FeatureName::parse(name)?,
                },
                ExprDtype::Unknown,
            )
        }
        ExprEntry::Boolean { value } => literal(Literal::Boolean(*value)),
        ExprEntry::Int { value } => literal(Literal::Int(*value)),
        ExprEntry::Float { value } => literal(Literal::Float(*value)),
        ExprEntry::String { value } => literal(Literal::String(value.clone())),
        ExprEntry::Binary {
            operator,
            left,
            right,
        } => Expr::binary(
            as_operand(operand(*left)?),
            (*operator).into(),
            as_operand(operand(*right)?),
        ),
        ExprEntry::Not { input } => operand(*input)?.not(),
        ExprEntry::StartsWith { prefix, input } => {
            operand(*input)?.str(StrFunction::StartsWith(prefix.clone()))
        }
        ExprEntry::EndsWith { suffix, input } => {
            operand(*input)?.str(StrFunction::EndsWith(suffix.clone()))
        }
        ExprEntry::IsNull { input } => operand(*input)?.null_test(NullTest::IsNull),
        ExprEntry::IsNotNull { input } => operand(*input)?.null_test(NullTest::IsNotNull),
    }
}

/// A literal stays a literal operand (so errors blame it as the constructors do).
fn as_operand<A: ExprAxis>(expr: Expr<A>) -> Operand<A> {
    match expr.node().as_ref() {
        ExprNode::Literal(literal) => Operand::Literal(literal.clone()),
        _ => Operand::Expr(expr),
    }
}

/// A float as a JSON number, or as `"NaN"`, `"inf"` or `"-inf"` when not finite.
mod float_text {
    use super::{Deserialize, Deserializer, Serializer, Unexpected, de};

    pub(super) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else if value.is_nan() {
            serializer.serialize_str("NaN")
        } else if *value > 0.0 {
            serializer.serialize_str("inf")
        } else {
            serializer.serialize_str("-inf")
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Number(f64),
        Text(String),
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        match Raw::deserialize(deserializer)? {
            Raw::Number(value) => Ok(value),
            Raw::Text(text) => match text.as_str() {
                "NaN" => Ok(f64::NAN),
                "inf" => Ok(f64::INFINITY),
                "-inf" => Ok(f64::NEG_INFINITY),
                _ => Err(de::Error::invalid_value(
                    Unexpected::Str(&text),
                    &"a number, \"NaN\", \"inf\" or \"-inf\"",
                )),
            },
        }
    }
}
