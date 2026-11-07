//! Typed expressions over one axis: [`CellExpr`] (one value per cell) and [`GeneExpr`]
//! (one value per gene).
//!
//! Does: build immutable expression trees from sources ([`Source`]), literals and
//! operations (comparison, arithmetic, `&`, `|`, `~`, `str.starts_with`,
//! `str.ends_with`, the null tests `is_null` and `is_not_null`); carry each expression's dtype kind ([`ExprDtype`]) and reject an
//! operand of the wrong kind with `KY-E205`; write an expression as the Python code that
//! builds it (`Display`, used in error messages).
//! Does not: resolve names against a schema (task 3.5), or evaluate anything.
//!
//! The axis is a type parameter, so Rust code cannot combine a cell expression with a
//! gene expression: `CellExpr::binary` only takes `CellExpr` operands. Python sees two
//! separate classes; the bindings report a mix with [`mixed_axes`] (`KY-E202`, R5).
//! Like Polars' `Expr`, a tree shares its children through `Arc`, so building `a & b`
//! leaves `a` and `b` unchanged and copies nothing.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use kyto_error::KytoError;

use crate::expr_dtype::ExprDtype;
use crate::expr_source::Source;
use crate::feature_name::FeatureName;
use crate::schema::Axis;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::CellAxis {}
    impl Sealed for super::GeneAxis {}
}

/// The axis an expression runs over: [`CellAxis`] or [`GeneAxis`]. Sealed.
pub trait ExprAxis: sealed::Sealed + Copy + fmt::Debug + PartialEq + 'static {
    /// The Python constructor of a column on this axis (`cell_column`, `gene_column`).
    const COLUMN: &'static str;
    /// The Python constructor of an embedding component on this axis.
    const EMBEDDING: &'static str;
    /// The axis, as the schema names it (its table and its embeddings).
    const AXIS: Axis;
}

/// One value per cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellAxis;

/// One value per gene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeneAxis;

impl ExprAxis for CellAxis {
    const COLUMN: &'static str = "cell_column";
    const EMBEDDING: &'static str = "cell_embedding";
    const AXIS: Axis = Axis::Cells;
}

impl ExprAxis for GeneAxis {
    const COLUMN: &'static str = "gene_column";
    const EMBEDDING: &'static str = "gene_embedding";
    const AXIS: Axis = Axis::Genes;
}

/// An expression with one value per cell.
pub type CellExpr = Expr<CellAxis>;
/// An expression with one value per gene.
pub type GeneExpr = Expr<GeneAxis>;

/// A literal operand.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// `True` / `False`.
    Boolean(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    String(String),
}

impl Literal {
    /// The literal's dtype kind.
    pub fn dtype(&self) -> ExprDtype {
        match self {
            Self::Boolean(_) => ExprDtype::Boolean,
            Self::Int(_) | Self::Float(_) => ExprDtype::Numeric,
            Self::String(_) => ExprDtype::String,
        }
    }
}

/// A binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `==`.
    Eq,
    /// `!=`.
    NotEq,
    /// `<`.
    Lt,
    /// `<=`.
    LtEq,
    /// `>`.
    Gt,
    /// `>=`.
    GtEq,
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/` (true division: the result is a float).
    Div,
    /// `&` (logical and).
    And,
    /// `|` (logical or).
    Or,
}

impl BinaryOp {
    /// The Python operator.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::And => "&",
            Self::Or => "|",
        }
    }

    /// The kinds each operand may have, and how messages describe them.
    fn accepted(self) -> (&'static [ExprDtype], &'static str) {
        use ExprDtype::{Boolean, Numeric, String};
        match self {
            Self::Eq | Self::NotEq => (
                &[Boolean, Numeric, String],
                "boolean, numeric or string operands",
            ),
            Self::Lt | Self::LtEq | Self::Gt | Self::GtEq => {
                (&[Numeric, String], "numeric or string operands")
            }
            Self::Add | Self::Sub | Self::Mul | Self::Div => (&[Numeric], "numeric operands"),
            Self::And | Self::Or => (&[Boolean], "boolean operands"),
        }
    }

    /// Whether both operands must have the same kind (comparisons).
    fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::NotEq | Self::Lt | Self::LtEq | Self::Gt | Self::GtEq
        )
    }

    /// The kind of the result.
    fn result(self) -> ExprDtype {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => ExprDtype::Numeric,
            _ => ExprDtype::Boolean,
        }
    }
}

/// A function of the `str` namespace. Each returns a boolean expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrFunction {
    /// `str.starts_with(prefix)`.
    StartsWith(String),
    /// `str.ends_with(suffix)`.
    EndsWith(String),
}

impl StrFunction {
    /// The Python method name, `str.` included.
    pub fn name(&self) -> &'static str {
        match self {
            Self::StartsWith(_) => "str.starts_with",
            Self::EndsWith(_) => "str.ends_with",
        }
    }

    fn argument(&self) -> &str {
        match self {
            Self::StartsWith(text) | Self::EndsWith(text) => text,
        }
    }
}

/// A test for missing values (task 3.7, R9). Each returns a boolean expression that is
/// never null itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NullTest {
    /// `is_null()`: true where the value is missing.
    IsNull,
    /// `is_not_null()`: true where the value is present.
    IsNotNull,
}

impl NullTest {
    /// The Python method name.
    pub fn name(self) -> &'static str {
        match self {
            Self::IsNull => "is_null",
            Self::IsNotNull => "is_not_null",
        }
    }
}

/// A node of an expression tree. Children are shared through `Arc`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprNode {
    /// A leaf reading values from a source.
    Source(Source),
    /// A literal, broadcast to every row.
    Literal(Literal),
    /// `left op right`.
    Binary {
        /// The operation.
        op: BinaryOp,
        /// The left operand.
        left: Arc<ExprNode>,
        /// The right operand.
        right: Arc<ExprNode>,
    },
    /// `~input`.
    Not(Arc<ExprNode>),
    /// `input.str.<function>(...)`.
    Str {
        /// The function and its argument.
        function: StrFunction,
        /// The string input.
        input: Arc<ExprNode>,
    },
    /// `input.is_null()` or `input.is_not_null()`.
    Null {
        /// The test.
        test: NullTest,
        /// The input, of any kind.
        input: Arc<ExprNode>,
    },
}

impl ExprNode {
    /// Whether every leaf is a metadata column or a literal, so the whole tree lowers
    /// to a Polars expression.
    pub fn is_metadata_only(&self) -> bool {
        match self {
            Self::Source(source) => source.is_metadata(),
            Self::Literal(_) => true,
            Self::Binary { left, right, .. } => left.is_metadata_only() && right.is_metadata_only(),
            Self::Not(input) | Self::Str { input, .. } | Self::Null { input, .. } => {
                input.is_metadata_only()
            }
        }
    }
}

/// A typed expression over axis `A`: [`CellExpr`] or [`GeneExpr`].
///
/// Immutable: every operation returns a new expression sharing this one's tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr<A: ExprAxis> {
    node: Arc<ExprNode>,
    dtype: ExprDtype,
    /// Levels in the tree: 1 for a source, so building deeper trees can be refused
    /// before rendering or lowering them (both recurse) can overflow the stack.
    depth: u32,
    axis: PhantomData<A>,
}

/// The deepest expression tree kyto builds (`KY-E207`). Far beyond any hand-written
/// expression. Rendering and lowering grow the stack on demand (`recursive`), but the
/// default drop of the `Arc` chain and Polars' own passes over a lowered tree recurse
/// on the thread's stack, so the depth stays bounded.
pub const MAX_EXPR_DEPTH: u32 = 1000;

/// One side of a binary operation: an expression of the same axis or a literal.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand<A: ExprAxis> {
    /// An expression.
    Expr(Expr<A>),
    /// A literal.
    Literal(Literal),
}

impl<A: ExprAxis> From<Expr<A>> for Operand<A> {
    fn from(expr: Expr<A>) -> Self {
        Self::Expr(expr)
    }
}

impl<A: ExprAxis> From<Literal> for Operand<A> {
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

impl<A: ExprAxis> Operand<A> {
    fn dtype(&self) -> ExprDtype {
        match self {
            Self::Expr(expr) => expr.dtype,
            Self::Literal(literal) => literal.dtype(),
        }
    }

    fn depth(&self) -> u32 {
        match self {
            Self::Expr(expr) => expr.depth,
            Self::Literal(_) => 1,
        }
    }

    fn node(self) -> Arc<ExprNode> {
        match self {
            Self::Expr(expr) => expr.node,
            Self::Literal(literal) => Arc::new(ExprNode::Literal(literal)),
        }
    }

    /// `a numeric expression`, `a string value`, ...
    fn describe(&self) -> String {
        let what = match self {
            Self::Expr(_) => "expression",
            Self::Literal(_) => "value",
        };
        describe_kind(self.dtype(), what)
    }

    fn code(&self) -> String {
        match self {
            Self::Expr(expr) => expr.to_string(),
            Self::Literal(literal) => {
                Code::<A>::new(&ExprNode::Literal(literal.clone())).to_string()
            }
        }
    }
}

/// The Python code that builds a leaf reading `source` on axis `A`, such as
/// `cell_column('n_genes')`.
pub(crate) fn source_code<A: ExprAxis>(source: &Source) -> String {
    Code::<A>::new(&ExprNode::Source(source.clone())).to_string()
}

/// `a boolean expression`, `an unknown value`, ...
fn describe_kind(dtype: ExprDtype, what: &str) -> String {
    let article = if dtype == ExprDtype::Unknown {
        "an"
    } else {
        "a"
    };
    format!("{article} {} {what}", dtype.name())
}

impl<A: ExprAxis> Expr<A> {
    pub(crate) fn from_node(node: ExprNode, dtype: ExprDtype) -> Self {
        Self::with_depth(node, dtype, 1)
    }

    fn with_depth(node: ExprNode, dtype: ExprDtype, depth: u32) -> Self {
        Self {
            node: Arc::new(node),
            dtype,
            depth,
            axis: PhantomData,
        }
    }

    /// The depth of a node built on children of depth `child_depth`.
    ///
    /// # Errors
    /// `KY-E207` beyond [`MAX_EXPR_DEPTH`].
    fn deeper(operation: &str, child_depth: u32) -> Result<u32, KytoError> {
        let depth = child_depth.saturating_add(1);
        if depth > MAX_EXPR_DEPTH {
            return Err(KytoError::ExpressionDepth {
                operation: operation.to_owned(),
                depth: u64::from(depth),
                limit: u64::from(MAX_EXPR_DEPTH),
            });
        }
        Ok(depth)
    }

    /// A column of this axis's table. Its dtype is `Unknown` until the plan resolves it.
    pub fn column(name: impl Into<String>) -> Self {
        Self::from_node(
            ExprNode::Source(Source::Column { name: name.into() }),
            ExprDtype::Unknown,
        )
    }

    /// Component `component` (0-based) of the embedding `name` of this axis. Numeric.
    pub fn embedding(name: impl Into<String>, component: u64) -> Self {
        Self::from_node(
            ExprNode::Source(Source::Embedding {
                name: name.into(),
                component,
            }),
            ExprDtype::Numeric,
        )
    }

    /// The expression tree.
    pub fn node(&self) -> &Arc<ExprNode> {
        &self.node
    }

    /// The dtype kind.
    pub fn dtype(&self) -> ExprDtype {
        self.dtype
    }

    /// Levels in the tree: 1 for a source.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Whether the whole expression reads metadata only (see [`Expr::to_polars`]).
    pub fn is_metadata_only(&self) -> bool {
        self.node.is_metadata_only()
    }

    /// `left op right`.
    ///
    /// # Errors
    /// `KY-E205` if an operand's kind is not accepted by `op` (arithmetic needs numeric
    /// operands, `&` and `|` boolean ones, ordering numeric or string ones), or if a
    /// comparison's operands have different kinds. `Unknown` operands are accepted and
    /// checked when the plan resolves them. `KY-E207` if the result would be deeper
    /// than [`MAX_EXPR_DEPTH`].
    pub fn binary(
        left: impl Into<Operand<A>>,
        op: BinaryOp,
        right: impl Into<Operand<A>>,
    ) -> Result<Self, KytoError> {
        let (left, right) = (left.into(), right.into());
        let (kinds, accepted) = op.accepted();
        for operand in [&left, &right] {
            if !operand.dtype().is_one_of(kinds) {
                return Err(operand_error(op.symbol(), accepted.to_owned(), operand));
            }
        }
        if op.is_comparison() && !left.dtype().matches(right.dtype()) {
            // Blame the literal when there is one, else the right side.
            let (known, blamed) = match (&left, &right) {
                (Operand::Literal(_), Operand::Expr(_)) => (&right, &left),
                _ => (&left, &right),
            };
            let accepted = format!(
                "{}, like the other side",
                describe_kind(known.dtype(), "operand")
            );
            return Err(operand_error(op.symbol(), accepted, blamed));
        }
        let depth = Self::deeper(op.symbol(), left.depth().max(right.depth()))?;
        Ok(Self::with_depth(
            ExprNode::Binary {
                op,
                left: left.node(),
                right: right.node(),
            },
            op.result(),
            depth,
        ))
    }

    /// `~self`, the logical not.
    ///
    /// # Errors
    /// `KY-E205` if this expression is not boolean (or unknown); `KY-E207` if the result
    /// would be deeper than [`MAX_EXPR_DEPTH`].
    pub fn not(&self) -> Result<Self, KytoError> {
        if !self.dtype.is_one_of(&[ExprDtype::Boolean]) {
            return Err(operand_error(
                "~",
                "a boolean expression".to_owned(),
                &Operand::Expr(self.clone()),
            ));
        }
        let depth = Self::deeper("~", self.depth)?;
        Ok(Self::with_depth(
            ExprNode::Not(Arc::clone(&self.node)),
            ExprDtype::Boolean,
            depth,
        ))
    }

    /// `self.str.<function>(...)`: a boolean expression.
    ///
    /// # Errors
    /// `KY-E205` if this expression is not a string (or unknown); `KY-E207` if the result
    /// would be deeper than [`MAX_EXPR_DEPTH`].
    pub fn str(&self, function: StrFunction) -> Result<Self, KytoError> {
        if !self.dtype.is_one_of(&[ExprDtype::String]) {
            return Err(operand_error(
                function.name(),
                "a string expression".to_owned(),
                &Operand::Expr(self.clone()),
            ));
        }
        let depth = Self::deeper(function.name(), self.depth)?;
        Ok(Self::with_depth(
            ExprNode::Str {
                function,
                input: Arc::clone(&self.node),
            },
            ExprDtype::Boolean,
            depth,
        ))
    }

    /// `self.is_null()` or `self.is_not_null()`: a boolean expression, true where the
    /// value is missing (or present). Accepts every kind. Metadata may be missing (R9);
    /// matrix values, embeddings and literals never are, so on them the test is constant.
    ///
    /// # Errors
    /// `KY-E207` if the result would be deeper than [`MAX_EXPR_DEPTH`].
    pub fn null_test(&self, test: NullTest) -> Result<Self, KytoError> {
        let depth = Self::deeper(test.name(), self.depth)?;
        Ok(Self::with_depth(
            ExprNode::Null {
                test,
                input: Arc::clone(&self.node),
            },
            ExprDtype::Boolean,
            depth,
        ))
    }

    /// The `KY-E206` error for using this expression as `True` / `False` in Python
    /// (`if`, `and`, `or`, `not`, chained comparisons).
    pub fn truth_value_error(&self) -> KytoError {
        KytoError::ExpressionTruthValue {
            expression: self.to_string(),
        }
    }
}

impl CellExpr {
    /// The values of gene `gene_id` across cells, from `layer` (`None`: the plan's
    /// default layer). Numeric.
    pub fn expression(layer: Option<String>, gene_id: impl Into<String>) -> Self {
        Self::from_node(
            ExprNode::Source(Source::Expression {
                layer,
                gene_id: gene_id.into(),
            }),
            ExprDtype::Numeric,
        )
    }
}

impl CellExpr {
    /// The feature `name` (short, such as `CD3E`, or full, such as `log1p:CD3E`), left
    /// unresolved as a [`Source::Feature`] leaf for the feature resolver (task 3.5).
    /// Its dtype is `Unknown` until then.
    ///
    /// # Errors
    /// `KY-E208` if `name` cannot name any feature (see [`FeatureName::parse`]).
    pub fn feature(name: &str) -> Result<Self, KytoError> {
        Ok(Self::from_node(
            ExprNode::Source(Source::Feature {
                name: FeatureName::parse(name)?,
            }),
            ExprDtype::Unknown,
        ))
    }
}

impl GeneExpr {
    /// The values of cell `cell_id` across genes, from `layer` (`None`: the plan's
    /// default layer). Numeric.
    pub fn profile(layer: Option<String>, cell_id: impl Into<String>) -> Self {
        Self::from_node(
            ExprNode::Source(Source::Profile {
                layer,
                cell_id: cell_id.into(),
            }),
            ExprDtype::Numeric,
        )
    }
}

/// The `KY-E202` error for combining `cell` and `gene` with `operation` (R5).
pub fn mixed_axes(operation: &str, cell: &CellExpr, gene: &GeneExpr) -> KytoError {
    KytoError::MixedAxes {
        operation: operation.to_owned(),
        cell_expression: cell.to_string(),
        gene_expression: gene.to_string(),
    }
}

/// The `KY-E205` error for an operand the operation does not accept.
fn operand_error<A: ExprAxis>(
    operation: &str,
    accepted: String,
    operand: &Operand<A>,
) -> KytoError {
    KytoError::ExpressionOperand {
        operation: operation.to_owned(),
        accepted,
        got: operand.describe(),
        operand: operand.code(),
    }
}

/// Writes a node as the Python code that builds it, on axis `A`.
struct Code<'a, A: ExprAxis> {
    node: &'a ExprNode,
    axis: PhantomData<A>,
}

impl<'a, A: ExprAxis> Code<'a, A> {
    fn new(node: &'a ExprNode) -> Self {
        Self {
            node,
            axis: PhantomData,
        }
    }

    /// The child, in parentheses if it is an operation Python would otherwise bind
    /// differently.
    fn child(node: &'a ExprNode, parenthesize: bool) -> impl fmt::Display + 'a {
        struct Child<'a, A: ExprAxis>(Code<'a, A>, bool);
        impl<A: ExprAxis> fmt::Display for Child<'_, A> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.1 {
                    write!(formatter, "({})", self.0)
                } else {
                    write!(formatter, "{}", self.0)
                }
            }
        }
        Child(Code::<A>::new(node), parenthesize)
    }
}

/// A Python string literal for `text`, in single quotes.
fn python_string(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('\'');
    for character in text.chars() {
        match character {
            '\\' => quoted.push_str("\\\\"),
            '\'' => quoted.push_str("\\'"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            other if needs_escape(other) => {
                let code = u32::from(other);
                let escaped = if code <= 0xff {
                    format!("\\x{code:02x}")
                } else if code <= 0xffff {
                    format!("\\u{code:04x}")
                } else {
                    format!("\\U{code:08x}")
                };
                quoted.push_str(&escaped);
            }
            other => quoted.push(other),
        }
    }
    quoted.push('\'');
    quoted
}

/// Whether `character` is written as an escape: control characters, every whitespace
/// but the plain space (line and paragraph separators, no-break spaces) and invisible
/// format characters, as Python's `repr` does for non-printable characters.
fn needs_escape(character: char) -> bool {
    character.is_control()
        || (character.is_whitespace() && character != ' ')
        || matches!(
            character,
            '\u{ad}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
                | '\u{2060}'..='\u{2064}' | '\u{feff}'
        )
}

fn python_float(value: f64) -> String {
    if value.is_nan() {
        "float('nan')".to_owned()
    } else if value.is_infinite() {
        if value > 0.0 {
            "float('inf')"
        } else {
            "float('-inf')"
        }
        .to_owned()
    } else {
        format!("{value:?}")
    }
}

fn layer_argument(layer: Option<&String>) -> String {
    layer.map_or_else(String::new, |layer| {
        format!(", layer={}", python_string(layer))
    })
}

impl<A: ExprAxis> fmt::Display for Code<'_, A> {
    #[recursive::recursive]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node {
            ExprNode::Source(Source::Column { name }) => {
                write!(formatter, "{}({})", A::COLUMN, python_string(name))
            }
            ExprNode::Source(Source::Expression { layer, gene_id }) => write!(
                formatter,
                "gene_expression({}{})",
                python_string(gene_id),
                layer_argument(layer.as_ref())
            ),
            ExprNode::Source(Source::Profile { layer, cell_id }) => write!(
                formatter,
                "cell_profile({}{})",
                python_string(cell_id),
                layer_argument(layer.as_ref())
            ),
            ExprNode::Source(Source::Embedding { name, component }) => write!(
                formatter,
                "{}({}, {component})",
                A::EMBEDDING,
                python_string(name)
            ),
            ExprNode::Source(Source::Feature { name }) => {
                write!(formatter, "feature({})", python_string(&name.to_string()))
            }
            ExprNode::Literal(Literal::Boolean(value)) => {
                formatter.write_str(if *value { "True" } else { "False" })
            }
            ExprNode::Literal(Literal::Int(value)) => write!(formatter, "{value}"),
            ExprNode::Literal(Literal::Float(value)) => formatter.write_str(&python_float(*value)),
            ExprNode::Literal(Literal::String(value)) => formatter.write_str(&python_string(value)),
            ExprNode::Binary { op, left, right } => {
                let nested = |node: &ExprNode| matches!(node, ExprNode::Binary { .. });
                write!(
                    formatter,
                    "{} {} {}",
                    Self::child(left, nested(left)),
                    op.symbol(),
                    Self::child(right, nested(right))
                )
            }
            ExprNode::Not(input) => {
                let nested = matches!(**input, ExprNode::Binary { .. });
                write!(formatter, "~{}", Self::child(input, nested))
            }
            ExprNode::Str { function, input } => {
                let nested = matches!(**input, ExprNode::Binary { .. } | ExprNode::Not(_));
                write!(
                    formatter,
                    "{}.{}({})",
                    Self::child(input, nested),
                    function.name(),
                    python_string(function.argument())
                )
            }
            ExprNode::Null { test, input } => {
                // A method binds tighter than `~` and every operator, so those need
                // parentheses; so does a literal (`(1).is_null()`, not `1.is_null()`).
                let nested = matches!(
                    **input,
                    ExprNode::Binary { .. } | ExprNode::Not(_) | ExprNode::Literal(_)
                );
                write!(
                    formatter,
                    "{}.{}()",
                    Self::child(input, nested),
                    test.name()
                )
            }
        }
    }
}

impl<A: ExprAxis> fmt::Display for Expr<A> {
    /// The Python code that builds this expression, such as
    /// `cell_column('n_genes') > 200`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", Code::<A>::new(&self.node))
    }
}
