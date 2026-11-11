//! The registered kyto errors. GENERATED from errors.toml by scripts/error_registry.py. Do not edit.

use crate::{FieldValue, quoted_list};

/// Every kyto error: one variant per registered `KY-Exxx` code in `errors.toml`.
///
/// `Display` gives `KY-Exxx: <message>`; [`KytoError::hint`] says how to fix it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KytoError {
    /// KY-E101: IDs written to a store are not unique.
    #[error("KY-E101: the {axis} table has {n_duplicated} IDs that occur more than once, first: {}", quoted_list(.ids))]
    DuplicateIds {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// How many distinct IDs occur more than once.
        n_duplicated: u64,
        /// The first duplicated IDs, in the order they repeat (at most 5).
        ids: Vec<String>,
    },
    /// KY-E102: A layer, embedding or graph name breaks the naming rules of format.md section 2.1.
    #[error("KY-E102: invalid {kind} name '{name}': {rule}")]
    InvalidMemberName {
        /// The member kind: `layer`, `embedding`, `graph` or `modality`.
        kind: String,
        /// The name as given.
        name: String,
        /// The rule it breaks.
        rule: String,
    },
    /// KY-E103: A member with the same name, compared case-insensitively, is already in the store.
    #[error("KY-E103: {kind} '{name}' already exists in the store as '{existing}'")]
    MemberExists {
        /// The member kind: `layer`, `embedding`, `graph` or `modality`.
        kind: String,
        /// The name as given.
        name: String,
        /// The name already in the store.
        existing: String,
    },
    /// KY-E104: A store writer parameter (tile or chunk size, compression level) is out of range.
    #[error("KY-E104: invalid {parameter} = {value}")]
    InvalidParameter {
        /// The parameter name.
        parameter: String,
        /// The value given.
        value: String,
        /// What the parameter accepts.
        accepted: String,
    },
    /// KY-E105: A size is above a limit of the store format (format.md section 4.5).
    #[error("KY-E105: {quantity} is {value}, above the format limit of {limit}")]
    LimitExceeded {
        /// What is too large.
        quantity: String,
        /// Its value.
        value: u64,
        /// The largest value allowed.
        limit: u64,
        /// How to get under the limit.
        advice: String,
    },
    /// KY-E106: A sparse matrix given to a store writer is not valid CSR of the stated shape.
    #[error("KY-E106: invalid sparse matrix for layer '{layer}': {problem}")]
    InvalidSparseMatrix {
        /// The layer being written.
        layer: String,
        /// What is wrong and where.
        problem: String,
    },
    /// KY-E107: A stored layer breaks the tiled sparse format (format.md section 4).
    #[error("KY-E107: layer '{layer}' is corrupt: {problem}")]
    CorruptLayer {
        /// The layer being read.
        layer: String,
        /// The invariant that fails and where.
        problem: String,
    },
    /// KY-E108: A layer is read with a value dtype other than the one it stores.
    #[error("KY-E108: layer '{layer}' stores {stored} values, not {requested}")]
    LayerDtype {
        /// The layer being read.
        layer: String,
        /// The Zarr dtype of its data array.
        stored: String,
        /// The dtype asked for.
        requested: String,
    },
    /// KY-E109: A layer, embedding, graph or modality is not in the store.
    #[error("KY-E109: no {kind} named '{name}' in the store at '{store}'")]
    MissingMember {
        /// The member kind: `layer`, `embedding`, `graph` or `modality`.
        kind: String,
        /// The name asked for.
        name: String,
        /// The store path.
        store: String,
    },
    /// KY-E110: The file system or the Zarr layer failed while reading or writing a store.
    #[error("KY-E110: could not {operation} '{path}': {detail}")]
    StoreIo {
        /// What was being done.
        operation: String,
        /// The store path or node.
        path: String,
        /// The underlying error.
        detail: String,
    },
    /// KY-E111: The ID column of a cell or gene table holds null or empty IDs.
    #[error("KY-E111: the {axis} table has {n_invalid} {problem} IDs, first at rows {}", quoted_list(.rows))]
    InvalidIds {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// How many IDs are invalid.
        n_invalid: u64,
        /// What is wrong with them: `null` or `empty`.
        problem: String,
        /// The first rows (0-based) with such an ID (at most 5).
        rows: Vec<String>,
    },
    /// KY-E112: A column of a cell or gene table breaks format.md section 5 (name, dtype or position).
    #[error("KY-E112: column '{column}' of the {axis} table {problem}")]
    InvalidTableColumn {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// The column name.
        column: String,
        /// What is wrong.
        problem: String,
        /// What would be accepted.
        accepted: String,
    },
    /// KY-E113: A stored cell or gene table breaks format.md section 5.
    #[error("KY-E113: table '{path}' is corrupt: {problem}")]
    CorruptTable {
        /// The table file.
        path: String,
        /// The invariant that fails.
        problem: String,
    },
    /// KY-E114: A cell or gene selection given to the tile reader is out of range, unsorted or repeats a position.
    #[error("KY-E114: invalid {axis} selection: {problem}")]
    InvalidSelection {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// What is wrong and where in the selection.
        problem: String,
        /// The axis length of the store.
        n_axis: u64,
    },
    /// KY-E115: A store has a format_version this kyto does not read (format.md section 11).
    #[error("KY-E115: the store at '{store}' has format_version '{found}', but this kyto reads only {}", quoted_list(.supported))]
    UnsupportedFormatVersion {
        /// The store path.
        store: String,
        /// The store's format_version.
        found: String,
        /// The format versions this kyto reads.
        supported: Vec<String>,
    },
    /// KY-E116: A path is not a kyto store: its root zarr.json is missing or has no format_version (format.md section 3).
    #[error("KY-E116: '{store}' is not a kyto store: {problem}")]
    NotAStore {
        /// The store path.
        store: String,
        /// What is missing or wrong.
        problem: String,
    },
    /// KY-E117: A CellBlock is not valid CSR, or does not fit the store or layer it is written to (format.md sections 4.8 and 10.2).
    #[error("KY-E117: invalid CellBlock ({context}): {problem}")]
    InvalidCellBlock {
        /// Where the block was given: the constructor, or the writer layer.
        context: String,
        /// What is wrong and where (row, position).
        problem: String,
    },
    /// KY-E118: A store writer's target path already exists (format.md section 10.1).
    #[error("KY-E118: cannot write a store to '{path}': {problem}")]
    StoreExists {
        /// The target path.
        path: String,
        /// What is there.
        problem: String,
    },
    /// KY-E119: A cell or gene table given to a store writer does not have one row per matrix row or column (format.md section 5.1).
    #[error("KY-E119: the {axis} table has {n_rows} rows, but the matrix has {n_axis} {axis}")]
    TableLength {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// Rows in the table.
        n_rows: u64,
        /// Length of the matrix axis.
        n_axis: u64,
    },
    /// KY-E120: An input file given to convert has a format or an encoding kyto does not import.
    #[error("KY-E120: cannot import '{path}': {problem}")]
    UnsupportedInput {
        /// The input path, or the member inside it.
        path: String,
        /// What is not supported.
        problem: String,
        /// What would be imported, and how to get there.
        accepted: String,
    },
    /// KY-E121: An input file given to convert holds a value or a structure that breaks its own format or cannot be stored.
    #[error("KY-E121: invalid input '{path}' at {location}: {problem}")]
    InvalidInput {
        /// The input path, or the member inside it.
        path: String,
        /// Where in the input: row, column, line or entry.
        location: String,
        /// What is wrong.
        problem: String,
        /// How to fix it.
        advice: String,
    },
    /// KY-E122: A store's schema.json is missing, is not JSON, or breaks format.md section 8.
    #[error("KY-E122: schema.json of the store at '{store}' is invalid: {problem}")]
    CorruptSchema {
        /// The store path.
        store: String,
        /// What is wrong, with the line and column of the key.
        problem: String,
    },
    /// KY-E123: An export's target path already exists (sink_h5ad, task 2.7).
    #[error("KY-E123: cannot write '{path}': {problem}")]
    OutputExists {
        /// The target path.
        path: String,
        /// What is there.
        problem: String,
    },
    /// KY-E124: A store holds a name or value the export format cannot hold (sink_h5ad, task 2.7).
    #[error("KY-E124: cannot export {location} to '{path}': {problem}")]
    UnsupportedExport {
        /// What cannot be exported: column, layer or value.
        location: String,
        /// The target path.
        path: String,
        /// Why the format cannot hold it.
        problem: String,
        /// How to fix it.
        advice: String,
    },
    /// KY-E125: A remote store URL or one of its storage options cannot be used to open an object store (task 2.8).
    #[error("KY-E125: cannot open remote store '{url}': {problem}")]
    InvalidRemoteStore {
        /// The store URL, with any password removed.
        url: String,
        /// What is wrong with the URL or the options.
        problem: String,
        /// How to fix it.
        advice: String,
    },
    /// KY-E201: A feature name matches more than one feature.
    #[error("KY-E201: ambiguous name '{name}': it matches {}", quoted_list(.candidates))]
    AmbiguousName {
        /// The name as written to feature(...).
        name: String,
        /// The full names it matches (docs/feature_names.md), each followed by its kind when two are written alike.
        candidates: Vec<String>,
    },
    /// KY-E202: A cell expression and a gene expression are combined.
    #[error("KY-E202: '{operation}' mixes a cell expression ({cell_expression}) with a gene expression ({gene_expression})")]
    MixedAxes {
        /// The operator or method that combined them.
        operation: String,
        /// The cell expression.
        cell_expression: String,
        /// The gene expression.
        gene_expression: String,
    },
    /// KY-E203: A gene removed by a filter is read later in the plan.
    #[error("KY-E203: gene '{gene}' was removed by '{removed_by}' earlier in the plan")]
    RemovedGene {
        /// The gene that was read.
        gene: String,
        /// The plan node that removed it.
        removed_by: String,
    },
    /// KY-E204: A gene symbol matches more than one gene ID.
    #[error("KY-E204: symbol '{symbol}' matches more than one gene: {}", quoted_list(.gene_ids))]
    AmbiguousSymbol {
        /// The symbol as written.
        symbol: String,
        /// The gene IDs that carry this symbol.
        gene_ids: Vec<String>,
    },
    /// KY-E205: An expression operation is given an operand of a dtype or type it does not accept.
    #[error("KY-E205: '{operation}' needs {accepted}, got {got}: {operand}")]
    ExpressionOperand {
        /// The operator or method, such as `+`, `&` or `str.starts_with`.
        operation: String,
        /// What the operation accepts.
        accepted: String,
        /// What it was given.
        got: String,
        /// The operand, written as code.
        operand: String,
    },
    /// KY-E206: An expression is used where Python needs True or False.
    #[error("KY-E206: the truth value of an expression is ambiguous: {expression}")]
    ExpressionTruthValue {
        /// The expression, written as code.
        expression: String,
    },
    /// KY-E207: An expression tree is deeper than kyto builds.
    #[error("KY-E207: '{operation}' would make an expression {depth} levels deep; the limit is {limit}")]
    ExpressionDepth {
        /// The operator or method that would exceed the limit.
        operation: String,
        /// The depth the result would have.
        depth: u64,
        /// The deepest expression kyto builds.
        limit: u64,
    },
    /// KY-E208: A feature name cannot name any feature.
    #[error("KY-E208: invalid feature name '{name}' in feature(...): {problem}")]
    FeatureNameSyntax {
        /// The name as written.
        name: String,
        /// What is wrong with it.
        problem: String,
    },
    /// KY-E209: An embedding component is not a non-negative int.
    #[error("KY-E209: '{constructor}' needs a non-negative int component (0-based), got {got}: {component}")]
    EmbeddingComponent {
        /// The constructor called.
        constructor: String,
        /// The embedding name passed.
        name: String,
        /// What the component was.
        got: String,
        /// The component, written as code.
        component: String,
    },
    /// KY-E210: A name does not match any column, gene, cell, embedding component or feature of the plan.
    #[error("KY-E210: unknown {kind} '{name}' in {expression}: {problem}")]
    UnknownName {
        /// What the name should name: `cell column`, `gene column`, `gene`, `gene symbol`, `cell`, `cell embedding`, `gene embedding`, `embedding component` or `feature`.
        kind: String,
        /// The name as written.
        name: String,
        /// The expression it was written in, as code.
        expression: String,
        /// Why nothing matches.
        problem: String,
    },
    /// KY-E211: A gene or cell is read from the default layer, but the plan has none.
    #[error("KY-E211: {expression} reads the default layer, but the plan has none")]
    NoDefaultLayer {
        /// The expression that reads it, as code.
        expression: String,
        /// The store's layers.
        layers: Vec<String>,
    },
    /// KY-E212: An expression is compared with None, which is not a test for missing values.
    #[error("KY-E212: '{operation}' with None does not test for missing values: {expression}")]
    NullComparison {
        /// The comparison, `==` or `!=`.
        operation: String,
        /// The comparison as written, as code.
        expression: String,
        /// The null test to write instead, as code.
        suggestion: String,
    },
    /// KY-E213: `.recompute()` is called on an expression that is not a plain column read.
    #[error("KY-E213: '.recompute()' needs a plain column read such as {example}, got {expression}")]
    RecomputeOperand {
        /// The expression it was called on, as code.
        expression: String,
        /// A plain column read of the same axis, as code.
        example: String,
    },
    /// KY-E214: An argument that picks how a name is looked up is given a value it does not accept.
    #[error("KY-E214: {function}(..., {parameter}={value}) is not accepted")]
    InvalidChoice {
        /// The function or method called.
        function: String,
        /// The argument name.
        parameter: String,
        /// The value given, as code.
        value: String,
        /// The values it accepts.
        accepted: Vec<String>,
    },
    /// KY-E215: A plan method is given an argument value outside what it accepts.
    #[error("KY-E215: {function}(..., {parameter}={value}) is invalid: {problem}")]
    InvalidArgument {
        /// The plan method called.
        function: String,
        /// The argument name.
        parameter: String,
        /// The value given, as code.
        value: String,
        /// What is wrong with it.
        problem: String,
        /// What the argument accepts.
        accepted: String,
    },
    /// KY-E216: A plan that ends in a sink is extended with another step.
    #[error("KY-E216: '{method}' cannot follow '{sink}': a plan ends at its sink")]
    PlanEnded {
        /// The plan method called.
        method: String,
        /// The sink that ends the plan.
        sink: String,
    },
    /// KY-E301: A derived column is read after an upstream step changed its inputs.
    #[error("KY-E301: column '{column}' was computed before '{changed_by}' changed its inputs")]
    StaleColumn {
        /// The derived column.
        column: String,
        /// The plan node that changed its inputs.
        changed_by: String,
    },
    /// KY-E302: A column without lineage is asked to be recomputed.
    #[error("KY-E302: {expression} asks to recompute {kind} '{column}', but the store records no lineage for it")]
    NoLineage {
        /// The expression, as code.
        expression: String,
        /// `cell column` or `gene column`.
        kind: String,
        /// The column.
        column: String,
    },
    /// KY-E401: A plan needs more memory than its limit, even with the smallest blocks.
    #[error("KY-E401: the plan needs at least {required_bytes} bytes but memory_limit is {limit_bytes} bytes")]
    MemoryLimit {
        /// The smallest peak memory the plan can run in, in bytes.
        required_bytes: u64,
        /// The memory limit given, in bytes.
        limit_bytes: u64,
    },
    /// KY-E501: The requested device has no kernel for an operation.
    #[error("KY-E501: no {device} kernel for '{operation}'")]
    MissingKernel {
        /// The operation without a kernel.
        operation: String,
        /// The device requested.
        device: String,
    },
    /// KY-E601: A documented method is called before the task that implements it is done.
    #[error("KY-E601: '{method}' is not implemented yet: it arrives with task {task} of the implementation plan")]
    NotImplementedYet {
        /// The method called, as Class.method.
        method: String,
        /// The task of the implementation plan that implements it.
        task: String,
    },
}

impl KytoError {
    /// Every registered code, in `errors.toml` order.
    pub const CODES: [&'static str; 46] = [
        "KY-E101",
        "KY-E102",
        "KY-E103",
        "KY-E104",
        "KY-E105",
        "KY-E106",
        "KY-E107",
        "KY-E108",
        "KY-E109",
        "KY-E110",
        "KY-E111",
        "KY-E112",
        "KY-E113",
        "KY-E114",
        "KY-E115",
        "KY-E116",
        "KY-E117",
        "KY-E118",
        "KY-E119",
        "KY-E120",
        "KY-E121",
        "KY-E122",
        "KY-E123",
        "KY-E124",
        "KY-E125",
        "KY-E201",
        "KY-E202",
        "KY-E203",
        "KY-E204",
        "KY-E205",
        "KY-E206",
        "KY-E207",
        "KY-E208",
        "KY-E209",
        "KY-E210",
        "KY-E211",
        "KY-E212",
        "KY-E213",
        "KY-E214",
        "KY-E215",
        "KY-E216",
        "KY-E301",
        "KY-E302",
        "KY-E401",
        "KY-E501",
        "KY-E601",
    ];

    /// The stable `KY-Exxx` code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::DuplicateIds { .. } => "KY-E101",
            Self::InvalidMemberName { .. } => "KY-E102",
            Self::MemberExists { .. } => "KY-E103",
            Self::InvalidParameter { .. } => "KY-E104",
            Self::LimitExceeded { .. } => "KY-E105",
            Self::InvalidSparseMatrix { .. } => "KY-E106",
            Self::CorruptLayer { .. } => "KY-E107",
            Self::LayerDtype { .. } => "KY-E108",
            Self::MissingMember { .. } => "KY-E109",
            Self::StoreIo { .. } => "KY-E110",
            Self::InvalidIds { .. } => "KY-E111",
            Self::InvalidTableColumn { .. } => "KY-E112",
            Self::CorruptTable { .. } => "KY-E113",
            Self::InvalidSelection { .. } => "KY-E114",
            Self::UnsupportedFormatVersion { .. } => "KY-E115",
            Self::NotAStore { .. } => "KY-E116",
            Self::InvalidCellBlock { .. } => "KY-E117",
            Self::StoreExists { .. } => "KY-E118",
            Self::TableLength { .. } => "KY-E119",
            Self::UnsupportedInput { .. } => "KY-E120",
            Self::InvalidInput { .. } => "KY-E121",
            Self::CorruptSchema { .. } => "KY-E122",
            Self::OutputExists { .. } => "KY-E123",
            Self::UnsupportedExport { .. } => "KY-E124",
            Self::InvalidRemoteStore { .. } => "KY-E125",
            Self::AmbiguousName { .. } => "KY-E201",
            Self::MixedAxes { .. } => "KY-E202",
            Self::RemovedGene { .. } => "KY-E203",
            Self::AmbiguousSymbol { .. } => "KY-E204",
            Self::ExpressionOperand { .. } => "KY-E205",
            Self::ExpressionTruthValue { .. } => "KY-E206",
            Self::ExpressionDepth { .. } => "KY-E207",
            Self::FeatureNameSyntax { .. } => "KY-E208",
            Self::EmbeddingComponent { .. } => "KY-E209",
            Self::UnknownName { .. } => "KY-E210",
            Self::NoDefaultLayer { .. } => "KY-E211",
            Self::NullComparison { .. } => "KY-E212",
            Self::RecomputeOperand { .. } => "KY-E213",
            Self::InvalidChoice { .. } => "KY-E214",
            Self::InvalidArgument { .. } => "KY-E215",
            Self::PlanEnded { .. } => "KY-E216",
            Self::StaleColumn { .. } => "KY-E301",
            Self::NoLineage { .. } => "KY-E302",
            Self::MemoryLimit { .. } => "KY-E401",
            Self::MissingKernel { .. } => "KY-E501",
            Self::NotImplementedYet { .. } => "KY-E601",
        }
    }

    /// The `kyto.errors` class this error converts to in Python.
    pub fn python_class(&self) -> &'static str {
        match self {
            Self::DuplicateIds { .. } => "DuplicateIdError",
            Self::InvalidMemberName { .. } => "StoreInputError",
            Self::MemberExists { .. } => "StoreInputError",
            Self::InvalidParameter { .. } => "StoreInputError",
            Self::LimitExceeded { .. } => "StoreInputError",
            Self::InvalidSparseMatrix { .. } => "StoreInputError",
            Self::CorruptLayer { .. } => "CorruptStoreError",
            Self::LayerDtype { .. } => "LayerDtypeError",
            Self::MissingMember { .. } => "MissingMemberError",
            Self::StoreIo { .. } => "StoreIOError",
            Self::InvalidIds { .. } => "StoreInputError",
            Self::InvalidTableColumn { .. } => "StoreInputError",
            Self::CorruptTable { .. } => "CorruptStoreError",
            Self::InvalidSelection { .. } => "StoreInputError",
            Self::UnsupportedFormatVersion { .. } => "FormatVersionError",
            Self::NotAStore { .. } => "CorruptStoreError",
            Self::InvalidCellBlock { .. } => "StoreInputError",
            Self::StoreExists { .. } => "StoreInputError",
            Self::TableLength { .. } => "StoreInputError",
            Self::UnsupportedInput { .. } => "InputFileError",
            Self::InvalidInput { .. } => "InputFileError",
            Self::CorruptSchema { .. } => "CorruptStoreError",
            Self::OutputExists { .. } => "StoreInputError",
            Self::UnsupportedExport { .. } => "StoreInputError",
            Self::InvalidRemoteStore { .. } => "StoreInputError",
            Self::AmbiguousName { .. } => "ResolutionError",
            Self::MixedAxes { .. } => "AxisError",
            Self::RemovedGene { .. } => "RemovedFeatureError",
            Self::AmbiguousSymbol { .. } => "ResolutionError",
            Self::ExpressionOperand { .. } => "ExpressionError",
            Self::ExpressionTruthValue { .. } => "ExpressionError",
            Self::ExpressionDepth { .. } => "ExpressionError",
            Self::FeatureNameSyntax { .. } => "ResolutionError",
            Self::EmbeddingComponent { .. } => "ExpressionError",
            Self::UnknownName { .. } => "ResolutionError",
            Self::NoDefaultLayer { .. } => "ResolutionError",
            Self::NullComparison { .. } => "ExpressionError",
            Self::RecomputeOperand { .. } => "ExpressionError",
            Self::InvalidChoice { .. } => "ResolutionError",
            Self::InvalidArgument { .. } => "PlanError",
            Self::PlanEnded { .. } => "PlanError",
            Self::StaleColumn { .. } => "StaleColumnError",
            Self::NoLineage { .. } => "LineageError",
            Self::MemoryLimit { .. } => "MemoryLimitError",
            Self::MissingKernel { .. } => "DeviceError",
            Self::NotImplementedYet { .. } => "NotImplementedYetError",
        }
    }

    /// The link to this code's documentation.
    pub fn docs_url(&self) -> &'static str {
        match self {
            Self::DuplicateIds { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e101"
            }
            Self::InvalidMemberName { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e102"
            }
            Self::MemberExists { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e103"
            }
            Self::InvalidParameter { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e104"
            }
            Self::LimitExceeded { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e105"
            }
            Self::InvalidSparseMatrix { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e106"
            }
            Self::CorruptLayer { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e107"
            }
            Self::LayerDtype { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e108"
            }
            Self::MissingMember { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e109"
            }
            Self::StoreIo { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e110"
            }
            Self::InvalidIds { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e111"
            }
            Self::InvalidTableColumn { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e112"
            }
            Self::CorruptTable { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e113"
            }
            Self::InvalidSelection { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e114"
            }
            Self::UnsupportedFormatVersion { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e115"
            }
            Self::NotAStore { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e116"
            }
            Self::InvalidCellBlock { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e117"
            }
            Self::StoreExists { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e118"
            }
            Self::TableLength { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e119"
            }
            Self::UnsupportedInput { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e120"
            }
            Self::InvalidInput { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e121"
            }
            Self::CorruptSchema { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e122"
            }
            Self::OutputExists { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e123"
            }
            Self::UnsupportedExport { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e124"
            }
            Self::InvalidRemoteStore { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e125"
            }
            Self::AmbiguousName { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e201"
            }
            Self::MixedAxes { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e202"
            }
            Self::RemovedGene { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e203"
            }
            Self::AmbiguousSymbol { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e204"
            }
            Self::ExpressionOperand { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e205"
            }
            Self::ExpressionTruthValue { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e206"
            }
            Self::ExpressionDepth { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e207"
            }
            Self::FeatureNameSyntax { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e208"
            }
            Self::EmbeddingComponent { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e209"
            }
            Self::UnknownName { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e210"
            }
            Self::NoDefaultLayer { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e211"
            }
            Self::NullComparison { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e212"
            }
            Self::RecomputeOperand { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e213"
            }
            Self::InvalidChoice { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e214"
            }
            Self::InvalidArgument { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e215"
            }
            Self::PlanEnded { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e216"
            }
            Self::StaleColumn { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e301"
            }
            Self::NoLineage { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e302"
            }
            Self::MemoryLimit { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e401"
            }
            Self::MissingKernel { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e501"
            }
            Self::NotImplementedYet { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e601"
            }
        }
    }

    /// How to fix the error, with its fields filled in.
    pub fn hint(&self) -> String {
        match self {
            Self::DuplicateIds { axis, .. } => {
                format!("IDs on the {axis} axis must be unique (exact byte comparison); drop or rename the duplicates before writing, or import with on_duplicate_ids='suffix' to make them unique")
            }
            Self::InvalidMemberName { .. } => {
                String::from("use 1 to 64 characters from A-Z a-z 0-9 _ . - that do not start with . or __, do not end with . and are not a Windows device name such as CON")
            }
            Self::MemberExists { .. } => {
                String::from("choose another name; names are compared case-insensitively because macOS and Windows file systems are")
            }
            Self::InvalidParameter { parameter, accepted, .. } => {
                format!("{parameter} must be {accepted}")
            }
            Self::LimitExceeded { advice, .. } => {
                format!("{advice} to stay within the limits of format.md section 4.5")
            }
            Self::InvalidSparseMatrix { .. } => {
                String::from("pass CSR with indptr of length n_cells + 1 starting at 0 and non-decreasing, and sorted, unique gene indices below n_genes in each row (format.md section 4.3)")
            }
            Self::CorruptLayer { .. } => {
                String::from("the store breaks docs/format.md section 4; rewrite the layer with kyto, or report a bug if kyto wrote it")
            }
            Self::LayerDtype { layer, stored, .. } => {
                format!("read layer '{layer}' as {stored}")
            }
            Self::MissingMember { .. } => {
                String::from("check the name against the store's schema; names are case-sensitive when read")
            }
            Self::StoreIo { .. } => {
                String::from("check that the path exists, that you may read and write it, and that the disk is not full")
            }
            Self::InvalidIds { axis, .. } => {
                format!("every {axis} ID must be a non-null, non-empty string; fill or drop those rows before writing")
            }
            Self::InvalidTableColumn { accepted, .. } => {
                format!("{accepted} (format.md section 5)")
            }
            Self::CorruptTable { .. } => {
                String::from("the store breaks docs/format.md section 5; rewrite the table with kyto, or report a bug if kyto wrote it")
            }
            Self::InvalidSelection { axis, n_axis, .. } => {
                format!("select {axis} by sorted, unique table positions below {n_axis}; the plan canonicalises other orders and repeats (format.md section 4.8)")
            }
            Self::UnsupportedFormatVersion { .. } => {
                String::from("open the store with the kyto version that wrote it, or rewrite it with this one")
            }
            Self::NotAStore { .. } => {
                String::from("check the path; a kyto store has a root zarr.json with attributes.kyto.format_version, written last, so an interrupted write leaves none")
            }
            Self::InvalidCellBlock { .. } => {
                String::from("build CellBlocks with indptr starting at 0, non-decreasing and ending at nnz, and sorted, unique gene positions below n_genes in each row; a store writer also needs one block per layer, in layer order, over all genes of the store, with the layer's dtype and the same rows in every layer")
            }
            Self::StoreExists { .. } => {
                String::from("choose another path, remove the existing one first, or write with overwrite=True to replace an existing kyto store (it is replaced only after the new store is complete)")
            }
            Self::TableLength { axis, .. } => {
                format!("pass one {axis} table row per matrix {axis} entry, in matrix order")
            }
            Self::UnsupportedInput { accepted, .. } => {
                format!("{accepted} (see the kyto.convert docstring for the inputs it reads)")
            }
            Self::InvalidInput { advice, .. } => {
                format!("{advice}, then convert again")
            }
            Self::CorruptSchema { .. } => {
                String::from("the store breaks docs/format.md section 8; write the store again with kyto, or report a bug if kyto wrote it")
            }
            Self::OutputExists { .. } => {
                String::from("choose another path, remove the existing one first, or pass overwrite=True to replace an existing file (it is replaced only after the new file is complete; a directory is never replaced)")
            }
            Self::UnsupportedExport { advice, .. } => {
                format!("{advice}, then export again")
            }
            Self::InvalidRemoteStore { advice, .. } => {
                format!("{advice}, then open the store again")
            }
            Self::AmbiguousName { candidates, .. } => {
                format!("pass one full name instead, one of {}; where two share a full name, use cell_column(...), gene_expression(..., layer=...) or cell_embedding(...)", quoted_list(candidates))
            }
            Self::MixedAxes { .. } => {
                String::from("combine expressions of one axis only: cell_column(...) with cell_column(...), gene_column(...) with gene_column(...)")
            }
            Self::RemovedGene { gene, removed_by } => {
                format!("read '{gene}' before '{removed_by}', or change the filter to keep it")
            }
            Self::AmbiguousSymbol { gene_ids, .. } => {
                format!("select the gene by ID with by='id', one of {}", quoted_list(gene_ids))
            }
            Self::ExpressionOperand { operation, accepted, got, .. } => {
                format!("pass {accepted} to '{operation}', or use an operation that accepts {got}")
            }
            Self::ExpressionTruthValue { .. } => {
                String::from("combine conditions with &, | and ~ instead of and, or and not, and write a < b < c as (a < b) & (b < c); an expression is evaluated only when the plan runs")
            }
            Self::ExpressionDepth { .. } => {
                String::from("build long chains as a balanced tree, combining pairs and then pairs of pairs, instead of adding one term at a time")
            }
            Self::FeatureNameSyntax { .. } => {
                String::from("write a short name such as 'CD3E' or 'cluster', or a full name '<prefix>:<name>' such as 'log1p:CD3E' or 'meta:cluster'")
            }
            Self::EmbeddingComponent { constructor, name, .. } => {
                format!("pass the component as an int from 0, such as {constructor}('{name}', 0) for the first one")
            }
            Self::UnknownName { name, .. } => {
                format!("names are matched exactly, case included; check '{name}' against the store's schema")
            }
            Self::NoDefaultLayer { layers, .. } => {
                format!("set one with .with_default_layer(...) before resolving, or name the layer; the store has {}", quoted_list(layers))
            }
            Self::NullComparison { suggestion, .. } => {
                format!("write {suggestion} instead; missing values are null (R9)")
            }
            Self::RecomputeOperand { example, .. } => {
                format!("call .recompute() on the column read itself, then build on the result: write {example} first")
            }
            Self::InvalidChoice { parameter, accepted, .. } => {
                format!("{parameter} takes one of {}", quoted_list(accepted))
            }
            Self::InvalidArgument { parameter, accepted, .. } => {
                format!("pass {accepted} as {parameter}")
            }
            Self::PlanEnded { method, sink } => {
                format!("add '{method}' before '{sink}', or build a new plan from the store that '{sink}' writes")
            }
            Self::StaleColumn { column, changed_by } => {
                format!("call .recompute() on '{column}' after '{changed_by}', or read it before that step")
            }
            Self::NoLineage { column, .. } => {
                format!("only derived columns written by kyto record their lineage (schema.json); read '{column}' as stored, or compute it again from the matrix")
            }
            Self::MemoryLimit { required_bytes, .. } => {
                format!("raise memory_limit to at least {required_bytes} bytes, or select fewer cells or genes before this step")
            }
            Self::MissingKernel { .. } => {
                String::from("pass device='auto' to fall back to the CPU for this step, or device='cpu'")
            }
            Self::NotImplementedYet { .. } => {
                String::from("inspect the data without reading it through names(), schema() and estimate() until then")
            }
        }
    }

    /// The structured fields as `(name, value)` pairs, in `errors.toml` order.
    pub fn fields(&self) -> Vec<(&'static str, FieldValue)> {
        match self {
            Self::DuplicateIds { axis, n_duplicated, ids } => vec![
                ("axis", FieldValue::Str(axis.clone())),
                ("n_duplicated", FieldValue::Int(*n_duplicated)),
                ("ids", FieldValue::StrList(ids.clone())),
            ],
            Self::InvalidMemberName { kind, name, rule } => vec![
                ("kind", FieldValue::Str(kind.clone())),
                ("name", FieldValue::Str(name.clone())),
                ("rule", FieldValue::Str(rule.clone())),
            ],
            Self::MemberExists { kind, name, existing } => vec![
                ("kind", FieldValue::Str(kind.clone())),
                ("name", FieldValue::Str(name.clone())),
                ("existing", FieldValue::Str(existing.clone())),
            ],
            Self::InvalidParameter { parameter, value, accepted } => vec![
                ("parameter", FieldValue::Str(parameter.clone())),
                ("value", FieldValue::Str(value.clone())),
                ("accepted", FieldValue::Str(accepted.clone())),
            ],
            Self::LimitExceeded { quantity, value, limit, advice } => vec![
                ("quantity", FieldValue::Str(quantity.clone())),
                ("value", FieldValue::Int(*value)),
                ("limit", FieldValue::Int(*limit)),
                ("advice", FieldValue::Str(advice.clone())),
            ],
            Self::InvalidSparseMatrix { layer, problem } => vec![
                ("layer", FieldValue::Str(layer.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::CorruptLayer { layer, problem } => vec![
                ("layer", FieldValue::Str(layer.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::LayerDtype { layer, stored, requested } => vec![
                ("layer", FieldValue::Str(layer.clone())),
                ("stored", FieldValue::Str(stored.clone())),
                ("requested", FieldValue::Str(requested.clone())),
            ],
            Self::MissingMember { kind, name, store } => vec![
                ("kind", FieldValue::Str(kind.clone())),
                ("name", FieldValue::Str(name.clone())),
                ("store", FieldValue::Str(store.clone())),
            ],
            Self::StoreIo { operation, path, detail } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("path", FieldValue::Str(path.clone())),
                ("detail", FieldValue::Str(detail.clone())),
            ],
            Self::InvalidIds { axis, n_invalid, problem, rows } => vec![
                ("axis", FieldValue::Str(axis.clone())),
                ("n_invalid", FieldValue::Int(*n_invalid)),
                ("problem", FieldValue::Str(problem.clone())),
                ("rows", FieldValue::StrList(rows.clone())),
            ],
            Self::InvalidTableColumn { axis, column, problem, accepted } => vec![
                ("axis", FieldValue::Str(axis.clone())),
                ("column", FieldValue::Str(column.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("accepted", FieldValue::Str(accepted.clone())),
            ],
            Self::CorruptTable { path, problem } => vec![
                ("path", FieldValue::Str(path.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::InvalidSelection { axis, problem, n_axis } => vec![
                ("axis", FieldValue::Str(axis.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("n_axis", FieldValue::Int(*n_axis)),
            ],
            Self::UnsupportedFormatVersion { store, found, supported } => vec![
                ("store", FieldValue::Str(store.clone())),
                ("found", FieldValue::Str(found.clone())),
                ("supported", FieldValue::StrList(supported.clone())),
            ],
            Self::NotAStore { store, problem } => vec![
                ("store", FieldValue::Str(store.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::InvalidCellBlock { context, problem } => vec![
                ("context", FieldValue::Str(context.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::StoreExists { path, problem } => vec![
                ("path", FieldValue::Str(path.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::TableLength { axis, n_rows, n_axis } => vec![
                ("axis", FieldValue::Str(axis.clone())),
                ("n_rows", FieldValue::Int(*n_rows)),
                ("n_axis", FieldValue::Int(*n_axis)),
            ],
            Self::UnsupportedInput { path, problem, accepted } => vec![
                ("path", FieldValue::Str(path.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("accepted", FieldValue::Str(accepted.clone())),
            ],
            Self::InvalidInput { path, location, problem, advice } => vec![
                ("path", FieldValue::Str(path.clone())),
                ("location", FieldValue::Str(location.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("advice", FieldValue::Str(advice.clone())),
            ],
            Self::CorruptSchema { store, problem } => vec![
                ("store", FieldValue::Str(store.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::OutputExists { path, problem } => vec![
                ("path", FieldValue::Str(path.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::UnsupportedExport { location, path, problem, advice } => vec![
                ("location", FieldValue::Str(location.clone())),
                ("path", FieldValue::Str(path.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("advice", FieldValue::Str(advice.clone())),
            ],
            Self::InvalidRemoteStore { url, problem, advice } => vec![
                ("url", FieldValue::Str(url.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("advice", FieldValue::Str(advice.clone())),
            ],
            Self::AmbiguousName { name, candidates } => vec![
                ("name", FieldValue::Str(name.clone())),
                ("candidates", FieldValue::StrList(candidates.clone())),
            ],
            Self::MixedAxes { operation, cell_expression, gene_expression } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("cell_expression", FieldValue::Str(cell_expression.clone())),
                ("gene_expression", FieldValue::Str(gene_expression.clone())),
            ],
            Self::RemovedGene { gene, removed_by } => vec![
                ("gene", FieldValue::Str(gene.clone())),
                ("removed_by", FieldValue::Str(removed_by.clone())),
            ],
            Self::AmbiguousSymbol { symbol, gene_ids } => vec![
                ("symbol", FieldValue::Str(symbol.clone())),
                ("gene_ids", FieldValue::StrList(gene_ids.clone())),
            ],
            Self::ExpressionOperand { operation, accepted, got, operand } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("accepted", FieldValue::Str(accepted.clone())),
                ("got", FieldValue::Str(got.clone())),
                ("operand", FieldValue::Str(operand.clone())),
            ],
            Self::ExpressionTruthValue { expression } => vec![
                ("expression", FieldValue::Str(expression.clone())),
            ],
            Self::ExpressionDepth { operation, depth, limit } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("depth", FieldValue::Int(*depth)),
                ("limit", FieldValue::Int(*limit)),
            ],
            Self::FeatureNameSyntax { name, problem } => vec![
                ("name", FieldValue::Str(name.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::EmbeddingComponent { constructor, name, got, component } => vec![
                ("constructor", FieldValue::Str(constructor.clone())),
                ("name", FieldValue::Str(name.clone())),
                ("got", FieldValue::Str(got.clone())),
                ("component", FieldValue::Str(component.clone())),
            ],
            Self::UnknownName { kind, name, expression, problem } => vec![
                ("kind", FieldValue::Str(kind.clone())),
                ("name", FieldValue::Str(name.clone())),
                ("expression", FieldValue::Str(expression.clone())),
                ("problem", FieldValue::Str(problem.clone())),
            ],
            Self::NoDefaultLayer { expression, layers } => vec![
                ("expression", FieldValue::Str(expression.clone())),
                ("layers", FieldValue::StrList(layers.clone())),
            ],
            Self::NullComparison { operation, expression, suggestion } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("expression", FieldValue::Str(expression.clone())),
                ("suggestion", FieldValue::Str(suggestion.clone())),
            ],
            Self::RecomputeOperand { expression, example } => vec![
                ("expression", FieldValue::Str(expression.clone())),
                ("example", FieldValue::Str(example.clone())),
            ],
            Self::InvalidChoice { function, parameter, value, accepted } => vec![
                ("function", FieldValue::Str(function.clone())),
                ("parameter", FieldValue::Str(parameter.clone())),
                ("value", FieldValue::Str(value.clone())),
                ("accepted", FieldValue::StrList(accepted.clone())),
            ],
            Self::InvalidArgument { function, parameter, value, problem, accepted } => vec![
                ("function", FieldValue::Str(function.clone())),
                ("parameter", FieldValue::Str(parameter.clone())),
                ("value", FieldValue::Str(value.clone())),
                ("problem", FieldValue::Str(problem.clone())),
                ("accepted", FieldValue::Str(accepted.clone())),
            ],
            Self::PlanEnded { method, sink } => vec![
                ("method", FieldValue::Str(method.clone())),
                ("sink", FieldValue::Str(sink.clone())),
            ],
            Self::StaleColumn { column, changed_by } => vec![
                ("column", FieldValue::Str(column.clone())),
                ("changed_by", FieldValue::Str(changed_by.clone())),
            ],
            Self::NoLineage { expression, kind, column } => vec![
                ("expression", FieldValue::Str(expression.clone())),
                ("kind", FieldValue::Str(kind.clone())),
                ("column", FieldValue::Str(column.clone())),
            ],
            Self::MemoryLimit { required_bytes, limit_bytes } => vec![
                ("required_bytes", FieldValue::Int(*required_bytes)),
                ("limit_bytes", FieldValue::Int(*limit_bytes)),
            ],
            Self::MissingKernel { operation, device } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("device", FieldValue::Str(device.clone())),
            ],
            Self::NotImplementedYet { method, task } => vec![
                ("method", FieldValue::Str(method.clone())),
                ("task", FieldValue::Str(task.clone())),
            ],
        }
    }

    /// One error per registered code, built from the `example` values in
    /// `errors.toml` (for tests and docs).
    pub fn examples() -> Vec<Self> {
        vec![
            Self::DuplicateIds {
                axis: "genes".to_owned(),
                n_duplicated: 12,
                ids: vec!["ENSG01".to_owned(), "ENSG02".to_owned()],
            },
            Self::InvalidMemberName {
                kind: "layer".to_owned(),
                name: "log1p/counts".to_owned(),
                rule: "character '/' is not allowed".to_owned(),
            },
            Self::MemberExists {
                kind: "layer".to_owned(),
                name: "Counts".to_owned(),
                existing: "counts".to_owned(),
            },
            Self::InvalidParameter {
                parameter: "nnz_shard_size".to_owned(),
                value: "1000".to_owned(),
                accepted: "a positive multiple of nnz_chunk_size (256)".to_owned(),
            },
            Self::LimitExceeded {
                quantity: "cell_block_size * gene_block_size".to_owned(),
                value: 8589934592,
                limit: 4294967295,
                advice: "use smaller cell_block_size or gene_block_size".to_owned(),
            },
            Self::InvalidSparseMatrix {
                layer: "counts".to_owned(),
                problem: "row 4: gene index 7 follows 9; indices must be strictly increasing".to_owned(),
            },
            Self::CorruptLayer {
                layer: "counts".to_owned(),
                problem: "tile (0, 2): indptr ends at 5 but tile_offsets gives 7 entries".to_owned(),
            },
            Self::LayerDtype {
                layer: "counts".to_owned(),
                stored: "uint32".to_owned(),
                requested: "float32".to_owned(),
            },
            Self::MissingMember {
                kind: "layer".to_owned(),
                name: "count".to_owned(),
                store: "pbmc.kyto".to_owned(),
            },
            Self::StoreIo {
                operation: "write array".to_owned(),
                path: "pbmc.kyto/layers/counts/data".to_owned(),
                detail: "No space left on device (os error 28)".to_owned(),
            },
            Self::InvalidIds {
                axis: "cells".to_owned(),
                n_invalid: 2,
                problem: "null".to_owned(),
                rows: vec!["3".to_owned(), "17".to_owned()],
            },
            Self::InvalidTableColumn {
                axis: "cells".to_owned(),
                column: "sample".to_owned(),
                problem: "has dtype Categorical, which v0 stores do not allow".to_owned(),
                accepted: "cast it to String or to an Enum with fixed categories".to_owned(),
            },
            Self::CorruptTable {
                path: "pbmc.kyto/cells.parquet".to_owned(),
                problem: "the first column is 'barcode', not 'cell_id'".to_owned(),
            },
            Self::InvalidSelection {
                axis: "genes".to_owned(),
                problem: "position 2 is gene 7, which does not follow gene 9".to_owned(),
                n_axis: 2000,
            },
            Self::UnsupportedFormatVersion {
                store: "pbmc.kyto".to_owned(),
                found: "0.2".to_owned(),
                supported: vec!["0.1".to_owned()],
            },
            Self::NotAStore {
                store: "pbmc.kyto".to_owned(),
                problem: "no root zarr.json".to_owned(),
            },
            Self::InvalidCellBlock {
                context: "StoreWriter.push, layer 'counts'".to_owned(),
                problem: "the block has 2000 genes but the store has 32738".to_owned(),
            },
            Self::StoreExists {
                path: "pbmc.kyto".to_owned(),
                problem: "a kyto store already exists there".to_owned(),
            },
            Self::TableLength {
                axis: "cells".to_owned(),
                n_rows: 2699,
                n_axis: 2700,
            },
            Self::UnsupportedInput {
                path: "pbmc.h5ad:/X".to_owned(),
                problem: "encoding-type 'awkward-array' is not supported".to_owned(),
                accepted: "store X as an array, csr_matrix or csc_matrix (anndata >= 0.8 writes these)".to_owned(),
            },
            Self::InvalidInput {
                path: "pbmc.h5ad:/X".to_owned(),
                location: "cell 4, gene 17".to_owned(),
                problem: "integer value -3 is negative".to_owned(),
                advice: "integer matrices become uint32 layers; cast X to float32 in anndata before converting to keep negative values".to_owned(),
            },
            Self::CorruptSchema {
                store: "pbmc.kyto".to_owned(),
                problem: "it does not match the format (unknown variant `int64`, expected `float32` or `uint32` at line 7 column 40)".to_owned(),
            },
            Self::OutputExists {
                path: "pbmc.h5ad".to_owned(),
                problem: "a file already exists there".to_owned(),
            },
            Self::UnsupportedExport {
                location: "cells column 'a/b'".to_owned(),
                path: "pbmc.h5ad".to_owned(),
                problem: "h5ad names cannot contain '/'".to_owned(),
                advice: "rename the column without '/'".to_owned(),
            },
            Self::InvalidRemoteStore {
                url: "s3://bucket/pbmc.kyto".to_owned(),
                problem: "unknown storage option 'aws_acess_key_id'".to_owned(),
                advice: "use an option key object_store accepts for s3, such as aws_access_key_id, aws_secret_access_key, aws_region, aws_endpoint or allow_http".to_owned(),
            },
            Self::AmbiguousName {
                name: "CD3E".to_owned(),
                candidates: vec!["meta:CD3E".to_owned(), "log1p:CD3E".to_owned()],
            },
            Self::MixedAxes {
                operation: "&".to_owned(),
                cell_expression: "cell_column('n_genes') > 200".to_owned(),
                gene_expression: "gene_column('highly_variable')".to_owned(),
            },
            Self::RemovedGene {
                gene: "MALAT1".to_owned(),
                removed_by: "filter_genes".to_owned(),
            },
            Self::AmbiguousSymbol {
                symbol: "TBCE".to_owned(),
                gene_ids: vec!["ENSG00000284770".to_owned(), "ENSG00000285053".to_owned()],
            },
            Self::ExpressionOperand {
                operation: "str.starts_with".to_owned(),
                accepted: "a string expression".to_owned(),
                got: "a numeric expression".to_owned(),
                operand: "cell_column('n_genes') + 1".to_owned(),
            },
            Self::ExpressionTruthValue {
                expression: "cell_column('n_genes') > 200".to_owned(),
            },
            Self::ExpressionDepth {
                operation: "+".to_owned(),
                depth: 1001,
                limit: 1000,
            },
            Self::FeatureNameSyntax {
                name: "log1p:".to_owned(),
                problem: "nothing follows 'log1p:'".to_owned(),
            },
            Self::EmbeddingComponent {
                constructor: "cell_embedding".to_owned(),
                name: "pca".to_owned(),
                got: "a negative int".to_owned(),
                component: "-1".to_owned(),
            },
            Self::UnknownName {
                kind: "feature".to_owned(),
                name: "CD3".to_owned(),
                expression: "feature('CD3')".to_owned(),
                problem: "it is not a cell column or a gene ID".to_owned(),
            },
            Self::NoDefaultLayer {
                expression: "feature('CD3E')".to_owned(),
                layers: vec!["counts".to_owned(), "log1p".to_owned()],
            },
            Self::NullComparison {
                operation: "==".to_owned(),
                expression: "cell_column('cell_type') == None".to_owned(),
                suggestion: "cell_column('cell_type').is_null()".to_owned(),
            },
            Self::RecomputeOperand {
                expression: "cell_column('n_genes') + 1".to_owned(),
                example: "cell_column('n_genes').recompute()".to_owned(),
            },
            Self::InvalidChoice {
                function: "gene_expression".to_owned(),
                parameter: "by".to_owned(),
                value: "'name'".to_owned(),
                accepted: vec!["id".to_owned(), "symbol".to_owned()],
            },
            Self::InvalidArgument {
                function: "pca".to_owned(),
                parameter: "n_components".to_owned(),
                value: "0".to_owned(),
                problem: "a PCA needs at least one component".to_owned(),
                accepted: "an int from 1 to 2000".to_owned(),
            },
            Self::PlanEnded {
                method: "log1p".to_owned(),
                sink: "sink_zarr".to_owned(),
            },
            Self::StaleColumn {
                column: "n_genes".to_owned(),
                changed_by: "filter_genes".to_owned(),
            },
            Self::NoLineage {
                expression: "cell_column('n_genes').recompute()".to_owned(),
                kind: "cell column".to_owned(),
                column: "n_genes".to_owned(),
            },
            Self::MemoryLimit {
                required_bytes: 6000000000,
                limit_bytes: 4000000000,
            },
            Self::MissingKernel {
                operation: "leiden".to_owned(),
                device: "cuda".to_owned(),
            },
            Self::NotImplementedYet {
                method: "Matrix.to_scipy".to_owned(),
                task: "7.5".to_owned(),
            },
        ]
    }
}
