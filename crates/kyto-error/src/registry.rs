//! The registered kyto errors. GENERATED from errors.toml by scripts/error_registry.py. Do not edit.

use crate::{FieldValue, quoted_list};

/// Every kyto error: one variant per registered `KY-Exxx` code in `errors.toml`.
///
/// `Display` gives `KY-Exxx: <message>`; [`KytoError::hint`] says how to fix it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KytoError {
    /// KY-E101: IDs written to a store are not unique.
    #[error("KY-E101: duplicate {axis} IDs: {}", quoted_list(.ids))]
    DuplicateIds {
        /// The axis, `cells` or `genes`.
        axis: String,
        /// The IDs that occur more than once.
        ids: Vec<String>,
    },
    /// KY-E102: A layer, embedding or graph name breaks the naming rules of format.md section 2.1.
    #[error("KY-E102: invalid {kind} name '{name}': {rule}")]
    InvalidMemberName {
        /// The member kind: `layer`, `embedding` or `graph`.
        kind: String,
        /// The name as given.
        name: String,
        /// The rule it breaks.
        rule: String,
    },
    /// KY-E103: A member with the same name, compared case-insensitively, is already in the store.
    #[error("KY-E103: {kind} '{name}' already exists in the store as '{existing}'")]
    MemberExists {
        /// The member kind: `layer`, `embedding` or `graph`.
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
    /// KY-E109: A layer, embedding or graph is not in the store.
    #[error("KY-E109: no {kind} named '{name}' in the store at '{store}'")]
    MissingMember {
        /// The member kind: `layer`, `embedding` or `graph`.
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
    /// KY-E201: A short name matches more than one feature.
    #[error("KY-E201: ambiguous name '{name}': it matches {}", quoted_list(.candidates))]
    AmbiguousName {
        /// The short name as written.
        name: String,
        /// The full names it matches.
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
    /// KY-E301: A derived column is read after an upstream step changed its inputs.
    #[error("KY-E301: column '{column}' was computed before '{changed_by}' changed its inputs")]
    StaleColumn {
        /// The derived column.
        column: String,
        /// The plan node that changed its inputs.
        changed_by: String,
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
}

impl KytoError {
    /// Every registered code, in `errors.toml` order.
    pub const CODES: [&'static str; 17] = [
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
        "KY-E201",
        "KY-E202",
        "KY-E203",
        "KY-E204",
        "KY-E301",
        "KY-E401",
        "KY-E501",
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
            Self::AmbiguousName { .. } => "KY-E201",
            Self::MixedAxes { .. } => "KY-E202",
            Self::RemovedGene { .. } => "KY-E203",
            Self::AmbiguousSymbol { .. } => "KY-E204",
            Self::StaleColumn { .. } => "KY-E301",
            Self::MemoryLimit { .. } => "KY-E401",
            Self::MissingKernel { .. } => "KY-E501",
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
            Self::AmbiguousName { .. } => "ResolutionError",
            Self::MixedAxes { .. } => "AxisError",
            Self::RemovedGene { .. } => "RemovedFeatureError",
            Self::AmbiguousSymbol { .. } => "ResolutionError",
            Self::StaleColumn { .. } => "StaleColumnError",
            Self::MemoryLimit { .. } => "MemoryLimitError",
            Self::MissingKernel { .. } => "DeviceError",
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
            Self::StaleColumn { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e301"
            }
            Self::MemoryLimit { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e401"
            }
            Self::MissingKernel { .. } => {
                "https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e501"
            }
        }
    }

    /// How to fix the error, with its fields filled in.
    pub fn hint(&self) -> String {
        match self {
            Self::DuplicateIds { axis, .. } => {
                format!("IDs on the {axis} axis must be unique; drop or rename the duplicates before writing")
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
            Self::AmbiguousName { candidates, .. } => {
                format!("pass the full name, one of {}", quoted_list(candidates))
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
            Self::StaleColumn { column, changed_by } => {
                format!("call .recompute() on '{column}' after '{changed_by}', or read it before that step")
            }
            Self::MemoryLimit { required_bytes, .. } => {
                format!("raise memory_limit to at least {required_bytes} bytes, or select fewer cells or genes before this step")
            }
            Self::MissingKernel { .. } => {
                String::from("pass device='auto' to fall back to the CPU for this step, or device='cpu'")
            }
        }
    }

    /// The structured fields as `(name, value)` pairs, in `errors.toml` order.
    pub fn fields(&self) -> Vec<(&'static str, FieldValue)> {
        match self {
            Self::DuplicateIds { axis, ids } => vec![
                ("axis", FieldValue::Str(axis.clone())),
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
            Self::StaleColumn { column, changed_by } => vec![
                ("column", FieldValue::Str(column.clone())),
                ("changed_by", FieldValue::Str(changed_by.clone())),
            ],
            Self::MemoryLimit { required_bytes, limit_bytes } => vec![
                ("required_bytes", FieldValue::Int(*required_bytes)),
                ("limit_bytes", FieldValue::Int(*limit_bytes)),
            ],
            Self::MissingKernel { operation, device } => vec![
                ("operation", FieldValue::Str(operation.clone())),
                ("device", FieldValue::Str(device.clone())),
            ],
        }
    }

    /// One error per registered code, built from the `example` values in
    /// `errors.toml` (for tests and docs).
    pub fn examples() -> Vec<Self> {
        vec![
            Self::DuplicateIds {
                axis: "genes".to_owned(),
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
            Self::AmbiguousName {
                name: "CD3E".to_owned(),
                candidates: vec!["log1p:CD3E".to_owned(), "counts:CD3E".to_owned()],
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
            Self::StaleColumn {
                column: "n_genes".to_owned(),
                changed_by: "filter_genes".to_owned(),
            },
            Self::MemoryLimit {
                required_bytes: 6000000000,
                limit_bytes: 4000000000,
            },
            Self::MissingKernel {
                operation: "leiden".to_owned(),
                device: "cuda".to_owned(),
            },
        ]
    }
}
