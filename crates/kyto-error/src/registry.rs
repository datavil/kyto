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
    pub const CODES: [&'static str; 8] = [
        "KY-E101",
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
