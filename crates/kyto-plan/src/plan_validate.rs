//! Checking a plan JSON without raising (task 8.4): its published JSON Schema and
//! [`validate_plan_json`], which lists what is wrong as diagnostics.
//!
//! Does: publish the JSON Schema of the plan JSON of 4.3 ([`PLAN_JSON_SCHEMA`], draft
//! 2020-12), and check a text in the order [`crate::Plan::from_json`] reads it: JSON syntax,
//! then the header (`version` first: another version is one `KY-E218` and nothing
//! else, its shape being unknown), then the shape against the schema (every mismatch,
//! each `KY-E217` at the JSON Pointer of the value), then, only when the shape is
//! right, a replay of each step through its builder, which stops at the first step
//! that does not build (later steps build on it) with that builder's own error. So a
//! text has no diagnostics exactly when `from_json` reads it, and when it has, the
//! first diagnostic has the code `from_json` raises.
//! Does not: raise, or read a store. The schema checks the shape only; names, layers,
//! expression kinds, IDs and inputs in order, and gene IDs against the scan's schema are
//! the replay's.

use std::sync::LazyLock;

use kyto_error::KytoError;
use serde_json::Value;

use crate::plan_json::{Located, check_header, read_plan_json};

/// The JSON Schema (draft 2020-12) of the plan JSON this kyto writes and reads
/// (version [`crate::PLAN_JSON_VERSION`]).
pub const PLAN_JSON_SCHEMA: &str = include_str!("plan_json.schema.json");

/// The compiled schema, built once.
static VALIDATOR: LazyLock<jsonschema::Validator> = LazyLock::new(|| {
    let schema: Value = serde_json::from_str(PLAN_JSON_SCHEMA)
        .unwrap_or_else(|error| unreachable!("the plan JSON schema is JSON: {error}"));
    jsonschema::validator_for(&schema)
        .unwrap_or_else(|error| unreachable!("the plan JSON schema is a valid schema: {error}"))
});

/// One problem with a plan JSON: a registered error and where it is in the text.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanDiagnostic {
    /// Where: a JSON Pointer (RFC 6901) into the text, such as `/nodes/2/step/layer`;
    /// empty for the whole text (a syntax error names its line and column instead).
    pub path: String,
    /// What: a registered error. Past the shape check, the error `from_json` raises; a
    /// shape problem is `KY-E217` worded by the schema check (`from_json` words it as
    /// serde does), with the same code as `from_json`'s error.
    pub error: KytoError,
}

impl PlanDiagnostic {
    /// The error code (`KY-E217`).
    pub fn code(&self) -> &'static str {
        self.error.code()
    }

    /// The error message without the leading code
    /// (`{code}: {message}` is the error's `Display`), as Python's `KytoError.message`.
    pub fn message(&self) -> String {
        let text = self.error.to_string();
        let prefix = format!("{}: ", self.code());
        match text.strip_prefix(&prefix) {
            Some(message) => message.to_owned(),
            None => text,
        }
    }

    /// How to fix it.
    pub fn hint(&self) -> String {
        self.error.hint()
    }
}

impl From<Located> for PlanDiagnostic {
    fn from(located: Located) -> Self {
        Self {
            path: located.pointer,
            error: *located.error,
        }
    }
}

/// What is wrong with the plan JSON `text`, in the order of the text; empty when
/// [`crate::Plan::from_json`] reads it, except for forms serde reads that `to_json` never
/// writes and the schema refuses: an array in place of an object, and an expression
/// entry's `op` given as a variant number (`{"op": 0}` for `column`). See the module documentation for the order of checks.
pub fn validate_plan_json(text: &str) -> Vec<PlanDiagnostic> {
    // Syntax, then the header, each a single diagnostic that ends the check.
    if let Err(located) = check_header(text) {
        return vec![located.into()];
    }
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        // `check_header` parsed the same text, so this is not reached; were it, the
        // syntax error is the diagnostic.
        Err(error) => return vec![Located::syntax(&error).into()],
    };
    let shape: Vec<PlanDiagnostic> = VALIDATOR
        .iter_errors(&value)
        .map(|error| {
            let path = error.instance_path().as_str().to_owned();
            let location = if path.is_empty() {
                "the top level".to_owned()
            } else {
                path.clone()
            };
            PlanDiagnostic {
                path,
                error: KytoError::InvalidPlanJson {
                    location,
                    problem: error.to_string(),
                },
            }
        })
        .collect();
    if !shape.is_empty() {
        return in_document_order(shape);
    }
    match read_plan_json(text) {
        Ok(_) => Vec::new(),
        Err(located) => vec![refined(located, &value)],
    }
}

/// `diagnostics` sorted by where they are in the text (a stable sort: the schema
/// check's order within one place), so `/nodes/2` comes before `/nodes/10`.
fn in_document_order(mut diagnostics: Vec<PlanDiagnostic>) -> Vec<PlanDiagnostic> {
    diagnostics.sort_by_cached_key(|diagnostic| {
        diagnostic
            .path
            .split('/')
            .skip(1)
            .map(|segment| match segment.parse::<u64>() {
                Ok(index) => (index, String::new()),
                Err(_) => (u64::MAX, segment.to_owned()),
            })
            .collect::<Vec<_>>()
    });
    diagnostics
}

/// `located` as a diagnostic, an invalid argument of a step pointed at that argument
/// when the step has a key of that name.
fn refined(located: Located, value: &Value) -> PlanDiagnostic {
    let mut diagnostic = PlanDiagnostic::from(located);
    if let KytoError::InvalidArgument { parameter, .. } = &diagnostic.error
        && diagnostic.path.ends_with("/step")
    {
        let escaped = parameter.replace('~', "~0").replace('/', "~1");
        let pointer = format!("{}/{escaped}", diagnostic.path);
        if value.pointer(&pointer).is_some() {
            diagnostic.path = pointer;
        }
    }
    diagnostic
}
