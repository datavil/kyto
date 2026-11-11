//! `kyto.validate_plan` and `kyto.plan_json_schema` (task 8.4).
//!
//! Does: return the diagnostics of `kyto_plan::validate_plan_json` as Python dicts, and
//! the published JSON Schema of the plan JSON as text.
//! Does not: check anything itself (kyto-plan does), or raise for a broken plan.

use kyto_plan::{PLAN_JSON_SCHEMA, validate_plan_json};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Check a plan JSON and list what is wrong with it, without raising.
///
/// The checks are the ones ``from_json`` makes, in its order: JSON syntax, the
/// ``version`` (another version is one ``KY-E218`` and nothing else), the shape
/// against the published JSON Schema (``plan_json_schema()``; every mismatch is
/// listed), then, when the shape is right, each step replayed through its plan method,
/// which stops at the first step that does not build (later steps build on it). No
/// store is opened or read.
///
/// Parameters
/// ----------
/// text : str
///     A plan JSON, as ``Atlas.to_json`` writes it (any formatting).
///
/// Returns
/// -------
/// list[PlanDiagnostic]
///     A new list, empty when ``from_json(text)`` reads the text; otherwise one dict
///     per problem with ``code`` (``"KY-E217"``), ``path`` (a JSON Pointer into the
///     text, such as ``"/nodes/2/step/layer"``; ``""`` for the whole text), ``message``
///     (what is wrong there) and ``hint`` (how to fix it). The first has the code
///     ``from_json`` raises; past the shape check, it is that error's message and hint
///     (a shape problem is worded by the schema check, more precisely than
///     ``from_json`` words it).
///
/// Examples
/// --------
/// >>> diagnostics = validate_plan('{"format": "kyto-plan", "version": 1, "nodes": []}')
/// >>> [(diagnostic["code"], diagnostic["path"]) for diagnostic in diagnostics]
/// [('KY-E217', '/nodes')]
/// >>> validate_plan('{"format": "kyto-plan", "version": 2, "nodes": []}')[0]["code"]
/// 'KY-E218'
/// >>> validate_plan("not json")[0]["message"]
/// 'the plan JSON is invalid at line 1 column 2: expected ident'
#[pyfunction]
#[pyo3(signature = (text, /))]
pub fn validate_plan<'py>(py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let diagnostics = py.detach(|| validate_plan_json(text));
    diagnostics
        .into_iter()
        .map(|diagnostic| {
            let dict = PyDict::new(py);
            dict.set_item("code", diagnostic.code())?;
            dict.set_item("path", &diagnostic.path)?;
            dict.set_item("message", diagnostic.message())?;
            dict.set_item("hint", diagnostic.hint())?;
            Ok(dict)
        })
        .collect()
}

/// The JSON Schema (draft 2020-12) of the plan JSON this kyto writes and reads.
///
/// It describes the shape of the text ``Atlas.to_json`` writes (version 1); a text can
/// match it and still not build (an unknown layer, a gene filter reading a cell column).
/// ``validate_plan`` checks both.
///
/// Returns
/// -------
/// str
///     The schema as JSON text, the same on every call (a new ``str``).
///
/// Examples
/// --------
/// >>> import json
/// >>> schema = json.loads(plan_json_schema())
/// >>> schema["$schema"]
/// 'https://json-schema.org/draft/2020-12/schema'
/// >>> schema["title"]
/// 'kyto plan JSON, version 1'
#[pyfunction]
pub fn plan_json_schema() -> &'static str {
    PLAN_JSON_SCHEMA
}
