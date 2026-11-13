//! Python arguments of the chain methods (task 7.2) turned into the plan's types.
//!
//! Does: map the string options (`flavor`, `metric`, `search`, `method`, `correction`,
//! `device`) to their enums and check int, float and list arguments (before PyO3's
//! own conversion, which raises uncoded `TypeError`/`OverflowError`), each refusal a `KY-E215`
//! naming the method, the argument and what is accepted (R10).
//! Does not: build plan nodes (`kyto_plan::Plan` does, with its own checks).

use kyto_algo::{DifferentialTest, PValueCorrection};
use kyto_error::KytoError;
use kyto_plan::{HvgFlavor, Metric, NeighborSearch, UmapInit};

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

use crate::errors::to_py_err;
use crate::terminal::invalid;

/// Pick the enum value `value` names among `choices`, or `KY-E215`.
fn choose<T: Copy>(
    function: &str,
    parameter: &str,
    value: &str,
    choices: &[(&str, T)],
) -> Result<T, KytoError> {
    choices
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, choice)| *choice)
        .ok_or_else(|| {
            let accepted = choices
                .iter()
                .map(|(name, _)| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ");
            invalid(
                function,
                parameter,
                format!("'{value}'"),
                "it is not one of the choices",
                &format!("one of {accepted}"),
            )
        })
}

/// `flavor` of `highly_variable_genes`.
pub fn hvg_flavor(value: &str) -> Result<HvgFlavor, KytoError> {
    choose(
        "highly_variable_genes",
        "flavor",
        value,
        &[
            ("seurat", HvgFlavor::Seurat),
            ("cell_ranger", HvgFlavor::CellRanger),
            ("seurat_v3", HvgFlavor::SeuratV3),
        ],
    )
}

/// `init` of `umap`.
pub fn umap_init(value: &str) -> Result<UmapInit, KytoError> {
    choose(
        "umap",
        "init",
        value,
        &[
            ("spectral", UmapInit::Spectral),
            ("random", UmapInit::Random),
        ],
    )
}

/// `metric` of `neighbors`.
pub fn metric(value: &str) -> Result<Metric, KytoError> {
    choose(
        "neighbors",
        "metric",
        value,
        &[("euclidean", Metric::Euclidean), ("cosine", Metric::Cosine)],
    )
}

/// `search` of `neighbors`.
pub fn search(value: &str) -> Result<NeighborSearch, KytoError> {
    choose(
        "neighbors",
        "search",
        value,
        &[
            ("exact", NeighborSearch::Exact),
            ("approximate", NeighborSearch::Approximate),
        ],
    )
}

/// `method` of `rank_genes_groups`.
pub fn differential_test(value: &str) -> Result<DifferentialTest, KytoError> {
    choose(
        "rank_genes_groups",
        "method",
        value,
        &[
            ("t-test", DifferentialTest::TTest),
            ("wilcoxon", DifferentialTest::Wilcoxon),
        ],
    )
}

/// `correction` of `rank_genes_groups`.
pub fn correction(value: &str) -> Result<PValueCorrection, KytoError> {
    choose(
        "rank_genes_groups",
        "correction",
        value,
        &[
            ("benjamini-hochberg", PValueCorrection::BenjaminiHochberg),
            ("bonferroni", PValueCorrection::Bonferroni),
        ],
    )
}

/// `KY-E215` for `value`, given as argument `parameter` of `function`.
fn refuse(
    function: &str,
    parameter: &str,
    value: &Bound<'_, PyAny>,
    problem: &str,
    accepted: &str,
) -> PyResult<KytoError> {
    Ok(invalid(
        function,
        parameter,
        value.repr()?.to_string(),
        problem,
        accepted,
    ))
}

/// A count or seed given as a Python int of at least `minimum` (and at most
/// `2^64 - 1`): `KY-E215` for anything else, `bool` included.
pub fn integer(
    py: Python<'_>,
    function: &str,
    parameter: &str,
    value: &Bound<'_, PyAny>,
    minimum: u64,
) -> PyResult<u64> {
    let accepted = format!("an int of {minimum} or more");
    let problem = if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        "it is not an int"
    } else {
        match value.extract::<u64>() {
            Ok(number) if number >= minimum => return Ok(number),
            Ok(_) => "it is too small",
            Err(_) => "it is negative or above 2^64 - 1",
        }
    };
    let error = refuse(function, parameter, value, problem, &accepted)?;
    Err(to_py_err(py, &error))
}

/// [`integer`] for an optional argument.
pub fn optional_integer(
    py: Python<'_>,
    function: &str,
    parameter: &str,
    value: Option<&Bound<'_, PyAny>>,
    minimum: u64,
) -> PyResult<Option<u64>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| integer(py, function, parameter, value, minimum))
        .transpose()
}

/// A number given as a Python int or float (not `bool`), or `None`: `KY-E215` for
/// anything else. Its range is the plan's to check.
pub fn optional_float(
    py: Python<'_>,
    function: &str,
    parameter: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<f64>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    if !value.is_instance_of::<PyBool>()
        && (value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>())
        && let Ok(number) = value.extract::<f64>()
    {
        return Ok(Some(number));
    }
    let error = refuse(
        function,
        parameter,
        value,
        "it is not a number",
        "an int or a float, or None",
    )?;
    Err(to_py_err(py, &error))
}

/// A list of names given as a Python sequence of `str` (a `str` itself is refused, it
/// is one name): `KY-E215` for anything else.
pub fn strings(
    py: Python<'_>,
    function: &str,
    parameter: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let accepted = "a list or tuple of str";
    if !value.is_instance_of::<PyString>()
        && let Ok(items) = value.try_iter()
    {
        let mut names = Vec::new();
        for item in items {
            let item = item?;
            match item.cast::<PyString>() {
                Ok(name) => names.push(name.to_cow()?.into_owned()),
                Err(_) => {
                    let error =
                        refuse(function, parameter, &item, "an item is not a str", accepted)?;
                    return Err(to_py_err(py, &error));
                }
            }
        }
        return Ok(names);
    }
    let problem = if value.is_instance_of::<PyString>() {
        "it is one str, not a list of them"
    } else {
        "it is not a list of str"
    };
    let error = refuse(function, parameter, value, problem, accepted)?;
    Err(to_py_err(py, &error))
}

/// `device` of `Atlas.on`: the CPU runs today; the GPU devices are task 9.1's
/// (`KY-E601`); anything else is `KY-E215`.
pub fn device(value: &str) -> Result<(), KytoError> {
    match value {
        "cpu" => Ok(()),
        "auto" | "cuda" | "wgpu" | "metal" => Err(KytoError::NotImplementedYet {
            method: format!("Atlas.on(device='{value}')"),
            task: "9.1".to_owned(),
        }),
        _ => Err(invalid(
            "on",
            "device",
            format!("'{value}'"),
            "it is not a device kyto knows",
            "'cpu' ('auto', 'cuda', 'wgpu' and 'metal' arrive with task 9.1)",
        )),
    }
}
