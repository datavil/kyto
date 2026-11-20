//! Prints the `pyo3-stub-gen` stub of `kyto._kyto` to stdout (task 7.6).
//!
//! Does: render what the `#[gen_stub_*]` annotations and `submit!` blocks of the
//! bindings registered, as one `.pyi` text, after two typing rules the annotations
//! would repeat everywhere (see `retype`).
//! Does not: write the file, merge the hand-written part or format it; that is
//! `scripts/stubs.py`, which runs this binary (`uv run python scripts/stubs.py`).

use pyo3_stub_gen::generate::{Module, Parameter, Parameters};
use pyo3_stub_gen::{ImportRef, TypeInfo};

/// The module the extension is imported as (`module-name` in `pyproject.toml`).
const MODULE: &str = "kyto._kyto";

/// The run arguments every terminal takes (task 7.2), by name. The bindings read them
/// as `Option<&PyAny>` (checked in `terminal.rs`), which `pyo3-stub-gen` types as
/// `typing.Any | None`; one rule types them all.
const RUN_ARGUMENTS: [(&str, &str); 3] = [
    ("memory_limit", "builtins.str | builtins.int | None"),
    ("threads", "builtins.int | None"),
    (
        "progress",
        "collections.abc.Callable[[builtins.int, builtins.int], builtins.object] | None",
    ),
];

/// How `pyo3-stub-gen` types a `PathBuf` argument (`os.PathLike` without its type
/// argument, which `pyright --strict` rejects), and what kyto's stub says instead.
const PATH: (&str, &str) = (
    "builtins.str | os.PathLike | pathlib.Path",
    "builtins.str | os.PathLike[builtins.str]",
);

fn main() -> pyo3_stub_gen::Result<()> {
    let mut info = _kyto::stub_info()?;
    if let Some(other) = info.modules.keys().find(|name| name.as_str() != MODULE) {
        return Err(std::io::Error::other(format!(
            "an item is annotated for module {other}; give its `#[gen_stub_*]` `module = \"{MODULE}\"`"
        ))
        .into());
    }
    let module = info.modules.get_mut(MODULE).ok_or_else(|| {
        std::io::Error::other(format!("pyo3-stub-gen gathered no module named {MODULE}"))
    })?;
    retype(module);
    print!("{}", module.format_with_config(false));
    Ok(())
}

/// Apply the two rules to every parameter of every function and method.
fn retype(module: &mut Module) {
    for functions in module.function.values_mut() {
        for function in functions {
            retype_parameters(&mut function.parameters);
        }
    }
    for class in module.class.values_mut() {
        for methods in class.methods.values_mut() {
            for method in methods {
                retype_parameters(&mut method.parameters);
            }
        }
    }
}

fn retype_parameters(parameters: &mut Parameters) {
    let all = parameters
        .positional_only
        .iter_mut()
        .chain(&mut parameters.positional_or_keyword)
        .chain(&mut parameters.keyword_only);
    for parameter in all {
        retype_parameter(parameter);
    }
}

fn retype_parameter(parameter: &mut Parameter) {
    let name = parameter.type_info.name.as_str();
    let run_argument = RUN_ARGUMENTS
        .iter()
        .find(|(argument, _)| *argument == parameter.name);
    let typed = match run_argument {
        // Only where the binding left the type open: an explicit type wins.
        Some((_, typed)) if name == "typing.Optional[typing.Any]" => (*typed).to_owned(),
        // A path, or a path inside a container (`Vec<PathBuf>`).
        _ if name.contains(PATH.0) => name.replace(PATH.0, PATH.1),
        _ => return,
    };
    let mut type_info = TypeInfo::unqualified(&typed);
    type_info.import = parameter.type_info.import.clone();
    type_info.import.remove(&ImportRef::from("pathlib"));
    for module in ["builtins", "collections.abc", "os"] {
        if typed.contains(&format!("{module}.")) {
            type_info.import.insert(ImportRef::from(module));
        }
    }
    parameter.type_info = type_info;
}
