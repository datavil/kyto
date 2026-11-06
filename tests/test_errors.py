"""The error registry (task 8.1, R10): generated code, round trip, no unregistered error."""

from __future__ import annotations

import doctest
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import kyto
import kyto.errors
import pytest
from kyto import _kyto

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "error_registry", ROOT / "scripts" / "error_registry.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


registry_script = _load_script()
REGISTRY = registry_script.load()
ENTRIES = {entry.code: entry for entry in REGISTRY.entries}
BASES = {item.name: item.base for item in REGISTRY.classes}
# Built so this file does not itself name an unregistered code.
UNREGISTERED = "KY-E" + "999"


def _example_fields(code: str) -> dict[str, str | int | tuple[str, ...]]:
    return {field.name: field.example for field in ENTRIES[code].fields}


# Generated files and the "Done when" check.


def test_generated_files_match_errors_toml() -> None:
    assert registry_script.stale_files(REGISTRY) == []


def test_no_unregistered_error_in_the_repo() -> None:
    assert [str(item) for item in registry_script.scan(REGISTRY)] == []


def test_check_command_passes() -> None:
    assert registry_script.main(["--check"]) == 0


def test_registry_has_the_seed_codes() -> None:
    seed = {"KY-E101", "KY-E201", "KY-E202", "KY-E203", "KY-E204"}
    assert seed | {"KY-E301", "KY-E401", "KY-E501"} <= set(ENTRIES)


# Round trip: every registered code, Rust -> Python.


@pytest.mark.parametrize("code", sorted(ENTRIES))
def test_round_trip_rust_to_python(code: str) -> None:
    entry = ENTRIES[code]
    expected_class = getattr(kyto.errors, entry.class_name)
    expected_base = getattr(__import__("builtins"), BASES[entry.class_name])
    with pytest.raises(kyto.KytoError) as caught:
        _kyto.raise_error_example(code)
    error = caught.value
    assert type(error) is expected_class
    assert isinstance(error, expected_base)
    assert error.code == code
    assert dict(error.fields) == _example_fields(code)
    rust = _kyto.error_example_text(code)
    assert rust is not None
    rust_message, rust_hint = rust
    lines = str(error).splitlines()
    assert lines == [
        rust_message,
        f"hint: {rust_hint}",
        f"docs: {REGISTRY.docs_base}#{code.lower()}",
    ]
    assert rust_message == f"{code}: {error.message}"
    assert error.hint == rust_hint
    for value in _example_fields(code).values():
        items = value if isinstance(value, tuple) else (value,)
        for item in items:
            assert str(item) in str(error)


def test_unknown_code_hooks_raise_nothing() -> None:
    assert _kyto.raise_error_example(UNREGISTERED) is False
    assert _kyto.error_example_text(UNREGISTERED) is None


# Python classes.


def test_classes_are_exported_and_subclass_the_builtin() -> None:
    for item in REGISTRY.classes:
        cls = getattr(kyto, item.name)
        assert cls is getattr(kyto.errors, item.name)
        assert issubclass(cls, kyto.KytoError)
        assert issubclass(cls, getattr(__import__("builtins"), item.base))
        assert cls.__doc__ and item.doc in cls.__doc__
    assert {item.name for item in REGISTRY.classes} | {"KytoError"} <= set(kyto.__all__)


def test_ambiguous_name_message() -> None:
    error = kyto.ResolutionError(
        "KY-E201", name="CD3", candidates=["log1p:CD3", "meta:CD3"]
    )
    assert isinstance(error, ValueError)
    assert error.message == "ambiguous name 'CD3': it matches 'log1p:CD3', 'meta:CD3'"
    assert "pass the full name" in error.hint
    assert error.fields["candidates"] == ("log1p:CD3", "meta:CD3")


def test_memory_limit_error_is_a_memory_error() -> None:
    with pytest.raises(MemoryError, match="KY-E401: the plan needs at least 9 bytes"):
        raise kyto.MemoryLimitError("KY-E401", required_bytes=9, limit_bytes=4)


def test_wrong_class_unknown_code_or_bad_fields_are_type_errors() -> None:
    fields = _example_fields("KY-E201")
    with pytest.raises(TypeError, match="not a registered AxisError code"):
        kyto.AxisError("KY-E201", **fields)
    with pytest.raises(TypeError, match="not a registered KytoError code"):
        kyto.KytoError("KY-E201", **fields)
    with pytest.raises(TypeError, match="not a registered"):
        kyto.ResolutionError(UNREGISTERED)
    with pytest.raises(TypeError, match="expected fields"):
        kyto.ResolutionError("KY-E201", name="CD3")
    with pytest.raises(TypeError, match="expected fields"):
        kyto.ResolutionError("KY-E201", **fields, extra="x")
    with pytest.raises(TypeError, match="'candidates' must be list"):
        kyto.ResolutionError("KY-E201", name="CD3", candidates="log1p:CD3")
    with pytest.raises(TypeError, match="'required_bytes' must be int"):
        kyto.MemoryLimitError("KY-E401", required_bytes=-1, limit_bytes=4)
    with pytest.raises(TypeError, match="'required_bytes' must be int"):
        kyto.MemoryLimitError("KY-E401", required_bytes=True, limit_bytes=4)


def test_user_subclass_keeps_its_codes() -> None:
    class MyResolutionError(kyto.ResolutionError):
        pass

    error = MyResolutionError("KY-E201", **_example_fields("KY-E201"))
    assert error.code == "KY-E201"


def test_fields_are_read_only() -> None:
    error = kyto.StaleColumnError("KY-E301", column="n_genes", changed_by="filter")
    with pytest.raises(TypeError):
        error.fields["column"] = "x"  # type: ignore[index]


def test_errors_pickle() -> None:
    error = kyto.ResolutionError("KY-E204", **_example_fields("KY-E204"))
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is kyto.ResolutionError
    assert (copy.code, str(copy), dict(copy.fields)) == (
        error.code,
        str(error),
        dict(error.fields),
    )


def test_docstring_examples_run() -> None:
    result = doctest.testmod(kyto.errors)
    assert result.attempted > 0
    assert result.failed == 0


# The registry validator rejects broken entries.

# Codes are written as @CODE@ so this file names no unregistered code itself.
VALID_TEMPLATE = """
docs_base = "https://example.org/errors.md"
[classes.ThingError]
base = "ValueError"
doc = "A thing."
[[errors]]
code = "@CODE@"
variant = "Thing"
class = "ThingError"
summary = "A thing went wrong."
message = "thing '{name}' failed"
hint = "fix {items}"
fields = [
    { name = "name", type = "str", doc = "The thing.", example = "a" },
    { name = "items", type = "list[str]", doc = "The items.", example = ["b"] },
]
"""
VALID = VALID_TEMPLATE.replace("@CODE@", "KY-E" + "901")


def test_validator_accepts_a_valid_registry() -> None:
    registry = registry_script.parse(VALID)
    assert [entry.code for entry in registry.entries] == ["KY-E" + "901"]


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ('code = "@CODE@"', 'code = "@CODE@0"', "three digits"),
        ('base = "ValueError"', 'base = "OSError"', "base"),
        ("thing '{name}' failed", "thing '{nam}' failed", "unknown fields"),
        ("thing '{name}' failed", "thing {{name}} failed", "braces"),
        ("thing '{name}' failed", 'thing \\"x\\" failed', "double quotes"),
        ('hint = "fix {items}"', 'hint = "fix it"', "appear in no template"),
        ('example = "a"', "example = 1", "non-empty string"),
        ('name = "name"', 'name = "type"', "not reserved"),
        ('class = "ThingError"', 'class = "OtherError"', "not declared"),
        ('variant = "Thing"', 'variant = "thing"', "CamelCase"),
        ('name = "name"', 'name = "source"', "not reserved"),
        ('name = "name"', 'name = "from"', "not reserved"),
    ],
)
def test_validator_rejects(old: str, new: str, reason: str) -> None:
    assert old in VALID_TEMPLATE
    broken = VALID_TEMPLATE.replace(old, new, 1).replace("@CODE@", "KY-E" + "901")
    with pytest.raises(registry_script.RegistryError, match=reason):
        registry_script.parse(broken)


def test_validator_rejects_duplicate_codes() -> None:
    entry = VALID[VALID.index("[[errors]]") :].replace('"Thing"', '"Other"')
    with pytest.raises(registry_script.RegistryError, match="duplicate code"):
        registry_script.parse(VALID + entry)


# The scanner catches unregistered errors (sabotage tests).


@pytest.mark.parametrize(
    "line",
    [
        "#[derive(Debug, thiserror::Error)]",
        "#[derive(Debug, Error)]",
        "impl std::error::Error for Mine {}",
        "impl fmt::Display for X {} impl Error for X {}",
        "fn f() -> Result<(), Box<dyn std::error::Error>> {",
        "fn f() -> anyhow::Result<()> {",
        'return Err(PyValueError::new_err("bad"));',
        'PyErr::new::<PyValueError, _>("bad")',
        "create_exception!(kyto, MyError, PyException);",
        'panic!("bad input");',
        "x.map_err(PyValueError::new_err)?;",
        "#[derive(\n    Debug,\n    Error,\n)]",
        "fn f() -> Box<dyn core::error::Error> {",
        'return Err("bad".into());',
        'return Err(format!("bad {x}"));',
        "fn f() -> Result<(), String> {",
        "#[cfg(test)]\nuse foo;\nfn f() { panic!() }",
    ],
)
def test_rust_scanner_flags(line: str) -> None:
    path = Path("crates/kyto-core/src/lib.rs")
    assert registry_script.scan_rust_text(path, f"fn main() {{}}\n{line}\n")


def test_rust_scanner_ignores_tests_and_conversions() -> None:
    path = Path("x.rs")
    text = (
        '// we avoid anyhow; panic!("never")\n'
        '/* thiserror */ let url = "https://x.org/a";\n'
        "let e = PyErr::from_value(x);\n#[cfg(test)]\nmod tests { panic!(); }\n"
    )
    assert registry_script.scan_rust_text(path, text) == []


@pytest.mark.parametrize(
    "source",
    [
        "raise ValueError('bad')",
        "raise RuntimeError",
        "import builtins\nraise builtins.TypeError('bad')",
        "class Mine(ValueError):\n    pass",
        "class Mine(Exception):\n    pass",
        "class Mine(StopIteration):\n    pass",
    ],
)
def test_python_scanner_flags(source: str) -> None:
    found = registry_script.scan_python_text(
        Path("python/kyto/x.py"), source, REGISTRY.class_names()
    )
    assert found


def test_python_scanner_allows_registered_classes_and_reraise() -> None:
    source = (
        "from kyto.errors import ResolutionError\n"
        "try:\n    pass\nexcept ResolutionError:\n    raise\n"
        "raise ResolutionError('KY-E201', name='a', candidates=['b'])\n"
    )
    assert (
        registry_script.scan_python_text(
            Path("python/kyto/x.py"), source, REGISTRY.class_names()
        )
        == []
    )


def test_code_scanner_flags_unregistered_codes() -> None:
    text = f"KY-E201 is fine, {UNREGISTERED} is not"
    found = registry_script.scan_codes_text(Path("x"), text, REGISTRY.codes())
    assert [item.reason for item in found] == [f"unregistered code {UNREGISTERED}"]


def test_scan_finds_a_sabotaged_tree(tmp_path: Path) -> None:
    rust = tmp_path / "crates" / "kyto-core" / "src" / "lib.rs"
    rust.parent.mkdir(parents=True)
    rust.write_text('fn f() { panic!("x") }\n', encoding="utf-8")
    python = tmp_path / "python" / "kyto" / "api.py"
    python.parent.mkdir(parents=True)
    python.write_text("raise KeyError('x')\n", encoding="utf-8")
    integration = tmp_path / "crates" / "kyto-core" / "tests" / "api.rs"
    integration.parent.mkdir(parents=True)
    integration.write_text('fn t() { panic!("x") }\n', encoding="utf-8")
    found = registry_script.scan(REGISTRY, tmp_path)
    assert len(found) == 2


def test_scan_works_under_a_directory_named_target(tmp_path: Path) -> None:
    root = tmp_path / "target" / "repo"
    rust = root / "crates" / "kyto-core" / "src" / "lib.rs"
    rust.parent.mkdir(parents=True)
    rust.write_text('fn f() { panic!("x") }\n', encoding="utf-8")
    assert len(registry_script.scan(REGISTRY, root)) == 1


def test_entry_without_fields_generates_formatted_python() -> None:
    text = (
        VALID_TEMPLATE.replace("@CODE@", "KY-E" + "901")
        .replace("'{name}'", "x")
        .replace("{items}", "it")
    )
    text = text[: text.index("fields = [")] + "fields = []\n"
    rendered = registry_script.render_python(registry_script.parse(text))
    assert "fields=()," in rendered
    assert "fields=(\n" not in rendered
