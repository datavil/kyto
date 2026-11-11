"""Generate and check kyto's error code from errors.toml (task 8.1, R10).

Does: read and validate `errors.toml`, write the Rust enum
(`crates/kyto-error/src/registry.rs`) and the Python classes
(`python/kyto/errors.py`, `python/kyto/errors.pyi`) from it. With `--check` it writes
nothing and fails if a generated file differs from what it would write, or if any
Rust or Python source raises an error that is not registered (see `scan_rust` and
`scan_python` for the rules).

Does not: run cargo or Python tests; the round trip of every code from Rust to Python
is tested in `tests/test_errors.py`.

Usage (from the repo root)::

    uv run python scripts/error_registry.py          # regenerate
    uv run python scripts/error_registry.py --check  # CI: drift and scan
"""

from __future__ import annotations

import argparse
import ast
import builtins
import keyword
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "errors.toml"
RUST_FILE = ROOT / "crates" / "kyto-error" / "src" / "registry.rs"
PYTHON_FILE = ROOT / "python" / "kyto" / "errors.py"
STUB_FILE = ROOT / "python" / "kyto" / "errors.pyi"

FieldType = Literal["str", "int", "list[str]", "suggestions"]
FIELD_TYPES: tuple[FieldType, ...] = ("str", "int", "list[str]", "suggestions")
# Field types holding a list of strings (`Vec<String>` in Rust, a tuple in Python).
LIST_TYPES: frozenset[FieldType] = frozenset({"list[str]", "suggestions"})
BUILTIN_BASES = (
    "ValueError",
    "TypeError",
    "MemoryError",
    "RuntimeError",
    "LookupError",
    "IndexError",
    "NotImplementedError",
)
# Names the Python base class already uses, and names that are not Rust identifiers.
RESERVED_FIELDS = frozenset(
    {"code", "message", "hint", "docs_url", "fields", "args", "self", "cls"}
    | {"as", "async", "await", "box", "break", "const", "continue", "crate", "dyn"}
    | {"else", "enum", "extern", "false", "fn", "for", "gen", "if", "impl", "in"}
    | {"let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return"}
    | {"static", "struct", "super", "trait", "true", "try", "type", "unsafe", "use"}
    | {"where", "while", "yield", "abstract", "become", "final", "macro", "override"}
    | {"priv", "typeof", "unsized", "virtual", "Self"}
    # thiserror treats `source` and `backtrace` specially; the rest shadow names the
    # generated Rust calls.
    | {"source", "backtrace", "quoted_list", "did_you_mean", "vec", "format", "String"}
    | {"FieldValue"}
    # Python keywords cannot be keyword arguments.
    | set(keyword.kwlist)
    | set(keyword.softkwlist)
)
CODE_PATTERN = re.compile(r"KY-E\d{3}")
ANY_CODE_PATTERN = re.compile(r"KY-E\d+")
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
# Plain printable ASCII without double quotes or backslashes: safe in Rust and Python
# string literals, docstrings and doc comments without escaping.
TEXT_PATTERN = re.compile(r"[ !#-\[\]-~]*")
GENERATED_NOTE = "GENERATED from errors.toml by scripts/error_registry.py. Do not edit."


class RegistryError(Exception):
    """errors.toml is invalid; the message says where and why."""


@dataclass(frozen=True)
class Field:
    """One structured field of an error."""

    name: str
    type: FieldType
    doc: str
    example: str | int | tuple[str, ...]


@dataclass(frozen=True)
class ErrorClass:
    """One Python exception class."""

    name: str
    base: str
    doc: str


@dataclass(frozen=True)
class Entry:
    """One registered error code."""

    code: str
    variant: str
    class_name: str
    summary: str
    message: str
    hint: str
    fields: tuple[Field, ...]

    def docs_url(self, docs_base: str) -> str:
        """Return the docs link of this code."""
        return f"{docs_base}#{self.code.lower()}"


@dataclass(frozen=True)
class Registry:
    """The parsed and validated errors.toml."""

    docs_base: str
    classes: tuple[ErrorClass, ...]
    entries: tuple[Entry, ...]

    def codes(self) -> frozenset[str]:
        """Return every registered code."""
        return frozenset(entry.code for entry in self.entries)

    def class_names(self) -> frozenset[str]:
        """Return every Python class name, `KytoError` included."""
        return frozenset({"KytoError"} | {item.name for item in self.classes})


# ---------------------------------------------------------------------------
# Loading and validation


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{where}: expected a non-empty string")
    if not TEXT_PATTERN.fullmatch(value):
        raise RegistryError(
            f"{where}: only printable ASCII without double quotes or backslashes"
        )
    return value


def _table(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RegistryError(f"{where}: expected a table")
    return cast(dict[str, object], value)


def _keys(table: dict[str, object], expected: set[str], where: str) -> None:
    if set(table) != expected:
        missing = sorted(expected - set(table))
        unknown = sorted(set(table) - expected)
        raise RegistryError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _example(
    value: object, field_type: FieldType, where: str
) -> str | int | tuple[str, ...]:
    if field_type == "str":
        return _text(value, where)
    if field_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise RegistryError(f"{where}: expected an integer")
        if not 0 <= value < 2**64:
            raise RegistryError(f"{where}: must fit in an unsigned 64-bit integer")
        return value
    if not isinstance(value, list):
        raise RegistryError(f"{where}: expected a list of strings")
    items = cast(list[object], value)
    return tuple(_text(item, f"{where}[{index}]") for index, item in enumerate(items))


def _template(value: object, names: set[str], where: str) -> tuple[str, set[str]]:
    text = _text(value, where)
    used = set(PLACEHOLDER.findall(text))
    if "{" in PLACEHOLDER.sub("", text) or "}" in PLACEHOLDER.sub("", text):
        raise RegistryError(f"{where}: braces only as {{field}} placeholders")
    unknown = sorted(used - names)
    if unknown:
        raise RegistryError(f"{where}: unknown fields {unknown}")
    return text, used


def parse(text: str) -> Registry:
    """Parse and validate the text of errors.toml.

    Parameters
    ----------
    text : str
        The TOML text.

    Returns
    -------
    Registry
        The registry, entries in file order.

    Raises
    ------
    RegistryError
        If the TOML is malformed or breaks a rule stated in errors.toml.
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RegistryError(f"errors.toml: {error}") from error
    _keys(document, {"docs_base", "classes", "errors"}, "errors.toml")
    docs_base = _text(document["docs_base"], "docs_base")
    if not docs_base.startswith("https://") or "#" in docs_base:
        raise RegistryError("docs_base: an https URL without a #fragment")

    classes: list[ErrorClass] = []
    for name, raw in _table(document["classes"], "classes").items():
        where = f"classes.{name}"
        if not re.fullmatch(r"[A-Z][A-Za-z0-9]*Error", name) or name == "KytoError":
            raise RegistryError(f"{where}: a CamelCase name ending in Error")
        table = _table(raw, where)
        _keys(table, {"base", "doc"}, where)
        base = _text(table["base"], f"{where}.base")
        if base not in BUILTIN_BASES:
            raise RegistryError(f"{where}.base: one of {list(BUILTIN_BASES)}")
        classes.append(ErrorClass(name, base, _text(table["doc"], f"{where}.doc")))
    class_names = {item.name for item in classes}

    raw_entries = document["errors"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RegistryError("errors: expected a non-empty array of tables")
    entries: list[Entry] = []
    seen_codes: set[str] = set()
    seen_variants: set[str] = set()
    for index, raw in enumerate(cast(list[object], raw_entries)):
        where = f"errors[{index}]"
        table = _table(raw, where)
        keys = {"code", "variant", "class", "summary", "message", "hint", "fields"}
        _keys(table, keys, where)
        code = _text(table["code"], f"{where}.code")
        where = code
        if not CODE_PATTERN.fullmatch(code):
            raise RegistryError(f"{where}: a code is KY-E and three digits")
        if code in seen_codes:
            raise RegistryError(f"{where}: duplicate code")
        seen_codes.add(code)
        variant = _text(table["variant"], f"{where}.variant")
        if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", variant) or variant in seen_variants:
            raise RegistryError(f"{where}.variant: a unique CamelCase name")
        seen_variants.add(variant)
        class_name = _text(table["class"], f"{where}.class")
        if class_name not in class_names:
            raise RegistryError(f"{where}.class: not declared under [classes]")

        raw_fields = table["fields"]
        if not isinstance(raw_fields, list):
            raise RegistryError(f"{where}.fields: expected an array of tables")
        fields: list[Field] = []
        for field_index, raw_field in enumerate(cast(list[object], raw_fields)):
            field_where = f"{where}.fields[{field_index}]"
            field_table = _table(raw_field, field_where)
            _keys(field_table, {"name", "type", "doc", "example"}, field_where)
            name = _text(field_table["name"], f"{field_where}.name")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in RESERVED_FIELDS:
                raise RegistryError(f"{field_where}.name: snake_case, not reserved")
            if name in {item.name for item in fields}:
                raise RegistryError(f"{field_where}.name: duplicate field")
            field_type = field_table["type"]
            if field_type not in FIELD_TYPES:
                raise RegistryError(f"{field_where}.type: one of {list(FIELD_TYPES)}")
            doc = _text(field_table["doc"], f"{field_where}.doc")
            example = _example(
                field_table["example"], field_type, f"{field_where}.example"
            )
            fields.append(Field(name, field_type, doc, example))

        names = {item.name for item in fields}
        message, in_message = _template(table["message"], names, f"{where}.message")
        hint, in_hint = _template(table["hint"], names, f"{where}.hint")
        misplaced = sorted(
            item.name
            for item in fields
            if item.type == "suggestions"
            and (item.name in in_hint or item.name not in in_message)
        )
        if misplaced:
            raise RegistryError(
                f"{where}: suggestions fields {misplaced} belong in the message only"
            )
        unused = sorted(names - in_message - in_hint)
        if unused:
            raise RegistryError(f"{where}: fields {unused} appear in no template")
        summary = _text(table["summary"], f"{where}.summary")
        entries.append(
            Entry(code, variant, class_name, summary, message, hint, tuple(fields))
        )

    unused_classes = sorted(class_names - {entry.class_name for entry in entries})
    if unused_classes:
        raise RegistryError(f"classes {unused_classes} have no error code")
    return Registry(docs_base, tuple(classes), tuple(entries))


def load(path: Path = REGISTRY_FILE) -> Registry:
    """Read and validate errors.toml (see `parse`)."""
    return parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Rust


def _rust_format(template: str, fields: tuple[Field, ...], prefix: str) -> str:
    """Return the format string and arguments for a template, `prefix` before lists.

    `str` and `int` fields are captured inline (`{name}`); `list[str]` fields become
    positional `{}` arguments rendered by `quoted_list`, `suggestions` fields by
    `did_you_mean`.
    """
    types = {item.name: item.type for item in fields}
    arguments: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if types[name] == "list[str]":
            arguments.append(f"quoted_list({prefix}{name})")
            return "{}"
        if types[name] == "suggestions":
            arguments.append(f"did_you_mean({prefix}{name})")
            return "{}"
        return match.group(0)

    text = PLACEHOLDER.sub(replace, template)
    return ", ".join([f'"{text}"', *arguments])


def _rust_literal(value: str | int | tuple[str, ...], field_type: FieldType) -> str:
    if field_type == "int":
        return str(value)
    if field_type == "str":
        return f'"{value}".to_owned()'
    items = cast(tuple[str, ...], value)
    return "vec![" + ", ".join(f'"{item}".to_owned()' for item in items) + "]"


def _rust_pattern(entry: Entry, used: set[str]) -> str:
    names = [item.name for item in entry.fields if item.name in used]
    if not names:
        return f"Self::{entry.variant} {{ .. }}"
    rest = ", .." if len(names) < len(entry.fields) else ""
    return f"Self::{entry.variant} {{ {', '.join(names)}{rest} }}"


def render_rust(registry: Registry) -> str:
    """Return the text of crates/kyto-error/src/registry.rs."""
    out: list[str] = [
        f"//! The registered kyto errors. {GENERATED_NOTE}",
        "",
        "use crate::{FieldValue, did_you_mean, quoted_list};",
        "",
        "/// Every kyto error: one variant per registered `KY-Exxx` code in `errors.toml`.",
        "///",
        "/// `Display` gives `KY-Exxx: <message>`; [`KytoError::hint`] says how to fix it.",
        "#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]",
        "pub enum KytoError {",
    ]
    for entry in registry.entries:
        message = _rust_format(f"{entry.code}: {entry.message}", entry.fields, ".")
        out.append(f"    /// {entry.code}: {entry.summary}")
        out.append(f"    #[error({message})]")
        if entry.fields:
            out.append(f"    {entry.variant} {{")
            for item in entry.fields:
                rust_type = {
                    "str": "String",
                    "int": "u64",
                    "list[str]": "Vec<String>",
                    # Boxed: keeps KytoError within clippy's result_large_err size.
                    "suggestions": "Box<[String]>",
                }[item.type]
                out.append(f"        /// {item.doc}")
                out.append(f"        {item.name}: {rust_type},")
            out.append("    },")
        else:
            out.append(f"    {entry.variant} {{}},")
    out.append("}")
    out.append("")
    count = len(registry.entries)
    out += [
        "impl KytoError {",
        "    /// Every registered code, in `errors.toml` order.",
        f"    pub const CODES: [&'static str; {count}] = [",
        *[f'        "{entry.code}",' for entry in registry.entries],
        "    ];",
        "",
        "    /// The stable `KY-Exxx` code.",
        "    pub fn code(&self) -> &'static str {",
        "        match self {",
        *[
            f'            Self::{entry.variant} {{ .. }} => "{entry.code}",'
            for entry in registry.entries
        ],
        "        }",
        "    }",
        "",
        "    /// The `kyto.errors` class this error converts to in Python.",
        "    pub fn python_class(&self) -> &'static str {",
        "        match self {",
        *[
            f'            Self::{entry.variant} {{ .. }} => "{entry.class_name}",'
            for entry in registry.entries
        ],
        "        }",
        "    }",
        "",
        "    /// The link to this code's documentation.",
        "    pub fn docs_url(&self) -> &'static str {",
        "        match self {",
    ]
    for entry in registry.entries:
        out.append(f"            Self::{entry.variant} {{ .. }} => {{")
        out.append(f'                "{entry.docs_url(registry.docs_base)}"')
        out.append("            }")
    out += [
        "        }",
        "    }",
        "",
        "    /// How to fix the error, with its fields filled in.",
        "    pub fn hint(&self) -> String {",
        "        match self {",
    ]
    for entry in registry.entries:
        used = set(PLACEHOLDER.findall(entry.hint))
        pattern = _rust_pattern(entry, used)
        if used:
            body = f"format!({_rust_format(entry.hint, entry.fields, '')})"
        else:
            body = f'String::from("{entry.hint}")'
        out.append(f"            {pattern} => {{")
        out.append(f"                {body}")
        out.append("            }")
    out += [
        "        }",
        "    }",
        "",
        "    /// The structured fields as `(name, value)` pairs, in `errors.toml` order.",
        "    pub fn fields(&self) -> Vec<(&'static str, FieldValue)> {",
        "        match self {",
    ]
    kinds = {
        "str": "Str",
        "int": "Int",
        "list[str]": "StrList",
        "suggestions": "StrList",
    }
    for entry in registry.entries:
        pattern = _rust_pattern(entry, {item.name for item in entry.fields})
        out.append(f"            {pattern} => vec![")
        for item in entry.fields:
            value = {
                "int": f"*{item.name}",
                "suggestions": f"{item.name}.to_vec()",
            }.get(item.type, f"{item.name}.clone()")
            out.append(
                f'                ("{item.name}", FieldValue::{kinds[item.type]}({value})),'
            )
        out.append("            ],")
    out += [
        "        }",
        "    }",
        "",
        "    /// One error per registered code, built from the `example` values in",
        "    /// `errors.toml` (for tests and docs).",
        "    pub fn examples() -> Vec<Self> {",
        "        vec![",
    ]
    for entry in registry.entries:
        out.append(f"            Self::{entry.variant} {{")
        for item in entry.fields:
            literal = _rust_literal(item.example, item.type)
            if item.type == "suggestions":
                literal += ".into_boxed_slice()"
            out.append(f"                {item.name}: {literal},")
        out.append("            },")
    out += ["        ]", "    }", "}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Python


def _python_value(value: str | int | tuple[str, ...]) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'


def _class_doc(registry: Registry, item: ErrorClass) -> list[str]:
    lines = [
        f'    """{item.doc}',
        "",
        f"    Subclasses `KytoError` and `{item.base}`. Raised with these codes:",
        "",
    ]
    for entry in registry.entries:
        if entry.class_name == item.name:
            lines.append(f"    - ``{entry.code}``: {entry.summary}")
    first = next(entry for entry in registry.entries if entry.class_name == item.name)
    arguments = ", ".join(
        f"{field.name}={_python_value(field.example)}" for field in first.fields
    )
    call = (
        f'{item.name}("{first.code}", {arguments})'
        if arguments
        else (f'{item.name}("{first.code}")')
    )
    lines += [
        "",
        "    See `KytoError` for the parameters and attributes.",
        "",
        "    Examples",
        "    --------",
        f"    >>> error = {call}",
        "    >>> error.code",
        f"    '{first.code}'",
        f"    >>> isinstance(error, {item.base})",
        "    True",
        '    """',
    ]
    return lines


def _python_fields(fields: tuple[Field, ...]) -> list[str]:
    """Render an entry's `fields=` tuple exactly as `ruff format` leaves it.

    Two or more fields keep one per line (ruff's magic trailing comma). A single
    field's trailing comma is required syntax, not magic, so ruff joins it onto one
    line whenever that fits the 88-character line length.
    """
    items = [f'_Field("{item.name}", "{item.type}"),' for item in fields]
    if not items:
        return ["        fields=(),"]
    if len(items) == 1:
        inline = f"        fields=({items[0]}),"
        if len(inline) <= 88:
            return [inline]
    return [
        "        fields=(",
        *[f"            {item}" for item in items],
        "        ),",
    ]


def render_python(registry: Registry) -> str:
    """Return the text of python/kyto/errors.py."""
    names = ["KytoError", *[item.name for item in registry.classes]]
    out = [
        '"""kyto exception classes (task 8.1, R10).',
        "",
        f"{GENERATED_NOTE}",
        "",
        "Every kyto error has a stable ``KY-Exxx`` code, a message, a fix hint and a docs",
        "link. Each class subclasses `KytoError` and the matching builtin, so for example",
        "``except ValueError`` also catches a `ResolutionError`.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Mapping, Sequence",
        "from types import MappingProxyType",
        "from typing import Final, Literal, NamedTuple, TypeAlias, cast",
        "",
        "__all__ = [",
        *[f'    "{name}",' for name in sorted([*names, "FieldValue"])],
        "]",
        "",
        "FieldValue: TypeAlias = str | int | tuple[str, ...]",
        '"""The type of an error field value (`list[str]` fields are stored as tuples)."""',
        "",
        "",
        "class _Field(NamedTuple):",
        "    name: str",
        '    type: Literal["str", "int", "list[str]", "suggestions"]',
        "",
        "",
        "class _Entry(NamedTuple):",
        "    class_name: str",
        "    message: str",
        "    hint: str",
        "    docs_url: str",
        "    fields: tuple[_Field, ...]",
        "",
        "",
        "def _check(code: str, field: _Field, value: object) -> FieldValue:",
        '    """Return `value` as stored, or raise TypeError if it has the wrong type."""',
        '    if field.type == "str" and isinstance(value, str):',
        "        return value",
        '    if field.type == "int" and type(value) is int and 0 <= value < 2**64:',
        "        return value",
        "    if (",
        '        field.type in ("list[str]", "suggestions")',
        "        and isinstance(value, Sequence)",
        "        and not isinstance(value, str)",
        "    ):",
        '        items = tuple(cast("Sequence[object]", value))',
        "        if all(isinstance(item, str) for item in items):",
        "            return tuple(str(item) for item in items)",
        "    raise TypeError(f\"{code}: field '{field.name}' must be {field.type}\")",
        "",
        "",
        "def _text(field: _Field, value: FieldValue) -> str:",
        '    """One field as a template shows it (the same as Rust).',
        "",
        "    Lists render as 'a', 'b'; suggestions as \"; did you mean 'a', 'b'?\", or as",
        "    nothing when there are none.",
        '    """',
        "    if not isinstance(value, tuple):",
        "        return str(value)",
        '    items = ", ".join(f"\'{item}\'" for item in value)',
        '    if field.type != "suggestions":',
        "        return items",
        '    return f"; did you mean {items}?" if items else ""',
        "",
        "",
        "def _render(",
        "    template: str, fields: tuple[_Field, ...], values: Mapping[str, FieldValue]",
        ") -> str:",
        '    """Fill a template with the values of `fields`."""',
        "    text = {field.name: _text(field, values[field.name]) for field in fields}",
        "    return template.format_map(text)",
        "",
        "",
        "class KytoError(Exception):",
        '    """Base class of every kyto error.',
        "",
        "    Catch it to handle any kyto error. Only its subclasses are raised, each with a",
        "    registered ``KY-Exxx`` code.",
        "",
        "    Parameters",
        "    ----------",
        "    code : str",
        "        A registered ``KY-Exxx`` code of this class.",
        "    **fields : str | int | Sequence[str]",
        "        The structured fields of that code, by name.",
        "",
        "    Attributes",
        "    ----------",
        "    code : str",
        "        The stable ``KY-Exxx`` code.",
        "    message : str",
        "        What was wrong and where, without the code.",
        "    hint : str",
        "        How to fix it.",
        "    docs_url : str",
        "        The link to this code's documentation.",
        "    fields : Mapping[str, FieldValue]",
        "        The structured fields, read-only (`list[str]` values as tuples).",
        "",
        "    Raises",
        "    ------",
        "    TypeError",
        "        If the code is not registered for this class, or a field is missing,",
        "        unknown or of the wrong type (a bug in kyto, not in the caller's input).",
        "",
        "    Examples",
        "    --------",
        "    >>> try:",
        '    ...     raise ResolutionError("KY-E201", name="CD3E", candidates=["log1p:CD3E"])',
        "    ... except KytoError as error:",
        '    ...     print(error.code, error.fields["name"])',
        "    KY-E201 CD3E",
        '    """',
        "",
        "    code: str",
        "    message: str",
        "    hint: str",
        "    docs_url: str",
        "    fields: Mapping[str, FieldValue]",
        "",
        "    def __init__(self, code: str, /, **fields: str | int | Sequence[str]) -> None:",
        "        entry = _REGISTRY.get(code)",
        "        if entry is None or entry.class_name != _class_name(type(self)):",
        '            raise TypeError(f"{code!r} is not a registered {type(self).__name__} code")',
        "        names = [field.name for field in entry.fields]",
        "        if sorted(fields) != sorted(names):",
        '            raise TypeError(f"{code}: expected fields {names}, got {sorted(fields)}")',
        "        values = {",
        "            field.name: _check(code, field, fields[field.name])",
        "            for field in entry.fields",
        "        }",
        "        self.code = code",
        "        self.fields = MappingProxyType(values)",
        "        self.message = _render(entry.message, entry.fields, values)",
        "        self.hint = _render(entry.hint, entry.fields, values)",
        "        self.docs_url = entry.docs_url",
        "        super().__init__(",
        '            f"{code}: {self.message}\\nhint: {self.hint}\\ndocs: {self.docs_url}"',
        "        )",
        "",
        "    def __reduce__(self) -> tuple[object, ...]:",
        '        """Pickle by code and fields, so the error survives a process boundary."""',
        "        return (_rebuild, (type(self), self.code, dict(self.fields)))",
        "",
        "",
        "def _class_name(cls: type[KytoError]) -> str:",
        '    """Return the registered class `cls` is or derives from (user subclasses too)."""',
        "    for parent in cls.__mro__:",
        "        if parent.__module__ == __name__ and parent is not KytoError:",
        "            return parent.__name__",
        '    return ""',
        "",
        "",
        "def _rebuild(",
        "    cls: type[KytoError], code: str, fields: dict[str, FieldValue]",
        ") -> KytoError:",
        "    return cls(code, **fields)",
        "",
    ]
    for item in registry.classes:
        out += ["", f"class {item.name}(KytoError, {item.base}):"]
        out += _class_doc(registry, item)
        out.append("")
    out += ["", "_REGISTRY: Final[dict[str, _Entry]] = {"]
    for entry in registry.entries:
        out += [
            f'    "{entry.code}": _Entry(',
            f'        class_name="{entry.class_name}",',
            f'        message="{entry.message}",',
            f'        hint="{entry.hint}",',
            f'        docs_url="{entry.docs_url(registry.docs_base)}",',
            *_python_fields(entry.fields),
            "    ),",
        ]
    out += ["}", ""]
    return "\n".join(out)


def render_stub(registry: Registry) -> str:
    """Return the text of python/kyto/errors.pyi."""
    names = ["KytoError", *[item.name for item in registry.classes]]
    out = [
        f'"""Type stub for `kyto.errors`. {GENERATED_NOTE}"""',
        "",
        "from collections.abc import Mapping, Sequence",
        "from typing import TypeAlias",
        "",
        "__all__ = [",
        *[f'    "{name}",' for name in sorted([*names, "FieldValue"])],
        "]",
        "",
        "FieldValue: TypeAlias = str | int | tuple[str, ...]",
        "",
        "class KytoError(Exception):",
        "    code: str",
        "    message: str",
        "    hint: str",
        "    docs_url: str",
        "    fields: Mapping[str, FieldValue]",
        "    def __init__(self, code: str, /, **fields: str | int | Sequence[str]) -> None: ...",
        "",
    ]
    for item in registry.classes:
        out.append(f"class {item.name}(KytoError, {item.base}): ...")
    out.append("")
    return "\n".join(out)


def render_all(registry: Registry) -> dict[Path, str]:
    """Return every generated file's path and text."""
    return {
        RUST_FILE: render_rust(registry),
        PYTHON_FILE: render_python(registry),
        STUB_FILE: render_stub(registry),
    }


# ---------------------------------------------------------------------------
# Scanning for unregistered errors


@dataclass(frozen=True)
class Violation:
    """One place that raises or names an error outside the registry."""

    path: Path
    line: int
    reason: str

    def __str__(self) -> str:
        try:
            shown = self.path.relative_to(ROOT)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: {self.reason}"


# Rust: only crates/kyto-error defines error types; nothing else builds an error or a
# Python exception by hand (kyto-py converts a KytoError with `PyErr::from_value`).
# Rules match across lines (a derive split by rustfmt); comments are stripped first.
RUST_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bthiserror\b"), "define errors in errors.toml, not with thiserror"),
    (re.compile(r"#\[derive\([^)]*\bError\b"), "define errors in errors.toml"),
    (re.compile(r"\bError\s+for\b"), "define errors in errors.toml"),
    (re.compile(r"\b(anyhow|eyre)\b"), "use KytoError, not anyhow or eyre"),
    (re.compile(r"\bdyn\s+[\w:]*\bError\b"), "return KytoError, not dyn Error"),
    (
        re.compile(r"\bResult<[^;{]*,\s*(String|&(\'\w+\s*)?str)\s*>"),
        "return KytoError",
    ),
    (re.compile(r"\bErr\(\s*(\"|format!)"), "return a KytoError, not a string"),
    (re.compile(r"::new_err\b"), "convert a KytoError, do not build a PyErr"),
    (
        re.compile(r"\bPyErr::(new|from_type)\b"),
        "convert a KytoError, do not build a PyErr",
    ),
    (re.compile(r"\b(create|import)_exception!"), "exceptions live in kyto.errors"),
    (re.compile(r"\b(panic|unimplemented)!\s*\("), "return a KytoError, do not panic"),
)
RUST_ERROR_CRATE = ROOT / "crates" / "kyto-error"


def _rust_sources(root: Path) -> Iterator[Path]:
    """Yield library sources: not `target/`, `tests/`, `benches/` or `examples/`."""
    skipped = {"target", "tests", "benches", "examples"}
    for path in sorted((root / "crates").rglob("*.rs")):
        if not skipped & set(path.relative_to(root).parts):
            yield path


TEST_MODULE = re.compile(r"^#\[cfg\(test\)\]\s*mod\s+\w+", re.MULTILINE)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


def _strip_comments(text: str) -> str:
    """Blank out comments (keeping line breaks), leaving string literals in place."""

    def blank(match: re.Match[str]) -> str:
        if match.group(0).startswith('"'):
            return match.group(0)
        return re.sub(r"[^\n]", " ", match.group(0))

    return re.compile(f"{STRING.pattern}|{COMMENT.pattern}", re.DOTALL).sub(blank, text)


def scan_rust_text(path: Path, text: str) -> list[Violation]:
    """Return the rule violations in one Rust source.

    Comments are ignored. Everything from a `#[cfg(test)] mod name` at the start of a
    line on is the test module (it sits at the end of each file) and is not checked.
    """
    found: list[Violation] = []
    code = _strip_comments(text)
    cut = TEST_MODULE.search(code)
    if cut is not None:
        code = code[: cut.start()]
    for pattern, reason in RUST_RULES:
        for match in pattern.finditer(code):
            number = code.count("\n", 0, match.start()) + 1
            found.append(Violation(path, number, reason))
    return sorted(found, key=lambda item: item.line)


def scan_codes_text(path: Path, text: str, codes: frozenset[str]) -> list[Violation]:
    """Return every `KY-E...` code in a text that is not registered."""
    found: list[Violation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in ANY_CODE_PATTERN.finditer(line):
            if match.group(0) not in codes:
                found.append(
                    Violation(path, number, f"unregistered code {match.group(0)}")
                )
    return found


def _raised_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ast.unparse(target)


def scan_python_text(path: Path, text: str, classes: frozenset[str]) -> list[Violation]:
    """Return the rule violations in one Python source.

    A `raise` must raise a registered class (`raise ResolutionError(...)`) or re-raise
    (`raise` alone). No class outside kyto/errors.py may subclass an exception.
    """
    found: list[Violation] = []
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Raise) and node.exc is not None:
            name = _raised_name(node.exc)
            if name not in classes:
                found.append(
                    Violation(path, node.lineno, f"raises unregistered {name}")
                )
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = ast.unparse(base).rsplit(".", 1)[-1]
                builtin = getattr(builtins, name, None)
                if name.endswith(("Error", "Exception", "Warning")) or (
                    isinstance(builtin, type) and issubclass(builtin, BaseException)
                ):
                    found.append(
                        Violation(path, node.lineno, f"defines exception {node.name}")
                    )
    return found


def scan(registry: Registry, root: Path = ROOT) -> list[Violation]:
    """Scan the repo for errors raised or named outside the registry.

    Parameters
    ----------
    registry : Registry
        The registry to check against.
    root : Path, default=ROOT
        The repo root.

    Returns
    -------
    list[Violation]
        Every violation, in path order; empty when the repo is clean.
    """
    codes = registry.codes()
    found: list[Violation] = []
    for path in _rust_sources(root):
        text = path.read_text(encoding="utf-8")
        found += scan_codes_text(path, text, codes)
        if RUST_ERROR_CRATE not in path.parents:
            found += scan_rust_text(path, text)
    for path in sorted((root / "python").rglob("*.py*")):
        if path.suffix not in {".py", ".pyi"}:
            continue
        text = path.read_text(encoding="utf-8")
        found += scan_codes_text(path, text, codes)
        if path.suffix == ".py" and path != root / "python" / "kyto" / "errors.py":
            found += scan_python_text(path, text, registry.class_names())
    for directory in ("tests", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            found += scan_codes_text(path, path.read_text(encoding="utf-8"), codes)
    return found


# ---------------------------------------------------------------------------
# Command line


def stale_files(registry: Registry) -> list[Path]:
    """Return the generated files whose text differs from errors.toml."""
    stale: list[Path] = []
    for path, text in render_all(registry).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(path)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Regenerate the files, or with `--check` report drift and violations."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail on drift or on an unregistered error",
    )
    arguments = parser.parse_args(argv)
    try:
        registry = load()
    except RegistryError as error:
        print(f"errors.toml is invalid: {error}", file=sys.stderr)
        return 1
    if not arguments.check:
        for path, text in render_all(registry).items():
            path.write_text(text, encoding="utf-8", newline="\n")
        return 0
    status = 0
    for path in stale_files(registry):
        print(
            f"{path.relative_to(ROOT)} is out of date with errors.toml; run "
            "`uv run python scripts/error_registry.py`",
            file=sys.stderr,
        )
        status = 1
    for violation in scan(registry):
        print(f"{violation} (R10, register it in errors.toml)", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
