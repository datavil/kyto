"""Task 7.6: the ``_kyto.pyi`` stub matches what the extension exposes at run time.

``scripts/stubs.py --check`` (the gate) proves the stub is what the bindings generate;
this test proves the generated and hand-written parts describe the built extension:
the same public names, for every class the same public members, for every function and
method the same parameters (names, kinds, defaults) as its ``__text_signature__``, and
the same docstring as its ``__doc__`` (R12: one docstring, written once in Rust).
"""

import ast
import inspect
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from kyto import _kyto

_STUB = ast.parse(Path(_kyto.__file__).with_name("_kyto.pyi").read_text("utf-8"))

# Names the stub declares for type checkers only: none exists at run time.
_TYPE_ONLY_BASES = {"TypedDict"}

# A member pyright needs that the run time has under another form: ``__hash__ = None``
# says expressions are unhashable (they define ``__eq__``).
_STUB_ONLY_MEMBERS = {"__hash__"}

# Run-time members a stub leaves out: ``__repr__`` and ``__str__`` return ``str`` as
# ``object``'s do, and ruff's PYI029 (run by the gate) removes them from stubs.
_RUNTIME_ONLY_MEMBERS = {"__repr__", "__str__"}

# Dunders compared: the ones the bindings define. ``object``'s own (``__init__``,
# ``__new__`` ...) are not part of the stub's contract.
_OBJECT_MEMBERS = set(vars(object))


def _stub_classes() -> dict[str, ast.ClassDef]:
    return {node.name: node for node in _STUB.body if isinstance(node, ast.ClassDef)}


def _is_type_only(node: ast.ClassDef) -> bool:
    return any(ast.unparse(base) in _TYPE_ONLY_BASES for base in node.bases)


def _functions(body: list[ast.stmt]) -> dict[str, list[ast.FunctionDef]]:
    """Every ``def`` of a stub body by name, overloads together."""
    found: dict[str, list[ast.FunctionDef]] = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, []).append(node)
    return found


def _members(node: ast.ClassDef) -> set[str]:
    """The public members and dunders of a stub class (``_private`` ones aside)."""
    names = set(_functions(node.body))
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            names.add(statement.target.id)
    return {name for name in names if _is_compared(name)}


def _is_compared(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _runtime_members(cls: type) -> set[str]:
    dunders = _dunders(cls)
    return {
        name
        for name in vars(cls)
        if (not name.startswith("_") or name in dunders) and _is_compared(name)
    }


def _dunders(cls: type) -> set[str]:
    """The dunders the binding defines (slots PyO3 fills), not ``object``'s."""
    return {
        name
        for name, value in vars(cls).items()
        if name.startswith("__")
        and name.endswith("__")
        and (name not in _OBJECT_MEMBERS or getattr(object, name, None) is not value)
        and name not in {"__doc__", "__module__", "__new__", "__hash__"}
    }


def _callables() -> Iterator[tuple[str, object, list[ast.FunctionDef]]]:
    """Each runtime function and method with a stub ``def``, and its stub defs."""
    functions = _functions(_STUB.body)
    for name, defs in functions.items():
        if hasattr(_kyto, name):
            yield name, getattr(_kyto, name), defs
    for class_name, node in _stub_classes().items():
        if _is_type_only(node):
            continue
        cls = getattr(_kyto, class_name)
        for name, defs in _functions(node.body).items():
            if name in _STUB_ONLY_MEMBERS:
                continue
            yield f"{class_name}.{name}", vars(cls)[name], defs


_CALLABLES = sorted(_callables(), key=lambda item: item[0])


def test_public_names_match() -> None:
    runtime = {name for name in dir(_kyto) if not name.startswith("_")}
    stub = {
        name
        for name, node in _stub_classes().items()
        if not _is_type_only(node) and not name.startswith("_")
    } | {name for name in _functions(_STUB.body) if not name.startswith("_")}
    assert stub == runtime


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, node in _stub_classes().items()
        if not _is_type_only(node) and not name.startswith("_")
    ),
)
def test_class_members_match(name: str) -> None:
    cls = getattr(_kyto, name)
    assert (
        _members(_stub_classes()[name]) - _STUB_ONLY_MEMBERS
        == _runtime_members(cls) - _RUNTIME_ONLY_MEMBERS
    )


def _stub_parameters(node: ast.FunctionDef) -> list[tuple[str, str, str | None]]:
    """(name, kind, default) of each parameter of a stub ``def``, ``self`` dropped."""
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    parameters: list[tuple[str, str, str | None]] = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = (
            "POSITIONAL_ONLY"
            if index < len(arguments.posonlyargs)
            else "POSITIONAL_OR_KEYWORD"
        )
        parameters.append(
            (argument.arg, kind, None if default is None else ast.unparse(default))
        )
    if arguments.vararg is not None:
        parameters.append((arguments.vararg.arg, "VAR_POSITIONAL", None))
    for argument, default in zip(
        arguments.kwonlyargs, arguments.kw_defaults, strict=True
    ):
        parameters.append(
            (
                argument.arg,
                "KEYWORD_ONLY",
                None if default is None else ast.unparse(default),
            )
        )
    if arguments.kwarg is not None:
        parameters.append((arguments.kwarg.arg, "VAR_KEYWORD", None))
    if parameters and parameters[0][0] in {"self", "cls"}:
        parameters = parameters[1:]
    return parameters


def _runtime_parameters(function: object) -> list[tuple[str, str, str | None]]:
    signature = inspect.signature(function)  # pyright: ignore[reportArgumentType]
    parameters: list[tuple[str, str, str | None]] = []
    for parameter in signature.parameters.values():
        default = (
            None
            if parameter.default is inspect.Parameter.empty
            else repr(parameter.default)
        )
        parameters.append((parameter.name, parameter.kind.name, default))
    if parameters and parameters[0][0] in {"self", "cls", "$self"}:
        parameters = parameters[1:]
    return parameters


def _is_slot(function: object) -> bool:
    """A dunder PyO3 fills a type slot with (``__add__``, ``__repr__`` ...): Python
    shows it with its own generic docstring and parameter names (``value``)."""
    return isinstance(function, types.WrapperDescriptorType)


def _same_default(stub: str | None, runtime: str | None) -> bool:
    # PyO3 writes ``...`` for a default it cannot show; the stub then has its value.
    if stub is None or runtime is None:
        return stub == runtime
    if runtime == "Ellipsis":
        return True
    stub_value, runtime_value = ast.literal_eval(stub), ast.literal_eval(runtime)
    # Same type too: ``1.0 == 1`` and ``False == 0`` must not pass.
    return type(stub_value) is type(runtime_value) and stub_value == runtime_value


@pytest.mark.parametrize(
    ("name", "function", "defs"), _CALLABLES, ids=[item[0] for item in _CALLABLES]
)
def test_signatures_match(
    name: str, function: object, defs: list[ast.FunctionDef]
) -> None:
    if getattr(function, "__text_signature__", None) is None:
        pytest.skip(f"{name}: the run time shows no signature (a slot or property)")
    runtime = _runtime_parameters(function)
    slot = _is_slot(function)
    for node in defs:
        stub = _stub_parameters(node)
        if slot:  # only the kinds: Python names a slot's parameters itself
            assert [k for _, k, _ in stub] == [k for _, k, _ in runtime], name
            continue
        assert [(n, k) for n, k, _ in stub] == [(n, k) for n, k, _ in runtime], name
        for (_, _, stub_default), (parameter, _, runtime_default) in zip(
            stub, runtime, strict=True
        ):
            # An overload may make an optional parameter required (``explain``'s
            # ``format="json"``); a default it does write must be the run time's.
            if stub_default is None and len(defs) > 1:
                continue
            assert _same_default(stub_default, runtime_default), (name, parameter)


@pytest.mark.parametrize(
    ("name", "function", "defs"), _CALLABLES, ids=[item[0] for item in _CALLABLES]
)
def test_docstrings_match(
    name: str, function: object, defs: list[ast.FunctionDef]
) -> None:
    documented = [doc for node in defs if (doc := ast.get_docstring(node))]
    assert documented, f"{name} has no stub docstring"
    if _is_slot(function):
        pytest.skip(f"{name}: a slot keeps Python's docstring, the stub has kyto's")
    assert documented[0] == inspect.cleandoc(getattr(function, "__doc__", None) or "")


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, node in _stub_classes().items()
        if not _is_type_only(node) and not name.startswith("_")
    ),
)
def test_class_docstrings_match(name: str) -> None:
    runtime = inspect.cleandoc(getattr(_kyto, name).__doc__ or "")
    assert ast.get_docstring(_stub_classes()[name]) == runtime


# ``Py_TPFLAGS_BASETYPE``: the type can be subclassed. PyO3 classes are not, unless
# declared ``subclass``.
_SUBCLASSABLE = 1 << 10


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, node in _stub_classes().items()
        if not _is_type_only(node) and not name.startswith("_")
    ),
)
def test_final_and_properties_match(name: str) -> None:
    """``@final`` where the run time refuses subclasses, ``@property`` exactly on the
    run time's attributes (a getter), so ``atlas.cells()`` cannot type-check."""
    node = _stub_classes()[name]
    cls: type = getattr(_kyto, name)
    decorators = {ast.unparse(decorator) for decorator in node.decorator_list}
    final = bool(decorators & {"final", "typing.final"})
    assert final == (not cls.__flags__ & _SUBCLASSABLE), name
    stub_properties = {
        member
        for member, defs in _functions(node.body).items()
        if any(ast.unparse(d) == "property" for d in defs[0].decorator_list)
    }
    runtime_properties: set[str] = {
        member
        for member, value in vars(cls).items()
        if isinstance(value, types.GetSetDescriptorType) and not member.startswith("_")
    }
    assert stub_properties == runtime_properties, name
