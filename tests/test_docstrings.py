"""R12: every function ``kyto`` exports has a runnable example, and the examples pass.

Collected by ``uv run pytest`` (the local gate), so a new export cannot skip its
doctest. Runs the examples of each function in ``kyto.__all__`` and in the public
namespaces' ``__all__`` (``kyto.qc``, task 7.1) twice: from the runtime docstring and
from the ``.pyi`` stub docstring (``_kyto.pyi`` for the extension), both in a namespace
holding every name these export. Classes (the exception classes and the expression
classes) keep their own doctest tests.
"""

import ast
import doctest
import inspect
from pathlib import Path
from types import BuiltinFunctionType, FunctionType

import kyto
import kyto.qc
import pytest
from kyto import _kyto

_EXPORTS: dict[str, object] = {
    name: getattr(module, name) for module in (kyto, kyto.qc) for name in module.__all__
}

_FUNCTIONS = sorted(
    name
    for name, value in _EXPORTS.items()
    if isinstance(value, BuiltinFunctionType | FunctionType)
)


def _namespace() -> dict[str, object]:
    return dict(_EXPORTS)


def _run(docstring: str, name: str) -> doctest.TestResults:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    test = parser.get_doctest(docstring, _namespace(), name, None, 0)
    runner.run(test)
    return runner.summarize(verbose=False)


def _stub_docstrings() -> dict[str, str]:
    """Top-level function docstrings of every stub in the package."""
    found: dict[str, str] = {}
    for stub in Path(_kyto.__file__).parent.glob("*.pyi"):
        tree = ast.parse(stub.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    found[node.name] = docstring
    return found


def test_every_exported_function_is_covered() -> None:
    # The constructors (3.4), scan and convert (7.1, 2.6) and the QC constructors
    # (kyto.qc, 6.1) at least: guards the filter above.
    assert {
        "cell_column",
        "feature",
        "gene_expression",
        "scan",
        "convert",
        "total_counts",
    } <= set(_FUNCTIONS)


@pytest.mark.parametrize("name", _FUNCTIONS)
def test_runtime_docstring_examples_run(name: str) -> None:
    docstring = inspect.getdoc(_EXPORTS[name])
    assert docstring is not None, f"{name} has no docstring"
    result = _run(docstring, name)
    assert result.attempted > 0, f"{name} has no example"
    assert result.failed == 0


@pytest.mark.parametrize("name", _FUNCTIONS)
def test_stub_docstring_examples_run(name: str) -> None:
    function = _EXPORTS[name]
    if not isinstance(function, BuiltinFunctionType):
        pytest.skip(f"{name} is Python; its source is its stub")
    docstring = _stub_docstrings().get(name)
    assert docstring is not None, f"{name} has no stub docstring"
    result = _run(docstring, name)
    assert result.attempted > 0, f"{name} has no stub example"
    assert result.failed == 0
