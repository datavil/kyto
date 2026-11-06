"""kyto exception classes (task 8.1, R10).

GENERATED from errors.toml by scripts/error_registry.py. Do not edit.

Every kyto error has a stable ``KY-Exxx`` code, a message, a fix hint and a docs
link. Each class subclasses `KytoError` and the matching builtin, so for example
``except ValueError`` also catches a `ResolutionError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, TypeAlias, cast

__all__ = [
    "AxisError",
    "DeviceError",
    "DuplicateIdError",
    "FieldValue",
    "KytoError",
    "MemoryLimitError",
    "RemovedFeatureError",
    "ResolutionError",
    "StaleColumnError",
]

FieldValue: TypeAlias = str | int | tuple[str, ...]
"""The type of an error field value (`list[str]` fields are stored as tuples)."""


class _Field(NamedTuple):
    name: str
    type: Literal["str", "int", "list[str]"]


class _Entry(NamedTuple):
    class_name: str
    message: str
    hint: str
    docs_url: str
    fields: tuple[_Field, ...]


def _check(code: str, field: _Field, value: object) -> FieldValue:
    """Return `value` as stored, or raise TypeError if it has the wrong type."""
    if field.type == "str" and isinstance(value, str):
        return value
    if field.type == "int" and type(value) is int and 0 <= value < 2**64:
        return value
    if (
        field.type == "list[str]"
        and isinstance(value, Sequence)
        and not isinstance(value, str)
    ):
        items = tuple(cast("Sequence[object]", value))
        if all(isinstance(item, str) for item in items):
            return tuple(str(item) for item in items)
    raise TypeError(f"{code}: field '{field.name}' must be {field.type}")


def _render(template: str, fields: Mapping[str, FieldValue]) -> str:
    """Fill a template; lists render as 'a', 'b' (the same as Rust)."""
    text = {
        name: ", ".join(f"'{item}'" for item in value)
        if isinstance(value, tuple)
        else str(value)
        for name, value in fields.items()
    }
    return template.format_map(text)


class KytoError(Exception):
    """Base class of every kyto error.

    Catch it to handle any kyto error. Only its subclasses are raised, each with a
    registered ``KY-Exxx`` code.

    Parameters
    ----------
    code : str
        A registered ``KY-Exxx`` code of this class.
    **fields : str | int | Sequence[str]
        The structured fields of that code, by name.

    Attributes
    ----------
    code : str
        The stable ``KY-Exxx`` code.
    message : str
        What was wrong and where, without the code.
    hint : str
        How to fix it.
    docs_url : str
        The link to this code's documentation.
    fields : Mapping[str, FieldValue]
        The structured fields, read-only (`list[str]` values as tuples).

    Raises
    ------
    TypeError
        If the code is not registered for this class, or a field is missing,
        unknown or of the wrong type (a bug in kyto, not in the caller's input).

    Examples
    --------
    >>> try:
    ...     raise ResolutionError("KY-E201", name="CD3E", candidates=["log1p:CD3E"])
    ... except KytoError as error:
    ...     print(error.code, error.fields["name"])
    KY-E201 CD3E
    """

    code: str
    message: str
    hint: str
    docs_url: str
    fields: Mapping[str, FieldValue]

    def __init__(self, code: str, /, **fields: str | int | Sequence[str]) -> None:
        entry = _REGISTRY.get(code)
        if entry is None or entry.class_name != _class_name(type(self)):
            raise TypeError(f"{code!r} is not a registered {type(self).__name__} code")
        names = [field.name for field in entry.fields]
        if sorted(fields) != sorted(names):
            raise TypeError(f"{code}: expected fields {names}, got {sorted(fields)}")
        values = {
            field.name: _check(code, field, fields[field.name])
            for field in entry.fields
        }
        self.code = code
        self.fields = MappingProxyType(values)
        self.message = _render(entry.message, values)
        self.hint = _render(entry.hint, values)
        self.docs_url = entry.docs_url
        super().__init__(
            f"{code}: {self.message}\nhint: {self.hint}\ndocs: {self.docs_url}"
        )

    def __reduce__(self) -> tuple[object, ...]:
        """Pickle by code and fields, so the error survives a process boundary."""
        return (_rebuild, (type(self), self.code, dict(self.fields)))


def _class_name(cls: type[KytoError]) -> str:
    """Return the registered class `cls` is or derives from (user subclasses too)."""
    for parent in cls.__mro__:
        if parent.__module__ == __name__ and parent is not KytoError:
            return parent.__name__
    return ""


def _rebuild(
    cls: type[KytoError], code: str, fields: dict[str, FieldValue]
) -> KytoError:
    return cls(code, **fields)


class DuplicateIdError(KytoError, ValueError):
    """Cell or gene IDs that must be unique are not.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E101``: IDs written to a store are not unique.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = DuplicateIdError("KY-E101", axis="genes", ids=["ENSG01", "ENSG02"])
    >>> error.code
    'KY-E101'
    >>> isinstance(error, ValueError)
    True
    """


class ResolutionError(KytoError, ValueError):
    """A name or symbol does not resolve to exactly one feature.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E201``: A short name matches more than one feature.
    - ``KY-E204``: A gene symbol matches more than one gene ID.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = ResolutionError("KY-E201", name="CD3E", candidates=["log1p:CD3E", "counts:CD3E"])
    >>> error.code
    'KY-E201'
    >>> isinstance(error, ValueError)
    True
    """


class AxisError(KytoError, TypeError):
    """An expression mixes the cells and genes axes (R5).

    Subclasses `KytoError` and `TypeError`. Raised with these codes:

    - ``KY-E202``: A cell expression and a gene expression are combined.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = AxisError("KY-E202", operation="&", cell_expression="cell_column('n_genes') > 200", gene_expression="gene_column('highly_variable')")
    >>> error.code
    'KY-E202'
    >>> isinstance(error, TypeError)
    True
    """


class RemovedFeatureError(KytoError, ValueError):
    """A feature removed earlier in the plan is read again (R9).

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E203``: A gene removed by a filter is read later in the plan.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = RemovedFeatureError("KY-E203", gene="MALAT1", removed_by="filter_genes")
    >>> error.code
    'KY-E203'
    >>> isinstance(error, ValueError)
    True
    """


class StaleColumnError(KytoError, ValueError):
    """A derived column is read after its inputs changed upstream.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E301``: A derived column is read after an upstream step changed its inputs.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = StaleColumnError("KY-E301", column="n_genes", changed_by="filter_genes")
    >>> error.code
    'KY-E301'
    >>> isinstance(error, ValueError)
    True
    """


class MemoryLimitError(KytoError, MemoryError):
    """A plan cannot run within its memory limit.

    Subclasses `KytoError` and `MemoryError`. Raised with these codes:

    - ``KY-E401``: A plan needs more memory than its limit, even with the smallest blocks.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = MemoryLimitError("KY-E401", required_bytes=6000000000, limit_bytes=4000000000)
    >>> error.code
    'KY-E401'
    >>> isinstance(error, MemoryError)
    True
    """


class DeviceError(KytoError, RuntimeError):
    """The requested device cannot run part of the plan.

    Subclasses `KytoError` and `RuntimeError`. Raised with these codes:

    - ``KY-E501``: The requested device has no kernel for an operation.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = DeviceError("KY-E501", operation="leiden", device="cuda")
    >>> error.code
    'KY-E501'
    >>> isinstance(error, RuntimeError)
    True
    """


_REGISTRY: Final[dict[str, _Entry]] = {
    "KY-E101": _Entry(
        class_name="DuplicateIdError",
        message="duplicate {axis} IDs: {ids}",
        hint="IDs on the {axis} axis must be unique; drop or rename the duplicates before writing",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e101",
        fields=(
            _Field("axis", "str"),
            _Field("ids", "list[str]"),
        ),
    ),
    "KY-E201": _Entry(
        class_name="ResolutionError",
        message="ambiguous name '{name}': it matches {candidates}",
        hint="pass the full name, one of {candidates}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e201",
        fields=(
            _Field("name", "str"),
            _Field("candidates", "list[str]"),
        ),
    ),
    "KY-E202": _Entry(
        class_name="AxisError",
        message="'{operation}' mixes a cell expression ({cell_expression}) with a gene expression ({gene_expression})",
        hint="combine expressions of one axis only: cell_column(...) with cell_column(...), gene_column(...) with gene_column(...)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e202",
        fields=(
            _Field("operation", "str"),
            _Field("cell_expression", "str"),
            _Field("gene_expression", "str"),
        ),
    ),
    "KY-E203": _Entry(
        class_name="RemovedFeatureError",
        message="gene '{gene}' was removed by '{removed_by}' earlier in the plan",
        hint="read '{gene}' before '{removed_by}', or change the filter to keep it",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e203",
        fields=(
            _Field("gene", "str"),
            _Field("removed_by", "str"),
        ),
    ),
    "KY-E204": _Entry(
        class_name="ResolutionError",
        message="symbol '{symbol}' matches more than one gene: {gene_ids}",
        hint="select the gene by ID with by='id', one of {gene_ids}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e204",
        fields=(
            _Field("symbol", "str"),
            _Field("gene_ids", "list[str]"),
        ),
    ),
    "KY-E301": _Entry(
        class_name="StaleColumnError",
        message="column '{column}' was computed before '{changed_by}' changed its inputs",
        hint="call .recompute() on '{column}' after '{changed_by}', or read it before that step",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e301",
        fields=(
            _Field("column", "str"),
            _Field("changed_by", "str"),
        ),
    ),
    "KY-E401": _Entry(
        class_name="MemoryLimitError",
        message="the plan needs at least {required_bytes} bytes but memory_limit is {limit_bytes} bytes",
        hint="raise memory_limit to at least {required_bytes} bytes, or select fewer cells or genes before this step",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e401",
        fields=(
            _Field("required_bytes", "int"),
            _Field("limit_bytes", "int"),
        ),
    ),
    "KY-E501": _Entry(
        class_name="DeviceError",
        message="no {device} kernel for '{operation}'",
        hint="pass device='auto' to fall back to the CPU for this step, or device='cpu'",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e501",
        fields=(
            _Field("operation", "str"),
            _Field("device", "str"),
        ),
    ),
}
