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
    "CorruptStoreError",
    "DeviceError",
    "DuplicateIdError",
    "ExpressionError",
    "FieldValue",
    "FormatVersionError",
    "InputFileError",
    "KytoError",
    "LayerDtypeError",
    "LineageError",
    "MemoryLimitError",
    "MissingMemberError",
    "NotImplementedYetError",
    "PlanError",
    "RemovedFeatureError",
    "ResolutionError",
    "StaleColumnError",
    "StoreIOError",
    "StoreInputError",
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
    >>> error = DuplicateIdError("KY-E101", axis="genes", n_duplicated=12, ids=["ENSG01", "ENSG02"])
    >>> error.code
    'KY-E101'
    >>> isinstance(error, ValueError)
    True
    """


class ResolutionError(KytoError, ValueError):
    """A name or symbol does not resolve to exactly one feature.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E201``: A feature name matches more than one feature.
    - ``KY-E204``: A gene symbol matches more than one gene ID.
    - ``KY-E208``: A feature name cannot name any feature.
    - ``KY-E210``: A name does not match any column, gene, cell, embedding component or feature of the plan.
    - ``KY-E211``: A gene or cell is read from the default layer, but the plan has none.
    - ``KY-E214``: An argument that picks how a name is looked up is given a value it does not accept.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = ResolutionError("KY-E201", name="CD3E", candidates=["meta:CD3E", "log1p:CD3E"])
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


class LineageError(KytoError, ValueError):
    """A column is asked to be recomputed but records no lineage to recompute it from.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E302``: A column without lineage is asked to be recomputed.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = LineageError("KY-E302", expression="cell_column('n_genes').recompute()", kind="cell column", column="n_genes")
    >>> error.code
    'KY-E302'
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
    - ``KY-E502``: The CPU executor cannot start the threads it was asked for.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = DeviceError("KY-E501", operation="leiden", device="cuda")
    >>> error.code
    'KY-E501'
    >>> isinstance(error, RuntimeError)
    True
    """


class StoreInputError(KytoError, ValueError):
    """Data, names, parameters or selections given to a store writer or reader break the store format (docs/format.md).

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E102``: A layer, embedding or graph name breaks the naming rules of format.md section 2.1.
    - ``KY-E103``: A member with the same name, compared case-insensitively, is already in the store.
    - ``KY-E104``: A store writer parameter (tile or chunk size, compression level) is out of range.
    - ``KY-E105``: A size is above a limit of the store format (format.md section 4.5).
    - ``KY-E106``: A sparse matrix given to a store writer is not valid CSR of the stated shape.
    - ``KY-E111``: The ID column of a cell or gene table holds null or empty IDs.
    - ``KY-E112``: A column of a cell or gene table breaks format.md section 5 (name, dtype or position).
    - ``KY-E114``: A cell or gene selection given to the tile reader is out of range, unsorted or repeats a position.
    - ``KY-E117``: A CellBlock is not valid CSR, or does not fit the store or layer it is written to (format.md sections 4.8 and 10.2).
    - ``KY-E118``: A store writer's target path already exists (format.md section 10.1).
    - ``KY-E119``: A cell or gene table given to a store writer does not have one row per matrix row or column (format.md section 5.1).
    - ``KY-E123``: An export's target path already exists (sink_h5ad, task 2.7).
    - ``KY-E124``: A store holds a name or value the export format cannot hold (sink_h5ad, task 2.7).
    - ``KY-E125``: A remote store URL or one of its storage options cannot be used to open an object store (task 2.8).

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = StoreInputError("KY-E102", kind="layer", name="log1p/counts", rule="character '/' is not allowed")
    >>> error.code
    'KY-E102'
    >>> isinstance(error, ValueError)
    True
    """


class CorruptStoreError(KytoError, ValueError):
    """A store on disk breaks the store format (docs/format.md).

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E107``: A stored layer breaks the tiled sparse format (format.md section 4).
    - ``KY-E113``: A stored cell or gene table breaks format.md section 5.
    - ``KY-E116``: A path is not a kyto store: its root zarr.json is missing or has no format_version (format.md section 3).
    - ``KY-E122``: A store's schema.json is missing, is not JSON, or breaks format.md section 8.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = CorruptStoreError("KY-E107", layer="counts", problem="tile (0, 2): indptr ends at 5 but tile_offsets gives 7 entries")
    >>> error.code
    'KY-E107'
    >>> isinstance(error, ValueError)
    True
    """


class FormatVersionError(KytoError, ValueError):
    """A store has a format_version this kyto does not read (format.md section 11).

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E115``: A store has a format_version this kyto does not read (format.md section 11).

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = FormatVersionError("KY-E115", store="pbmc.kyto", found="0.2", supported=["0.1"])
    >>> error.code
    'KY-E115'
    >>> isinstance(error, ValueError)
    True
    """


class StoreIOError(KytoError, RuntimeError):
    """The file system or the Zarr layer failed while reading or writing a store.

    Subclasses `KytoError` and `RuntimeError`. Raised with these codes:

    - ``KY-E110``: The file system or the Zarr layer failed while reading or writing a store.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = StoreIOError("KY-E110", operation="write array", path="pbmc.kyto/layers/counts/data", detail="No space left on device (os error 28)")
    >>> error.code
    'KY-E110'
    >>> isinstance(error, RuntimeError)
    True
    """


class MissingMemberError(KytoError, LookupError):
    """A layer, embedding, graph or modality is not in the store.

    Subclasses `KytoError` and `LookupError`. Raised with these codes:

    - ``KY-E109``: A layer, embedding, graph or modality is not in the store.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = MissingMemberError("KY-E109", kind="layer", name="count", store="pbmc.kyto")
    >>> error.code
    'KY-E109'
    >>> isinstance(error, LookupError)
    True
    """


class LayerDtypeError(KytoError, TypeError):
    """A layer is read with a value dtype other than the one it stores.

    Subclasses `KytoError` and `TypeError`. Raised with these codes:

    - ``KY-E108``: A layer is read with a value dtype other than the one it stores.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = LayerDtypeError("KY-E108", layer="counts", stored="uint32", requested="float32")
    >>> error.code
    'KY-E108'
    >>> isinstance(error, TypeError)
    True
    """


class ExpressionError(KytoError, TypeError):
    """An expression is given an operand, or used in a context, that it does not support.

    Subclasses `KytoError` and `TypeError`. Raised with these codes:

    - ``KY-E205``: An expression operation is given an operand of a dtype or type it does not accept.
    - ``KY-E206``: An expression is used where Python needs True or False.
    - ``KY-E207``: An expression tree is deeper than kyto builds.
    - ``KY-E209``: An embedding component is not a non-negative int.
    - ``KY-E212``: An expression is compared with None, which is not a test for missing values.
    - ``KY-E213``: `.recompute()` is called on an expression that is not a plain column read.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = ExpressionError("KY-E205", operation="str.starts_with", accepted="a string expression", got="a numeric expression", operand="cell_column('n_genes') + 1")
    >>> error.code
    'KY-E205'
    >>> isinstance(error, TypeError)
    True
    """


class InputFileError(KytoError, ValueError):
    """An input file given to convert has a format, encoding or value kyto cannot import.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E120``: An input file given to convert has a format or an encoding kyto does not import.
    - ``KY-E121``: An input file given to convert holds a value or a structure that breaks its own format or cannot be stored.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = InputFileError("KY-E120", path="pbmc.h5ad:/X", problem="encoding-type 'awkward-array' is not supported", accepted="store X as an array, csr_matrix or csc_matrix (anndata >= 0.8 writes these)")
    >>> error.code
    'KY-E120'
    >>> isinstance(error, ValueError)
    True
    """


class PlanError(KytoError, ValueError):
    """A plan method is given an argument outside what it accepts, or cannot extend this plan.

    Subclasses `KytoError` and `ValueError`. Raised with these codes:

    - ``KY-E215``: A plan method is given an argument value outside what it accepts.
    - ``KY-E216``: A plan that ends in a sink is extended with another step.
    - ``KY-E217``: Text given to from_json is not a plan JSON, or breaks its format.
    - ``KY-E218``: A plan JSON has a version this kyto does not read.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = PlanError("KY-E215", function="pca", parameter="n_components", value="0", problem="a PCA needs at least one component", accepted="an int from 1 to 2000")
    >>> error.code
    'KY-E215'
    >>> isinstance(error, ValueError)
    True
    """


class NotImplementedYetError(KytoError, NotImplementedError):
    """A documented method is not implemented yet in this kyto version; its task in the implementation plan adds it.

    Subclasses `KytoError` and `NotImplementedError`. Raised with these codes:

    - ``KY-E601``: A documented method is called before the task that implements it is done.

    See `KytoError` for the parameters and attributes.

    Examples
    --------
    >>> error = NotImplementedYetError("KY-E601", method="Matrix.to_scipy", task="7.5")
    >>> error.code
    'KY-E601'
    >>> isinstance(error, NotImplementedError)
    True
    """


_REGISTRY: Final[dict[str, _Entry]] = {
    "KY-E101": _Entry(
        class_name="DuplicateIdError",
        message="the {axis} table has {n_duplicated} IDs that occur more than once, first: {ids}",
        hint="IDs on the {axis} axis must be unique (exact byte comparison); drop or rename the duplicates before writing, or import with on_duplicate_ids='suffix' to make them unique",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e101",
        fields=(
            _Field("axis", "str"),
            _Field("n_duplicated", "int"),
            _Field("ids", "list[str]"),
        ),
    ),
    "KY-E102": _Entry(
        class_name="StoreInputError",
        message="invalid {kind} name '{name}': {rule}",
        hint="use 1 to 64 characters from A-Z a-z 0-9 _ . - that do not start with . or __, do not end with . and are not a Windows device name such as CON",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e102",
        fields=(
            _Field("kind", "str"),
            _Field("name", "str"),
            _Field("rule", "str"),
        ),
    ),
    "KY-E103": _Entry(
        class_name="StoreInputError",
        message="{kind} '{name}' already exists in the store as '{existing}'",
        hint="choose another name; names are compared case-insensitively because macOS and Windows file systems are",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e103",
        fields=(
            _Field("kind", "str"),
            _Field("name", "str"),
            _Field("existing", "str"),
        ),
    ),
    "KY-E104": _Entry(
        class_name="StoreInputError",
        message="invalid {parameter} = {value}",
        hint="{parameter} must be {accepted}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e104",
        fields=(
            _Field("parameter", "str"),
            _Field("value", "str"),
            _Field("accepted", "str"),
        ),
    ),
    "KY-E105": _Entry(
        class_name="StoreInputError",
        message="{quantity} is {value}, above the format limit of {limit}",
        hint="{advice} to stay within the limits of format.md section 4.5",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e105",
        fields=(
            _Field("quantity", "str"),
            _Field("value", "int"),
            _Field("limit", "int"),
            _Field("advice", "str"),
        ),
    ),
    "KY-E106": _Entry(
        class_name="StoreInputError",
        message="invalid sparse matrix for layer '{layer}': {problem}",
        hint="pass CSR with indptr of length n_cells + 1 starting at 0 and non-decreasing, and sorted, unique gene indices below n_genes in each row (format.md section 4.3)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e106",
        fields=(
            _Field("layer", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E107": _Entry(
        class_name="CorruptStoreError",
        message="layer '{layer}' is corrupt: {problem}",
        hint="the store breaks docs/format.md section 4; rewrite the layer with kyto, or report a bug if kyto wrote it",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e107",
        fields=(
            _Field("layer", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E108": _Entry(
        class_name="LayerDtypeError",
        message="layer '{layer}' stores {stored} values, not {requested}",
        hint="read layer '{layer}' as {stored}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e108",
        fields=(
            _Field("layer", "str"),
            _Field("stored", "str"),
            _Field("requested", "str"),
        ),
    ),
    "KY-E109": _Entry(
        class_name="MissingMemberError",
        message="no {kind} named '{name}' in the store at '{store}'",
        hint="check the name against the store's schema; names are case-sensitive when read",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e109",
        fields=(
            _Field("kind", "str"),
            _Field("name", "str"),
            _Field("store", "str"),
        ),
    ),
    "KY-E110": _Entry(
        class_name="StoreIOError",
        message="could not {operation} '{path}': {detail}",
        hint="check that the path exists, that you may read and write it, and that the disk is not full",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e110",
        fields=(
            _Field("operation", "str"),
            _Field("path", "str"),
            _Field("detail", "str"),
        ),
    ),
    "KY-E111": _Entry(
        class_name="StoreInputError",
        message="the {axis} table has {n_invalid} {problem} IDs, first at rows {rows}",
        hint="every {axis} ID must be a non-null, non-empty string; fill or drop those rows before writing",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e111",
        fields=(
            _Field("axis", "str"),
            _Field("n_invalid", "int"),
            _Field("problem", "str"),
            _Field("rows", "list[str]"),
        ),
    ),
    "KY-E112": _Entry(
        class_name="StoreInputError",
        message="column '{column}' of the {axis} table {problem}",
        hint="{accepted} (format.md section 5)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e112",
        fields=(
            _Field("axis", "str"),
            _Field("column", "str"),
            _Field("problem", "str"),
            _Field("accepted", "str"),
        ),
    ),
    "KY-E113": _Entry(
        class_name="CorruptStoreError",
        message="table '{path}' is corrupt: {problem}",
        hint="the store breaks docs/format.md section 5; rewrite the table with kyto, or report a bug if kyto wrote it",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e113",
        fields=(
            _Field("path", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E114": _Entry(
        class_name="StoreInputError",
        message="invalid {axis} selection: {problem}",
        hint="select {axis} by sorted, unique table positions below {n_axis}; the plan canonicalises other orders and repeats (format.md section 4.8)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e114",
        fields=(
            _Field("axis", "str"),
            _Field("problem", "str"),
            _Field("n_axis", "int"),
        ),
    ),
    "KY-E115": _Entry(
        class_name="FormatVersionError",
        message="the store at '{store}' has format_version '{found}', but this kyto reads only {supported}",
        hint="open the store with the kyto version that wrote it, or rewrite it with this one",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e115",
        fields=(
            _Field("store", "str"),
            _Field("found", "str"),
            _Field("supported", "list[str]"),
        ),
    ),
    "KY-E116": _Entry(
        class_name="CorruptStoreError",
        message="'{store}' is not a kyto store: {problem}",
        hint="check the path; a kyto store has a root zarr.json with attributes.kyto.format_version, written last, so an interrupted write leaves none",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e116",
        fields=(
            _Field("store", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E117": _Entry(
        class_name="StoreInputError",
        message="invalid CellBlock ({context}): {problem}",
        hint="build CellBlocks with indptr starting at 0, non-decreasing and ending at nnz, and sorted, unique gene positions below n_genes in each row; a store writer also needs one block per layer, in layer order, over all genes of the store, with the layer's dtype and the same rows in every layer",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e117",
        fields=(
            _Field("context", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E118": _Entry(
        class_name="StoreInputError",
        message="cannot write a store to '{path}': {problem}",
        hint="choose another path, remove the existing one first, or write with overwrite=True to replace an existing kyto store (it is replaced only after the new store is complete)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e118",
        fields=(
            _Field("path", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E119": _Entry(
        class_name="StoreInputError",
        message="the {axis} table has {n_rows} rows, but the matrix has {n_axis} {axis}",
        hint="pass one {axis} table row per matrix {axis} entry, in matrix order",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e119",
        fields=(
            _Field("axis", "str"),
            _Field("n_rows", "int"),
            _Field("n_axis", "int"),
        ),
    ),
    "KY-E120": _Entry(
        class_name="InputFileError",
        message="cannot import '{path}': {problem}",
        hint="{accepted} (see the kyto.convert docstring for the inputs it reads)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e120",
        fields=(
            _Field("path", "str"),
            _Field("problem", "str"),
            _Field("accepted", "str"),
        ),
    ),
    "KY-E121": _Entry(
        class_name="InputFileError",
        message="invalid input '{path}' at {location}: {problem}",
        hint="{advice}, then convert again",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e121",
        fields=(
            _Field("path", "str"),
            _Field("location", "str"),
            _Field("problem", "str"),
            _Field("advice", "str"),
        ),
    ),
    "KY-E122": _Entry(
        class_name="CorruptStoreError",
        message="schema.json of the store at '{store}' is invalid: {problem}",
        hint="the store breaks docs/format.md section 8; write the store again with kyto, or report a bug if kyto wrote it",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e122",
        fields=(
            _Field("store", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E123": _Entry(
        class_name="StoreInputError",
        message="cannot write '{path}': {problem}",
        hint="choose another path, remove the existing one first, or pass overwrite=True to replace an existing file (it is replaced only after the new file is complete; a directory is never replaced)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e123",
        fields=(
            _Field("path", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E124": _Entry(
        class_name="StoreInputError",
        message="cannot export {location} to '{path}': {problem}",
        hint="{advice}, then export again",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e124",
        fields=(
            _Field("location", "str"),
            _Field("path", "str"),
            _Field("problem", "str"),
            _Field("advice", "str"),
        ),
    ),
    "KY-E125": _Entry(
        class_name="StoreInputError",
        message="cannot open remote store '{url}': {problem}",
        hint="{advice}, then open the store again",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e125",
        fields=(
            _Field("url", "str"),
            _Field("problem", "str"),
            _Field("advice", "str"),
        ),
    ),
    "KY-E201": _Entry(
        class_name="ResolutionError",
        message="ambiguous name '{name}': it matches {candidates}",
        hint="pass one full name instead, one of {candidates}; where two share a full name, use cell_column(...), gene_expression(..., layer=...) or cell_embedding(...)",
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
    "KY-E205": _Entry(
        class_name="ExpressionError",
        message="'{operation}' needs {accepted}, got {got}: {operand}",
        hint="pass {accepted} to '{operation}', or use an operation that accepts {got}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e205",
        fields=(
            _Field("operation", "str"),
            _Field("accepted", "str"),
            _Field("got", "str"),
            _Field("operand", "str"),
        ),
    ),
    "KY-E206": _Entry(
        class_name="ExpressionError",
        message="the truth value of an expression is ambiguous: {expression}",
        hint="combine conditions with &, | and ~ instead of and, or and not, and write a < b < c as (a < b) & (b < c); an expression is evaluated only when the plan runs",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e206",
        fields=(_Field("expression", "str"),),
    ),
    "KY-E207": _Entry(
        class_name="ExpressionError",
        message="'{operation}' would make an expression {depth} levels deep; the limit is {limit}",
        hint="build long chains as a balanced tree, combining pairs and then pairs of pairs, instead of adding one term at a time",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e207",
        fields=(
            _Field("operation", "str"),
            _Field("depth", "int"),
            _Field("limit", "int"),
        ),
    ),
    "KY-E208": _Entry(
        class_name="ResolutionError",
        message="invalid feature name '{name}' in feature(...): {problem}",
        hint="write a short name such as 'CD3E' or 'cluster', or a full name '<prefix>:<name>' such as 'log1p:CD3E' or 'meta:cluster'",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e208",
        fields=(
            _Field("name", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E209": _Entry(
        class_name="ExpressionError",
        message="'{constructor}' needs a non-negative int component (0-based), got {got}: {component}",
        hint="pass the component as an int from 0, such as {constructor}('{name}', 0) for the first one",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e209",
        fields=(
            _Field("constructor", "str"),
            _Field("name", "str"),
            _Field("got", "str"),
            _Field("component", "str"),
        ),
    ),
    "KY-E210": _Entry(
        class_name="ResolutionError",
        message="unknown {kind} '{name}' in {expression}: {problem}",
        hint="names are matched exactly, case included; check '{name}' against the store's schema",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e210",
        fields=(
            _Field("kind", "str"),
            _Field("name", "str"),
            _Field("expression", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E211": _Entry(
        class_name="ResolutionError",
        message="{expression} reads the default layer, but the plan has none",
        hint="set one with .with_default_layer(...) before resolving, or name the layer; the store has {layers}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e211",
        fields=(
            _Field("expression", "str"),
            _Field("layers", "list[str]"),
        ),
    ),
    "KY-E212": _Entry(
        class_name="ExpressionError",
        message="'{operation}' with None does not test for missing values: {expression}",
        hint="write {suggestion} instead; missing values are null (R9)",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e212",
        fields=(
            _Field("operation", "str"),
            _Field("expression", "str"),
            _Field("suggestion", "str"),
        ),
    ),
    "KY-E213": _Entry(
        class_name="ExpressionError",
        message="'.recompute()' needs a plain column read such as {example}, got {expression}",
        hint="call .recompute() on the column read itself, then build on the result: write {example} first",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e213",
        fields=(
            _Field("expression", "str"),
            _Field("example", "str"),
        ),
    ),
    "KY-E214": _Entry(
        class_name="ResolutionError",
        message="{function}(..., {parameter}={value}) is not accepted",
        hint="{parameter} takes one of {accepted}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e214",
        fields=(
            _Field("function", "str"),
            _Field("parameter", "str"),
            _Field("value", "str"),
            _Field("accepted", "list[str]"),
        ),
    ),
    "KY-E215": _Entry(
        class_name="PlanError",
        message="{function}(..., {parameter}={value}) is invalid: {problem}",
        hint="pass {accepted} as {parameter}",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e215",
        fields=(
            _Field("function", "str"),
            _Field("parameter", "str"),
            _Field("value", "str"),
            _Field("problem", "str"),
            _Field("accepted", "str"),
        ),
    ),
    "KY-E216": _Entry(
        class_name="PlanError",
        message="'{method}' cannot follow '{sink}': a plan ends at its sink",
        hint="add '{method}' before '{sink}', or build a new plan from the store that '{sink}' writes",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e216",
        fields=(
            _Field("method", "str"),
            _Field("sink", "str"),
        ),
    ),
    "KY-E217": _Entry(
        class_name="PlanError",
        message="the plan JSON is invalid at {location}: {problem}",
        hint="pass the text Atlas.to_json() wrote, unchanged; to change a plan, build it with plan methods and write it again",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e217",
        fields=(
            _Field("location", "str"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E218": _Entry(
        class_name="PlanError",
        message="the plan JSON has version {found}, but this kyto reads only version {supported}",
        hint="load it with the kyto version that wrote it, or build the plan again with this one and write it with Atlas.to_json()",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e218",
        fields=(
            _Field("found", "int"),
            _Field("supported", "int"),
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
    "KY-E302": _Entry(
        class_name="LineageError",
        message="{expression} asks to recompute {kind} '{column}', but the store records no lineage for it",
        hint="only derived columns written by kyto record their lineage (schema.json); read '{column}' as stored, or compute it again from the matrix",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e302",
        fields=(
            _Field("expression", "str"),
            _Field("kind", "str"),
            _Field("column", "str"),
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
    "KY-E502": _Entry(
        class_name="DeviceError",
        message="could not start {threads} executor threads: {problem}",
        hint="pass fewer threads, or raise the process's limit on threads",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e502",
        fields=(
            _Field("threads", "int"),
            _Field("problem", "str"),
        ),
    ),
    "KY-E601": _Entry(
        class_name="NotImplementedYetError",
        message="'{method}' is not implemented yet: it arrives with task {task} of the implementation plan",
        hint="inspect the data without reading it through names(), schema() and estimate() until then",
        docs_url="https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e601",
        fields=(
            _Field("method", "str"),
            _Field("task", "str"),
        ),
    ),
}
