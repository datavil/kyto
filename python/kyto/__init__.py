"""kyto: lazy, streaming single-cell analysis, written in Rust, used from Python.

The top level holds what task 7.1 lists (``scan``, ``convert``, ``from_json``, the
expression constructors and the exception classes) and the plan JSON checks of task
8.4 (``validate_plan``, ``plan_json_schema``). Everything else lives in a namespace:
``kyto.qc`` (QC metric expressions), ``kyto.errors`` (the exception classes again),
``kyto.dtypes`` (expression dtype markers). ``__version__`` and these namespaces are
module attributes, not in ``__all__``.
"""

from kyto import dtypes as dtypes
from kyto import errors as errors
from kyto import qc as qc
from kyto._convert import convert
from kyto._kyto import (
    __version__ as __version__,
)
from kyto._kyto import (
    cell_column,
    cell_embedding,
    cell_profile,
    feature,
    from_json,
    gene_column,
    gene_embedding,
    gene_expression,
    plan_json_schema,
    scan,
    validate_plan,
)
from kyto.errors import (
    AxisError,
    CorruptStoreError,
    DeviceError,
    DuplicateIdError,
    ExpressionError,
    FormatVersionError,
    InputFileError,
    KytoError,
    LayerDtypeError,
    LineageError,
    MemoryLimitError,
    MissingMemberError,
    NotImplementedYetError,
    PlanError,
    RemovedFeatureError,
    ResolutionError,
    RunCancelledError,
    StaleColumnError,
    StoreInputError,
    StoreIOError,
)

__all__ = [
    "AxisError",
    "CorruptStoreError",
    "DeviceError",
    "DuplicateIdError",
    "ExpressionError",
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
    "RunCancelledError",
    "StaleColumnError",
    "StoreIOError",
    "StoreInputError",
    "cell_column",
    "cell_embedding",
    "cell_profile",
    "convert",
    "feature",
    "from_json",
    "gene_column",
    "gene_embedding",
    "gene_expression",
    "plan_json_schema",
    "scan",
    "validate_plan",
]
