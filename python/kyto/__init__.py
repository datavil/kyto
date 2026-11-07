"""kyto: lazy, streaming single-cell analysis, written in Rust, used from Python."""

from kyto._convert import convert
from kyto._kyto import __version__, cell_column, gene_column
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
    MemoryLimitError,
    MissingMemberError,
    RemovedFeatureError,
    ResolutionError,
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
    "MemoryLimitError",
    "MissingMemberError",
    "RemovedFeatureError",
    "ResolutionError",
    "StaleColumnError",
    "StoreIOError",
    "StoreInputError",
    "__version__",
    "cell_column",
    "convert",
    "gene_column",
]
