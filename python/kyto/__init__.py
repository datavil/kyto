"""kyto: lazy, streaming single-cell analysis, written in Rust, used from Python."""

from kyto._kyto import __version__
from kyto.errors import (
    AxisError,
    DeviceError,
    DuplicateIdError,
    KytoError,
    MemoryLimitError,
    RemovedFeatureError,
    ResolutionError,
    StaleColumnError,
)

__all__ = [
    "AxisError",
    "DeviceError",
    "DuplicateIdError",
    "KytoError",
    "MemoryLimitError",
    "RemovedFeatureError",
    "ResolutionError",
    "StaleColumnError",
    "__version__",
]
