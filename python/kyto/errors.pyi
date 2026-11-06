"""Type stub for `kyto.errors`. GENERATED from errors.toml by scripts/error_registry.py. Do not edit."""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

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

class KytoError(Exception):
    code: str
    message: str
    hint: str
    docs_url: str
    fields: Mapping[str, FieldValue]
    def __init__(self, code: str, /, **fields: str | int | Sequence[str]) -> None: ...

class DuplicateIdError(KytoError, ValueError): ...
class ResolutionError(KytoError, ValueError): ...
class AxisError(KytoError, TypeError): ...
class RemovedFeatureError(KytoError, ValueError): ...
class StaleColumnError(KytoError, ValueError): ...
class MemoryLimitError(KytoError, MemoryError): ...
class DeviceError(KytoError, RuntimeError): ...
