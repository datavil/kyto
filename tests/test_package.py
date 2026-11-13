"""Smoke tests for the installed package: import and version."""

import re
from pathlib import Path
from types import ModuleType

import kyto

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_workspace_cargo_toml() -> None:
    # A regex, not tomllib: tomllib needs Python 3.11 and kyto supports 3.10.
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    assert kyto.__version__ == match.group(1)


def test_version_is_pep440_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", kyto.__version__)


def test_extension_is_compiled() -> None:
    from kyto import _kyto

    assert _kyto.__file__ is not None
    assert Path(_kyto.__file__).suffix in {".so", ".pyd"}


# Task 7.1: the top-level surface, checked in. `scan`, `convert`, `from_json`, the
# expression constructors (3.4, with the embedding constructors 3.4 added), the plan
# JSON checks of 8.4 (`ky.validate_plan`, as that task names it, and its schema) and
# the exception classes (8.1). Change it only on purpose, with the plan entry.
EXPECTED_ALL = [
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

EXPECTED_QC_ALL = [
    "genes_detected",
    "percent_counts_in",
    "total_counts",
    "total_counts_in",
]

# Public namespaces: module attributes of `kyto` (imported by `import kyto`), not exports.
NAMESPACES = {"dtypes", "errors", "qc"}


def test_all_matches_checked_in_list() -> None:
    assert kyto.__all__ == EXPECTED_ALL


def test_qc_namespace_matches_checked_in_list() -> None:
    import kyto.qc

    assert kyto.qc.__all__ == EXPECTED_QC_ALL
    assert not set(EXPECTED_QC_ALL) & set(dir(kyto))


def test_nothing_else_at_top_level() -> None:
    public = {name for name in dir(kyto) if not name.startswith("_")}
    modules = {name for name in public if isinstance(getattr(kyto, name), ModuleType)}
    assert modules == NAMESPACES
    assert public - modules == set(EXPECTED_ALL)
    assert "__version__" not in kyto.__all__


def test_every_exception_class_is_exported() -> None:
    classes = {
        name
        for name in kyto.errors.__all__
        if isinstance(value := getattr(kyto.errors, name), type)
        and issubclass(value, kyto.KytoError)
    }
    assert len(classes) == 20
    assert classes == {name for name in EXPECTED_ALL if name[0].isupper()}
    for name in classes:
        assert getattr(kyto, name) is getattr(kyto.errors, name)


def test_every_export_resolves() -> None:
    for name in EXPECTED_ALL:
        assert getattr(kyto, name) is not None
