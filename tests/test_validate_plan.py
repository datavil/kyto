"""Task 8.4: ``validate_plan`` lists what is wrong with a plan JSON, without raising.

Done when: deliberately broken plans each return the expected diagnostic. The full
table (27 broken plans over every check: syntax, version, shape against the schema,
replay through the plan methods) is in ``crates/kyto-plan/tests/plan_validate.rs``;
here the Python surface: the dict shape, agreement with ``from_json`` (no diagnostics
exactly when it reads the text; the first diagnostic has the code it raises, and past
the shape check is that error), a set of broken plans with their code and path, and the
published schema.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest
from kyto import KytoError, from_json, gene_column, plan_json_schema, validate_plan

# A plan JSON written by hand: 2 cells, 3 genes, a `counts` layer.
_SCAN: dict[str, Any] = {
    "id": 0,
    "input": None,
    "step": {
        "method": "scan",
        "source": "pbmc.kyto",
        "schema": {
            "n_cells": 2,
            "n_genes": 3,
            "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
            "cells": {"columns": [{"name": "cell_id", "dtype": "String"}]},
            "genes": {
                "columns": [
                    {"name": "gene_id", "dtype": "String"},
                    {"name": "highly_variable", "dtype": "Boolean"},
                ]
            },
            "layers": [{"name": "counts", "dtype": "uint32", "nnz": 3}],
            "embeddings": [],
            "graphs": [],
        },
        "gene_ids": ["ENSG01", "ENSG02", "ENSG03"],
        "symbols": None,
    },
}


def _document() -> dict[str, Any]:
    """Nodes: 0 scan, 1 with_default_layer, 2 filter_genes."""
    scan = from_json(
        json.dumps({"format": "kyto-plan", "version": 1, "nodes": [_SCAN]})
    )
    plan = scan.with_default_layer("counts").filter_genes(
        gene_column("highly_variable") & gene_column("gene_id").str.starts_with("ENSG")
    )
    document: dict[str, Any] = json.loads(plan.to_json())
    return document


def _edited(edit: Callable[[dict[str, Any]], None]) -> str:
    document = _document()
    edit(document)
    return json.dumps(document)


def _raised(text: str) -> KytoError:
    with pytest.raises(KytoError) as caught:
        from_json(text)
    return caught.value


def test_a_valid_plan_has_no_diagnostics() -> None:
    text = json.dumps(_document())
    assert validate_plan(text) == []
    assert validate_plan(json.dumps(_document(), indent=2)) == []
    from_json(text)


def test_diagnostics_are_fresh_dicts_of_four_strings() -> None:
    text = '{"format": "kyto-plan", "version": 1, "nodes": []}'
    first = validate_plan(text)
    second = validate_plan(text)
    assert first == second
    assert first is not second
    assert first[0] is not second[0]
    assert list(first[0]) == ["code", "path", "message", "hint"]
    assert all(isinstance(value, str) for value in first[0].values())


def _set(path: list[str | int], value: object) -> Callable[[dict[str, Any]], None]:
    def edit(document: dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _delete(path: list[str | int]) -> Callable[[dict[str, Any]], None]:
    def edit(document: dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return edit


_STEP_2 = ["nodes", 2, "step"]
_SCHEMA = ["nodes", 0, "step", "schema"]

# (name, edit, code, path): the diagnostic each broken plan must return.
_BROKEN: list[tuple[str, Callable[[dict[str, Any]], None], str, str]] = [
    ("version 2", _set(["version"], 2), "KY-E218", "/version"),
    ("version text", _set(["version"], "1"), "KY-E217", "/version"),
    ("format", _set(["format"], "kyto"), "KY-E217", "/format"),
    ("unknown key", _set(["comment"], "mine"), "KY-E217", ""),
    ("no nodes", _set(["nodes"], []), "KY-E217", "/nodes"),
    ("method", _set([*_STEP_2, "method"], "filter_gene"), "KY-E217", "/nodes/2/step/method"),
    ("missing layer key", _delete(["nodes", 1, "step", "layer"]), "KY-E217", "/nodes/1/step"),
    ("layer type", _set(["nodes", 1, "step", "layer"], 3), "KY-E217", "/nodes/1/step/layer"),
    ("op", _set([*_STEP_2, "predicate", 0, "op"], "col"), "KY-E217", "/nodes/2/step/predicate/0/op"),
    ("block size", _set([*_SCHEMA, "tiling", "gene_block_size"], 0), "KY-E217", "/nodes/0/step/schema/tiling/gene_block_size"),
    ("id", _set(["nodes", 2, "id"], 7), "KY-E217", "/nodes/2/id"),
    ("input", _set(["nodes", 2, "input"], 0), "KY-E217", "/nodes/2/input"),
    ("gene count", _set(["nodes", 0, "step", "gene_ids"], ["ENSG01"]), "KY-E217", "/nodes/0/step/gene_ids"),
    ("operand order", _set([*_STEP_2, "predicate", -1, "left"], 9), "KY-E217", "/nodes/2/step/predicate/3"),
    ("missing layer", _set(["nodes", 1, "step", "layer"], "spliced"), "KY-E109", "/nodes/1/step"),
    ("unknown column", _set([*_STEP_2, "predicate", 0, "name"], "highly_variabel"), "KY-E210", "/nodes/2/step"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("edit", "code", "path"),
    [case[1:] for case in _BROKEN],
    ids=[case[0] for case in _BROKEN],
)
def test_broken_plans_return_the_expected_diagnostic(
    edit: Callable[[dict[str, Any]], None], code: str, path: str
) -> None:
    text = _edited(edit)
    diagnostics = validate_plan(text)
    assert any(
        diagnostic["code"] == code and diagnostic["path"] == path
        for diagnostic in diagnostics
    ), diagnostics
    # The first diagnostic has the code `from_json` raises.
    assert diagnostics[0]["code"] == _raised(text).code


def test_past_the_shape_a_diagnostic_is_the_error_from_json_raises() -> None:
    text = _edited(_set(["nodes", 1, "step", "layer"], "spliced"))
    error = _raised(text)
    [diagnostic] = validate_plan(text)
    assert diagnostic == {
        "code": error.code,
        "path": "/nodes/1/step",
        "message": error.message,
        "hint": error.hint,
    }


def test_every_shape_problem_is_listed() -> None:
    def edit(document: dict[str, Any]) -> None:
        document["nodes"][1]["step"]["layer"] = 3
        document["nodes"][2]["step"]["predicate"][0]["op"] = "col"

    paths = [diagnostic["path"] for diagnostic in validate_plan(_edited(edit))]
    assert "/nodes/1/step/layer" in paths
    assert "/nodes/2/step/predicate/0/op" in paths


def test_the_schema_is_published_as_json_text() -> None:
    text = plan_json_schema()
    assert text == plan_json_schema()
    schema = json.loads(text)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["version"] == {
        "const": 1,
        "description": "The plan JSON version. Another version is KY-E218.",
    }
    methods = schema["$defs"]["later_node"]["allOf"][1]["properties"]["step"][
        "properties"
    ]["method"]["enum"]
    assert "filter_genes" in methods
    assert "scan" not in methods
