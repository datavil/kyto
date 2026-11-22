"""Provenance (task 8.5): every store a kyto sink writes holds the plan, versions,
seeds, device and timings that made it, and ``Atlas.provenance()`` rebuilds a plan
that writes the same store again (format.md sections 9 and 10.3, R11).

The Done-when: on the synthetic fixture, a seeded analysis (``normalize_total ->
log1p -> pca -> neighbors -> umap -> leiden``, behind a cell filter) is sunk; the plan
``provenance()`` rebuilds from that store is sunk again, on another thread count, and
every file of the two stores has the same bytes, ``provenance.json`` too once its
``run`` object (timestamps and timings) is left out.
"""

import json
from pathlib import Path
from typing import Any

import kyto
import pytest
from fixtures.interop import to_anndata
from fixtures.registry import load_fixture
from kyto import _kyto, cell_column


def _store(tmp_path: Path) -> Path:
    source = tmp_path / "source.h5ad"
    to_anndata(load_fixture("synthetic")).write_h5ad(source)
    store = tmp_path / "store.kyto"
    kyto.convert(source, store)
    return store


def _analysis(store: Path) -> _kyto.Atlas:
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .with_cells(total=kyto.qc.total_counts())
        .filter(cell_column("total") > 0)
        .normalize_total(1e4, into="norm")
        .log1p(layer="norm", into="log1p")
        .pca(layer="log1p", into="X_pca", n_components=10, seed=3)
        .neighbors("X_pca", n_neighbors=8, seed=5)
        .umap(into="X_umap", n_epochs=30, seed=7)
        .leiden(seed=11, into="leiden")
    )


def _files(root: Path) -> dict[str, bytes]:
    """Every file below ``root`` by relative path; ``provenance.json`` without ``run``."""
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if path.name == "provenance.json":
                provenance = json.loads(data)
                del provenance["run"]
                data = json.dumps(provenance).encode()
            found[path.relative_to(root).as_posix()] = data
    return found


def _provenance(store: Path) -> dict[str, Any]:
    return json.loads((store / "provenance.json").read_text())


def test_provenance_rebuilds_a_plan_that_reproduces_the_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    analysis = _analysis(store)
    first = tmp_path / "first.kyto"
    analysis.sink_zarr(first, threads=2)

    rebuilt = _kyto.scan(first).provenance()
    # The same plan: every step, name and seed as built (R6, R7).
    assert rebuilt.to_json() == analysis.to_json()
    second = tmp_path / "second.kyto"
    rebuilt.sink_zarr(second, threads=1)

    files = _files(first)
    assert files.keys() == _files(second).keys()
    differing = [name for name, data in _files(second).items() if files[name] != data]
    assert differing == []
    # Only `run` differs; the target path is not recorded.
    assert str(first) not in (first / "provenance.json").read_text()


def test_provenance_records_seeds_device_and_timings(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = tmp_path / "out.kyto"
    _analysis(store).sink_zarr(target)
    provenance = _provenance(target)
    assert list(provenance) == [
        "kyto_version",
        "dependencies",
        "plan",
        "source",
        "seeds",
        "device",
        "devices",
        "run",
    ]
    # Node IDs from the scan (0): pca 6, neighbors 7, umap 8, leiden 9; the sink is 10.
    assert provenance["seeds"] == {"n6": 3, "n7": 5, "n8": 7, "n9": 11}
    assert provenance["device"] == "cpu"
    # No `on` step: asked for the CPU, and every plan node ran there (task 9.9).
    devices = provenance["devices"]
    assert devices["requested"] == "cpu"
    assert devices["found"].split(", ")[0] == "cpu"
    assert set(devices["nodes"]) == {f"n{node}" for node in range(11)}
    assert all(
        placement == {"device": "cpu", "device_note": None}
        for placement in devices["nodes"].values()
    )
    assert provenance["source"] is None
    assert [node["step"]["method"] for node in provenance["plan"]["nodes"]][
        -1
    ] == "leiden"
    run = provenance["run"]
    assert run["started_at"] <= run["finished_at"]
    # Each node with its own passes or in-memory step, and the sink's stream and writes
    # (normalize_total to a fixed target is a per-cell map in the sink's stream).
    assert set(run["timings"]) == {"n6", "n7", "n8", "n9", "n10"}
    assert all(seconds >= 0 for seconds in run["timings"].values())
    assert _kyto.scan(target).metadata() == provenance


def test_provenance_of_a_plan_without_seeds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = tmp_path / "logged.kyto"
    _kyto.scan(store).log1p(layer="counts", into="log1p").sink_zarr(target)
    provenance = _provenance(target)
    assert provenance["seeds"] == {}
    assert provenance["run"]["timings"].keys() == {"n2"}
    assert _kyto.scan(target).provenance().to_json() == (
        _kyto.scan(store).log1p(layer="counts", into="log1p").to_json()
    )


def test_a_recorded_sink_step_is_dropped(tmp_path: Path) -> None:
    # Stores written before task 8.5 recorded the plan with its sink step.
    store = _store(tmp_path)
    target = tmp_path / "logged.kyto"
    plan = _kyto.scan(store).log1p(layer="counts", into="log1p")
    plan.sink_zarr(target)
    provenance = _provenance(target)
    provenance["plan"]["nodes"].append(
        {"id": 2, "input": 1, "step": {"method": "sink_zarr", "path": "old.kyto"}}
    )
    (target / "provenance.json").write_text(json.dumps(provenance))
    assert _kyto.scan(target).provenance().to_json() == plan.to_json()


def test_an_import_has_no_plan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(kyto.LineageError) as raised:
        _kyto.scan(store).provenance()
    assert raised.value.code == "KY-E304"
    assert "records no plan" in str(raised.value)
    assert str(store) in str(raised.value)


def test_an_unreadable_provenance_is_coded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store / "provenance.json").write_text("{not json")
    with pytest.raises(kyto.LineageError, match="KY-E304.*is not JSON"):
        _kyto.scan(store).provenance()
