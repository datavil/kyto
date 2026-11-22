"""Tasks 9.1 and 9.2: ``on(device=...)``, the device of every step in ``explain()``,
and ``log1p`` on a GPU (9.2: within 1e-6 relative of the CPU, noted in ``explain()``).

The devices found depend on the machine and on the build (the default wheel has no GPU
feature, so it finds only the CPU); the tests read them from ``explain(format="json")``
and check the rules for whatever was found. Every machine (no GPU, Metal, CUDA) is
covered by the Rust tests with an injected probe (``crates/kyto-plan/tests/it/devices.rs``).
"""

import gzip
import json
from pathlib import Path

import kyto
import numpy as np
import pytest
from kyto import _kyto


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 2-cell, 3-gene store with a ``counts`` layer."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    features = b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\nENSG03\tMS4A1\tGene Expression\n"
    (source / "features.tsv.gz").write_bytes(gzip.compress(features))
    target = tmp_path / "pbmc.kyto"
    kyto.convert(source, target, layer="counts")
    return target


def _plan(store: Path, device: str) -> _kyto.Atlas:
    return kyto.scan(store).log1p(layer="counts", into="log").on(device=device)  # pyright: ignore[reportArgumentType]


def test_explain_names_the_device_of_every_step(store: Path) -> None:
    explained = _plan(store, "auto").explain(format="json")
    device = explained["optimized"]["device"]
    assert device["requested"] == "auto"
    found = device["found"]
    assert found.startswith("cpu")
    # log1p has a GPU kernel (9.2): on the first GPU found, noted; else the CPU.
    gpu = found.split(", ")[1] if found != "cpu" else None
    device_name, note = (
        ("cpu", "auto: no gpu found") if gpu is None else (gpu, "within 1e-6 of cpu")
    )
    nodes = {node["name"]: node for node in explained["optimized"]["nodes"]}
    assert set(nodes) == {
        "LOG1P",
        "SCAN",
    }  # `on` runs nothing: not in the optimized plan
    for host in ("SCAN",):
        assert (nodes[host]["device"], nodes[host]["device_note"]) == ("cpu", None)
    log1p_node = nodes["LOG1P"]
    assert (log1p_node["device"], log1p_node["device_note"]) == (device_name, note)
    text = _plan(store, "auto").explain()
    assert f"\n  device auto (found: {found})\n" in text
    log1p = next(
        line for line in text.splitlines() if " LOG1P " in line and "[device" in line
    )
    assert f"[device={device_name} ({note}) passes=1" in log1p
    # A CPU plan prints no device line and no note.
    cpu = _plan(store, "cpu").explain()
    assert "\n  device " not in cpu
    assert "(auto" not in cpu
    assert _plan(store, "auto").estimate()["device"] == "auto"


def test_auto_runs_on_the_cpu_with_the_same_bits(store: Path) -> None:
    results = [
        _plan(store, device).layer("log").to_dense(max_bytes=1_000)
        for device in ("cpu", "auto")
    ]
    assert results[0].dtype == results[1].dtype
    found = _plan(store, "auto").explain(format="json")["optimized"]["device"]["found"]
    if found == "cpu":
        np.testing.assert_array_equal(results[0], results[1])
    else:
        # On a GPU (9.2): within 1e-6 relative, not the same bits.
        np.testing.assert_allclose(results[1], results[0], rtol=1e-6, atol=0)


@pytest.mark.parametrize("device", ["cuda", "metal", "wgpu"])
def test_a_named_gpu_runs_log1p_or_is_refused_first(store: Path, device: str) -> None:
    plan = _plan(store, device)
    explained = plan.explain(format="json")
    log1p = next(
        node for node in explained["optimized"]["nodes"] if node["name"] == "LOG1P"
    )
    if log1p["device_note"] == "within 1e-6 of cpu":
        # The GPU was found: log1p runs there (9.2), within 1e-6 of the CPU.
        on_gpu = plan.layer("log").to_dense(max_bytes=1_000)
        on_cpu = _plan(store, "cpu").layer("log").to_dense(max_bytes=1_000)
        np.testing.assert_allclose(on_gpu, on_cpu, rtol=1e-6, atol=0)
        return
    with pytest.raises(kyto.DeviceError) as raised:
        plan.collect()
    if log1p["device_note"] == "not found: KY-E504 when run":
        assert raised.value.code == "KY-E504"
        assert f"device '{device}'" in str(raised.value)
        assert f"(found: {explained['optimized']['device']['found']})" in str(
            raised.value
        )
    else:
        pytest.fail(f"unexpected note {log1p['device_note']!r}: {raised.value}")


def test_the_device_is_part_of_the_plan(store: Path) -> None:
    plan = _plan(store, "auto")
    assert '"method": "on"' in plan.to_json() or '"method":"on"' in plan.to_json()
    assert kyto.from_json(plan.to_json()).explain() == plan.explain()
    assert str(plan).splitlines()[0] == 'ON_DEVICE "auto"'
    with pytest.raises(kyto.PlanError) as raised:
        kyto.scan(store).on(device="gpu")  # pyright: ignore[reportArgumentType]
    assert raised.value.code == "KY-E215"
    assert "'cpu', 'auto', 'cuda', 'wgpu' or 'metal'" in str(raised.value)


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_provenance_records_the_device_the_run_used(
    store: Path, tmp_path: Path, device: str
) -> None:
    plan = _plan(store, device)
    target = tmp_path / f"{device}.kyto"
    plan.sink_zarr(target, threads=1)
    provenance = json.loads((target / "provenance.json").read_text())
    # The resolved device, not the request: every step ran where explain() said.
    explained = plan.explain(format="json")
    ran = {node["device"] for node in explained["optimized"]["nodes"]}
    gpus = ran - {"cpu"}
    # A GPU only where `auto` put log1p on one (9.2): provenance names it.
    assert len(gpus) <= 1 and (device == "auto" or not gpus)
    assert provenance["device"] == (gpus.pop() if gpus else "cpu")
    # The request is the plan's `on` step, which provenance() rebuilds.
    steps = [node["step"] for node in provenance["plan"]["nodes"]]
    assert {"method": "on", "device": device} in steps
    rebuilt = kyto.scan(target).provenance()
    assert rebuilt.explain(format="json")["optimized"]["device"]["requested"] == device
