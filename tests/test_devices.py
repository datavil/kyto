"""Tasks 9.1, 9.2, 9.6, 9.9 and 9.10: ``on(device=...)``, the device of every step in
``explain()``, ``log1p`` on a GPU (9.2: within 1e-6 relative of the CPU, noted in
``explain()``), the fallback rules once some steps have GPU kernels (9.6): ``auto``
puts a chain's steps on a GPU or the CPU one by one, says why at each CPU step and
names each GPU step's block moves (``cpu->metal->cpu``: blocks live on the host
between steps); a named GPU without a kernel for one step refuses the whole run with
``KY-E501``, never the CPU; and 9.9: ``provenance.json`` records the request, the
devices found and each step's device; 9.10: a ``with_cells`` or ``filter`` on QC
metrics runs on a GPU (counts: the CPU's bits).

The devices found depend on the machine and on the build (the default wheel has no GPU
feature, so it finds only the CPU); the tests read them from ``explain(format="json")``
and check the rules for whatever was found. Every machine (no GPU, Metal, CUDA) is
covered by the Rust tests with an injected probe (``crates/kyto-plan/tests/it/devices.rs``).
"""

import gzip
import json
from pathlib import Path

import kyto
import kyto.qc
import numpy as np
import pytest
from kyto import _kyto
from polars.testing import assert_frame_equal


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


def _files(store: Path) -> dict[str, bytes]:
    """Every file of ``store`` by relative path, ``provenance.json`` without ``run``."""
    files: dict[str, bytes] = {}
    for path in sorted(store.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if path.name == "provenance.json":
                provenance = json.loads(data)
                del provenance["run"]
                data = json.dumps(provenance).encode()
            files[str(path.relative_to(store))] = data
    return files


def test_provenance_records_each_steps_device(store: Path, tmp_path: Path) -> None:
    plan = _plan(store, "auto")
    written: list[Path] = []
    for threads in (1, 4):
        target = tmp_path / f"auto-{threads}.kyto"
        plan.sink_zarr(target, threads=threads)
        written.append(target)
    provenance = json.loads((written[0] / "provenance.json").read_text())
    devices = provenance["devices"]
    explained = plan.explain(format="json")["optimized"]
    # The request and the devices found, as explain() gives them.
    assert list(devices) == ["requested", "found", "nodes"]
    assert devices["requested"] == "auto"
    assert devices["found"] == explained["device"]["found"]
    # Each plan node that runs: the device and note of the optimized node it ran in.
    # Nodes 0 (scan), 1 (log1p) and 3 (the sink); the `on` step (2) does no work and
    # has no optimized node, so neither explain() nor provenance places it.
    assert list(devices["nodes"]) == ["n0", "n1", "n3"]
    for node in explained["nodes"]:
        for origin in node["origin"]:
            assert devices["nodes"][f"n{origin}"] == {
                "device": node["device"],
                "device_note": node["device_note"],
            }
    # The sink writes on the CPU; `device` is the GPU of a step placed on one.
    assert devices["nodes"]["n3"] == {"device": "cpu", "device_note": None}
    gpus = {placement["device"] for placement in devices["nodes"].values()} - {"cpu"}
    assert provenance["device"] == (gpus.pop() if gpus else "cpu")
    # Byte-identical on any thread count, `run` aside (R11).
    assert _files(written[0]) == _files(written[1])


def _alternating(store: Path, device: str) -> _kyto.Atlas:
    """GPU maps, a filter on a gene's values (no GPU kernel), a GPU map on its output."""
    return (
        kyto.scan(store)
        .normalize_total(1e4, layer="counts", into="norm")
        .log1p(layer="norm", into="logn")
        .filter(kyto.gene_expression("ENSG01", layer="logn") >= 0)
        .normalize_total(1e4, layer="logn", into="renorm")
        .on(device=device)  # pyright: ignore[reportArgumentType]
    )


def test_auto_alternates_devices_and_says_so_at_every_step(store: Path) -> None:
    plan = _alternating(store, "auto")
    explained = plan.explain(format="json")
    found = explained["optimized"]["device"]["found"]
    gpu = found.split(", ")[1] if found != "cpu" else None
    nodes = [(node["name"], node) for node in explained["optimized"]["nodes"]]
    assert [name for name, _ in nodes] == [
        "NORMALIZE_TOTAL",
        "FILTER_CELLS",
        "FUSED_ELEMENTWISE",
        "SCAN",
    ]
    fallback = "auto: no gpu found" if gpu is None else "auto: no gpu kernel"
    for name, node in nodes:
        on_gpu = gpu is not None and name in ("NORMALIZE_TOTAL", "FUSED_ELEMENTWISE")
        if on_gpu:
            expected = (gpu, "within 1e-6 of cpu", f"cpu->{gpu}->cpu")
        elif name == "SCAN":
            expected = ("cpu", None, None)
        else:
            # Never silent: every compute step on the CPU says why.
            expected = ("cpu", fallback, None)
        assert (node["device"], node["device_note"], node["moves"]) == expected, name
    text = plan.explain()
    filter_line = next(
        line
        for line in text.splitlines()
        if " FILTER_CELLS " in line and "[device" in line
    )
    assert f"[device=cpu ({fallback}) passes=1" in filter_line
    if gpu is not None:
        moved = [line for line in text.splitlines() if f"moves=cpu->{gpu}->cpu" in line]
        assert len(moved) == 2
    # It runs, each step reading the previous one's output through the host.
    results = [
        _alternating(store, device).layer("renorm").to_dense(max_bytes=1_000)
        for device in ("cpu", "auto")
    ]
    assert results[0].shape == (2, 3)
    if gpu is None:
        np.testing.assert_array_equal(results[0], results[1])
    else:
        np.testing.assert_allclose(results[1], results[0], rtol=1e-6, atol=0)


@pytest.mark.parametrize("device", ["cuda", "metal", "wgpu"])
def test_a_named_gpu_without_a_kernel_for_a_step_raises_e501(
    store: Path, device: str
) -> None:
    plan = _alternating(store, device)
    nodes = {
        node["name"]: node for node in plan.explain(format="json")["optimized"]["nodes"]
    }
    note = nodes["FILTER_CELLS"]["device_note"]
    with pytest.raises(kyto.DeviceError) as raised:
        plan.layer("renorm").to_dense(max_bytes=1_000)
    if note == "no kernel: KY-E501 when run":
        # The GPU was found: it has the maps' kernels, not the filter's.
        assert nodes["FUSED_ELEMENTWISE"]["device_note"] == "within 1e-6 of cpu"
        assert raised.value.code == "KY-E501"
        assert "'filter_cells'" in str(raised.value)
        assert "device='auto'" in str(raised.value)  # the hint: fall back out loud
    else:
        assert note == "not found: KY-E504 when run"
        assert raised.value.code == "KY-E504"


def _qc(store: Path, device: str) -> _kyto.Atlas:
    """QC columns and a filter on a QC metric: GPU kernels since 9.10."""
    return (
        kyto.scan(store)
        .with_cells(
            total=kyto.qc.total_counts(layer="counts"),
            detected=kyto.qc.genes_detected(layer="counts"),
            percent=kyto.qc.percent_counts_in(
                kyto.gene_column("symbol") == "CD4", layer="counts"
            ),
        )
        .filter(kyto.qc.total_counts(layer="counts") > 0)
        .on(device=device)  # pyright: ignore[reportArgumentType]
    )


def test_qc_steps_run_on_a_gpu_with_the_cpus_bits(store: Path) -> None:
    explained = _qc(store, "auto").explain(format="json")
    found = explained["optimized"]["device"]["found"]
    gpu = found.split(", ")[1] if found != "cpu" else None
    expected = (
        ("cpu", "auto: no gpu found") if gpu is None else (gpu, "within 1e-6 of cpu")
    )
    nodes = {node["name"]: node for node in explained["optimized"]["nodes"]}
    for name in ("WITH_CELLS", "FILTER_CELLS"):
        assert (nodes[name]["device"], nodes[name]["device_note"]) == expected, name
    # On a counts layer a GPU gives the CPU's bits (integer sums).
    cpu, auto = (_qc(store, device).cells.to_polars() for device in ("cpu", "auto"))
    assert_frame_equal(auto, cpu, check_exact=True)
    assert cpu["total"].to_list() == [6.0, 7.0]
