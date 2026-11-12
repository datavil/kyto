"""Task 5.7: the GIL is released while a plan computes; progress; Ctrl-C cancels cleanly.

Done when: a Python test cancels a run and the process stays usable.

From Python the plans the executor runs are scans (and ``with_default_layer``), run by
the private ``Atlas._execute`` (task 5.7's hook until 7.2/7.3's terminals). A store of
400 000 cells x 2 000 genes (20 million entries, written by anndata, converted once per
module) takes about 0.3 s on one thread on an M-series Mac, many times the 20 ms the
calling thread waits between two looks at signals and progress, so a cancel lands
while the run computes. Ctrl-C is simulated in the main thread both ways it can come:
``_thread.interrupt_main`` from another thread while the main thread waits without the
GIL (``check_signals`` raises it), and ``signal.raise_signal`` inside the progress
callback (the callback raises it). After each cancel the same process runs the plan to
the end. The executor side (cancel in a statistics pass, no reads after a cancel) is
``crates/kyto-exec/tests/it/run_control.rs``. Skips where anndata has no wheels.
"""

import _thread
import doctest
import gzip
import json
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from kyto import _kyto, gene_column

# pandas, scipy and anndata ship without complete type information; strict pyright
# sees their members as unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
anndata = pytest.importorskip("anndata")
pandas = pytest.importorskip("pandas")
scipy_sparse = pytest.importorskip("scipy.sparse")

N_CELLS = 400_000
N_GENES = 2_000
PER_CELL = 50


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """400 000 cells with 50 entries each, one gene per 40-gene stripe."""
    work = tmp_path_factory.mktemp("run_control")
    generator = np.random.default_rng(57)
    stripe = N_GENES // PER_CELL
    indices = (
        generator.integers(0, stripe, size=(N_CELLS, PER_CELL))
        + np.arange(PER_CELL) * stripe
    )
    matrix: Any = scipy_sparse.csr_matrix(
        (
            generator.integers(1, 50, size=N_CELLS * PER_CELL).astype(np.float32),
            indices.ravel().astype(np.int32),
            np.arange(0, (N_CELLS + 1) * PER_CELL, PER_CELL, dtype=np.int64),
        ),
        shape=(N_CELLS, N_GENES),
    )
    adata: Any = anndata.AnnData(
        X=matrix,
        obs=pandas.DataFrame(index=[f"c{cell}" for cell in range(N_CELLS)]),
        var=pandas.DataFrame(index=[f"g{gene}" for gene in range(N_GENES)]),
    )
    source = work / "big.h5ad"
    adata.write_h5ad(source)
    target = work / "big.kyto"
    kyto.convert(source, target, layer="counts")
    source.unlink()
    return target


@pytest.fixture(autouse=True)
def default_sigint() -> Iterator[None]:
    """Ctrl-C raises ``KeyboardInterrupt``, whatever handler the runner installed."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_ctrl_c_while_computing_cancels_and_the_process_stays_usable(
    store: Path,
) -> None:
    atlas = _kyto.scan(store)
    calls: list[tuple[int, int]] = []
    started = threading.Event()

    def progress(done: int, total: int) -> None:
        calls.append((done, total))
        started.set()

    def press_ctrl_c() -> None:
        started.wait()
        _thread.interrupt_main(signal.SIGINT)

    presser = threading.Thread(target=press_ctrl_c)
    presser.start()
    began = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        atlas._execute(threads=1, progress=progress)
    cancelled_after = time.perf_counter() - began
    presser.join()
    # Cancelled in the middle: never reported done, and stopped long before the end.
    assert calls, "no progress before the cancel"
    assert all(total == N_CELLS for _, total in calls)
    assert calls[-1][0] < N_CELLS

    # The same process runs the plan again, to the end.
    began = time.perf_counter()
    assert atlas._execute(threads=1) == N_CELLS
    whole_run = time.perf_counter() - began
    assert cancelled_after < whole_run
    assert atlas.with_default_layer("counts")._execute() == N_CELLS


def test_ctrl_c_in_the_progress_callback_cancels(store: Path) -> None:
    atlas = _kyto.scan(store)
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))
        signal.raise_signal(signal.SIGINT)  # raises KeyboardInterrupt here

    with pytest.raises(KeyboardInterrupt):
        atlas._execute(threads=1, progress=progress)
    assert len(calls) == 1
    assert calls[0][0] < N_CELLS
    assert atlas._execute(threads=2) == N_CELLS


def test_an_exception_from_the_callback_cancels_and_is_raised(store: Path) -> None:
    atlas = _kyto.scan(store)

    def progress(done: int, total: int) -> None:
        raise ValueError(f"stop at {done} of {total}")

    with pytest.raises(ValueError, match="^stop at "):
        atlas._execute(threads=1, progress=progress)
    assert atlas._execute(threads=1) == N_CELLS


def test_other_python_threads_run_while_the_plan_computes(store: Path) -> None:
    atlas = _kyto.scan(store)
    counter = [0]
    stop = threading.Event()

    def count() -> None:
        while not stop.is_set():
            counter[0] += 1

    in_call: list[int] = []

    def progress(done: int, total: int) -> None:
        in_call.append(counter[0])

    counting = threading.Thread(target=count)
    counting.start()
    try:
        before = counter[0]
        assert atlas._execute(threads=1, progress=progress) == N_CELLS
        after = counter[0]
    finally:
        stop.set()
        counting.join()
    # Held through the call, the GIL would let the counter move only at the progress
    # callbacks; it counts on between them, by far more than the callbacks.
    assert len(in_call) >= 2
    assert in_call[-1] > in_call[0] + 1_000
    assert after - before > 10_000


def test_progress_counts_every_cell_once_per_pass(store: Path) -> None:
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))

    assert _kyto.scan(store)._execute(threads=4, progress=progress) == N_CELLS
    assert len(calls) >= 2
    dones = [done for done, _ in calls]
    assert dones == sorted(set(dones)), "progress only moves forward, once per change"
    assert {total for _, total in calls} == {N_CELLS}
    assert calls[-1] == (N_CELLS, N_CELLS)
    assert calls[0][0] < N_CELLS


def test_execute_without_progress_and_on_any_threads(store: Path) -> None:
    atlas = _kyto.scan(store)
    assert atlas._execute() == N_CELLS
    assert atlas._execute(threads=3, progress=None) == N_CELLS


def test_arguments_and_unsupported_plans_are_coded(store: Path, tmp_path: Path) -> None:
    atlas = _kyto.scan(store)
    with pytest.raises(kyto.PlanError) as raised:
        atlas._execute(threads=0)
    assert raised.value.code == "KY-E215"
    assert "threads" in str(raised.value)
    with pytest.raises(kyto.PlanError) as raised:
        atlas._execute(progress=3)  # pyright: ignore[reportArgumentType]
    assert raised.value.code == "KY-E215"
    assert "not callable" in str(raised.value)
    # A gene filter runs since task 5.8 (every cell streams, over one gene).
    filtered = atlas.filter_genes(gene_column("gene_id") == "g1")
    assert filtered._execute(threads=1) == N_CELLS
    # A sink writes its target since task 5.9 (built from plan JSON: Python has no sink
    # method yet), never over an existing one.
    target = tmp_path / "out.kyto"
    plan = json.loads(filtered.to_json())
    plan["nodes"].append(
        {
            "id": len(plan["nodes"]),
            "input": len(plan["nodes"]) - 1,
            "step": {"method": "sink_zarr", "path": str(target)},
        }
    )
    sink = kyto.from_json(json.dumps(plan))
    assert sink._execute(threads=1) == N_CELLS
    assert json.loads((target / "schema.json").read_text())["n_genes"] == 1
    with pytest.raises(kyto.KytoError) as raised:
        sink._execute(threads=1)
    assert raised.value.code == "KY-E118"


def test_docstring_example_runs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            b"G1\tA\tGene Expression\nG2\tB\tGene Expression\nG3\tC\tGene Expression\n"
        )
    )
    small = tmp_path / "small.kyto"
    kyto.convert(source, small, layer="counts")
    docstring = _kyto.Atlas._execute.__doc__
    assert docstring is not None
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs = {"scan": _kyto.scan, "store": small}
    runner.run(parser.get_doctest(docstring, globs, "Atlas._execute", None, 0))
    assert runner.failures == 0
    assert runner.tries == 3
