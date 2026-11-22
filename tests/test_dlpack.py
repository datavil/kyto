"""Task 9.5: DLPack export, ``to_torch(device=)`` and ``__dlpack__``, copy-free on the CPU.

Done when: a torch tensor from kyto shares the device pointer. kyto's results are on
the host, so the shared device is the CPU: ``torch.from_dlpack(embedding)`` and both
``to_torch(device="cpu")`` give a tensor whose ``data_ptr()`` is the address kyto
exported (read from the DLPack capsule, or from the array kyto hands to
``torch.from_dlpack``). Another device (``mps``, ``cuda``) gets one copy, made by
PyTorch; those tests run where the device exists and skip with a notice elsewhere.

The PyTorch tests skip, visibly, without ``torch`` (it is not a dependency); the
NumPy ones (``numpy.from_dlpack``, the same protocol) always run.
"""

import ctypes
import gzip
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import kyto
import numpy as np
import pytest
from kyto import _kyto
from kyto.errors import MemoryLimitError
from typing_extensions import CapsuleType

# The cells x genes counts of the store below (Matrix Market is genes x cells).
COUNTS = np.array([[5, 0, 1], [0, 7, 0]], dtype=np.uint32)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A 2-cell, 3-gene store with one layer ``counts`` (as ``tests/test_terminals.py``)."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 5\n3 1 1\n2 2 7\n"
    )
    (source / "barcodes.tsv").write_text("AAAC-1\nAAAG-1\n")
    (source / "features.tsv.gz").write_bytes(
        gzip.compress(
            b"ENSG01\tCD3E\tGene Expression\nENSG02\tCD4\tGene Expression\n"
            b"ENSG03\tMS4A1\tGene Expression\n"
        )
    )
    path = tmp_path / "pbmc.kyto"
    kyto.convert(source, path, layer="counts")
    return path


def _pca(store: Path) -> _kyto.Atlas:
    """A plan with a ``log1p`` layer and an embedding ``X_pca``."""
    return (
        kyto.scan(store)
        .log1p(layer="counts", into="log1p")
        .pca(layer="log1p", into="X_pca", n_components=1, seed=0)
    )


def _torch() -> Any:
    """The ``torch`` module, or skip the test with a visible reason."""
    return pytest.importorskip(
        "torch", reason="torch is not installed (not a kyto dependency)"
    )


def _capsule_data_pointer(capsule: object) -> int:
    """The ``data`` address of a legacy ``"dltensor"`` capsule's ``DLTensor``.

    ``DLManagedTensor`` starts with its ``DLTensor``, which starts with ``void* data``.
    """
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    managed = get_pointer(capsule, b"dltensor")
    assert managed is not None
    data = ctypes.c_void_p.from_address(managed).value
    assert data is not None
    return data


class _Exported:
    """One capsule kyto exported, offered again to a consumer's ``from_dlpack``."""

    def __init__(self, capsule: CapsuleType) -> None:
        self.capsule = capsule

    def __dlpack__(self, **_: object) -> CapsuleType:
        return self.capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


@pytest.fixture
def spy_from_dlpack() -> Iterator[list[int]]:
    """Record the address of every array kyto hands to ``torch.from_dlpack``."""
    torch = _torch()
    original: Callable[[Any], Any] = torch.from_dlpack
    addresses: list[int] = []

    def spy(array: Any) -> Any:
        addresses.append(int(array.ctypes.data))
        return original(array)

    torch.from_dlpack = spy
    try:
        yield addresses
    finally:
        torch.from_dlpack = original


# The protocol, through NumPy (always runs).


def test_dlpack_device_is_the_cpu_and_reads_nothing(store: Path) -> None:
    # The device comes from metadata: kyto's results are on the host.
    embedding = _pca(store).embedding("X_pca")
    assert embedding.__dlpack_device__() == (1, 0)


def test_numpy_from_dlpack_equals_to_numpy(store: Path) -> None:
    embedding = _pca(store).embedding("X_pca")
    array = np.from_dlpack(embedding)
    expected = embedding.to_numpy(threads=1)
    assert array.dtype == np.float32
    assert array.shape == (2, 1)
    np.testing.assert_array_equal(array, expected)
    # A collected Atlas (read in place) exports the same values.
    collected = _pca(store).collect(threads=1).embedding("X_pca")
    np.testing.assert_array_equal(np.from_dlpack(collected), expected)


def test_consumer_shares_the_exported_pointer(store: Path) -> None:
    capsule = _pca(store).embedding("X_pca").__dlpack__()
    pointer = _capsule_data_pointer(capsule)
    array = np.from_dlpack(_Exported(capsule))
    assert array.ctypes.data == pointer


def test_versioned_capsule_and_copy(store: Path) -> None:
    embedding = _pca(store).embedding("X_pca")
    capsule = embedding.__dlpack__(max_version=(1, 0))
    assert "dltensor_versioned" in repr(capsule)
    # copy=True still exports (a copy NumPy makes), copy=False refuses nothing on the CPU.
    copied = embedding.__dlpack__(copy=True)
    assert "dltensor" in repr(copied)
    assert "dltensor" in repr(embedding.__dlpack__(copy=False))


def test_other_device_is_refused(store: Path) -> None:
    embedding = _pca(store).embedding("X_pca")
    with pytest.raises(BufferError):
        embedding.__dlpack__(dl_device=(2, 0))  # kDLCUDA


def test_matrix_has_no_dlpack(store: Path) -> None:
    # R8: a Matrix densifies only through max_bytes; its dense array speaks DLPack.
    matrix = kyto.scan(store).layer("counts")
    assert not hasattr(matrix, "__dlpack__")
    dense = matrix.to_dense(max_bytes=1024)
    np.testing.assert_array_equal(np.from_dlpack(dense), COUNTS)


# PyTorch (skips without torch).


def test_torch_from_dlpack_shares_the_pointer(store: Path) -> None:
    torch = _torch()
    embedding = _pca(store).embedding("X_pca")
    tensor = torch.from_dlpack(embedding)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    np.testing.assert_array_equal(tensor.numpy(), embedding.to_numpy())
    # The tensor's memory is the memory kyto exported: no copy.
    capsule = embedding.__dlpack__()
    pointer = _capsule_data_pointer(capsule)
    assert torch.from_dlpack(_Exported(capsule)).data_ptr() == pointer


def test_embedding_to_torch_shares_the_pointer(
    store: Path, spy_from_dlpack: list[int]
) -> None:
    torch = _torch()
    embedding = _pca(store).embedding("X_pca")
    tensor = embedding.to_torch(device="cpu", threads=1)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (2, 1)
    assert spy_from_dlpack and tensor.data_ptr() == spy_from_dlpack[-1]
    np.testing.assert_array_equal(tensor.numpy(), embedding.to_numpy(threads=1))


def test_matrix_to_torch_shares_the_pointer(
    store: Path, spy_from_dlpack: list[int]
) -> None:
    torch = _torch()
    atlas = _pca(store)
    counts = atlas.layer("counts").to_torch(device="cpu", max_bytes=1024)
    assert counts.dtype == torch.uint32
    assert counts.data_ptr() == spy_from_dlpack[-1]
    np.testing.assert_array_equal(counts.numpy(), COUNTS)
    log1p = atlas.layer("log1p").to_torch(device="cpu", max_bytes=1024, threads=1)
    assert log1p.dtype == torch.float32
    assert log1p.data_ptr() == spy_from_dlpack[-1]
    np.testing.assert_array_equal(
        log1p.numpy(), atlas.layer("log1p").to_dense(max_bytes=1024, threads=1)
    )


def test_matrix_to_torch_checks_max_bytes_first(store: Path) -> None:
    # R8: refused before the plan runs (after torch and the device are checked).
    _torch()
    with pytest.raises(MemoryLimitError, match=r"KY-E402: 'Matrix\.to_torch'"):
        kyto.scan(store).layer("counts").to_torch(device="cpu", max_bytes=8)


def test_unknown_device_fails_before_the_plan_runs(store: Path) -> None:
    _torch()
    atlas = _pca(store)
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))

    with pytest.raises(RuntimeError):
        atlas.embedding("X_pca").to_torch(device="bogus", progress=progress)
    with pytest.raises(RuntimeError):
        atlas.layer("log1p").to_torch(device="bogus", max_bytes=1024, progress=progress)
    assert calls == []
    # The progress callback is called when the plan does run.
    atlas.embedding("X_pca").to_torch(device="cpu", progress=progress)
    assert calls


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_to_torch_on_a_gpu(store: Path, device: str) -> None:
    torch = _torch()
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }[device]
    if not available:
        pytest.skip(f"no {device} device here")
    atlas = _pca(store)
    tensor = atlas.embedding("X_pca").to_torch(device=device)
    assert tensor.device.type == device
    np.testing.assert_array_equal(
        tensor.cpu().numpy(), atlas.embedding("X_pca").to_numpy()
    )
    log1p = atlas.layer("log1p").to_torch(device=device, max_bytes=1024)
    assert log1p.device.type == device
