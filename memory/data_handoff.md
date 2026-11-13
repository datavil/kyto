# Handing data to Python without a copy (task 7.5)

**Works:** Rust Polars frames reach Python Polars through the Arrow PyCapsule interface,
not `pyo3-polars`: a private `#[pyclass]` with `__arrow_c_stream__` returning
`PyCapsule::new_with_value(py, stream, c"arrow_array_stream")` (a
`polars_arrow::ffi::ArrowArrayStream` from `export_iterator` over
`frame.into_struct("")`'s chunks), read by `pl.DataFrame(obj)`. Python takes the Rust
buffers (checked by address through `_kyto.handoff_example()`), `Enum` categories kept.
No `unsafe` (kyto-py forbids it): `ArrowArrayStream` is `Send` and releases itself on
drop if nobody read it. Python Polars' `to_numpy(allow_copy=False).ctypes.data` gives
a column's buffer address (`_get_buffer_info` is gone in Polars 2).

**Broke:** `pyo3-polars` 0.28 needs pyo3 0.29; kyto is on 0.30 and two pyo3 versions
cannot link (`pyo3-ffi` links `python`). Check its index entry before planning on it.

**Broke:** `Column::new(name, vec)` copies the vector; `UInt32Chunked::from_vec(name,
vec).into_column()` moves it. A `Vec` pointer taken before `Column::new` is not the
frame's buffer.

**Works:** `bytemuck::cast_vec` turns `Vec<u32>` into `Vec<i32>` (and `u64` into
`i64`) in place, safely, for SciPy's signed index arrays.

**Broke:** a Rust unit test that embeds Python (`Python::initialize`) dies under
`uv run cargo test` ("No module named 'encodings'": uv's Python is relocatable and
the embedded one finds no stdlib), and setting `PYTHONHOME` from the test needs
`unsafe` (`set_var`, edition 2024). Test the hand-off from pytest through a hook.
