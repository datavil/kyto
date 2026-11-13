# Python chain methods and terminals (task 7.2)

**Works:** one `#[pymethods]` block per pyclass (pyo3's `multiple-pymethods` feature is
off), so `atlas.rs` holds thin methods that build a node through `Atlas::derive` (which
also carries `collect()`'s temporary store) and leave parsing to `chain.rs` and the
shared terminal arguments (`memory_limit`, `threads`, `progress`) to `terminal.rs`. A
new chain method is one method there, its stub in `_kyto.pyi` (same docstring), and a
line in `tests/test_chain.py`'s method list (its doctests run from both sources).

**Works:** a result that is a frame, not a plan member (6.10's DE), stays lazy through
`pl.defer(function, schema=...)` with a `PyCFunction::new_closure` that runs the
algorithm with the GIL released: one return type (`pl.LazyFrame`), nothing read until
`.collect()` (R1). Check names against the plan when the method is called (`Output::members`)
so errors still fire at build time.

**Broke:** adding a `[project] dependencies` entry made uv 0.11.7 relock `uv.lock` at
an older `revision` (5 -> 3) and add markers everywhere: a large, conflict-prone diff.
Leave the lock alone unless the task needs the dependency, and use the uv that wrote it.
