# Generated stubs with pyo3-stub-gen (task 7.6)

**Works:** `python/kyto/_kyto.pyi` is generated, never edited: `uv run python
scripts/stubs.py` (the gate runs it with `--check`). A new binding is annotated
(`#[gen_stub_pyclass(module = "kyto._kyto")]` before `#[pyclass]`, `#[gen_stub_pymethods]`
before `#[pymethods]`, `#[gen_stub_pyfunction]` before `#[pyfunction]`), its docstring is
its Rust `///` comment (the one source: `tests/test_stubs.py` compares stub and
`__doc__`), and every argument the binding takes as `&Bound<PyAny>` gets
`#[gen_stub(override_type(type_repr = "...", imports = ("module",)))]`, returns
`override_return_type` on the method. Write types fully qualified (`builtins.str`,
`polars.DataFrame`, `collections.abc.Sequence[...]`) with their modules in `imports`; a
bare name (`CellExpr[Numeric]`, `_GeneExprBoolean`) is taken from the stub as is. The
run arguments `memory_limit`/`threads`/`progress` need nothing: `stub_gen.rs` types them
by name. Extra overloads of a method: one `gen_methods_from_python!` block (each `def`
with `@typing.overload`); the Rust one becomes an overload too and keeps the docstring.

**Hand-written:** what pyo3-stub-gen cannot say (generic classes such as
`CellExpr(Generic[_D_co])`, methods typed on `self`, `TypeVar`s, aliases,
`TypedDict`s) lives in `crates/kyto-py/stubs/expressions.pyi`, merged in by the script.
Its docstrings must equal the run-time ones (the test fails otherwise); edit the Rust
doc comment and copy it.

**Broke (each cost a debug round):**
- `StubInfo::from_pyproject_toml` evaluates `env::var("CARGO_MANIFEST_DIR").unwrap()`
  eagerly: run the binary through `cargo run`, never `target/debug/stub_gen` directly.
- `infer_signature` (real defaults instead of `...`) starts an embedded interpreter,
  which dies with `No module named 'encodings'` unless `PYTHONHOME` is set (the script
  sets it to the running Python's `sys.base_prefix`).
- `ruff --stdin-filename` does not see `kyto` as first-party and sorts its import into
  the third-party block; the gate's ruff on the file does. Pass
  `lint.isort.known-first-party`. Apply the project's full `ruff check --fix` (UP007
  turns `typing.Optional` into `X | None`, PYI029 drops `__repr__`/`__str__`), not only
  `I`, or the gate's ruff rewrites the generated file and the tree is dirty.
- An argument no overload of `filter` accepts makes pyright report both
  `reportCallIssue` and `reportArgumentType`: an ignore comment must name both.
- `scipy-stubs` types `to_scipy()` results (added by 7.9). In its stubs neither
  `spmatrix` nor `sparray` is a sparse base, so `csr_matrix(x)` accepts only a
  concrete sparse class or an array: narrow anndata's loosely typed `X`/`obsp` with
  `isinstance(x, csr_matrix)` (`make_reference._sparse`), not with `spmatrix`.
