# One build path: uv, never `--no-sync` or mixed `maturin develop`

(from ggrust; re-verify once task 1.2 lands)

**Broke (in ggrust):** the test target ran `uv run --no-sync pytest` after
`maturin develop`. Editing Rust and running the tests alone tested the *old*
extension: false green. On a fresh clone `--no-sync` also picked a global pytest from
PATH and failed to import the package.

**Works:** let uv own the build. `uv run pytest` rebuilds the extension when any
`[tool.uv] cache-keys` file changes (Rust sources, `Cargo.toml`, `Cargo.lock`), so list
them there. Force a rebuild with `uv sync --reinstall-package kyto`.

Also: cargo (clippy, test) must see the project interpreter, so export
`PYO3_PYTHON=.venv/bin/python` (an abi3 floor fails against an older system python3).
Verify the setup by sabotaging a Rust function and confirming the Python tests fail.
