# Static HDF5 through hdf5-metno, CMake from PyPI (task 2.6)

**Decision (user, Nov 7, 2026):** the importers read HDF5 with `hdf5-metno` 0.15 (the
maintained fork of the unmaintained `hdf5` 0.8 crate) with features `static`, `zlib`,
`lzf`: HDF5 1.14 is built from bundled C source with CMake, zlib and lzf from bundled
C, so wheels need no system libhdf5. Rejected: the pure-Rust `hdf5-reader` (no LZF
filter, young), and linking a system libhdf5 (wheels would have to bundle it).

**Works:** CMake comes from the `cmake` PyPI package, listed twice in `pyproject.toml`:
in the uv dev group (so `uv run cargo ...` finds it on PATH) and in
`[build-system] requires` (`uv sync` builds the extension in an isolated PEP 517 env
that does not see the dev group; without it the hdf5-metno-src build script fails
with "is `cmake` not installed?"). CMake 4.x builds the bundled HDF5 fine. A clean
build of the HDF5 C sources adds about 1 minute.

**Not verified yet:** the 6-target build (Windows ARM64 above all; manylinux
containers in maturin-action, which build with plain `maturin build` and so rely on
the image's own `cmake`). CI runs only on demand: check the next on-demand or release
run and record the outcome here.

**Also:**
- hdf5-metno's default reader conversion is `Soft`, so integer datasets read as `f64`
  or `i64` directly.
- h5py writes `numpy.bool_` as an HDF5 enum; hdf5-metno reads it as `bool`.
- 10x files use fixed-length ASCII strings: read through `FixedAscii<N>` with `N` at
  least the stored size (HDF5 pads); use `as_bytes` + `String::from_utf8`, not
  `as_str` (unchecked).
- anndata 0.13 lists `X` as `layers[None]`; filter `None` when iterating layers in
  Python. An empty `column-order` attribute is written as a `float64` array.
