# Reproducibility: checksums, and HDF5 object times (task 8.7)

**Works:** the R11 byte check is `tests/test_reproducibility.py`: a pbmc3k pipeline that
writes every kind of member (two layers, HVG gene columns, a leiden cell column, `X_pca`,
`X_pca_loadings`, `X_umap`, both graphs; approximate neighbours) sunk at `threads=1` and
at one per core, SHA-256 per file. `provenance.json` is hashed after deleting `run` and
`source.path` (format.md §10.3, "The checksum rule"); nothing else is left out. Both
writes scan the store by the same path (8.5: a relative vs absolute scan path changes
`plan`). ~6 s in the gate.

**Broke:** `.h5ad` exports of one store differed in bytes. HDF5's default object creation
lists track times, so every group (and the root group, through the file creation list)
got a modification time in whole seconds; hdf5-metno's datasets already turn it off (as
h5py's). Fix in `export_h5.rs`: `with_fcpl(|fcpl| fcpl.obj_track_times(false))` and
groups through `create_group_builder().obj_track_times(false)`. A new HDF5 object
created another way (plain `create_group`, an intermediate group made by a link)
brings the times back.

**Also:** two quick writes can land in the same second and match anyway, so the test
also asserts every object's `h5o.get_info` times are 0 (h5py has no stubs: a `Protocol`
plus `cast`). Since 8.7 the 5.20 thread test compares `.h5ad` bytes directly instead of
converting them back to stores.
