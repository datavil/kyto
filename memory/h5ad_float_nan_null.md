# `.h5ad` float NaN is missing, kyto NaN is a value (task 3.7)

**Decided (user, Nov 7, 2026):** anndata has no nullable float encoding, so a `NaN` in an
`obs`/`var` float column means missing. The `.h5ad` importer (`nan_to_null` in
`crates/kyto-store/src/import_h5ad.rs`) makes it null; matrix values are never touched.
kyto's own writers keep `NaN` and null apart (format.md §5.5).

**Consequences to remember:**
- The export writes float nulls as `NaN`, so a store's `NaN` comes back **null** after
  `sink_h5ad` then `convert`. Round-trip tests compare "NaN or null" against null
  (`assert_frames_equal` in `crates/kyto-store/tests/export_h5ad.rs`).
- Parity references: `tests/fixtures/import_inputs.py` expects float columns with
  `fill_nan(None)`.
- Lowered expressions follow Polars, where `NaN > 1` is `True`; that now only concerns
  `NaN` that kyto itself writes.
