# serde validation and the R10 string-error check (task 3.1)

**Broke:** `#[serde(try_from = "Raw")]` with `type Error = String` and `Err(format!(...))`
fails `scripts/error_registry.py --check` ("return a KytoError, not a string"), and
kyto-core cannot build the store-specific `KytoError` (`KY-E122` needs the store path).

**Works:** implement `Deserialize` by hand: deserialize the raw struct, then fail with
serde's structured constructors (`D::Error::missing_field`, `unknown_variant`,
`unknown_field`). No strings, standard serde messages with line and column, and the
caller (kyto-store `read_atlas_schema`) wraps the serde error in `KY-E122`.

**Also:** for exact JSON round trips, optional keys need `deserialize_with` that rejects
`null` (`#[serde(default)]` alone turns `"key": null` into `None`, which is then not
written back). See `crates/kyto-core/src/schema_column.rs`.
