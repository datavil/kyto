//! The store's `format_version`, checked when a reader opens a store (format.md §3, §11).
//!
//! Does: read the root `zarr.json`, require `attributes.kyto.format_version`, and accept
//! exactly the `0.x` versions this kyto implements.
//! Does not: write the root group (see `store_writer`, task 2.5) or check `schema.json` (§8).

use kyto_error::KytoError;
use serde_json::Value;
use zarrs::storage::{ReadableStorageTraits, StoreKey};

use crate::zarr_arrays::store_io;

/// The `format_version` this kyto writes (format.md §3).
pub const FORMAT_VERSION: &str = "0.1";

/// The `format_version`s this kyto reads. While MAJOR is 0 a reader accepts exactly
/// the versions it implements (format.md §11).
pub const SUPPORTED_FORMAT_VERSIONS: &[&str] = &[FORMAT_VERSION];

/// Check that `storage` holds a kyto store of a supported `format_version`.
///
/// `store` names the store in error messages. Returns the version found.
///
/// # Errors
/// - `KY-E116` if there is no root `zarr.json`, it is not a Zarr v3 group in JSON, or it
///   has no string `attributes.kyto.format_version` (format.md §3).
/// - `KY-E115` if the version is not one of [`SUPPORTED_FORMAT_VERSIONS`] (§11).
/// - `KY-E110` if the storage fails.
pub fn check_format_version<S: ?Sized + ReadableStorageTraits>(
    storage: &S,
    store: &str,
) -> Result<String, KytoError> {
    let not_a_store = |problem: &str| KytoError::NotAStore {
        store: store.to_owned(),
        problem: problem.to_owned(),
    };
    let key = StoreKey::new("zarr.json").map_err(|error| store_io("read", store, error))?;
    let bytes = storage
        .get(&key)
        .map_err(|error| store_io("read the root zarr.json of", store, error))?
        .ok_or_else(|| not_a_store("no root zarr.json"))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|_| not_a_store("the root zarr.json is not valid JSON"))?;
    if root.get("zarr_format") != Some(&Value::from(3))
        || root.get("node_type") != Some(&Value::from("group"))
    {
        return Err(not_a_store("the root zarr.json is not a Zarr v3 group"));
    }
    let version = root
        .get("attributes")
        .and_then(|attributes| attributes.get("kyto"))
        .and_then(|kyto| kyto.get("format_version"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            not_a_store("the root zarr.json has no string attributes.kyto.format_version")
        })?;
    if !SUPPORTED_FORMAT_VERSIONS.contains(&version) {
        return Err(KytoError::UnsupportedFormatVersion {
            store: store.to_owned(),
            found: version.to_owned(),
            supported: SUPPORTED_FORMAT_VERSIONS
                .iter()
                .map(|&supported| supported.to_owned())
                .collect(),
        });
    }
    Ok(version.to_owned())
}
