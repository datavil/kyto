//! Read a store from S3, GCS or HTTP through `object_store` (task 2.8).
//!
//! Does: parse a store URL (`s3://`, `gs://`, `http(s)://`), build the matching
//! `object_store` backend from the explicit options plus that backend's standard
//! environment chain, and expose it as read-only zarrs storage, so [`LayerReader`] and
//! every other generic reader read a remote store exactly like a local one. Every read
//! is a ranged GET (a suffix range for a shard index), and the ranges of one call run
//! concurrently on a tokio runtime the storage owns (no global state, R7).
//! Does not: write, list or cache; open local paths (use `FilesystemStore`); or support
//! Azure and other backends.
//!
//! [`LayerReader`]: crate::LayerReader

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, stream};
use kyto_error::KytoError;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::http::HttpBuilder;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use object_store::{
    ClientConfigKey, GetOptions, GetRange, ObjectStore, ObjectStoreExt, ObjectStoreScheme,
};
use tokio::runtime::Runtime;
use url::Url;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{Bytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

/// Worker threads of the runtime that drives a remote store's requests. They only poll
/// sockets; the calling (rayon) threads wait on their own reads, so reads of different
/// tiles stay parallel.
const RUNTIME_WORKERS: usize = 2;

/// Most requests one `get_partial_many` call has in flight, so a read of many inner
/// chunks of one shard cannot open thousands of connections at once.
const REQUESTS_IN_FLIGHT: usize = 32;

/// Read-only zarrs storage over an `object_store` backend (S3, GCS or HTTP).
///
/// Built by [`RemoteStorage::open`]. Pass it in an `Arc` to [`LayerReader::open`]. Call
/// it only from synchronous code: a read from inside a tokio runtime panics (zarrs'
/// storage traits are blocking). Dropping it is safe anywhere.
///
/// [`LayerReader::open`]: crate::LayerReader::open
pub struct RemoteStorage {
    store: Arc<dyn ObjectStore>,
    /// Always `Some` until dropped; `Drop` takes it to shut it down without blocking.
    runtime: Option<Runtime>,
}

impl Drop for RemoteStorage {
    fn drop(&mut self) {
        // A plain drop blocks, which panics inside an async context.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

impl std::fmt::Debug for RemoteStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The store's Debug output can hold credentials; print only its kind.
        write!(formatter, "RemoteStorage({})", self.store)
    }
}

impl RemoteStorage {
    /// Open the store at `url` for reading.
    ///
    /// `url` is `s3://bucket/path/to/store.kyto`, `gs://bucket/path/to/store.kyto`, or an
    /// `http://` or `https://` URL of the store's root directory (an `https://` URL of
    /// an S3 or GCS host opens that backend). `options` are `object_store` config keys
    /// and values for that backend (for S3 `aws_access_key_id`, `aws_secret_access_key`,
    /// `aws_region`, `aws_endpoint`; for GCS `google_service_account`; for all of them
    /// client keys such as `allow_http` and `timeout`). S3 and GCS first read their
    /// standard environment variables (`AWS_*`, `GOOGLE_*`), then `options` override
    /// them; without keys, the backend's default credential chain applies (instance
    /// metadata, web identity). HTTP reads no environment. Nothing is read from the
    /// store until a reader opens it.
    ///
    /// # Errors
    /// `KY-E125` if `url` does not parse or names another scheme, an option key is not
    /// one the backend accepts, or the backend rejects the configuration (no bucket, a
    /// bad value). A URL with a query, a fragment or a user name and password, and the
    /// bucket options (`bucket`, `aws_bucket`, `google_bucket`, which would override the
    /// URL's bucket) are refused too. Errors never echo option values.
    pub fn open<I, K, V>(url: &str, options: I) -> Result<Self, KytoError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let parsed = Url::parse(url).map_err(|error| {
            invalid(
                url,
                format!("the URL does not parse ({error})"),
                "give a full URL such as s3://bucket/pbmc.kyto, gs://bucket/pbmc.kyto or https://host/pbmc.kyto",
            )
        })?;
        let shown = redacted(&parsed);
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(invalid(
                &shown,
                "the URL has a query or a fragment, which object stores do not send".to_owned(),
                "remove it; pass credentials and tokens as options",
            ));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(invalid(
                &shown,
                "the URL holds a user name or password".to_owned(),
                "remove them from the URL; pass credentials as options",
            ));
        }
        let unsupported = || {
            invalid(
                &shown,
                format!("scheme '{}' is not a remote store", parsed.scheme()),
                "use s3://, gs://, http:// or https://; open a local store by its path",
            )
        };
        let (scheme, prefix) = ObjectStoreScheme::parse(&parsed).map_err(|_| unsupported())?;
        let options: Vec<(String, String)> = options
            .into_iter()
            .map(|(key, value)| (key.as_ref().to_ascii_lowercase(), value.into()))
            .collect();
        let store: Arc<dyn ObjectStore> = match scheme {
            ObjectStoreScheme::AmazonS3 => {
                let builder = AmazonS3Builder::from_env().with_url(parsed.as_str());
                let builder = configure(&shown, "s3", builder, &options, |builder, key, value| {
                    match key.parse::<AmazonS3ConfigKey>()? {
                        AmazonS3ConfigKey::Bucket => Err(Refused::Bucket),
                        key => Ok(builder.with_config(key, value)),
                    }
                })?;
                Arc::new(builder.build().map_err(|error| rejected(&shown, error))?)
            }
            ObjectStoreScheme::GoogleCloudStorage => {
                let builder = GoogleCloudStorageBuilder::from_env().with_url(parsed.as_str());
                let builder = configure(&shown, "gs", builder, &options, |builder, key, value| {
                    match key.parse::<GoogleConfigKey>()? {
                        GoogleConfigKey::Bucket => Err(Refused::Bucket),
                        key => Ok(builder.with_config(key, value)),
                    }
                })?;
                Arc::new(builder.build().map_err(|error| rejected(&shown, error))?)
            }
            ObjectStoreScheme::Http => {
                let origin = &parsed[..url::Position::BeforePath];
                let builder = HttpBuilder::new().with_url(origin);
                let builder =
                    configure(&shown, "http", builder, &options, |builder, key, value| {
                        Ok(builder.with_config(key.parse::<ClientConfigKey>()?, value))
                    })?;
                Arc::new(builder.build().map_err(|error| rejected(&shown, error))?)
            }
            _ => return Err(unsupported()),
        };
        let store: Arc<dyn ObjectStore> = if prefix.as_ref().is_empty() {
            store
        } else {
            Arc::new(PrefixStore::new(store, prefix))
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(RUNTIME_WORKERS)
            .thread_name("kyto-remote")
            .enable_all()
            .build()
            .map_err(|error| {
                invalid(
                    &shown,
                    format!("cannot start the I/O runtime ({error})"),
                    "check that the process may start threads",
                )
            })?;
        Ok(Self {
            store,
            runtime: Some(runtime),
        })
    }

    /// Read one range of the object at `path`; `Ok(None)` if there is no such object.
    async fn read_range(
        store: &dyn ObjectStore,
        path: &Path,
        range: ByteRange,
    ) -> Result<Option<Bytes>, object_store::Error> {
        let (range, expected) = match range {
            // object_store rejects an empty range; it reads nothing anyway.
            ByteRange::FromStart(_, Some(0)) | ByteRange::Suffix(0) => {
                return Ok(Some(Bytes::new()));
            }
            ByteRange::FromStart(offset, Some(length)) => {
                let end =
                    offset
                        .checked_add(length)
                        .ok_or_else(|| object_store::Error::Generic {
                            store: "kyto",
                            source: format!("byte range {offset} + {length} of {path} overflows")
                                .into(),
                        })?;
                (GetRange::Bounded(offset..end), Some(length))
            }
            ByteRange::FromStart(offset, None) => (GetRange::Offset(offset), None),
            ByteRange::Suffix(length) => (GetRange::Suffix(length), Some(length)),
        };
        let options = GetOptions {
            range: Some(range),
            ..GetOptions::default()
        };
        let bytes = match store.get_opts(path, options).await {
            Ok(result) => result.bytes().await?,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        // object_store returns less than asked when a range passes the end of the
        // object; a filesystem store fails there, and so do we.
        if let Some(expected) = expected
            && bytes.len() as u64 != expected
        {
            return Err(object_store::Error::Generic {
                store: "kyto",
                source: format!(
                    "asked for {expected} bytes of {path}, got {}: the range passes the end of the object",
                    bytes.len()
                )
                .into(),
            });
        }
        Ok(Some(bytes))
    }
}

impl RemoteStorage {
    fn block_on<F: std::future::Future>(&self, future: F) -> Result<F::Output, StorageError> {
        match &self.runtime {
            Some(runtime) => Ok(runtime.block_on(future)),
            None => Err(StorageError::Other(
                "the remote store was dropped".to_owned(),
            )),
        }
    }
}

/// The object path of a store key. Zarr keys are already path text: parse, do not
/// encode, or a `%` in a key would be encoded twice.
fn object_path(key: &StoreKey) -> Result<Path, StorageError> {
    Path::parse(key.as_str())
        .map_err(|error| StorageError::Other(format!("key {key} is not an object path: {error}")))
}

impl ReadableStorageTraits for RemoteStorage {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let path = object_path(key)?;
        let ranges: Vec<ByteRange> = byte_ranges.collect();
        // Empty ranges send no request, so they cannot tell a missing key: ask first.
        let all_empty = ranges.iter().all(|range| {
            matches!(
                range,
                ByteRange::FromStart(_, Some(0)) | ByteRange::Suffix(0)
            )
        });
        if all_empty && self.size_key(key)?.is_none() {
            return Ok(None);
        }
        let store = &*self.store;
        // `buffered` keeps the ranges' order and caps the requests in flight.
        let reads = stream::iter(ranges)
            .map(|range| Self::read_range(store, &path, range))
            .buffered(REQUESTS_IN_FLIGHT)
            .try_collect::<Vec<_>>();
        let results = self
            .block_on(reads)?
            .map_err(|error| StorageError::Other(error.to_string()))?;
        // A key is missing or present for all its ranges; one missing read means the
        // object is not there (or vanished mid-read, which reads as missing too).
        if results.iter().any(Option::is_none) {
            return Ok(None);
        }
        let parts: Vec<Result<Bytes, StorageError>> =
            results.into_iter().flatten().map(Ok).collect();
        Ok(Some(Box::new(parts.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let path = object_path(key)?;
        match self.block_on(self.store.head(&path))? {
            Ok(meta) => Ok(Some(meta.size)),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(error) => Err(StorageError::Other(error.to_string())),
        }
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

/// Apply `options` to `builder`, rejecting keys the backend does not accept.
fn configure<B>(
    url: &str,
    scheme: &str,
    builder: B,
    options: &[(String, String)],
    apply: impl Fn(B, &str, String) -> Result<B, Refused>,
) -> Result<B, KytoError> {
    options
        .iter()
        .try_fold(builder, |builder, (key, value)| {
            apply(builder, key, value.clone()).map_err(|refused| {
                if matches!(refused, Refused::Bucket) {
                    return invalid(
                        url,
                        format!("option '{key}' would override the bucket in the URL"),
                        "name the bucket in the URL only, as in s3://bucket/pbmc.kyto",
                    );
                }
                let advice = match scheme {
                    "s3" => {
                        "use an object_store key for s3, such as aws_access_key_id, aws_secret_access_key, aws_session_token, aws_region, aws_endpoint, aws_skip_signature or allow_http"
                    }
                    "gs" => {
                        "use an object_store key for gcs, such as google_service_account, google_service_account_key, google_application_credentials or allow_http"
                    }
                    _ => {
                        "use an object_store client key for http, such as allow_http, timeout, connect_timeout or user_agent"
                    }
                };
                invalid(url, format!("unknown storage option '{key}' for {scheme}"), advice)
            })
        })
}

/// Why an option was not applied.
enum Refused {
    /// The backend does not know the key.
    Unknown,
    /// The key names the bucket, which only the URL may do.
    Bucket,
}

impl From<object_store::Error> for Refused {
    fn from(_: object_store::Error) -> Self {
        Self::Unknown
    }
}

/// `KY-E125` for a configuration the backend refused.
fn rejected(url: &str, error: object_store::Error) -> KytoError {
    invalid(
        url,
        format!("the backend rejected the configuration ({error})"),
        "check the bucket name in the URL and the option values",
    )
}

fn invalid(url: &str, problem: String, advice: &str) -> KytoError {
    KytoError::InvalidRemoteStore {
        url: url.to_owned(),
        problem,
        advice: advice.to_owned(),
    }
}

/// `url` without a password, for error messages.
fn redacted(url: &Url) -> String {
    let mut shown = url.clone();
    if shown.password().is_some() {
        // set_password fails only for URLs that cannot have one, which have none.
        let _ = shown.set_password(Some("***"));
    }
    shown.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_of(result: Result<RemoteStorage, KytoError>) -> KytoError {
        match result {
            Ok(storage) => panic!("expected an error, opened {storage:?}"),
            Err(error) => error,
        }
    }

    const NO_OPTIONS: [(&str, &str); 0] = [];

    #[test]
    fn bad_urls_and_schemes_fail_with_e125() {
        for url in ["not a url", "file:///tmp/pbmc.kyto", "ftp://host/pbmc.kyto"] {
            let error = error_of(RemoteStorage::open(url, NO_OPTIONS));
            assert_eq!(error.code(), "KY-E125", "{url}: {error}");
        }
        let error = error_of(RemoteStorage::open("file:///tmp/x.kyto", NO_OPTIONS));
        assert!(error.to_string().contains("scheme 'file'"), "{error}");
    }

    #[test]
    fn unknown_options_fail_without_echoing_values() {
        let error = error_of(RemoteStorage::open(
            "s3://bucket/pbmc.kyto",
            [("aws_acess_key_id", "SECRET-VALUE")],
        ));
        assert_eq!(error.code(), "KY-E125");
        let message = error.to_string();
        assert!(message.contains("'aws_acess_key_id'"), "{message}");
        assert!(!message.contains("SECRET-VALUE"), "{message}");
        let error = error_of(RemoteStorage::open(
            "https://example.com/pbmc.kyto",
            [("aws_region", "eu-west-1")],
        ));
        assert!(error.to_string().contains("for http"), "{error}");
    }

    #[test]
    fn credentials_and_queries_in_urls_are_refused_and_redacted() {
        let error = error_of(RemoteStorage::open(
            "https://user:hunter2@example.com/pbmc.kyto",
            NO_OPTIONS,
        ));
        assert_eq!(error.code(), "KY-E125");
        let message = error.to_string();
        assert!(!message.contains("hunter2"), "{message}");
        assert!(message.contains("user:***@example.com"), "{message}");
        assert!(message.contains("user name or password"), "{message}");
        for url in [
            "https://example.com/pbmc.kyto?token=x",
            "https://example.com/pbmc.kyto#part",
        ] {
            let error = error_of(RemoteStorage::open(url, NO_OPTIONS));
            assert!(error.to_string().contains("query or a fragment"), "{error}");
        }
    }

    #[test]
    fn bucket_options_are_refused() {
        for (url, key) in [
            ("s3://bucket/pbmc.kyto", "aws_bucket"),
            ("s3://bucket/pbmc.kyto", "bucket_name"),
            ("gs://bucket/pbmc.kyto", "google_bucket"),
        ] {
            let error = error_of(RemoteStorage::open(url, [(key, "other")]));
            assert_eq!(error.code(), "KY-E125");
            assert!(error.to_string().contains("override the bucket"), "{error}");
        }
    }

    #[test]
    fn dropping_inside_a_runtime_does_not_panic() {
        let storage = RemoteStorage::open("http://127.0.0.1:1/pbmc.kyto", NO_OPTIONS).unwrap();
        let outer = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        outer.block_on(async move { drop(storage) });
    }

    #[test]
    fn valid_configurations_open_without_reading() {
        let s3 = RemoteStorage::open(
            "s3://bucket/path/pbmc.kyto",
            [
                ("aws_access_key_id", "key"),
                ("aws_secret_access_key", "secret"),
                ("aws_region", "us-east-1"),
                ("aws_endpoint", "http://127.0.0.1:1"),
                ("allow_http", "true"),
            ],
        );
        assert!(s3.is_ok());
        assert!(RemoteStorage::open("http://127.0.0.1:1/pbmc.kyto", NO_OPTIONS).is_ok());
        assert!(RemoteStorage::open("gs://bucket/pbmc.kyto", [("allow_http", "false")]).is_ok());
    }
}
