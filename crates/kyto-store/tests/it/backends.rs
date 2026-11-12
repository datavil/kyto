//! Storage backends the reader test suite runs on (task 2.8): the local filesystem,
//! HTTP (an in-process server with byte-range support, always on), and S3 (a MinIO or
//! other S3-compatible server, only when `KYTO_TEST_S3_ENDPOINT` is set).
//!
//! Each call to [`storage`] serves the store directory as it is at that moment: HTTP
//! serves the directory live, S3 uploads it to a fresh prefix of the test bucket. So a
//! test that edits files must call [`storage`] after editing.
//!
//! S3 settings, all read from the environment of the test run (set by
//! `scripts/remote_store_tests.py`, which starts MinIO):
//! `KYTO_TEST_S3_ENDPOINT` (`http://127.0.0.1:9000`), `KYTO_TEST_S3_BUCKET`,
//! `KYTO_TEST_S3_ACCESS_KEY_ID`, `KYTO_TEST_S3_SECRET_ACCESS_KEY`.

// Test helpers may unwrap (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use kyto_store::RemoteStorage;
use object_store::ObjectStoreExt;
use object_store::aws::AmazonS3Builder;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

/// Any readable storage.
pub type Storage = dyn ReadableStorageTraits;

/// A backend of the reader test suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// `FilesystemStore` on the directory.
    Local,
    /// `RemoteStorage` over HTTP from an in-process server.
    Http,
    /// `RemoteStorage` over S3 (MinIO), when configured.
    S3,
}

/// The backends to run: S3 only when `KYTO_TEST_S3_ENDPOINT` is set.
pub fn backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Local, Backend::Http];
    if std::env::var_os("KYTO_TEST_S3_ENDPOINT").is_some() {
        backends.push(Backend::S3);
    } else if std::env::var_os("KYTO_TEST_REQUIRE_S3").is_some() {
        panic!("KYTO_TEST_REQUIRE_S3 is set but KYTO_TEST_S3_ENDPOINT is not");
    }
    backends
}

/// Storage for the store in `dir` on `backend`.
pub fn storage(backend: Backend, dir: &Path) -> Arc<Storage> {
    match backend {
        Backend::Local => Arc::new(FilesystemStore::new(dir).unwrap()),
        Backend::Http => {
            let url = serve_http(dir.to_owned());
            Arc::new(RemoteStorage::open(&url, [("allow_http", "true")]).unwrap())
        }
        Backend::S3 => Arc::new(upload_s3(dir)),
    }
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

/// Upload every file under `dir` to a fresh prefix and open it.
fn upload_s3(dir: &Path) -> RemoteStorage {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let endpoint = env("KYTO_TEST_S3_ENDPOINT");
    let bucket = env("KYTO_TEST_S3_BUCKET");
    let key_id = env("KYTO_TEST_S3_ACCESS_KEY_ID");
    let secret = env("KYTO_TEST_S3_SECRET_ACCESS_KEY");
    let prefix = format!(
        "run-{}-{}-{}.kyto",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let store = AmazonS3Builder::new()
        .with_endpoint(&endpoint)
        .with_bucket_name(&bucket)
        .with_access_key_id(&key_id)
        .with_secret_access_key(&secret)
        .with_region("us-east-1")
        .with_allow_http(true)
        .build()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for file in files(dir) {
            let key = file
                .strip_prefix(dir)
                .unwrap()
                .components()
                .map(|part| part.as_os_str().to_str().unwrap())
                .collect::<Vec<_>>()
                .join("/");
            let bytes = std::fs::read(&file).unwrap();
            store
                .put(
                    &object_store::path::Path::from(format!("{prefix}/{key}")),
                    bytes.into(),
                )
                .await
                .unwrap();
        }
    });
    RemoteStorage::open(
        &format!("s3://{bucket}/{prefix}"),
        [
            ("aws_endpoint", endpoint.as_str()),
            ("aws_access_key_id", key_id.as_str()),
            ("aws_secret_access_key", secret.as_str()),
            ("aws_region", "us-east-1"),
            ("allow_http", "true"),
        ],
    )
    .unwrap()
}

fn files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(files(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// Serve `root` over HTTP on a free local port (GET and HEAD, with single byte
/// ranges) until the test process ends. Returns the store URL.
fn serve_http(root: PathBuf) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for connection in listener.incoming().flatten() {
            let root = root.clone();
            std::thread::spawn(move || {
                // A failed connection only fails the read that used it.
                let _ = handle(connection, &root);
            });
        }
    });
    format!("http://{address}/some/dir/store.kyto")
}

/// Answer the requests on one connection, keeping it alive between them.
fn handle(connection: TcpStream, root: &Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(connection.try_clone()?);
    let mut writer = connection;
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Ok(());
        }
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_owned();
        let target = parts.next().unwrap_or_default().to_owned();
        let mut range = None;
        loop {
            let mut header = String::new();
            reader.read_line(&mut header)?;
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':')
                && name.eq_ignore_ascii_case("range")
            {
                range = Some(value.trim().to_owned());
            }
        }
        respond(&mut writer, root, &method, &target, range.as_deref())?;
    }
}

fn respond(
    writer: &mut TcpStream,
    root: &Path,
    method: &str,
    target: &str,
    range: Option<&str>,
) -> std::io::Result<()> {
    let status = |writer: &mut TcpStream, code: &str| {
        write!(writer, "HTTP/1.1 {code}\r\ncontent-length: 0\r\n\r\n")
    };
    let Some(relative) = target.strip_prefix("/some/dir/store.kyto/") else {
        return status(writer, "404 Not Found");
    };
    if relative
        .split('/')
        .any(|part| part == ".." || part.is_empty())
    {
        return status(writer, "404 Not Found");
    }
    let path = root.join(relative);
    let Ok(mut file) = std::fs::File::open(&path) else {
        return status(writer, "404 Not Found");
    };
    if !path.is_file() {
        return status(writer, "404 Not Found");
    }
    let size = file.metadata()?.len();
    let (start, end, partial) = match range.and_then(|range| range.strip_prefix("bytes=")) {
        None => (0, size, false),
        Some(spec) => {
            let (from, to) = spec.split_once('-').unwrap_or((spec, ""));
            let bounds = match (from.parse::<u64>().ok(), to.parse::<u64>().ok()) {
                (Some(from), Some(to)) => Some((from, (to + 1).min(size))),
                (Some(from), None) => Some((from, size)),
                (None, Some(suffix)) => Some((size.saturating_sub(suffix), size)),
                (None, None) => None,
            };
            match bounds {
                Some((from, to)) if from < to => (from, to, true),
                _ => {
                    return write!(
                        writer,
                        "HTTP/1.1 416 Range Not Satisfiable\r\ncontent-range: bytes */{size}\r\ncontent-length: 0\r\n\r\n"
                    );
                }
            }
        }
    };
    let length = end - start;
    if partial {
        write!(
            writer,
            "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes {start}-{}/{size}\r\n",
            end - 1
        )?;
    } else {
        write!(writer, "HTTP/1.1 200 OK\r\n")?;
    }
    write!(
        writer,
        "content-length: {length}\r\naccept-ranges: bytes\r\n\r\n"
    )?;
    if method == "GET" {
        let mut body = vec![0; length as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut body)?;
        writer.write_all(&body)?;
    }
    writer.flush()
}
