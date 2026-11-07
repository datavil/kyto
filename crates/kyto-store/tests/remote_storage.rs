//! `RemoteStorage` answers like a filesystem store for the reads zarrs makes (task 2.8):
//! missing keys are `None` even for empty ranges, many ranges come back in order.

#![allow(clippy::unwrap_used)]

#[path = "support/backends.rs"]
mod backends;

use backends::{Backend, backends, storage};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{ReadableStorageTraits, StoreKey};

#[test]
fn missing_keys_are_none_for_any_ranges() {
    let dir = tempfile::tempdir().unwrap();
    for backend in backends() {
        let storage = storage(backend, dir.path());
        let missing = StoreKey::new("nothing/here").unwrap();
        for ranges in [
            vec![],
            vec![ByteRange::FromStart(0, Some(0))],
            vec![ByteRange::Suffix(0)],
            vec![ByteRange::FromStart(0, Some(4))],
        ] {
            let found = storage
                .get_partial_many(&missing, Box::new(ranges.clone().into_iter()))
                .unwrap();
            assert!(found.is_none(), "{backend:?}: {ranges:?}");
        }
        assert_eq!(storage.size_key(&missing).unwrap(), None);
    }
}

#[test]
fn many_ranges_come_back_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = (0..=255).cycle().take(3000).collect();
    std::fs::write(dir.path().join("blob"), &bytes).unwrap();
    for backend in backends() {
        let storage = storage(backend, dir.path());
        let key = StoreKey::new("blob").unwrap();
        let ranges: Vec<ByteRange> = (0..500)
            .map(|at| ByteRange::FromStart(at * 6, Some(3)))
            .chain([ByteRange::Suffix(5), ByteRange::FromStart(2990, None)])
            .collect();
        let parts: Vec<Vec<u8>> = storage
            .get_partial_many(&key, Box::new(ranges.clone().into_iter()))
            .unwrap()
            .unwrap()
            .map(|part| part.unwrap().to_vec())
            .collect();
        assert_eq!(parts.len(), ranges.len(), "{backend:?}");
        for (range, part) in ranges.iter().zip(&parts) {
            let start = range.start(3000) as usize;
            assert_eq!(
                part,
                &bytes[start..start + part.len()],
                "{backend:?} {range:?}"
            );
            assert_eq!(
                part.len() as u64,
                range.length(3000),
                "{backend:?} {range:?}"
            );
        }
        if backend != Backend::Local {
            assert_eq!(storage.size_key(&key).unwrap(), Some(3000));
        }
    }
}
