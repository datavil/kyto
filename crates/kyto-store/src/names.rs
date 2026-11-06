//! Member names: the rules of format.md §2.1 for layers, embeddings and graphs.
//!
//! Does: check one name's syntax, and find a member of a group directory whose name
//! equals a given one case-insensitively.
//! Does not: check column names (format.md §5.6, task 2.3).

use std::path::Path;

use kyto_error::KytoError;

/// Longest member name in bytes.
const MAX_NAME_BYTES: usize = 64;

/// Windows device names, reserved with or without an extension, in any case.
const WINDOWS_DEVICES: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Check a member name against format.md §2.1.
///
/// `kind` is `layer`, `embedding` or `graph`, used in the error.
///
/// # Errors
/// `KY-E102` naming the rule the name breaks.
pub fn validate_member_name(kind: &str, name: &str) -> Result<(), KytoError> {
    let broken = |rule: String| {
        Err(KytoError::InvalidMemberName {
            kind: kind.to_owned(),
            name: name.to_owned(),
            rule,
        })
    };
    if name.is_empty() {
        return broken("the name is empty".to_owned());
    }
    if name.len() > MAX_NAME_BYTES {
        return broken(format!(
            "it is {} bytes long, more than {MAX_NAME_BYTES}",
            name.len()
        ));
    }
    if let Some(character) = name
        .chars()
        .find(|&character| !(character.is_ascii_alphanumeric() || "_.-".contains(character)))
    {
        return broken(format!("character {character:?} is not allowed"));
    }
    if name.starts_with('.') || name.starts_with("__") {
        return broken("it starts with '.' or '__'".to_owned());
    }
    if name.ends_with('.') {
        return broken("it ends with '.'".to_owned());
    }
    let stem = name.split('.').next().unwrap_or(name);
    if WINDOWS_DEVICES
        .iter()
        .any(|device| device.eq_ignore_ascii_case(stem))
    {
        return broken(format!("'{stem}' is a reserved Windows device name"));
    }
    Ok(())
}

/// Return the name of the entry of directory `group` that equals `name` ignoring
/// ASCII case, if any. A missing directory has no entries.
///
/// # Errors
/// `KY-E110` if the directory exists but cannot be listed.
pub(crate) fn find_case_insensitive(group: &Path, name: &str) -> Result<Option<String>, KytoError> {
    let io_error = |error: std::io::Error| KytoError::StoreIo {
        operation: "list directory".to_owned(),
        path: group.display().to_string(),
        detail: error.to_string(),
    };
    let entries = match std::fs::read_dir(group) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error(error)),
    };
    for entry in entries {
        let entry_name = entry.map_err(io_error)?.file_name();
        if let Some(entry_name) = entry_name.to_str()
            && entry_name.eq_ignore_ascii_case(name)
        {
            return Ok(Some(entry_name.to_owned()));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{find_case_insensitive, validate_member_name};

    fn rule(name: &str) -> String {
        let error = validate_member_name("layer", name).unwrap_err();
        assert_eq!(error.code(), "KY-E102");
        error.to_string()
    }

    #[test]
    fn valid_names_pass() {
        for name in [
            "counts", "log1p", "X_pca", "a.b-c_d", "CONSOLE", "com10", "_x", "a__b",
        ] {
            validate_member_name("layer", name).unwrap();
        }
        validate_member_name("layer", &"a".repeat(64)).unwrap();
    }

    #[test]
    fn each_rule_is_reported() {
        assert!(rule("").contains("empty"));
        assert!(rule(&"a".repeat(65)).contains("65 bytes"));
        assert!(rule("log1p/counts").contains("'/' is not allowed"));
        assert!(rule("caf\u{e9}").contains("not allowed"));
        assert!(rule("a b").contains("' ' is not allowed"));
        assert!(rule("..").contains("starts with"));
        assert!(rule(".hidden").contains("starts with"));
        assert!(rule("__zarr").contains("starts with"));
        assert!(rule("name.").contains("ends with"));
        assert!(rule("con").contains("'con' is a reserved Windows device name"));
        assert!(rule("Lpt9.txt").contains("'Lpt9'"));
        assert!(rule("NUL.tar.gz").contains("'NUL'"));
    }

    #[test]
    fn finds_names_ignoring_case() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("counts")).unwrap();
        let found = find_case_insensitive(directory.path(), "Counts").unwrap();
        assert_eq!(found.as_deref(), Some("counts"));
        assert_eq!(
            find_case_insensitive(directory.path(), "log1p").unwrap(),
            None
        );
        let missing = directory.path().join("missing");
        assert_eq!(find_case_insensitive(&missing, "counts").unwrap(), None);
    }
}
