//! Member names: the rules of format.md §2.1 for layers, embeddings and graphs.
//!
//! Does: check one name's syntax (the rules are `kyto_core::member_name_problem`), and
//! find a member of a group directory whose name equals a given one case-insensitively.
//! Does not: check column names (format.md §5.6, task 2.3).

use std::path::Path;

use kyto_error::KytoError;

/// Check a member name against format.md §2.1 (the rules live in kyto-core, shared
/// with the plan).
///
/// `kind` is `layer`, `embedding` or `graph`, used in the error.
///
/// # Errors
/// `KY-E102` naming the rule the name breaks.
pub fn validate_member_name(kind: &str, name: &str) -> Result<(), KytoError> {
    match kyto_core::member_name_problem(name) {
        None => Ok(()),
        Some(rule) => Err(KytoError::InvalidMemberName {
            kind: kind.to_owned(),
            name: name.to_owned(),
            rule,
        }),
    }
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
