//! kyto-error: every kyto error, one enum generated from `errors.toml` (task 8.1, R10).
//!
//! Does: define [`KytoError`] with one variant per registered `KY-Exxx` code, each
//! with structured fields, a message (`Display`), a fix hint and a docs link. Every
//! kyto crate depends on this crate and returns `KytoError` across its boundary.
//! Does not: convert to Python (that is `kyto-py`) or hold any other logic.
//!
//! `registry.rs` is generated: edit `errors.toml` and run
//! `uv run python scripts/error_registry.py`.
#![forbid(unsafe_code)]

// Generated code is left as the generator writes it, so `--check` stays exact.
#[rustfmt::skip]
mod registry;

pub use registry::KytoError;

/// The value of one structured error field, for conversion to Python (kyto-py).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldValue {
    /// A `str` field.
    Str(String),
    /// An `int` field (non-negative).
    Int(u64),
    /// A `list[str]` field.
    StrList(Vec<String>),
}

/// Render a list as `'a', 'b', 'c'`, the way every error message shows lists.
pub fn quoted_list(items: &[String]) -> String {
    items
        .iter()
        .map(|item| format!("'{item}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::{FieldValue, KytoError, quoted_list};
    use std::collections::HashSet;

    #[test]
    fn examples_cover_every_code_once_in_order() {
        let codes: Vec<&str> = KytoError::examples().iter().map(KytoError::code).collect();
        assert_eq!(codes, KytoError::CODES);
        assert_eq!(codes.iter().collect::<HashSet<_>>().len(), codes.len());
    }

    #[test]
    fn display_starts_with_the_code() {
        for error in KytoError::examples() {
            let text = error.to_string();
            assert!(text.starts_with(&format!("{}: ", error.code())), "{text}");
            assert!(!error.hint().is_empty());
            let anchor = format!("#{}", error.code().to_lowercase());
            assert!(error.docs_url().ends_with(&anchor));
        }
    }

    #[test]
    fn ambiguous_name_says_what_and_how_to_fix() {
        let error = KytoError::AmbiguousName {
            name: "CD3".to_owned(),
            candidates: vec!["log1p:CD3".to_owned(), "meta:CD3".to_owned()],
        };
        assert_eq!(error.code(), "KY-E201");
        assert_eq!(error.python_class(), "ResolutionError");
        assert_eq!(
            error.to_string(),
            "KY-E201: ambiguous name 'CD3': it matches 'log1p:CD3', 'meta:CD3'"
        );
        assert_eq!(
            error.hint(),
            "pass the full name, one of 'log1p:CD3', 'meta:CD3'"
        );
        assert_eq!(
            error.fields(),
            vec![
                ("name", FieldValue::Str("CD3".to_owned())),
                (
                    "candidates",
                    FieldValue::StrList(vec!["log1p:CD3".to_owned(), "meta:CD3".to_owned()])
                ),
            ]
        );
    }

    #[test]
    fn int_fields_render_plainly() {
        let error = KytoError::MemoryLimit {
            required_bytes: 10,
            limit_bytes: 4,
        };
        assert_eq!(
            error.to_string(),
            "KY-E401: the plan needs at least 10 bytes but memory_limit is 4 bytes"
        );
        assert!(error.hint().contains("at least 10 bytes"));
        assert_eq!(error.fields()[1], ("limit_bytes", FieldValue::Int(4)));
    }

    #[test]
    fn quoted_list_handles_empty_and_single() {
        assert_eq!(quoted_list(&[]), "");
        assert_eq!(quoted_list(&["a".to_owned()]), "'a'");
    }
}
