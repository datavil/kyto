//! The syntax of a feature name as passed to `feature(...)`: short (`CD3E`, `cluster`)
//! or full (`log1p:CD3E`, `meta:cluster`).
//!
//! Does: split a name at its first `:` into a prefix and a name, and reject names that
//! can never resolve (empty, empty prefix or name, a prefix no layer, embedding or
//! `meta` can have) with `KY-E208`.
//! Does not: look at the schema. Which feature a name means (a layer and gene, a table
//! column, an embedding component), and whether a short name is ambiguous (`KY-E201`),
//! is task 3.5, the feature resolver.

use std::fmt;

use kyto_error::KytoError;

/// A feature name, parsed but not resolved.
///
/// Its `Display` is the text it was parsed from, so `parse(name.to_string())` gives the
/// same name back.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureName {
    /// A name without a `:`, such as `CD3E` or `cluster`. Resolving it needs the schema
    /// and, for genes, the plan's default layer (task 3.5).
    Short {
        /// The name as written.
        name: String,
    },
    /// A name with a prefix, such as `log1p:CD3E` or `meta:cluster`: split at the first
    /// `:`, so the name part may contain `:` itself.
    Full {
        /// The part before the first `:`, such as `log1p` or `meta`.
        prefix: String,
        /// The part after it, such as `CD3E` or `cluster`.
        name: String,
    },
}

impl FeatureName {
    /// Parse `text` as a short or full feature name.
    ///
    /// # Errors
    /// `KY-E208` if `text` is empty, has an empty part around its first `:`, or has a
    /// prefix that is not a valid store name (`[A-Za-z0-9_.-]+`, docs/format.md §2).
    pub fn parse(text: &str) -> Result<Self, KytoError> {
        let invalid = |problem: String| KytoError::FeatureNameSyntax {
            name: text.to_owned(),
            problem,
        };
        if text.is_empty() {
            return Err(invalid("it is empty".to_owned()));
        }
        let Some((prefix, name)) = text.split_once(':') else {
            return Ok(Self::Short {
                name: text.to_owned(),
            });
        };
        if prefix.is_empty() {
            return Err(invalid("nothing comes before ':'".to_owned()));
        }
        if name.is_empty() {
            return Err(invalid(format!("nothing follows '{prefix}:'")));
        }
        if !prefix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
        {
            return Err(invalid(format!(
                "the prefix '{prefix}' cannot be a layer or embedding name or 'meta': \
                 names match [A-Za-z0-9_.-]+"
            )));
        }
        Ok(Self::Full {
            prefix: prefix.to_owned(),
            name: name.to_owned(),
        })
    }
}

impl fmt::Display for FeatureName {
    /// The text the name was parsed from.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short { name } => formatter.write_str(name),
            Self::Full { prefix, name } => write!(formatter, "{prefix}:{name}"),
        }
    }
}
