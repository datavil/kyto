//! The columns of the cell and gene tables as `schema.json` lists them (format.md §5.3,
//! §5.4, §8).
//!
//! Does: name the v0 column dtypes and (de)serialize one `columns` entry: `name`,
//! `dtype`, then `n_categories` and `ordered` exactly when the dtype is `Enum`, then the
//! optional `lineage`.
//! Does not: map Polars dtypes (kyto-store does, it owns the Parquet files) or check
//! column names (format.md §5.6, checked where tables are written).

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

/// A column dtype allowed in v0 tables (format.md §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnDtype {
    /// `Boolean`.
    Boolean,
    /// `Int8`.
    Int8,
    /// `Int16`.
    Int16,
    /// `Int32`.
    Int32,
    /// `Int64`.
    Int64,
    /// `UInt8`.
    UInt8,
    /// `UInt16`.
    UInt16,
    /// `UInt32`.
    UInt32,
    /// `UInt64`.
    UInt64,
    /// `Float32`.
    Float32,
    /// `Float64`.
    Float64,
    /// `String`: the ID column only.
    String,
    /// `Enum` with a fixed category list (format.md §5.4).
    Enum {
        /// Number of categories, used or not.
        n_categories: u64,
        /// Whether the categories are ordered. Polars `Enum` does not carry this; it is
        /// kept in `schema.json` only.
        ordered: bool,
    },
}

/// Every dtype name `schema.json` accepts, for error messages.
const DTYPE_NAMES: &[&str] = &[
    "Boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32",
    "Float64", "String", "Enum",
];

/// The keys a non-`Enum` column may have.
const PLAIN_KEYS: &[&str] = &["name", "dtype", "lineage"];

impl ColumnDtype {
    /// The Polars dtype name, as `schema.json` writes it (format.md §8).
    pub fn name(self) -> &'static str {
        match self {
            Self::Boolean => "Boolean",
            Self::Int8 => "Int8",
            Self::Int16 => "Int16",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::UInt8 => "UInt8",
            Self::UInt16 => "UInt16",
            Self::UInt32 => "UInt32",
            Self::UInt64 => "UInt64",
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::String => "String",
            Self::Enum { .. } => "Enum",
        }
    }

    /// The non-`Enum` dtype named `name`, or `None` (also for `"Enum"`, which needs its
    /// category count and `ordered` flag).
    fn plain_from_name(name: &str) -> Option<Self> {
        Some(match name {
            "Boolean" => Self::Boolean,
            "Int8" => Self::Int8,
            "Int16" => Self::Int16,
            "Int32" => Self::Int32,
            "Int64" => Self::Int64,
            "UInt8" => Self::UInt8,
            "UInt16" => Self::UInt16,
            "UInt32" => Self::UInt32,
            "UInt64" => Self::UInt64,
            "Float32" => Self::Float32,
            "Float64" => Self::Float64,
            "String" => Self::String,
            _ => return None,
        })
    }
}

/// Where a stored derived column came from (format.md §8; its use is task 3.8).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lineage {
    /// The layer the column was computed from.
    pub layer: String,
    /// The stable plan node ID that computed it (task 4.3).
    pub node: String,
}

/// One column of a cell or gene table: one entry of `columns` in `schema.json`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(into = "RawColumn")]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// Column dtype.
    pub dtype: ColumnDtype,
    /// Where a derived column came from; `None` for imported columns.
    pub lineage: Option<Lineage>,
}

impl ColumnSchema {
    /// A column without lineage.
    pub fn new(name: impl Into<String>, dtype: ColumnDtype) -> Self {
        Self {
            name: name.into(),
            dtype,
            lineage: None,
        }
    }
}

/// The JSON shape of a column, in key order (format.md §8).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawColumn {
    name: String,
    dtype: String,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    n_categories: Option<u64>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    ordered: Option<bool>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    lineage: Option<Lineage>,
}

/// An optional key that, when present, holds a value: `null` is rejected, so every
/// accepted column writes back exactly as read.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

impl<'de> Deserialize<'de> for ColumnSchema {
    /// Reads the raw JSON object, then checks the `Enum` keys with serde's own error kinds
    /// (callers wrap them in a coded error, for `schema.json` `KY-E122`).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let RawColumn {
            name,
            dtype,
            n_categories,
            ordered,
            lineage,
        } = RawColumn::deserialize(deserializer)?;
        let dtype = if dtype == "Enum" {
            ColumnDtype::Enum {
                n_categories: n_categories
                    .ok_or_else(|| D::Error::missing_field("n_categories"))?,
                ordered: ordered.ok_or_else(|| D::Error::missing_field("ordered"))?,
            }
        } else {
            let plain = ColumnDtype::plain_from_name(&dtype)
                .ok_or_else(|| D::Error::unknown_variant(&dtype, DTYPE_NAMES))?;
            if n_categories.is_some() {
                return Err(D::Error::unknown_field("n_categories", PLAIN_KEYS));
            }
            if ordered.is_some() {
                return Err(D::Error::unknown_field("ordered", PLAIN_KEYS));
            }
            plain
        };
        Ok(Self {
            name,
            dtype,
            lineage,
        })
    }
}

impl From<ColumnSchema> for RawColumn {
    fn from(column: ColumnSchema) -> Self {
        let (n_categories, ordered) = match column.dtype {
            ColumnDtype::Enum {
                n_categories,
                ordered,
            } => (Some(n_categories), Some(ordered)),
            _ => (None, None),
        };
        Self {
            name: column.name,
            dtype: column.dtype.name().to_owned(),
            n_categories,
            ordered,
            lineage: column.lineage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<ColumnSchema, String> {
        serde_json::from_str(text).map_err(|error| error.to_string())
    }

    #[test]
    fn writes_enum_keys_only_for_enums() {
        let plain = ColumnSchema::new("n_genes", ColumnDtype::UInt32);
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            r#"{"name":"n_genes","dtype":"UInt32"}"#
        );
        let mut categorical = ColumnSchema::new(
            "louvain",
            ColumnDtype::Enum {
                n_categories: 3,
                ordered: false,
            },
        );
        categorical.lineage = Some(Lineage {
            layer: "counts".to_owned(),
            node: "n3".to_owned(),
        });
        assert_eq!(
            serde_json::to_string(&categorical).unwrap(),
            r#"{"name":"louvain","dtype":"Enum","n_categories":3,"ordered":false,"lineage":{"layer":"counts","node":"n3"}}"#
        );
    }

    #[test]
    fn rejects_inconsistent_columns() {
        for (text, needle) in [
            (
                r#"{"name":"a","dtype":"Enum","n_categories":2}"#,
                "missing field `ordered`",
            ),
            (
                r#"{"name":"a","dtype":"Enum","ordered":true}"#,
                "missing field `n_categories`",
            ),
            (
                r#"{"name":"a","dtype":"Int8","ordered":true}"#,
                "unknown field `ordered`",
            ),
            (
                r#"{"name":"a","dtype":"Categorical"}"#,
                "unknown variant `Categorical`",
            ),
            (
                r#"{"name":"a","dtype":"Int8","extra":1}"#,
                "unknown field `extra`",
            ),
            (
                r#"{"name":"a","dtype":"Int8","n_categories":1}"#,
                "unknown field `n_categories`",
            ),
            (r#"{"dtype":"Int8"}"#, "missing field `name`"),
            (
                r#"{"name":"a","dtype":"Int8","lineage":null}"#,
                "invalid type: null",
            ),
            (
                r#"{"name":"a","dtype":"Int8","ordered":null}"#,
                "invalid type: null",
            ),
            (
                r#"{"name":"a","dtype":"Enum","n_categories":-1,"ordered":true}"#,
                "invalid value",
            ),
        ] {
            let error = parse(text).unwrap_err();
            assert!(error.contains(needle), "{text}: {error}");
        }
    }

    #[test]
    fn every_dtype_name_round_trips() {
        let dtypes = [
            ColumnDtype::Boolean,
            ColumnDtype::Int8,
            ColumnDtype::Int16,
            ColumnDtype::Int32,
            ColumnDtype::Int64,
            ColumnDtype::UInt8,
            ColumnDtype::UInt16,
            ColumnDtype::UInt32,
            ColumnDtype::UInt64,
            ColumnDtype::Float32,
            ColumnDtype::Float64,
            ColumnDtype::String,
            ColumnDtype::Enum {
                n_categories: u64::MAX,
                ordered: true,
            },
        ];
        for dtype in dtypes {
            let column = ColumnSchema::new("x", dtype);
            let text = serde_json::to_string(&column).unwrap();
            assert_eq!(parse(&text).unwrap(), column, "{text}");
        }
    }
}
