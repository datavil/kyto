//! The store's JSON files: `schema.json` (format.md §8) and `provenance.json` (§9).
//!
//! Does: build `schema.json` as a `kyto_core::AtlasSchema` (whose serde order is the
//! spec's key order) with member lists sorted by name (format.md §10.3), build
//! `provenance.json` with keys in spec order, and format the `run` timestamps as RFC 3339
//! UTC.
//! Does not: write files (see `store_writer`), or fill in a plan, seeds or timings
//! (tasks 4.3 and 8.5); v0 writes `null`, `{}` and `{}`.

use std::time::{SystemTime, UNIX_EPOCH};

use kyto_core::{AtlasSchema, ColumnSchema, LayerSchema, TableSchema, Tiling};
use serde_json::{Value, json};

/// The input file of an import, recorded as `source` in `provenance.json` (format.md §9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportSource {
    /// Input kind, for example `h5ad`, `10x_h5` or `mtx`.
    pub kind: String,
    /// The input path as the caller gave it.
    pub path: String,
    /// Lowercase hex SHA-256 of the input file.
    pub sha256: String,
}

/// The [`AtlasSchema`] of a new store (format.md §8), layers sorted by name in byte
/// order. The store has no embeddings or graphs yet (their writers come with 6.x).
pub(crate) fn atlas_schema(
    n_cells: u64,
    n_genes: u64,
    tiling: Tiling,
    cell_columns: Vec<ColumnSchema>,
    gene_columns: Vec<ColumnSchema>,
    mut layers: Vec<LayerSchema>,
) -> AtlasSchema {
    layers.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    AtlasSchema {
        n_cells,
        n_genes,
        tiling,
        cells: TableSchema {
            columns: cell_columns,
        },
        genes: TableSchema {
            columns: gene_columns,
        },
        layers,
        embeddings: Vec::new(),
        graphs: Vec::new(),
        modalities: Vec::new(),
    }
}

/// `provenance.json` (format.md §9). `run` holds the only varying values besides
/// `source.path` (§10.3).
pub(crate) fn provenance_json(
    source: Option<&ImportSource>,
    started_at: SystemTime,
    finished_at: SystemTime,
) -> Value {
    json!({
        "kyto_version": env!("CARGO_PKG_VERSION"),
        "dependencies": {
            "polars": polars::VERSION,
            "zarrs": zarrs::version::version_str(),
            "zstd": zstd_safe::version_string(),
        },
        "plan": null,
        "source": source.map(|source| json!({
            "kind": source.kind,
            "path": source.path,
            "sha256": source.sha256,
        })),
        "seeds": {},
        "device": "cpu",
        "run": {
            "started_at": rfc3339_utc(started_at),
            "finished_at": rfc3339_utc(finished_at),
            "timings": {},
        },
    })
}

/// `YYYY-MM-DDTHH:MM:SSZ`, whole seconds, UTC. Times before 1970 give the epoch.
pub(crate) fn rfc3339_utc(time: SystemTime) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let days = seconds / 86_400;
    let day_seconds = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3600,
        day_seconds % 3600 / 60,
        day_seconds % 60
    )
}

/// Proleptic Gregorian date of `days` since 1970-01-01 (H. Hinnant's `civil_from_days`).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::rfc3339_utc;

    #[test]
    fn timestamps_are_rfc3339_utc() {
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        let leap_day = UNIX_EPOCH + Duration::from_secs(951_782_400 + 3_723);
        assert_eq!(rfc3339_utc(leap_day), "2000-02-29T01:02:03Z");
        let spec_example = UNIX_EPOCH + Duration::from_secs(1_794_045_600);
        assert_eq!(rfc3339_utc(spec_example), "2026-11-07T10:00:00Z");
        let year_end = UNIX_EPOCH + Duration::from_secs(1_798_761_599);
        assert_eq!(rfc3339_utc(year_end), "2026-12-31T23:59:59Z");
    }
}
