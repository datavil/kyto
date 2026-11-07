# Polars Parquet for the metadata tables (task 2.3)

**Broke:** `ParquetWriter::with_row_group_size(n).finish()` does not give row groups of
exactly `n` rows: it re-chunks the frame and merged a short last group into the one
before (7 rows at `n = 2` gave groups of 2, 2, 3), breaking format.md §5.7.

**Works:** `ParquetWriter::batched(schema)`, then one `write_batch` per `n`-row slice of
a rechunked frame: each single-chunk batch is one row group (`table_io::write_table`;
`tests/table_format.rs::parquet_uses_zstd_statistics_and_block_row_groups` guards it).

**Also:**
- Without the `dtype-i8`, `dtype-i16`, `dtype-u8`, `dtype-u16` features the polars crate
  has no Int8/Int16/UInt8/UInt16 series at all; all four are needed for §5.3.
- polars 0.55 does not re-export `ZstdLevel`; `ParquetCompression::Zstd(None)` is level 3.
- polars publishes no `rust-version` but needs a recent stable Rust (sysinfo 0.39 wants
  1.95), so the workspace `rust-version = "1.85"` no longer holds; CI builds on stable.
- Polars `Enum` has no `ordered` flag; §8 keeps it in `schema.json` (`TableColumn`).
- Rust polars 0.55.2 files read in Python polars 2.x with the same dtypes, Enum categories
  included (`tests/table_round_trip.rs::python_polars_reads_the_dtypes`, needs `uv run`).
