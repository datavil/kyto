//! The two metadata tables of a store, `cells.parquet` and `genes.parquet` (format.md §5).
//!
//! Does: name each table's axis, file, ID column and row group size.
//! Does not: check or read a table (see `table_check` and `table_io`).

use crate::tiling::Tiling;

/// Which metadata table: one row per cell or one row per gene (format.md §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableAxis {
    /// `cells.parquet`, one row per matrix row.
    Cells,
    /// `genes.parquet`, one row per matrix column.
    Genes,
}

impl TableAxis {
    /// The axis name used in errors: `cells` or `genes`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cells => "cells",
            Self::Genes => "genes",
        }
    }

    /// The table's file name at the store root.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Cells => "cells.parquet",
            Self::Genes => "genes.parquet",
        }
    }

    /// The ID column, which comes first (format.md §5.2).
    pub fn id_column(self) -> &'static str {
        match self {
            Self::Cells => "cell_id",
            Self::Genes => "gene_id",
        }
    }

    /// Rows per Parquet row group: one cell block for cells, one gene block for genes
    /// (format.md §5.7).
    pub fn row_group_size(self, tiling: Tiling) -> u64 {
        match self {
            Self::Cells => tiling.cell_block_size,
            Self::Genes => tiling.gene_block_size,
        }
    }
}

/// Column names no non-ID column of either table may use (format.md §5.2).
pub(crate) const RESERVED_ID_COLUMNS: [&str; 2] = ["cell_id", "gene_id"];
