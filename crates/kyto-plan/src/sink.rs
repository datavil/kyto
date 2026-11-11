//! The node that ends a plan by writing it: a kyto store or an `.h5ad` file.
//!
//! Does: hold where and how to write. Its output schema is its input's: what the
//! sink writes. A plan that ends in a sink cannot be extended (`KY-E216`). Changes no
//! axis; per-cell (the writers stream blocks, 2.5 and 2.7).
//! Does not: write (5.x runs the plan into the 2.5/2.7 writers), or decide the
//! lineage of written columns (3.8's open item, for the 5.x sink).

use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// Where a sink writes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "format")]
pub enum Sink {
    /// A kyto store (`sink_zarr`).
    Zarr {
        /// The store path or URL.
        path: String,
    },
    /// An `.h5ad` file for scanpy (`sink_h5ad`).
    H5ad {
        /// The file path.
        path: String,
    },
}

impl Sink {
    /// The target path.
    pub fn path(&self) -> &str {
        match self {
            Self::Zarr { path } | Self::H5ad { path } => path,
        }
    }
}

impl LogicalNode for Sink {
    fn name(&self) -> &'static str {
        match self {
            Self::Zarr { .. } => "SINK_ZARR",
            Self::H5ad { .. } => "SINK_H5AD",
        }
    }

    fn method(&self) -> &'static str {
        match self {
            Self::Zarr { .. } => "sink_zarr",
            Self::H5ad { .. } => "sink_h5ad",
        }
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.path())
    }
}

impl Transform for Sink {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone()
    }
}

impl Plan {
    /// This plan ending in `sink`. Nothing can follow it.
    ///
    /// # Errors
    /// `KY-E215` for an empty path; `KY-E216` if this plan already ends in a sink.
    pub fn sink(self: &Arc<Self>, sink: Sink) -> Result<Arc<Self>, KytoError> {
        self.check_open(sink.method())?;
        Self::check_name(sink.method(), "path", sink.path())?;
        Ok(self.extend(sink, Self::Sink))
    }
}
