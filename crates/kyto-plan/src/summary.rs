//! The plan in one line (task 7.7): node count, sizes and layers, for `repr()`.
//!
//! Does: format what the plan already holds (its node chain and its cached output
//! schema). A size that only the data can tell (after a filter) is shown as `?`.
//! Does not: read the store, optimize the plan or estimate anything.

use std::fmt::Write as _;

use crate::plan::Plan;

impl Plan {
    /// The plan in one line, without reading anything:
    /// `plan: 7 nodes · cells: ? · genes: 2000 · layers: counts, log1p`.
    ///
    /// `nodes` counts this node down to the scan. `cells` and `genes` are exact, or `?`
    /// when only an upper bound is known (after a filter on that axis). `layers` are the
    /// plan's output layers in schema order, `none` if there are none.
    pub fn summary(&self) -> String {
        let schema = self.output_schema();
        let nodes = self.nodes().count();
        let mut line = format!(
            "plan: {nodes} {} · cells: {} · genes: {} · layers: ",
            if nodes == 1 { "node" } else { "nodes" },
            size(schema.n_cells()),
            size(schema.n_genes()),
        );
        let layers = &schema.atlas().layers;
        if layers.is_empty() {
            line.push_str("none");
        }
        for (position, layer) in layers.iter().enumerate() {
            let separator = if position > 0 { ", " } else { "" };
            // A stored name has no control character (format.md 2.1), but a schema given
            // through `Plan::scan` or the plan JSON is not checked for that: escape, so
            // the line stays one line. Writing to a `String` cannot fail.
            let _ = write!(line, "{separator}{}", layer.name.escape_debug());
        }
        line
    }
}

/// An exact size, or `?` when only a bound is known.
fn size(exact: Option<u64>) -> String {
    exact.map_or_else(|| "?".to_owned(), |exact| exact.to_string())
}
