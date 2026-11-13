//! The graphs Leiden and Louvain work on (task 6.8).
//!
//! Does: [`CellGraph`] is a borrowed, checked CSR view of a weighted cell graph (a
//! [`ConnectivityGraph`], a [`NeighborGraph`] or scanpy's arrays). [`Level`] is one
//! level of the optimisation as leidenalg holds it: the graph as directed arcs (scanpy
//! builds a directed igraph graph from the matrix), `f64` weights, self loops apart,
//! each node's out and in strength and its size (the cells it stands for), and the
//! incoming arcs only when the graph is not symmetric (a symmetric graph's are its
//! outgoing ones). [`Level::aggregate`] collapses a level by a partition (each
//! community a node, the arcs between two communities summed, those inside one its
//! self loop), summing in a fixed order (communities, then members, by index).
//! Does not: optimise anything (see `leiden.rs`).

use kyto_error::KytoError;

use crate::connectivities::ConnectivityGraph;
use crate::neighbors::NeighborGraph;

/// A weighted cell graph in CSR form: row `i` holds the arcs from cell `i`.
#[derive(Debug, Clone, Copy)]
pub struct CellGraph<'a> {
    indptr: &'a [u64],
    indices: &'a [u32],
    weights: &'a [f32],
}

impl<'a> CellGraph<'a> {
    /// A graph of `indptr.len() - 1` cells, checked.
    ///
    /// # Errors
    /// `KY-E215` unless `indptr` starts at 0, never decreases and ends at the number of
    /// `indices`, `weights` has as many values, every index lies below the number of
    /// cells (fewer than 2^32), and every weight is finite and not negative.
    pub fn new(
        indptr: &'a [u64],
        indices: &'a [u32],
        weights: &'a [f32],
    ) -> Result<Self, KytoError> {
        let refuse = |parameter: &str, value: String, problem: String, accepted: &str| {
            KytoError::InvalidArgument {
                function: "CellGraph::new".to_owned(),
                parameter: parameter.to_owned(),
                value,
                problem,
                accepted: accepted.to_owned(),
            }
        };
        let n_cells = indptr.len().saturating_sub(1);
        // Cells and communities are numbered in `u32`, 0..n included.
        if u32::try_from(n_cells).is_err() {
            return Err(refuse(
                "indptr",
                format!("{n_cells} rows"),
                "cells are numbered in uint32 (format.md §7)".to_owned(),
                "fewer than 2^32 rows",
            ));
        }
        let well_formed = indptr.first() == Some(&0)
            && indptr.windows(2).all(|pair| pair[0] <= pair[1])
            && indptr.last().copied() == Some(indices.len() as u64);
        if !well_formed {
            return Err(refuse(
                "indptr",
                format!("{} offsets for {} entries", indptr.len(), indices.len()),
                "it does not run from 0 to the number of entries without decreasing".to_owned(),
                "n_cells + 1 non-decreasing offsets from 0 to the number of entries",
            ));
        }
        if weights.len() != indices.len() {
            return Err(refuse(
                "weights",
                format!("{} weights", weights.len()),
                format!("there are {} indices", indices.len()),
                "one weight an entry",
            ));
        }
        if let Some(&index) = indices.iter().find(|&&index| index as usize >= n_cells) {
            return Err(refuse(
                "indices",
                index.to_string(),
                format!("the graph has {n_cells} cells"),
                "row positions below the number of cells",
            ));
        }
        if let Some(entry) = weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
            return Err(refuse(
                "weights",
                weights[entry].to_string(),
                format!("entry {entry} is not a finite weight of at least 0"),
                "finite weights of at least 0",
            ));
        }
        Ok(Self {
            indptr,
            indices,
            weights,
        })
    }

    /// The number of cells (rows and columns).
    pub fn n_cells(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// The arcs from cell `row`: (cell, weight).
    fn row(&self, row: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let range = self.indptr[row] as usize..self.indptr[row + 1] as usize;
        self.indices[range.clone()]
            .iter()
            .copied()
            .zip(self.weights[range].iter().copied())
    }
}

impl<'a> From<&'a ConnectivityGraph> for CellGraph<'a> {
    fn from(graph: &'a ConnectivityGraph) -> Self {
        Self {
            indptr: graph.indptr(),
            indices: graph.indices(),
            weights: graph.weights(),
        }
    }
}

impl<'a> From<&'a NeighborGraph> for CellGraph<'a> {
    fn from(graph: &'a NeighborGraph) -> Self {
        Self {
            indptr: graph.indptr(),
            indices: graph.indices(),
            weights: graph.distances(),
        }
    }
}

/// Arcs in CSR form, without self loops.
#[derive(Debug, Clone, Default)]
pub(crate) struct Arcs {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    weights: Vec<f64>,
}

impl Arcs {
    /// The arcs of node `node`: (node, weight).
    pub(crate) fn of(&self, node: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let range = self.indptr[node]..self.indptr[node + 1];
        self.indices[range.clone()]
            .iter()
            .copied()
            .zip(self.weights[range].iter().copied())
    }

    /// The same arcs reversed (row `j` holds the arcs into `j`), rows in source order.
    fn transposed(&self, n: usize) -> Self {
        let mut counts = vec![0_usize; n + 1];
        for &index in &self.indices {
            counts[index as usize + 1] += 1;
        }
        for node in 0..n {
            counts[node + 1] += counts[node];
        }
        let mut next = counts.clone();
        let mut indices = vec![0_u32; self.indices.len()];
        let mut weights = vec![0.0; self.weights.len()];
        for source in 0..n {
            for (target, weight) in self.of(source) {
                let slot = &mut next[target as usize];
                indices[*slot] = source as u32;
                weights[*slot] = weight;
                *slot += 1;
            }
        }
        Self {
            indptr: counts,
            indices,
            weights,
        }
    }

    /// Whether every arc has its reverse of the same weight (same bits), so the arcs
    /// into a node are its arcs out.
    fn is_symmetric(&self, n: usize) -> bool {
        let transposed = self.transposed(n);
        // Rows of the transpose are in source order; compare each row as a sorted list.
        (0..n).all(|node| {
            let mut out: Vec<(u32, u64)> = self
                .of(node)
                .map(|(index, weight)| (index, weight.to_bits()))
                .collect();
            let mut into: Vec<(u32, u64)> = transposed
                .of(node)
                .map(|(index, weight)| (index, weight.to_bits()))
                .collect();
            out.sort_unstable();
            into.sort_unstable();
            out == into
        })
    }
}

/// One level of the optimisation: a directed weighted graph of nodes, each standing
/// for `size` cells.
#[derive(Debug, Clone)]
pub(crate) struct Level {
    pub(crate) n: usize,
    pub(crate) out: Arcs,
    /// The arcs into each node; `None` when the graph is symmetric (they are `out`).
    pub(crate) into: Option<Arcs>,
    pub(crate) self_weight: Vec<f64>,
    pub(crate) out_strength: Vec<f64>,
    pub(crate) in_strength: Vec<f64>,
    pub(crate) size: Vec<u64>,
}

impl Level {
    /// The cells of `graph` as the first level: each weight as `f64` (1 when not
    /// `weighted`, scanpy's `use_weights=False`), arcs of weight 0 left out (scanpy
    /// builds the igraph graph from the matrix's non-zeros), self loops apart.
    pub(crate) fn from_graph(graph: CellGraph<'_>, weighted: bool) -> Self {
        let n = graph.n_cells();
        let mut out = Arcs {
            indptr: Vec::with_capacity(n + 1),
            indices: Vec::with_capacity(graph.indices.len()),
            weights: Vec::with_capacity(graph.indices.len()),
        };
        out.indptr.push(0);
        let mut self_weight = vec![0.0; n];
        for (row, loops) in self_weight.iter_mut().enumerate() {
            for (column, weight) in graph.row(row) {
                if weight == 0.0 {
                    continue;
                }
                let weight = if weighted { f64::from(weight) } else { 1.0 };
                if column as usize == row {
                    *loops += weight;
                } else {
                    out.indices.push(column);
                    out.weights.push(weight);
                }
            }
            out.indptr.push(out.indices.len());
        }
        let into = (!out.is_symmetric(n)).then(|| out.transposed(n));
        Self::with_strengths(n, out, into, self_weight, vec![1; n])
    }

    /// A level from its arcs, computing each node's strengths (arcs in row order, then
    /// the self loop).
    fn with_strengths(
        n: usize,
        out: Arcs,
        into: Option<Arcs>,
        self_weight: Vec<f64>,
        size: Vec<u64>,
    ) -> Self {
        let strengths = |arcs: &Arcs| -> Vec<f64> {
            (0..n)
                .map(|node| arcs.of(node).map(|(_, w)| w).sum::<f64>() + self_weight[node])
                .collect()
        };
        let out_strength = strengths(&out);
        let in_strength = match &into {
            Some(into) => strengths(into),
            None => out_strength.clone(),
        };
        Self {
            n,
            out,
            into,
            self_weight,
            out_strength,
            in_strength,
            size,
        }
    }

    /// The arcs into each node.
    pub(crate) fn incoming(&self) -> &Arcs {
        self.into.as_ref().unwrap_or(&self.out)
    }

    /// The total weight of the arcs (self loops included): leidenalg's `m`.
    pub(crate) fn total_weight(&self) -> f64 {
        self.out_strength.iter().sum()
    }

    /// This level collapsed by `membership` (communities `0..n_communities`, each with
    /// at least one node): node `c` of the result is community `c`.
    pub(crate) fn aggregate(&self, membership: &[u32], n_communities: usize) -> Self {
        // The members of each community, by node index (a counting sort).
        let mut starts = vec![0_usize; n_communities + 1];
        for &community in membership {
            starts[community as usize + 1] += 1;
        }
        for community in 0..n_communities {
            starts[community + 1] += starts[community];
        }
        let mut next = starts.clone();
        let mut members = vec![0_u32; self.n];
        for (node, &community) in membership.iter().enumerate() {
            members[next[community as usize]] = node as u32;
            next[community as usize] += 1;
        }
        let mut out = Arcs {
            indptr: Vec::with_capacity(n_communities + 1),
            indices: Vec::new(),
            weights: Vec::new(),
        };
        out.indptr.push(0);
        let mut self_weight = vec![0.0; n_communities];
        let mut size = vec![0_u64; n_communities];
        let mut weight_to = vec![0.0_f64; n_communities];
        let mut seen = vec![false; n_communities];
        let mut touched: Vec<u32> = Vec::new();
        for community in 0..n_communities {
            for &node in &members[starts[community]..starts[community + 1]] {
                let node = node as usize;
                size[community] += self.size[node];
                self_weight[community] += self.self_weight[node];
                for (target, weight) in self.out.of(node) {
                    let target = membership[target as usize];
                    if target as usize == community {
                        self_weight[community] += weight;
                    } else {
                        if !seen[target as usize] {
                            seen[target as usize] = true;
                            touched.push(target);
                        }
                        weight_to[target as usize] += weight;
                    }
                }
            }
            touched.sort_unstable();
            for &target in &touched {
                out.indices.push(target);
                out.weights.push(weight_to[target as usize]);
                weight_to[target as usize] = 0.0;
                seen[target as usize] = false;
            }
            touched.clear();
            out.indptr.push(out.indices.len());
        }
        let into = self.into.as_ref().map(|_| out.transposed(n_communities));
        Self::with_strengths(n_communities, out, into, self_weight, size)
    }
}
