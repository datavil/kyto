//! Leiden and Louvain clustering of a cell graph (task 6.8), as scanpy runs them.
//!
//! Does: [`cluster`] optimises leidenalg's `RBConfigurationVertexPartition` (modularity
//! with a resolution `γ`) of the graph as scanpy hands it over (`directed=True`: each
//! stored entry an arc), with leidenalg's moves:
//! - **Move nodes** (both methods): nodes in an order shuffled by the seed, in a
//!   queue; each goes to the neighbouring community (or an empty one, if it is not
//!   alone) with the largest gain above `10 ε`, and its neighbours outside that
//!   community rejoin the queue. The gain of moving `v` to `c` is
//!   `w(v, c) - γ (k_out(v) K_in(c) + k_in(v) K_out(c)) / m`, against staying, `w`
//!   counting arcs both ways, `K` the community's strengths without `v`, `m` the
//!   total weight.
//! - **Leiden** (`tl.leiden`: weighted, `n_iterations=-1`): after moving, each
//!   community is refined (leidenalg's `merge_nodes_constrained`: nodes in a shuffled
//!   order, each still alone joins the neighbouring refined community inside its
//!   community with the largest gain of at least 0), the graph is collapsed by the
//!   refined communities, starting from the moved ones, while that shrinks it; whole
//!   iterations repeat until one moves no node.
//! - **Louvain** (`tl.louvain`'s `vtraag` flavour: every arc of weight 1, scanpy's
//!   `use_weights=False`): move, collapse by the communities, until a level moves no
//!   node.
//!
//! Labels number the clusters by size, largest first (leidenalg's
//! `renumber_communities`), ties by their first cell. One thread, `f64` sums in a fixed
//! order, randomness from SplitMix64 on the seed: the same labels on any thread count
//! (R11) and every run.
//! Does not: copy leidenalg's random number stream (labels agree with it up to the
//! partition found, reported as an adjusted Rand index by the parity tests), or offer
//! scanpy's `n_iterations`, `use_weights`, `directed`, `partition_type`,
//! `restrict_to` or the `igraph` flavour (deviations.md).

use std::collections::VecDeque;
use std::sync::Arc;

use kyto_error::KytoError;
use kyto_plan::{Cluster, ClusterMethod};

use crate::cluster_graph::{CellGraph, Level};
use crate::pca::mix;
use crate::run_control::RunControl;

/// The smallest gain a move must exceed (leidenalg's `10 * DBL_EPSILON`).
const MIN_GAIN: f64 = 10.0 * f64::EPSILON;

/// Queue steps between cancellation checks.
const CHECK_EVERY: usize = 1 << 16;

/// A clustering of a graph's cells.
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    labels: Vec<u32>,
    n_clusters: u32,
    quality: f64,
}

impl Clustering {
    /// Each cell's cluster, `0..n_clusters()`, cluster 0 the largest.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// The number of clusters.
    pub fn n_clusters(&self) -> u32 {
        self.n_clusters
    }

    /// The partition's quality: modularity with the resolution on the graph clustered
    /// (unweighted for Louvain), `sum_c (W_c - γ K_out(c) K_in(c) / m) / m`; see
    /// [`partition_quality`].
    pub fn quality(&self) -> f64 {
        self.quality
    }
}

/// What a clustering node computed: its clustering of the graph's cells.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterStatistics {
    pub(crate) cells: Arc<Vec<u64>>,
    pub(crate) clustering: Clustering,
}

impl ClusterStatistics {
    /// The store table position of each clustered cell (the graph's rows).
    pub fn cells(&self) -> &[u64] {
        &self.cells
    }

    /// The clustering, in the order of [`ClusterStatistics::cells`].
    pub fn clustering(&self) -> &Clustering {
        &self.clustering
    }
}

/// The clusters of `graph`'s cells by `params.method` at `params.resolution` with
/// `params.seed` (`params.graph` and `params.into` are not read).
///
/// # Errors
/// `KY-E215` for a resolution that is not positive and finite.
pub fn cluster(graph: CellGraph<'_>, params: &Cluster) -> Result<Clustering, KytoError> {
    run(graph, params, &RunControl::new())
}

/// `cluster`, checking `control` for a cancel.
pub(crate) fn run(
    graph: CellGraph<'_>,
    params: &Cluster,
    control: &RunControl,
) -> Result<Clustering, KytoError> {
    check_resolution(params)?;
    let weighted = params.method == ClusterMethod::Leiden;
    let level = Level::from_graph(graph, weighted);
    let mut optimiser = Optimiser::new(&level, params.resolution, params.seed);
    let membership = match params.method {
        ClusterMethod::Leiden => optimiser.leiden(&level, control)?,
        ClusterMethod::Louvain => optimiser.louvain(&level, control)?,
    };
    let (labels, n_clusters) = renumber_by_size(&membership);
    let quality = quality_of(&level, &labels, params.resolution);
    Ok(Clustering {
        labels,
        n_clusters,
        quality,
    })
}

/// The quality [`Clustering::quality`] reports of `labels` on `graph`, with the
/// weights for `ClusterMethod::Leiden` and without for `ClusterMethod::Louvain`.
///
/// # Errors
/// `KY-E215` for a resolution that is not positive and finite, or `labels` of another
/// length than the graph's cells.
pub fn partition_quality(
    graph: CellGraph<'_>,
    labels: &[u32],
    params: &Cluster,
) -> Result<f64, KytoError> {
    check_resolution(params)?;
    if labels.len() != graph.n_cells() {
        return Err(KytoError::InvalidArgument {
            function: "partition_quality".to_owned(),
            parameter: "labels".to_owned(),
            value: format!("{} labels", labels.len()),
            problem: format!("the graph has {} cells", graph.n_cells()),
            accepted: "one label a cell".to_owned(),
        });
    }
    let level = Level::from_graph(graph, params.method == ClusterMethod::Leiden);
    Ok(quality_of(&level, labels, params.resolution))
}

fn check_resolution(params: &Cluster) -> Result<(), KytoError> {
    if params.resolution.is_finite() && params.resolution > 0.0 {
        return Ok(());
    }
    Err(KytoError::InvalidArgument {
        function: params.method.name().to_owned(),
        parameter: "resolution".to_owned(),
        value: params.resolution.to_string(),
        problem: "it is not positive and finite".to_owned(),
        accepted: "a positive finite float".to_owned(),
    })
}

/// `sum_c (W_c - γ K_out(c) K_in(c) / m) / m` on `level` (0 without arcs).
fn quality_of(level: &Level, labels: &[u32], resolution: f64) -> f64 {
    let total = level.total_weight();
    if total <= 0.0 {
        return 0.0;
    }
    let n_labels = labels
        .iter()
        .map(|&label| label as usize + 1)
        .max()
        .unwrap_or(0);
    let mut inside = vec![0.0_f64; n_labels];
    let mut out_total = vec![0.0_f64; n_labels];
    let mut in_total = vec![0.0_f64; n_labels];
    for node in 0..level.n {
        let label = labels[node] as usize;
        inside[label] += level.self_weight[node];
        for (target, weight) in level.out.of(node) {
            if labels[target as usize] as usize == label {
                inside[label] += weight;
            }
        }
        out_total[label] += level.out_strength[node];
        in_total[label] += level.in_strength[node];
    }
    let sum: f64 = (0..n_labels)
        .map(|label| inside[label] - resolution * out_total[label] * in_total[label] / total)
        .sum();
    sum / total
}

/// Labels `0..k` for the communities of `membership`, by size (cells) descending, ties
/// by the first cell, and `k`.
fn renumber_by_size(membership: &[u32]) -> (Vec<u32>, u32) {
    let (compact, n) = compact(membership);
    let mut sizes = vec![0_u64; n];
    let mut first = vec![usize::MAX; n];
    for (cell, &community) in compact.iter().enumerate() {
        sizes[community as usize] += 1;
        first[community as usize] = first[community as usize].min(cell);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&community| (std::cmp::Reverse(sizes[community]), first[community]));
    let mut label_of = vec![0_u32; n];
    for (label, &community) in order.iter().enumerate() {
        label_of[community] = label as u32;
    }
    let labels = compact
        .iter()
        .map(|&community| label_of[community as usize])
        .collect();
    (labels, n as u32)
}

/// Community ids `0..k` in order of first appearance, and `k`.
fn compact(membership: &[u32]) -> (Vec<u32>, usize) {
    let mut id = vec![u32::MAX; membership.len()];
    let mut n = 0_u32;
    let compacted = membership
        .iter()
        .map(|&community| {
            let slot = &mut id[community as usize];
            if *slot == u32::MAX {
                *slot = n;
                n += 1;
            }
            *slot
        })
        .collect();
    (compacted, n as usize)
}

/// SplitMix64 on the seed, integer arithmetic only.
struct Random {
    state: u64,
}

impl Random {
    fn new(seed: u64) -> Self {
        Self { state: mix(seed) }
    }

    /// `0..n` shuffled (Fisher-Yates).
    fn shuffled(&mut self, n: usize) -> Vec<u32> {
        let mut order: Vec<u32> = (0..n as u32).collect();
        for last in (1..n).rev() {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let pick = (u128::from(mix(self.state)) * (last as u128 + 1)) >> 64;
            order.swap(last, pick as usize);
        }
        order
    }
}

/// A partition of a level's nodes into community slots `0..n`.
struct Partition {
    membership: Vec<u32>,
    out_total: Vec<f64>,
    in_total: Vec<f64>,
    /// Nodes in each community.
    n_nodes: Vec<u32>,
    /// Empty communities; the last is the one offered to a move.
    empty: Vec<u32>,
}

impl Partition {
    /// `membership` (ids below `level.n`) on `level`.
    fn new(level: &Level, membership: Vec<u32>) -> Self {
        let mut out_total = vec![0.0; level.n];
        let mut in_total = vec![0.0; level.n];
        let mut n_nodes = vec![0_u32; level.n];
        for (node, &community) in membership.iter().enumerate() {
            out_total[community as usize] += level.out_strength[node];
            in_total[community as usize] += level.in_strength[node];
            n_nodes[community as usize] += 1;
        }
        let empty = (0..level.n as u32)
            .rev()
            .filter(|&community| n_nodes[community as usize] == 0)
            .collect();
        Self {
            membership,
            out_total,
            in_total,
            n_nodes,
            empty,
        }
    }

    fn singletons(level: &Level) -> Self {
        Self::new(level, (0..level.n as u32).collect())
    }

    fn move_node(&mut self, level: &Level, node: usize, to: u32) {
        let from = self.membership[node] as usize;
        let to_slot = to as usize;
        if self.n_nodes[to_slot] == 0 {
            let position = self.empty.iter().rposition(|&community| community == to);
            if let Some(position) = position {
                self.empty.remove(position);
            }
        }
        self.out_total[from] -= level.out_strength[node];
        self.in_total[from] -= level.in_strength[node];
        self.n_nodes[from] -= 1;
        if self.n_nodes[from] == 0 {
            // Exactly 0, not a rounding residue, for an empty community.
            self.out_total[from] = 0.0;
            self.in_total[from] = 0.0;
            self.empty.push(from as u32);
        }
        self.out_total[to_slot] += level.out_strength[node];
        self.in_total[to_slot] += level.in_strength[node];
        self.n_nodes[to_slot] += 1;
        self.membership[node] = to;
    }
}

/// The run's state: the random stream and the scratch of one node's neighbourhood.
struct Optimiser {
    random: Random,
    /// `γ / m`.
    scale: f64,
    weight_to: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<u32>,
}

impl Optimiser {
    fn new(level: &Level, resolution: f64, seed: u64) -> Self {
        let total = level.total_weight();
        Self {
            random: Random::new(seed),
            scale: if total > 0.0 { resolution / total } else { 0.0 },
            weight_to: vec![0.0; level.n],
            seen: vec![false; level.n],
            touched: Vec::new(),
        }
    }

    /// Sum the weight of `node`'s arcs (both ways) to each community of `membership`,
    /// among the neighbours `allowed` keeps; communities in order of first appearance.
    fn gather(
        &mut self,
        level: &Level,
        membership: &[u32],
        node: usize,
        allowed: impl Fn(usize) -> bool,
    ) {
        let both_ways = level.into.is_none();
        let mut add = |target: u32, weight: f64| {
            if !allowed(target as usize) {
                return;
            }
            let community = membership[target as usize];
            if !self.seen[community as usize] {
                self.seen[community as usize] = true;
                self.touched.push(community);
            }
            self.weight_to[community as usize] += weight;
        };
        if both_ways {
            for (target, weight) in level.out.of(node) {
                add(target, 2.0 * weight);
            }
        } else {
            for (target, weight) in level.out.of(node) {
                add(target, weight);
            }
            for (source, weight) in level.incoming().of(node) {
                add(source, weight);
            }
        }
    }

    fn clear(&mut self) {
        for &community in &self.touched {
            self.weight_to[community as usize] = 0.0;
            self.seen[community as usize] = false;
        }
        self.touched.clear();
    }

    /// The quality gain of `node` joining `community` (its own one counted without it),
    /// up to the constant of leaving its own; `gather` ran first.
    fn gain(&self, level: &Level, partition: &Partition, node: usize, community: u32) -> f64 {
        let slot = community as usize;
        let (mut out_total, mut in_total) = (partition.out_total[slot], partition.in_total[slot]);
        if partition.membership[node] == community {
            out_total -= level.out_strength[node];
            in_total -= level.in_strength[node];
        }
        let weight = if self.seen[slot] {
            self.weight_to[slot]
        } else {
            0.0
        };
        weight
            - self.scale
                * (level.out_strength[node] * in_total + level.in_strength[node] * out_total)
    }

    /// leidenalg's `move_nodes`: whether a node moved.
    fn move_nodes(
        &mut self,
        level: &Level,
        partition: &mut Partition,
        control: &RunControl,
    ) -> Result<bool, KytoError> {
        let mut queue: VecDeque<u32> = self.random.shuffled(level.n).into();
        let mut queued = vec![true; level.n];
        let mut moved = false;
        let mut steps = 0_usize;
        while let Some(node) = queue.pop_front() {
            steps += 1;
            if steps.is_multiple_of(CHECK_EVERY) {
                control.check()?;
            }
            let node = node as usize;
            queued[node] = false;
            let own = partition.membership[node];
            self.gather(level, &partition.membership, node, |_| true);
            let stay = self.gain(level, partition, node, own);
            let (mut best, mut best_gain) = (own, MIN_GAIN);
            for &community in &self.touched {
                if community == own {
                    continue;
                }
                let gain = self.gain(level, partition, node, community) - stay;
                if gain > best_gain {
                    (best, best_gain) = (community, gain);
                }
            }
            if partition.n_nodes[own as usize] > 1
                && let Some(&empty) = partition.empty.last()
                && -stay > best_gain
            {
                best = empty;
            }
            self.clear();
            if best == own {
                continue;
            }
            partition.move_node(level, node, best);
            moved = true;
            let mut requeue = |neighbour: u32| {
                let neighbour = neighbour as usize;
                if !queued[neighbour] && partition.membership[neighbour] != best {
                    queued[neighbour] = true;
                    queue.push_back(neighbour as u32);
                }
            };
            for (target, _) in level.out.of(node) {
                requeue(target);
            }
            if level.into.is_some() {
                for (source, _) in level.incoming().of(node) {
                    requeue(source);
                }
            }
        }
        Ok(moved)
    }

    /// leidenalg's `merge_nodes_constrained` from singletons: the refined partition.
    fn refine(&mut self, level: &Level, constraint: &[u32]) -> Partition {
        let mut refined = Partition::singletons(level);
        for node in self.random.shuffled(level.n) {
            let node = node as usize;
            let own = refined.membership[node];
            if refined.n_nodes[own as usize] != 1 {
                continue;
            }
            let group = constraint[node];
            self.gather(level, &refined.membership, node, |other| {
                constraint[other] == group
            });
            let stay = self.gain(level, &refined, node, own);
            let (mut best, mut best_gain) = (own, 0.0);
            for &community in &self.touched {
                if community == own {
                    continue;
                }
                let gain = self.gain(level, &refined, node, community) - stay;
                if gain >= best_gain {
                    (best, best_gain) = (community, gain);
                }
            }
            self.clear();
            if best != own {
                refined.move_node(level, node, best);
            }
        }
        refined
    }

    /// leidenalg's `find_partition` with `n_iterations=-1`: optimise from singletons
    /// until an iteration moves nothing; each cell's community.
    fn leiden(&mut self, base: &Level, control: &RunControl) -> Result<Vec<u32>, KytoError> {
        let mut membership: Vec<u32> = (0..base.n as u32).collect();
        while self.leiden_iteration(base, &mut membership, control)? {}
        Ok(membership)
    }

    /// One `optimise_partition` from `membership`; whether a node moved.
    fn leiden_iteration(
        &mut self,
        base: &Level,
        membership: &mut [u32],
        control: &RunControl,
    ) -> Result<bool, KytoError> {
        control.check()?;
        let mut owned: Option<Level> = None;
        let mut partition = Partition::new(base, membership.to_vec());
        // Each cell's node at the current level.
        let mut node_of: Vec<u32> = (0..base.n as u32).collect();
        let mut moved = false;
        loop {
            let level = owned.as_ref().unwrap_or(base);
            moved |= self.move_nodes(level, &mut partition, control)?;
            let (communities, n_communities) = compact(&partition.membership);
            let refined = self.refine(level, &communities);
            let (refined, n_refined) = compact(&refined.membership);
            if n_refined >= level.n || level.n <= n_communities {
                break;
            }
            let next = level.aggregate(&refined, n_refined);
            let mut next_membership = vec![0_u32; n_refined];
            for (node, &group) in refined.iter().enumerate() {
                next_membership[group as usize] = communities[node];
            }
            for node in &mut node_of {
                *node = refined[*node as usize];
            }
            partition = Partition::new(&next, next_membership);
            owned = Some(next);
            control.check()?;
        }
        for (cell, node) in node_of.iter().enumerate() {
            membership[cell] = partition.membership[*node as usize];
        }
        Ok(moved)
    }

    /// leidenalg's Louvain loop: move, collapse, until a level moves nothing.
    fn louvain(&mut self, base: &Level, control: &RunControl) -> Result<Vec<u32>, KytoError> {
        let mut owned: Option<Level> = None;
        let mut partition = Partition::singletons(base);
        let mut node_of: Vec<u32> = (0..base.n as u32).collect();
        loop {
            control.check()?;
            let level = owned.as_ref().unwrap_or(base);
            if !self.move_nodes(level, &mut partition, control)? {
                break;
            }
            let (communities, n_communities) = compact(&partition.membership);
            let next = level.aggregate(&communities, n_communities);
            for node in &mut node_of {
                *node = communities[*node as usize];
            }
            partition = Partition::singletons(&next);
            owned = Some(next);
        }
        Ok(node_of
            .iter()
            .map(|&node| partition.membership[node as usize])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(method: ClusterMethod, resolution: f64, seed: u64) -> Cluster {
        Cluster {
            method,
            graph: "connectivities".to_owned(),
            resolution,
            seed,
            into: method.name().to_owned(),
        }
    }

    /// `n_groups` cliques of `size` cells (weight 1), each joined to the next by one
    /// edge of weight `bridge`; symmetric CSR.
    fn cliques(n_groups: usize, size: usize, bridge: f32) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
        let n = n_groups * size;
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for group in 0..n_groups {
            for i in 0..size {
                for j in 0..size {
                    if i != j {
                        rows[group * size + i].push(((group * size + j) as u32, 1.0));
                    }
                }
            }
            let (a, b) = (group * size, ((group + 1) % n_groups) * size + 1);
            rows[a].push((b as u32, bridge));
            rows[b].push((a as u32, bridge));
        }
        let mut indptr = vec![0_u64];
        let (mut indices, mut weights) = (Vec::new(), Vec::new());
        for mut row in rows {
            row.sort_by_key(|&(index, _)| index);
            for (index, weight) in row {
                indices.push(index);
                weights.push(weight);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, weights)
    }

    #[test]
    fn both_methods_find_the_cliques() {
        let (indptr, indices, weights) = cliques(6, 7, 0.1);
        let graph = CellGraph::new(&indptr, &indices, &weights).unwrap();
        for method in [ClusterMethod::Leiden, ClusterMethod::Louvain] {
            for seed in [0, 1, 99] {
                let found = cluster(graph, &params(method, 1.0, seed)).unwrap();
                assert_eq!(found.n_clusters(), 6, "{method:?} seed {seed}");
                for group in found.labels().chunks(7) {
                    assert!(group.iter().all(|&label| label == group[0]));
                }
                let expected = partition_quality(graph, found.labels(), &params(method, 1.0, 0));
                assert_eq!(found.quality(), expected.unwrap());
            }
        }
    }

    #[test]
    fn labels_number_clusters_by_size() {
        // Cliques of 3, 6 and 4 cells, unconnected.
        let sizes = [3_usize, 6, 4];
        let mut indptr = vec![0_u64];
        let (mut indices, mut weights) = (Vec::new(), Vec::new());
        let mut start = 0;
        for size in sizes {
            for i in 0..size {
                for j in 0..size {
                    if i != j {
                        indices.push((start + j) as u32);
                        weights.push(1.0);
                    }
                }
                indptr.push(indices.len() as u64);
            }
            start += size;
        }
        let graph = CellGraph::new(&indptr, &indices, &weights).unwrap();
        let found = cluster(graph, &params(ClusterMethod::Leiden, 1.0, 3)).unwrap();
        let expected: Vec<u32> = [2, 0, 1]
            .iter()
            .zip(sizes)
            .flat_map(|(&label, size)| std::iter::repeat_n(label, size))
            .collect();
        assert_eq!(found.labels(), expected);
    }

    #[test]
    fn a_graph_without_arcs_keeps_every_cell_apart() {
        let indptr = [0_u64; 5];
        let graph = CellGraph::new(&indptr, &[], &[]).unwrap();
        for method in [ClusterMethod::Leiden, ClusterMethod::Louvain] {
            let found = cluster(graph, &params(method, 1.0, 0)).unwrap();
            assert_eq!(found.labels(), [0, 1, 2, 3]);
            assert_eq!(found.quality(), 0.0);
        }
        let empty = CellGraph::new(&[0], &[], &[]).unwrap();
        let found = cluster(empty, &params(ClusterMethod::Leiden, 1.0, 0)).unwrap();
        assert!(found.labels().is_empty());
        assert_eq!(found.n_clusters(), 0);
    }

    #[test]
    fn resolution_trades_size_for_number() {
        let (indptr, indices, weights) = cliques(8, 5, 0.5);
        let graph = CellGraph::new(&indptr, &indices, &weights).unwrap();
        let low = cluster(graph, &params(ClusterMethod::Leiden, 0.01, 0)).unwrap();
        let high = cluster(graph, &params(ClusterMethod::Leiden, 1.0, 0)).unwrap();
        assert_eq!(low.n_clusters(), 1);
        assert_eq!(high.n_clusters(), 8);
    }

    #[test]
    fn a_directed_graph_and_self_loops_are_clustered() {
        // Two triangles as one-way cycles, joined by one arc; self loops on two cells.
        let indptr = [0_u64, 2, 3, 4, 5, 6, 8];
        let indices = [0_u32, 1, 2, 0, 4, 5, 3, 5];
        let weights = [2.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 3.0];
        let graph = CellGraph::new(&indptr, &indices, &weights).unwrap();
        let found = cluster(graph, &params(ClusterMethod::Leiden, 1.0, 0)).unwrap();
        assert_eq!(found.n_clusters(), 2);
        assert_eq!(found.labels()[0], found.labels()[2]);
        assert_ne!(found.labels()[0], found.labels()[3]);
        let quality = partition_quality(
            graph,
            found.labels(),
            &params(ClusterMethod::Leiden, 1.0, 0),
        );
        assert_eq!(found.quality(), quality.unwrap());
    }

    #[test]
    fn refuses_a_bad_resolution_and_malformed_graphs() {
        let graph = CellGraph::new(&[0, 0], &[], &[]).unwrap();
        for resolution in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let error = cluster(graph, &params(ClusterMethod::Louvain, resolution, 0)).unwrap_err();
            assert_eq!(error.code(), "KY-E215");
        }
        let cases: [(&[u64], &[u32], &[f32]); 6] = [
            (&[], &[], &[]),
            (&[1, 1], &[0], &[1.0]),
            (&[0, 2, 1], &[0, 0], &[1.0, 1.0]),
            (&[0, 1], &[0], &[]),
            (&[0, 1], &[1], &[1.0]),
            (&[0, 1], &[0], &[f32::NAN]),
        ];
        for (indptr, indices, weights) in cases {
            let error = CellGraph::new(indptr, indices, weights).unwrap_err();
            assert_eq!(error.code(), "KY-E215", "{indptr:?}");
        }
        assert!(CellGraph::new(&[0, 1], &[0], &[-1.0]).is_err());
    }

    #[test]
    fn renumbering_breaks_ties_by_first_cell() {
        assert_eq!(renumber_by_size(&[4, 4, 2, 2, 3]), (vec![0, 0, 1, 1, 2], 3));
        assert_eq!(renumber_by_size(&[1, 0, 0, 1, 1]), (vec![0, 1, 1, 0, 0], 2));
    }
}
