# Leiden and Louvain references (task 6.8)

**Breaks:** the `louvain` package (scanpy's `tl.louvain` default `vtraag` flavour) has
no Python 3.13 wheel and its sdist fails to build, so it is not a dev dependency.
leidenalg 0.10.2's `Optimiser.refine_partition = False` (Louvain inside leidenalg)
aborts the process ("Given vertex is not in the graph", igraph C error, no Python
traceback).

**Works:** the Louvain reference runs leidenalg's documented loop
(`aggregate = partition.aggregate_partition()`; `while optimiser.move_nodes(aggregate) > 0:
partition.from_coarse_partition(aggregate); aggregate = aggregate.aggregate_partition()`),
unweighted on the directed graph, as scanpy's `tl.louvain` defaults.

**Comparing:** leidenalg's random stream is igraph's and cannot be copied, so parity is
an adjusted Rand index plus the partition quality (modularity with resolution,
`sum_c (W_c - γ K_out K_in / m) / m`, the same formula in `make_reference.py` and
`partition_quality`). Louvain partitions vary much more with the node order than
Leiden's (two kyto seeds: ARI 0.75 vs 0.95 on pbmc3k), so its ARI floor is lower.

**Lock:** this machine's uv 0.11.7 rewrites `uv.lock` `revision = 5` as `revision = 3`
on any relock (content equivalent).
