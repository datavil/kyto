"""e-means (Bessa et al., arXiv 2611.09501) in NumPy: kyto's reference for ``metacells``.

The paper's Algorithm 1, written to replay kyto's Rust implementation step for step
(task 6.16): the same SplitMix64 draws for the initial cells, the shuffles and the
revival donors, float64 throughout, and distances and sums added in the same order, so
the two agree bit for bit on the fixtures. ``fast=True`` computes distances with one
matrix product instead (other float bits, same algorithm): the step and batch law
sweep of task 6.16 used it.

The batch size and first step are the paper's Eq. 7 and 6 (``paper_batch_size``,
``paper_first_step``), kept after task 6.16 measured them on single-cell PCA. Where
kyto departs from the paper (deviations.md, "metacells"): each epoch's last batch
takes the remainder; a cluster is revived below the lower of 10% of the mean count
and two cells' worth (``revival_fraction``); after the last epoch every cell is
assigned once more to the nearest final centroid and the centroids become their
members' means (``final_assignment``).

It is not used by kyto itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
#: The cosine schedule's floor, a fraction of the first step (the paper's epsilon).
SCHEDULE_FLOOR = 0.1
#: Clusters whose count falls below this fraction of the mean count are revived
#: (the paper's tau), or below ``REVIVAL_CELLS`` cells' worth if that is lower.
REVIVAL_FRACTION = 0.1
#: Cells' worth of count under which a cluster is revived (task 6.16).
REVIVAL_CELLS = 2.0
#: Revival is checked after every this many batches.
REVIVAL_EVERY = 5
#: The starting count of every cluster.
INITIAL_COUNT = 1e-10
#: The relative perturbation of a split centroid (Faiss's ``EPS``).
SPLIT_PERTURBATION = 1.0 / 1024.0
#: Stream purposes, so the draws for each use are unrelated.
PURPOSE_INIT = 1
PURPOSE_SHUFFLE = 2
PURPOSE_REVIVAL = 3


def _mix(value: int) -> int:
    """SplitMix64's output function."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


class Draws:
    """A seeded SplitMix64 stream, as kyto-exec's ``Draws`` (integer arithmetic only)."""

    def __init__(self, seed: int, purpose: int) -> None:
        self.state = _mix(seed ^ _mix((purpose + GOLDEN) & MASK))

    def next_u64(self) -> int:
        """The next 64 random bits."""
        self.state = (self.state + GOLDEN) & MASK
        return _mix(self.state)

    def below(self, n: int) -> int:
        """Uniform in ``0..n`` (multiply-shift, as the Rust side)."""
        return (self.next_u64() * n) >> 64

    def uniform(self) -> float:
        """Uniform in ``[0, 1)``: 53 random bits."""
        return float(self.next_u64() >> 11) * 2.0**-53

    def shuffled(self, n: int) -> NDArray[np.int64]:
        """``0..n`` shuffled (Fisher-Yates from the last slot)."""
        order = np.arange(n, dtype=np.int64)
        for last in range(n - 1, 0, -1):
            pick = self.below(last + 1)
            order[last], order[pick] = order[pick], order[last]
        return order

    def sample(self, n: int, k: int) -> NDArray[np.int64]:
        """``k`` distinct values of ``0..n`` (the first ``k`` slots of a partial shuffle)."""
        order = np.arange(n, dtype=np.int64)
        for slot in range(k):
            pick = slot + self.below(n - slot)
            order[slot], order[pick] = order[pick], order[slot]
        return order[:k].copy()


@dataclass(frozen=True)
class Result:
    """What e-means returns."""

    labels: NDArray[np.int64]
    """Each cell's cluster in the last epoch."""
    centroids: NDArray[np.float64]
    """The mean of each cluster's cells in the last epoch (the running centroid if empty)."""
    mse: float
    """Mean squared distance of each cell to its cluster's centroid."""
    batch_size: int
    first_step: float


def step(t: int, total: int, first_step: float) -> float:
    """The step size (``1 - gamma``) of update ``t`` of ``total``: the cosine schedule."""
    return first_step * (
        SCHEDULE_FLOOR
        + (1.0 - SCHEDULE_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * t / total))
    )


def _distances(
    batch: NDArray[np.float64], centroids: NDArray[np.float64], fast: bool
) -> NDArray[np.float64]:
    """Squared distances, ``(len(batch), k)``; exact mode adds components in order."""
    if fast:
        return (
            (batch * batch).sum(1)[:, None]
            - 2.0 * batch @ centroids.T
            + (centroids * centroids).sum(1)[None, :]
        )
    total = np.zeros((batch.shape[0], centroids.shape[0]))
    for component in range(batch.shape[1]):
        difference = batch[:, component, None] - centroids[None, :, component]
        total += difference * difference
    return total


def revival_fraction(n: int, k: int) -> float:
    """The fraction of the mean count under which a cluster is revived."""
    return min(REVIVAL_FRACTION, REVIVAL_CELLS * k / n)


def _revive(
    sums: NDArray[np.float64],
    counts: NDArray[np.float64],
    draws: Draws,
    fraction: float,
) -> None:
    """Split a populous cluster into each nearly empty one (in place)."""
    k = counts.shape[0]
    mean = sum(float(count) for count in counts) / k
    for dead in range(k):
        if counts[dead] >= fraction * mean:
            continue
        # A donor with probability proportional to its count, drawn by walking the
        # running total (dead clusters can donate too, with their tiny weight).
        total = sum(float(count) for count in counts)
        target = draws.uniform() * total
        donor = k - 1
        running = 0.0
        for cluster in range(k):
            running += float(counts[cluster])
            if target < running:
                donor = cluster
                break
        if donor == dead or counts[donor] <= 0.0:
            continue
        centroid: NDArray[np.float64] = sums[donor] / counts[donor]
        width = sums.shape[1]
        signs = np.tile(np.array([1.0, -1.0]), (width + 1) // 2)[:width]
        half = counts[donor] * 0.5
        counts[dead] = half
        counts[donor] = half
        sums[dead] = centroid * (1.0 + signs * SPLIT_PERTURBATION) * half
        sums[donor] = centroid * (1.0 - signs * SPLIT_PERTURBATION) * half


def emeans(
    x: NDArray[np.floating],
    k: int,
    epochs: int,
    seed: int,
    batch_size: int | None = None,
    first_step: float | None = None,
    *,
    fast: bool = False,
    final_assignment: bool = True,
) -> Result:
    """Run e-means on the rows of ``x``.

    Parameters
    ----------
    x : numpy.ndarray
        Points, ``(n, d)``; computed in float64.
    k : int
        Clusters, ``1 <= k <= n``.
    epochs : int
        Passes over the points.
    seed : int
        Seeds the initial cells, the shuffles and the revival draws.
    batch_size : int, optional
        Points per update (default ``paper_batch_size``); each epoch's last batch
        takes the remainder.
    first_step : float, optional
        The first step size (``1 - gamma``), the cosine schedule's peak (default
        ``paper_first_step``).
    fast : bool, default=False
        Distances by a matrix product (not kyto's bits).
    final_assignment : bool, default=True
        Assign every cell once more after the last epoch (kyto); ``False`` keeps the
        last epoch's assignments, as the paper.

    Returns
    -------
    Result
        Labels, centroids and the training MSE.
    """
    points = np.ascontiguousarray(x, dtype=np.float64)
    n, d = points.shape
    if batch_size is None:
        batch_size = paper_batch_size(n, k, epochs)
    if first_step is None:
        first_step = paper_first_step(n, k, epochs, batch_size)
    fraction = revival_fraction(n, k)
    initial = points[Draws(seed, PURPOSE_INIT).sample(n, k)]
    counts = np.full(k, INITIAL_COUNT)
    sums = initial * INITIAL_COUNT
    # The last centroids read: a cluster whose count is 0 (a first step of 1 gives
    # gamma = 0) keeps its centroid instead of 0 / 0.
    previous = initial.copy()
    shuffles = Draws(seed, PURPOSE_SHUFFLE)
    revivals = Draws(seed, PURPOSE_REVIVAL)
    per_epoch = -(-n // batch_size)
    total = epochs * per_epoch
    labels = np.zeros(n, dtype=np.int64)
    t = 0
    for _ in range(epochs):
        order = shuffles.shuffled(n)
        for start in range(0, n, batch_size):
            members = order[start : start + batch_size]
            batch = points[members]
            positive = counts > 0.0
            previous[positive] = sums[positive] / counts[positive, None]
            centroids = previous.copy()
            assigned = np.argmin(_distances(batch, centroids, fast), axis=1)
            labels[members] = assigned
            batch_counts = np.bincount(assigned, minlength=k).astype(np.float64)
            batch_sums = np.zeros((k, d))
            np.add.at(batch_sums, assigned, batch)
            alpha = step(t, total, first_step)
            gamma = 1.0 - alpha
            counts = gamma * counts + alpha * batch_counts
            sums = gamma * sums + alpha * batch_sums
            t += 1
            if t % REVIVAL_EVERY == 0:
                _revive(sums, counts, revivals, fraction)
    positive = counts > 0.0
    previous[positive] = sums[positive] / counts[positive, None]
    centroids = _means(points, labels, previous)
    if final_assignment:
        # One more assignment, to the nearest final centroid, and the means again.
        labels = np.argmin(_distances(points, centroids, fast), axis=1)
        centroids = _means(points, labels, centroids)
    residual = points - centroids[labels]
    mse = float((residual * residual).sum() / n)
    return Result(labels, centroids, mse, batch_size, first_step)


def _means(
    points: NDArray[np.float64],
    labels: NDArray[np.int64],
    fallback: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Each cluster's mean (members added in cell order), ``fallback`` where empty."""
    k, d = fallback.shape
    member_counts = np.bincount(labels, minlength=k)
    member_sums = np.zeros((k, d))
    np.add.at(member_sums, labels, points)
    centroids = fallback.copy()
    filled = member_counts > 0
    centroids[filled] = member_sums[filled] / member_counts[filled, None]
    return centroids


def paper_batch_size(n: int, k: int, epochs: int) -> int:
    """The paper's Eq. 7 (knee at 0.2%), rounded to a power of two, capped at ``n``."""
    raw = 0.035 * k**0.13 * n**0.88 * epochs**0.81
    if n > 2_000_000:
        raw /= 8.0
    # Half away from zero (Rust's rounding), at least 2^0.
    exponent = min(62, max(0, math.floor(math.log2(raw) + 0.5)))
    return min(n, 2**exponent)


def paper_first_step(n: int, k: int, epochs: int, batch_size: int) -> float:
    """The paper's Eq. 6."""
    return min(0.824 * k**-0.46 * batch_size**0.90 * n**-0.45 * epochs**-0.11, 1.0)
