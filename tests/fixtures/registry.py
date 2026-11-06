"""The list of fixtures, their builders and their expected digests.

What it does: maps each fixture name to a builder and to the digest of its content
(:meth:`fixtures.dataset.Dataset.digest`), and caches built fixtures per process.

What it does not: build anything at import time.

Changing a builder changes its digest on purpose only: update ``EXPECTED_DIGESTS``,
regenerate the references (:mod:`fixtures.make_reference`) and say so in the commit.
"""

from collections.abc import Callable
from functools import cache
from typing import Literal

from fixtures.dataset import Dataset
from fixtures.pbmc import load_pbmc3k
from fixtures.synthetic import make_synthetic

FixtureName = Literal["synthetic", "pbmc3k"]

FIXTURE_NAMES: tuple[FixtureName, ...] = ("synthetic", "pbmc3k")

_BUILDERS: dict[FixtureName, Callable[[], Dataset]] = {
    "synthetic": make_synthetic,
    "pbmc3k": load_pbmc3k,
}

EXPECTED_DIGESTS: dict[FixtureName, str] = {
    "synthetic": "ad7cb28a0480c507397b70d3d14b77cda161174a201f5c7a13d44167a9c28820",
    "pbmc3k": "6566b9312f97756266a4b0e1fec5d82606a630ce55b1a429a616d03b52131579",
}


@cache
def load_fixture(name: FixtureName) -> Dataset:
    """Return the named fixture, built once per process and shared (read-only arrays).

    Parameters
    ----------
    name : {"synthetic", "pbmc3k"}
        Fixture to build. ``pbmc3k`` downloads its archive on first use.

    Returns
    -------
    Dataset
        The cached dataset; not copied, its arrays are read-only.
    """
    return _BUILDERS[name]()
