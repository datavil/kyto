"""Test fixtures for kyto (task 1.5): datasets, their hashes and scanpy references.

What it does: builds the reference datasets every parity test runs on, in memory, with
numpy only:

- ``synthetic``: a seeded generator (cells, genes, sparsity, batches, cell types, edge
  cases), see :mod:`fixtures.synthetic`.
- ``pbmc3k``: the 10x Genomics 3k PBMC dataset (the standard scanpy dataset), downloaded
  once into a gitignored cache and checked by SHA-256, see :mod:`fixtures.pbmc`.

Each dataset has a content digest (:mod:`fixtures.hashing`) that is stable across
operating systems and architectures; :mod:`fixtures.registry` lists the expected ones.
Reference outputs from scanpy live in ``fixtures/reference`` and are loaded with
:func:`fixtures.reference.load_reference`; :mod:`fixtures.make_reference` regenerates
them.

What it does not: hold kyto code or parity tolerances (task 10.2 owns those).
"""
