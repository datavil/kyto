"""Task 6.14: ``pca(genes=...)``, scanpy's ``mask_var``.

Done when: ``pca(genes=gene_column("highly_variable"))`` gives the same embedding as
``filter(gene_column("highly_variable")).pca(...)`` bit for bit, with the plan keeping
every gene. Checked on pbmc3k (1000 highly variable genes: more than the Krylov basis
holds, so the result is sketched and any difference in the order of the sums would
show), with the flags computed by ``highly_variable_genes`` in the same plan: the
embedding's bytes, the loadings (the filter's rows in the set, zeros outside, as
scanpy's ``varm['PCs']``), and that a later ``rank_genes_groups`` ranks every gene.
The clustering tutorial's run against scanpy is ``tests/test_clustering_tutorial.py``.
"""

import tarfile
from pathlib import Path

import kyto
import numpy as np
import pytest
from fixtures.pbmc import _MEMBER_PREFIX, fetch_pbmc3k  # pyright: ignore[reportPrivateUsage]
from kyto import _kyto, cell_column, gene_column

N_TOP_GENES = 1000
N_COMPONENTS = 50


@pytest.fixture(scope="module")
def pbmc3k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """pbmc3k as a store, converted from its 10x matrix directory."""
    directory = tmp_path_factory.mktemp("pbmc3k")
    matrix = directory / "hg19"
    matrix.mkdir()
    with tarfile.open(fetch_pbmc3k()) as archive:
        for name in ["matrix.mtx", "genes.tsv", "barcodes.tsv"]:
            member = archive.extractfile(_MEMBER_PREFIX + name)
            assert member is not None
            (matrix / name).write_bytes(member.read())
    store = directory / "pbmc3k.kyto"
    kyto.convert(matrix, store)
    return store


def _preprocessed(store: Path) -> _kyto.Atlas:
    return (
        _kyto.scan(store)
        .with_default_layer("counts")
        .normalize_total(1e4, into="normalized")
        .log1p(layer="normalized", into="log1p")
        .with_default_layer("log1p")
        .highly_variable_genes(n_top_genes=N_TOP_GENES)
    )


def test_a_gene_set_gives_the_filters_embedding_and_keeps_every_gene(
    pbmc3k: Path,
) -> None:
    base = _preprocessed(pbmc3k)
    highly_variable = gene_column("highly_variable")
    masked = base.pca(
        into="pca", n_components=N_COMPONENTS, seed=0, genes=highly_variable
    )
    filtered = base.filter(highly_variable).pca(
        into="pca", n_components=N_COMPONENTS, seed=0
    )
    assert "genes=gene_column('highly_variable')" in str(masked)

    embedding = masked.embedding("pca").to_numpy()
    expected = filtered.embedding("pca").to_numpy()
    assert embedding.shape == expected.shape == (2700, N_COMPONENTS)
    assert embedding.tobytes() == expected.tobytes()

    # The plan keeps every gene; the filtered one only the set.
    flags = masked.pull(highly_variable).to_numpy()
    assert flags.sum() == N_TOP_GENES
    assert len(masked.gene_ids()) == len(flags) == 32738
    assert len(filtered.gene_ids()) == N_TOP_GENES
    loadings = masked.embedding("pca_loadings").to_numpy()
    assert loadings.shape == (32738, N_COMPONENTS)
    assert not loadings[~flags].any()
    expected_loadings = filtered.embedding("pca_loadings").to_numpy()
    assert loadings[flags].tobytes() == expected_loadings.tobytes()
    # Not the PCA of every gene.
    every = base.pca(into="pca", n_components=N_COMPONENTS, seed=0)
    assert not np.array_equal(every.embedding("pca").to_numpy(), embedding)


def test_marker_ranking_after_a_gene_set_pca_sees_every_gene(pbmc3k: Path) -> None:
    # One plan (the clustering tutorial's shape): the components from the highly
    # variable genes, the Wilcoxon ranking over all the genes.
    clustered = (
        _preprocessed(pbmc3k)
        .pca(
            into="pca",
            n_components=20,
            seed=0,
            genes=gene_column("highly_variable"),
        )
        .neighbors("pca")
        .leiden(into="leiden", seed=0)
    )
    ranked = clustered.rank_genes_groups("leiden", method="wilcoxon").collect()
    assert ranked["gene_id"].n_unique() == 32738
    assert ranked["group"].n_unique() >= 2


def test_gene_set_arguments_are_coded(pbmc3k: Path) -> None:
    base = _preprocessed(pbmc3k)
    cases: list[object] = [
        "highly_variable",
        cell_column("cell_id") == "x",
        gene_column("means"),
        kyto.cell_profile("AAACATACAACCAC-1") > 0,
    ]
    for genes in cases:
        with pytest.raises(kyto.ExpressionError) as raised:
            base.pca(into="pca", seed=0, genes=genes)  # pyright: ignore[reportArgumentType]
        assert raised.value.code == "KY-E205", genes
        assert "pca" in str(raised.value)
    with pytest.raises(kyto.ResolutionError) as unknown:
        base.pca(into="pca", seed=0, genes=gene_column("hvg"))
    assert unknown.value.code == "KY-E210"
    # An empty set is known only when the plan runs.
    empty = base.pca(into="pca", seed=0, genes=gene_column("gene_id") == "none")
    with pytest.raises(kyto.PlanError) as none:
        empty.embedding("pca").to_numpy()
    assert none.value.code == "KY-E215"
    assert "true for none of the genes reaching pca" in str(none.value)
