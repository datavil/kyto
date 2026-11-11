"""Task 8.2: close matches for unknown names (Jaro-Winkler, top 3) in the messages.

Done when: ``pull("CD3")`` suggests ``CD3E``, ``CD3D``, ``CD3G`` on the PBMC fixture.
``pull`` is task 7.4; it resolves its name as ``feature(...)`` does, so this resolves
``feature("CD3")`` on pbmc3k, converted from the 10x matrix directory (gene IDs are
Ensembl IDs; symbols sit in the ``symbol`` column). Every case is also covered without
a store by ``crates/kyto-core/tests/suggestions.rs``.
"""

import tarfile
from pathlib import Path

import kyto
import pytest
from fixtures.pbmc import (
    _MEMBER_PREFIX,  # pyright: ignore[reportPrivateUsage]
    fetch_pbmc3k,
)
from kyto import _kyto

CD3 = ("CD3E", "CD3D", "CD3G")


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


def _suggestions(raised: pytest.ExceptionInfo[kyto.KytoError]) -> tuple[str, ...]:
    value = raised.value.fields["suggestions"]
    assert isinstance(value, tuple)
    return value


def test_cd3_suggests_the_cd3_complex_on_pbmc3k(pbmc3k: Path) -> None:
    atlas = _kyto.scan(pbmc3k).with_default_layer("counts")
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.feature("CD3"))
    assert raised.value.code == "KY-E210"
    labelled = tuple(f"{symbol} (gene symbol)" for symbol in CD3)
    assert _suggestions(raised) == labelled
    assert (
        str(raised.value)
        .splitlines()[0]
        .endswith(
            "; did you mean 'CD3E (gene symbol)', 'CD3D (gene symbol)', "
            "'CD3G (gene symbol)'?"
        )
    )
    assert "by='symbol'" in raised.value.hint
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.gene_expression("CD3", by="symbol"))
    assert _suggestions(raised) == CD3
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.gene_expression("cd3e", by="symbol"))
    assert _suggestions(raised)[0] == "CD3E"


def test_layers_and_columns_get_suggestions(pbmc3k: Path) -> None:
    atlas = _kyto.scan(pbmc3k)
    with pytest.raises(kyto.MissingMemberError) as raised:
        atlas.resolve(kyto.gene_expression("ENSG00000198851", layer="count"))
    assert raised.value.message == (
        f"no layer named 'count' in the store at '{pbmc3k}'; did you mean 'counts'?"
    )
    with pytest.raises(kyto.ResolutionError) as raised:
        atlas.resolve(kyto.gene_column("symbl"))
    assert _suggestions(raised) == ("symbol",)


def test_nothing_close_adds_nothing(pbmc3k: Path) -> None:
    atlas = _kyto.scan(pbmc3k)
    with pytest.raises(kyto.MissingMemberError) as raised:
        atlas.resolve(kyto.gene_expression("ENSG00000198851", layer="zzz"))
    assert _suggestions(raised) == ()
    assert "did you mean" not in str(raised.value)
