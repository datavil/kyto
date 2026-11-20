"""The parity tolerance table (task 10.2) for the Python parity tests.

``tolerance(name)`` reads a value from ``tests/parity/tolerances.toml``, the one place
where the tolerances against scanpy and anndata live (each with its reason); the Rust
tests read the same table through ``tests/parity/tolerances.rs``. The table itself is
checked by ``tests/test_parity_suite.py``.
"""

from functools import cache
from pathlib import Path
from typing import cast

import tomllib

TABLE = Path(__file__).with_name("tolerances.toml")


@cache
def entries() -> dict[str, dict[str, object]]:
    """Every entry of the table by its name (``<algorithm>.<what>``), in table order."""
    with TABLE.open("rb") as file:
        table = tomllib.load(file)
    return {
        f"{algorithm}.{name}": cast("dict[str, object]", entry)
        for algorithm, group in table.items()
        for name, entry in cast("dict[str, object]", group).items()
    }


def tolerance(name: str) -> float:
    """The value of the tolerance ``name``.

    Raises
    ------
    KeyError
        If the table has no entry ``name``.
    """
    try:
        value = entries()[name]["value"]
    except KeyError:
        message = f"no tolerance {name!r} in {TABLE}"
        raise KeyError(message) from None
    return float(cast("float", value))
