"""Type stub for the compiled extension module `kyto._kyto`."""

__version__: str

def raise_error_example(code: str, /) -> bool:
    """Raise the registered example error for `code`; return False if unknown."""

def error_example_text(code: str, /) -> tuple[str, str] | None:
    """Return the Rust message and hint of the example error for `code`."""

def convert(
    source: str,
    target: str,
    /,
    *,
    layer: str,
    on_duplicate_ids: str,
    overwrite: bool,
) -> None:
    """Import `source` into a new store at `target` (see `kyto.convert`)."""

def sink_h5ad(
    store: str,
    target: str,
    /,
    *,
    x_layer: str | None,
    overwrite: bool,
) -> None:
    """Export the store at `store` as an `.h5ad` file at `target` (private, task 2.7)."""
