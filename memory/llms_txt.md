# `llms.txt` from the stubs (task 8.6)

**Works:** `llms.txt` (signature, summary, example per public item) and `llms-full.txt`
(whole docstrings) are generated, never edited: `uv run python scripts/llms_txt.py`
(the gate runs it with `--check`). It reads only sources, by `ast`: the `__all__` of
`kyto`, `kyto.qc`, `kyto.dtypes` followed through imports to the defining module
(`_kyto.pyi` for the extension), and every public non-TypedDict class of `_kyto.pyi`
with its public methods and properties. `tests/test_llms_txt.py` holds it to the built
package: the runtime surface equals the `### name` headings, every entry has an
example (only `Modality`'s three methods may lack one), and every example runs as a
doctest from the llms.txt text, each in a fresh folder with `store` (2 cells x 3
genes) and `store_with_members`. About 2.5 s for 136 examples.

**Regenerate in this order** after a change to a docstring or binding: `scripts/stubs.py`
(and `scripts/error_registry.py` for errors), then `scripts/llms_txt.py`; the stub feeds
llms.txt, so a stale stub makes llms.txt stale too.

**A new public method needs an example**, or `test_every_entry_has_an_example` fails.

**Broke (each cost a debug round):**
- In the shared `expr_class!` macro, examples are written per class with
  `#[doc = concat!(" >>> ", $column, ...)]`. PyO3 strips one leading space from literal
  doc lines (`///` and `#[doc = "..."]`) but not from `concat!` ones, so every line of
  such an example, outputs too, must be a `concat!` (or doctest sees inconsistent
  indentation). The hand-written copy in `crates/kyto-py/stubs/expressions.pyi` keeps
  the extra space (indent 9, not 8) so `test_stubs.py` sees the same docstring.
- `ast.unparse(arguments)` writes `layer: str | None=None`; the script renders
  parameters itself (`name: type = default`).
- An example's `is` check on a resolved dtype (`... .dtype is String`) fails
  `test_doc_examples_typecheck.py`: statically `resolve` keeps `Unknown`. Show the
  value instead.
