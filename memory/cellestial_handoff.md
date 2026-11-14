# Plotting through cellestial (task 7.8)

**Works:** `Atlas.to_anndata(var_names="symbol")` (sink to a temporary `.h5ad`, read
by anndata, scanpy's names applied in `kyto/_anndata.py`) is everything cellestial
needs: `X_<name>` `obsm` keys, genes by symbol, `Enum` columns as categoricals. Check a
plot with ggrust's `PlotSpec.build()` (scales and layer data, no pixels); a multi-panel
plot (`cl.violins`, `cl.umaps`) is a `SupPlotsSpec` with no `build`: build each of
`plot.figures` (`None` for an empty cell). `plot.data` is the Polars frame plotted.

**Broke (dependencies):** cellestial 0.62.1 pins `polars<2` and `pandas<3`, the dev env
runs polars 2.1 and pandas 3.1. Its plots build fine on both (checked in a scratch
venv with `--no-deps`), so `[tool.uv] override-dependencies` lifts the caps for the dev
env only; users of `kyto[plot]` get cellestial's pins (kyto itself pins no polars).

**Upstream, not kyto (cellestial 0.62.1 / ggrust 0.8.2):** a categorical's legend in
`cl.umap` is in order of appearance, not category order, even for a plain AnnData
(`0, 1, 2` categories, legend `3, 1, 0, 2`); `cl.dotplot` keeps category order. Its
theme sets the font face `"Bold"`, which ggrust warns about (`GgrustWarning`, expects
`bold`).

**Also:**
- anndata 0.12+ lists `None` among `layers.keys()` (an alias for `X`).
- `scripts/error_registry.py --check` flags `Result<BTreeMap<String, String>,
  KytoError>` as a `Result<_, String>` (its regex sees the map's `String`); a type alias
  avoids it.
- Doctest a markdown file with `DocTestParser().get_doctest(text, ...)` and
  `runner.run(test, clear_globs=False)` to inspect the objects the examples made.
