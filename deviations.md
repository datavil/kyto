# Deviations from scanpy and anndata

Every place where kyto gives different results or semantics than scanpy or anndata.
Rules in [AGENTS.md](AGENTS.md) ("Record every deviation"). Add a row in the same commit
that introduces the deviation. Keep rows short; long reasoning goes in the topic
`memory/` entry. API shape differences required by the design rules (R1-R12, for
example no `.X` and no in-place mutation) are not listed here.

Kind: `chosen` (deliberate), `gap` (not implemented yet or known difference to fix),
`inherited` (kyto follows scanpy where another reference differs).

| Topic | scanpy / anndata | Other reference | kyto | Kind | Why | Decided |
|---|---|---|---|---|---|---|
| Layer value dtype | sparse/dense `X` and `layers` of any NumPy dtype | - | layers are `float32` or `uint32` only (`docs/format.md` §4.2); the importer (2.6) converts other dtypes; negative or `> 2^32 - 1` integers cannot become `uint32` | chosen | one CellBlock type pair (5.1); `float64` inputs lose precision on import | Nov 7, 2026, plan 5.1, spec 2.1 |
| Explicit zeros in layers | CSR may store explicit zeros; `nnz` counts them | scipy keeps them unless `eliminate_zeros()` | never stored (§4.6); `-0.0` reads back as `+0.0`; `nnz` can be smaller | chosen | canonical CSR: one stored form per matrix, byte-identical stores (8.7) | Nov 7, 2026, user (2.1) |
| Unique IDs | duplicate `obs_names`/`var_names` allowed with a warning | - | `cell_id` and `gene_id` must be unique, `KY-E101` on write (§5.2) | chosen | IDs are the join key for every lookup (3.9) | Nov 6, 2026, plan 2.3, 2.6 |
| Names | any string for columns, layers, `obsm` keys | - | layer, embedding and graph names match `[A-Za-z0-9_.-]+`; column names have no `:` (§2.1, §5.6) | chosen | names are Zarr node names and appear in `kind:name` feature names (3.5) | Nov 7, 2026, spec 2.1 |
| Unstored members | `varp`, `uns`, `raw`, dense layers stored | - | not stored in format v0 (§12) | gap | out of scope for v0; each needs a format minor version | Nov 7, 2026, spec 2.1 |
| String metadata columns | `obs`/`var` strings may be plain `object`/`string` or `category` | - | every non-ID string column is a Polars `Enum`; categorical code `-1` becomes null (§5.4) | chosen | one string encoding; category order kept | Nov 7, 2026, user (2.1) |
| Metadata dtypes | any pandas dtype (datetime, nested, complex, ...) | - | only Boolean, integers, floats, `String` IDs and `Enum` in format v0 (§5.3) | gap | v0 scope; later minor versions can add dtypes | Nov 7, 2026, spec 2.1 |
| Unsorted or duplicate CSR indices | allowed in stored CSR | scipy sums duplicates (`sum_duplicates`) | canonicalised on import: sorted, duplicates summed (§4.3, §12) | chosen | canonical CSR, sorted CellBlock indices (5.1) | Nov 7, 2026, spec 2.1 |
| Zeros in graphs | `obsp` CSR may store zeros | - | stored zero-weight entries kept as edges, unlike layers (§7); open until 6.6 | chosen | a distance of 0 to a duplicate cell is an edge | Nov 7, 2026, spec 2.1 |
