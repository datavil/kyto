# Feature names

How kyto turns the names in an expression into the data they read (task 3.5, R6).
The resolver is `kyto_core::Resolver`; Python reaches it through `Atlas.resolve`.

## 1. When names resolve

A plan resolves every name once, when the expression is given to it, against what the
plan saw when it was built: the store's `schema.json`, the `gene_id` column of
`genes.parquet` (both read by the scan) and the plan's default layer
(`with_default_layer`). The resolved expression names everything in full and is what
the plan stores. Adding a column, a layer or a gene to the store, or setting another
default layer downstream, never changes an expression that is already resolved.

## 2. Full names

`feature(name)` takes a short or a full name. A full name is `<prefix>:<rest>`, split
at the **first** `:` (so a gene ID may contain `:`). There are three forms:

| Form | Means | Built by |
| --- | --- | --- |
| `meta:<column>` | a column of the cell table | `cell_column(column)` |
| `<layer>:<gene_id>` | gene `gene_id` read from layer `layer` | `gene_expression(gene_id, layer=layer)` |
| `<embedding>:<component>` | component `component` (0-based) of a cell embedding | `cell_embedding(embedding, component)` |

- `<component>` is written in decimal, without a sign or leading zeros (`0`, `12`).
- Column names never contain `:` (format.md §5.6); member names match
  `[A-Za-z0-9_.-]+` (format.md §2.1).

**Decided by kind, not by syntax.** No prefix is reserved and no layer or embedding
name is forbidden: a layer may be named `meta`, and a layer and an embedding may share a
name (format.md §2.1). The resolver tries every form against the schema and keeps
those that exist:

- `meta:<column>` matches if the cell table has that column.
- `<layer>:<gene_id>` matches if `layer` is a layer and `gene_id` is a gene ID.
- `<embedding>:<component>` matches if `embedding` is a cell embedding and
  `component` is below its width.

## 3. Short names

A short name has no `:`. It matches:

- the cell table column of that name, and
- the gene of that ID, read from the plan's default layer.

A short name never names an embedding (it needs a component). A `feature(...)` name
is never a gene symbol: full names hold gene IDs.

## 3.1 Gene symbols

`gene_expression(symbol, by="symbol")` names a gene by its symbol, the `symbol` column
of `genes.parquet` (format.md §5.2), matched exactly (case included; nulls match
nothing). The resolver turns it into the gene's ID once, when the plan is built (R6):
the stored expression is `gene_expression(<gene_id>, layer=...)`, so its full name is
`<layer>:<gene_id>`. Symbols need not be unique, so:

| Genes with this symbol | Result |
| --- | --- |
| exactly one | that gene's ID |
| none, or no string `symbol` column | `KY-E210` (kind `gene symbol`) |
| two or more | `KY-E204`, listing their IDs in table order |

## 4. One match or an error

There is no priority order.

| Matches | Result |
| --- | --- |
| exactly one | that feature |
| none | `KY-E210`, saying why for each form that could apply |
| two or more | `KY-E201`, listing every full name it matches |
| a gene, no default layer, and no column | `KY-E211`, listing the layers |

Where two matches have the same full name (a layer `meta` with a gene `cluster`, and a
column `cluster`), `KY-E201` adds the kind to each (`meta:cluster (cell column)`,
`meta:cluster (gene)`); name the one you mean with its constructor instead. A short
gene name with no default layer that also names a column is `KY-E201` and lists the
gene in every layer.

## 5. The other constructors

The resolver checks the names given to every constructor too:

- `cell_column` / `gene_column`: the column must exist in the cell / gene table
  (`KY-E210`); its stored dtype sets the expression's kind, and every operation on it
  is checked again (`KY-E205`). Ordering (`<`, `<=`, `>`, `>=`) an unordered
  categorical column is `KY-E205`, as pandas raises for unordered categoricals; an
  ordered one compares in category order.
- `gene_expression`: the gene ID must exist (`KY-E210`); a named layer must exist
  (`KY-E109`); without one the default layer is written in, or `KY-E211`.
- `cell_profile`: the cell ID must not be empty (`KY-E210`); the layer as above. The
  plan holds no cell IDs, so whether the cell exists is checked when the plan runs.
- `cell_embedding` / `gene_embedding`: the embedding must exist (`KY-E109`), be of the
  expression's axis and have the component (`KY-E210`).

Names are compared exactly, case included. Suggestions for close matches are task 8.2.
