# kyto store format, v0 (`format_version` "0.1")

Status: v0, task 2.1. This document is the contract between every kyto writer and
reader. Cite it by section number, for example `format.md §4.3`. Section numbers are
stable: new sections are appended, removed ones stay as "withdrawn". Inside this
document `§4.3` means a section here, and a bare number such as `5.1` or `plan 5.1`
means a task in [Implementation Plan.md](../Implementation%20Plan.md).

Design goal (plan section 2): reading a few genes costs about the same as reading a
few cells. The matrix is cut into tiles of cell block x gene block, so a gene
selection reads one gene block per cell block and a cell selection reads only the
cell blocks it touches.

## 1. Conventions

### 1.1 Terms

| Term | Meaning |
| --- | --- |
| store | A directory (or object-store prefix) laid out as in §2. It is a Zarr v3 hierarchy plus a few non-Zarr files. |
| cell, gene | The two named axes (R4). Cells are rows of every layer, genes are columns. |
| `n_cells`, `n_genes` | Axis lengths, recorded in `schema.json` (§8). |
| `C`, `G` | `cell_block_size` and `gene_block_size` (§4.1). Same for every layer of a store. |
| cell block `b`, gene block `g` | Consecutive runs of `C` cells and `G` genes (§4.1). |
| tile `(b, g)` | The submatrix of one layer at cell block `b` and gene block `g`. |
| tile number `t` | `t = b * n_gene_blocks + g` (cell-block major). |
| nnz | Number of stored entries. |
| CellBlock | The executor's in-memory batch of CSR rows (plan 5.1). Not a storage unit. |

### 1.2 Requirement words

MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. A store that breaks a MUST is
invalid. Readers are not required to validate everything, but they MUST check the
cheap invariants: `format_version` (§11), `schema.json` against member shapes and
dtypes on open (§8), `tile_offsets[n_tiles] == nnz` on open, and
`indptr[b, g, C] == tile_offsets[t+1] - tile_offsets[t]` for every tile they read. A
reader that detects a violation raises a coded `KY-E1xx` storage error (R10). The codes
are registered by the task that implements the check, not by this spec.

### 1.3 Numbers and bytes

- Integers are written `u32`, `u64` (unsigned) and `i64`. Zarr data types use their
  Zarr v3 names (`uint32`, `uint64`, `float32`, `float64`).
- All binary data is **little endian**. Every Zarr array uses the `bytes` codec with
  `"endian": "little"`. A reader MAY reject any other endianness.
- All arithmetic on sizes and offsets (`C + 1`, `C * G`, `b * C`, `t`) MUST be done in
  `u64`. Doing it in `u32` overflows at the limits of §4.5.
- JSON files are UTF-8, with no byte order mark.

## 2. Directory tree

```
<store>/
  zarr.json                  root Zarr group; holds format_version (§3)       required
  schema.json                axis lengths, tiling, tables, members (§8)        required
  provenance.json            how the store was made (§9)                       required
  cells.parquet              cell table, one row per cell (§5)                 required
  genes.parquet              gene table, one row per gene (§5)                 required
  layers/                    Zarr group                                        required
    zarr.json
    <layer>/                 Zarr group, one tiled sparse matrix (§4)          zero or more
      zarr.json
      indptr/                uint32 [n_cell_blocks, n_gene_blocks, C + 1]
      tile_offsets/          uint64 [n_tiles + 1]
      indices/               uint32 [nnz]
      data/                  float32 or uint32 [nnz]
  embeddings/                Zarr group                                        required
    zarr.json
    <embedding>/             Zarr array, dense [n_axis, width] (§6)            zero or more
  graphs/                    Zarr group                                        required
    zarr.json
    <graph>/                 Zarr group, CSR cell x cell (§7)                  zero or more
      zarr.json
      indptr/                uint64 [n_cells + 1]
      indices/               uint32 [nnz]
      data/                  float32 or float64 [nnz]
```

- Each Zarr array directory holds its own `zarr.json` and its chunk keys (`c/...`) as
  the Zarr v3 spec defines. They are not repeated above.
- The three groups `layers`, `embeddings` and `graphs` MUST exist, even when empty.
  Zarr v3 has no implicit groups, so each has a `zarr.json`.
- `cells.parquet`, `genes.parquet`, `schema.json` and `provenance.json` are plain
  files. Zarr readers ignore them; they are not Zarr nodes.
- The name `modalities` is reserved at the root. v0 writers MUST NOT create it.
- Readers MUST ignore entries this section does not name, at the root and inside
  member groups (for example `.DS_Store`). Writers MUST NOT create any.

### 2.1 Member names

Names of layers, embeddings and graphs are Zarr node names and also appear in
resolved feature names (`log1p:CD3E`, R6). They MUST:

- match `[A-Za-z0-9_.-]+` and be at most 64 bytes,
- not start with `.` or `__` (Zarr reserves `__`), and not end with `.`,
- not be a Windows reserved device name (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`,
  `LPT1`-`LPT9`, in any case, with or without an extension).

Names MUST be unique within their group, compared case-insensitively (macOS and
Windows file systems are case-insensitive). A layer, an embedding and a graph MAY share a
name; the feature resolver (3.5) tells them apart by kind
([feature_names.md](feature_names.md)). Column names follow §5.6.

## 3. Root group and `format_version`

The root `zarr.json` is a Zarr v3 group whose attributes carry the version under the
`kyto` key:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "kyto": {
      "format_version": "0.1"
    }
  }
}
```

- `format_version` is a string `"MAJOR.MINOR"`, both parts decimal integers without
  leading zeros. This document defines `"0.1"`. Compatibility rules are in §11.
- A directory is a kyto store only if the root `zarr.json` exists and has
  `attributes.kyto.format_version`. Writers write the root `zarr.json` **last** (§10.2),
  so it doubles as the commit marker.
- `format_version` lives only here. No other member repeats it.

## 4. Tiled sparse layers

A layer is one `n_cells x n_genes` sparse matrix. Every layer of a store has the same
shape and the same tiling, so tiles of two layers line up.

### 4.1 Tiling

- `C = cell_block_size` and `G = gene_block_size` are positive integers recorded once
  per store in `schema.json` `tiling` (§8). They are writer parameters (plan 2.2).
- `n_cell_blocks = ceil(n_cells / C)`, `n_gene_blocks = ceil(n_genes / G)`,
  `n_tiles = n_cell_blocks * n_gene_blocks`.
- Cell block `b` holds cells `[b*C, min((b+1)*C, n_cells))`; its row count
  `rows(b)` is `C` except possibly for the last block.
- Gene block `g` holds genes `[g*G, min((g+1)*G, n_genes))`; its width `width(g)` is
  `G` except possibly for the last block.
- **Genes are tiled in table order** (decided by the user, Nov 7, 2026): matrix column
  `j` is row `j` of `genes.parquet`, and gene `j` lives in gene block `floor(j / G)`.
  There is no permutation. Likewise matrix row `i` is row `i` of `cells.parquet`.
- Every cell block except the last has exactly `C` rows, in every layer. A writer whose
  input arrives in other batch sizes re-blocks it (§10.2).

**Defaults.** The format has no default tile sizes; any valid `C` and `G` are
readable. Writer defaults are set by task 10.4, which records them and its benchmark
table here.

### 4.2 Arrays of a layer

Layer `<layer>` is a Zarr group `layers/<layer>/` with attributes
`{"kyto": {"kind": "tiled_csr"}}` and exactly these four arrays:

| Array | Shape | Dtype | `dimension_names` | Meaning |
| --- | --- | --- | --- | --- |
| `indptr` | `[n_cell_blocks, n_gene_blocks, C + 1]` | `uint32` | `["cell_block", "gene_block", "row"]` | Per-tile row pointers, local to the tile (§4.3) |
| `tile_offsets` | `[n_tiles + 1]` | `uint64` | `["tile"]` | Start of tile `t` in `indices` and `data` |
| `indices` | `[nnz]` | `uint32` | `["entry"]` | Gene index local to the gene block |
| `data` | `[nnz]` | `float32` or `uint32` | `["entry"]` | Stored values |

- The value dtype of a layer is the dtype of its `data` array: `float32` or `uint32`,
  matching the CellBlock (plan 5.1). No other dtype is valid in v0. `schema.json`
  repeats it (§8) and the two MUST agree.
- `fill_value` is `0` for all four arrays.
- `nnz` is the layer's total stored entries; `tile_offsets[n_tiles] == nnz`.

### 4.3 Tile encoding (CSR inside each tile)

Tile `t = (b, g)` is a CSR matrix of `rows(b)` rows and `width(g)` columns:

- Its entries occupy positions `[tile_offsets[t], tile_offsets[t+1])` of `indices` and
  `data`. Tiles are stored back to back in tile-number order (cell-block major), so
  `tile_offsets` is non-decreasing, `tile_offsets[0] == 0`, and
  `tile_offsets[t+1] - tile_offsets[t]` is the tile's nnz.
- `indptr[b, g, r]` is the number of entries of the tile in local rows `0..r`, so
  `indptr[b, g, 0] == 0` and row `r` holds entries
  `tile_offsets[t] + indptr[b, g, r] .. tile_offsets[t] + indptr[b, g, r+1]`.
- For `r > rows(b)` (padding of the last cell block), `indptr[b, g, r] ==
  indptr[b, g, rows(b)]`. Hence `indptr[b, g, C]` is always the tile's nnz and MUST
  equal `tile_offsets[t+1] - tile_offsets[t]`.
- Within a row, `indices` are **strictly increasing** (sorted, no duplicates) and each
  is `< width(g)`. The global gene index is `g*G + indices[k]`; the global cell index
  is `b*C + r`.
- An empty tile has all-zero `indptr` and `tile_offsets[t] == tile_offsets[t+1]`.

`tile_offsets` is redundant with `indptr` (it is the prefix sum of the tiles' nnz). It
is stored so a reader finds a tile's byte range without reading every `indptr` chunk;
the two MUST agree.

### 4.4 Chunks, shards and codecs

All four arrays use the `regular` chunk grid. `indptr`, `indices` and `data` use the
Zarr v3 `sharding_indexed` codec; `tile_offsets` does not.

| Array | Chunk (shard) shape | Inner chunk shape |
| --- | --- | --- |
| `indptr` | `[1, max(1, n_gene_blocks), C + 1]`: one shard per cell block | `[1, 1, C + 1]`: one tile |
| `indices`, `data` | `[S]` | `[K]` |
| `tile_offsets` | `[65536]`, not sharded | none |

- `K` (`nnz_chunk_size`) and `S` (`nnz_shard_size`) are writer parameters with
  `1 <= K`, `S` a multiple of `K`, and the same values for `indices` and `data` of one
  layer. Readers take them from `zarr.json`. Their defaults come from 10.4.
- Codec chain of a sharded array:

  ```json
  {
    "name": "sharding_indexed",
    "configuration": {
      "chunk_shape": ["<inner chunk shape>"],
      "codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": "<L>", "checksum": false}}
      ],
      "index_codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "crc32c"}
      ],
      "index_location": "end"
    }
  }
  ```

  `tile_offsets` uses `[bytes(little), zstd(L)]` directly. The zstd level `L` is a
  writer parameter recorded in each `zarr.json`.
- Writers MUST NOT store a chunk or inner chunk whose values all equal the fill value
  (Zarr reads a missing chunk as fill values). This keeps empty tiles free and makes
  stores byte-reproducible (§10.3). Readers MUST treat a missing chunk as all fill
  values, as Zarr requires.
- Because entry chunks are cut by count, not by tile, reading one tile decodes the inner
  chunks that overlap its entry range: at most two partially used ones at its edges.
  If `K` is much larger than a typical tile's nnz, each tile read decodes entries of
  neighbouring tiles. 10.4 MUST choose the default `K` relative to typical tile nnz
  (`C * G * density`), not on its own.
- Edge chunks and the last shard always have the full chunk shape; the part beyond the
  array shape holds the fill value (Zarr v3 rule). A chunk shape larger than the array
  shape is valid.

### 4.5 Index dtypes and overflow limits

Writers MUST check these before writing and fail with a coded error; readers MAY trust
them.

| Quantity | Limit | Why |
| --- | --- | --- |
| `C * G` | `<= 2^32 - 1` | tile nnz `<= rows(b) * width(g) <= C * G` fits `indptr` (`u32`) |
| `n_genes` | `<= 2^32 - 1` | global gene index (`u32` in CellBlock `indices`, 5.1) |
| `n_cells` | `<= 2^32 - 1` | graph `indices` (`u32`, §7) and Polars' default row index (`u32`) |
| layer or graph `nnz` | `<= 2^63 - 1` | `tile_offsets` and graph `indptr` are `u64`; the `i64` bound keeps NumPy, Arrow and SciPy interop safe |
| local gene index | `< width(g) <= G` | fits `u32` because `G <= 2^32 - 1` |
| one `indptr` shard | `max(1, n_gene_blocks) * (C + 1) * 4 <= 2^31` bytes | the writer buffers one shard per cell block; also caps tiny `G` with large `C` |
| `K`, `S` (entries) | `K * 4 <= 2^31` and `S * 4 <= 2^31` bytes | one inner chunk and one shard are buffered whole |
| embedding chunk | `C * width * itemsize <= 2^31` bytes (cells; `G` for genes) | one chunk is buffered whole |

The CellBlock's `indptr` is `u64` (5.1) because one CellBlock spans all selected gene
blocks, so its nnz can exceed `u32`. A tile's cannot, by the first row.

### 4.6 Zeros, absence and special values (R9)

- **An absent entry is exactly 0.** A layer has no nulls.
- **Explicit zeros are forbidden** (decided by the user, Nov 7, 2026): every stored
  `data` value is non-zero. For `float32`, `-0.0` counts as zero and is not stored, so
  it reads back as `+0.0`. With sorted, unique indices (§4.3) this makes the stored form
  of a matrix unique (canonical CSR), which 8.7 needs.
- Writers drop zeros produced by computation (for example after a subtraction) before
  writing. Readers MAY assume none are stored.
- `NaN` and `+-inf` are values, not absence, and are stored like any non-zero value.

### 4.7 Reading a tile

To read tile `(b, g)` of a layer (the unit task 2.4 counts):

1. Read `tile_offsets` once when the store is opened (it is small: 8 bytes per tile).
2. Let `t = b * n_gene_blocks + g`, `start = tile_offsets[t]`, `end = tile_offsets[t+1]`.
   If `start == end` the tile is empty: stop.
3. Read inner chunk `(b, g)` of `indptr` (one shard index lookup plus one chunk).
4. Read `indices[start..end]` and `data[start..end]`.

A gene selection maps to the set of gene blocks `{floor(j / G)}` of its genes; a cell
selection maps to the set of cell blocks `{floor(i / C)}`. A reader reads exactly the
tiles in the product of the two sets and never touches others. Entries of genes or
cells outside the selection inside a read tile are filtered in memory.

### 4.8 From tiles to a CellBlock

A CellBlock (5.1) holds rows of one or more cells over the selected genes. To build it
for cell block `b`, take the read tiles `(b, g)` in increasing `g`, and for each row
concatenate that row's entries from each tile, converting local gene indices to global
ones (`g*G + local`). CellBlock `indices` are positions in the gene selection, and the
selection is always held as sorted, unique table positions, so increasing `g` keeps
indices sorted. A user selection in another order or with repeats is canonicalised
by the plan; any reordering for output happens at the terminal, not in the CellBlock. A CellBlock MAY cover only part of a cell block (5.5 shrinks blocks to
fit a memory limit); tiles stay the storage unit.

## 5. Metadata tables

### 5.1 Files and row order

- `cells.parquet` has exactly `n_cells` rows; row `i` describes matrix row `i`.
- `genes.parquet` has exactly `n_genes` rows; row `j` describes matrix column `j`.
- Each is a single Parquet file written with Polars (plan 2.3).

### 5.2 ID columns

- The first column of `cells.parquet` is `cell_id`; of `genes.parquet`, `gene_id`.
- Its dtype is `String`. Values are non-null, non-empty and **unique** (exact byte
  comparison, no normalisation). Writers check this and raise `KY-E101` on duplicates
  (plan 2.3).
- `cell_id` and `gene_id` are reserved: no other column of either table may use them.
- `symbol` is reserved in `genes.parquet` for gene symbols (plan 3.9). It is optional;
  when present it follows §5.3 like any string column. Symbols need not be unique.

### 5.3 Column dtypes

Allowed Polars dtypes in v0: `Boolean`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`,
`UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `String` (ID columns only) and
`Enum` (§5.4). Nested, temporal, binary, `Categorical` and `Object` dtypes are not
valid in v0.

### 5.4 Categoricals (Enum)

Decided by the user, Nov 7, 2026: **every string column other than the ID column is a
Polars `Enum`.**

- The category list is fixed per column, unique, non-null, and kept in order,
  including categories no row uses (as anndata categoricals do).
- The categories live in the Parquet file only: the column is dictionary encoded and
  carries Polars' Enum type metadata. `schema.json` records the category count and an
  `ordered` flag (§8), so a column with millions of categories (barcodes, free text)
  does not bloat it. `ordered` is kept for a lossless round trip to anndata, and it
  decides comparisons in expressions: ordering (`<`, `<=`, `>`, `>=`) an unordered
  Enum is `KY-E205`, an ordered one compares in category order
  ([feature_names.md](feature_names.md) §5, task 3.5).
- An Enum with zero categories (an all-null column) is valid. The empty string is a
  valid category.
- A writer given a plain `String` column stores it as an `Enum` whose categories are
  the column's distinct non-null values in byte order, as anndata sorts the categories
  of string columns. An `Enum` given to a writer keeps its category list as is.

### 5.5 Nulls

- Any non-ID column may hold nulls. Null means missing (R9).
- `NaN` in a float column is a value, not null. kyto's own writers (the store writer,
  plan sinks) never turn one into the other.
- Importers map their source's missing values to null (§12). The `.h5ad` importer also
  makes `NaN` in an `obs` or `var` float column null: anndata has no nullable float
  encoding, so there `NaN` means missing (decided by the user, Nov 7, 2026, task 3.7).
  Layer values are never changed: a `NaN` in a matrix stays a value. The 10x and mtx
  importers read string metadata only. The `.h5ad` export writes float nulls as `NaN`,
  so after a round trip through `.h5ad` a float `NaN` of a store comes back as null.

### 5.6 Column names

Column names are unique within a table, non-empty, valid UTF-8, and MUST NOT contain
`:` (the feature resolver uses `<prefix>:<name>`, [feature_names.md](feature_names.md)).

### 5.7 Parquet settings

- Compression `zstd`. Column statistics on (they drive predicate pushdown, 2.4).
- `cells.parquet` SHOULD use one row group per cell block (`C` rows), so row-group
  statistics map to cell blocks. `genes.parquet` SHOULD use row groups of `G` rows.
  Readers MUST NOT rely on row-group boundaries.

## 6. Embeddings

Embedding `<embedding>` is a Zarr array `embeddings/<embedding>`:

| Field | Value |
| --- | --- |
| attributes | `{"kyto": {"axis": "cells"}}` or `{"kyto": {"axis": "genes"}}` |
| shape | `[n_cells, width]` or `[n_genes, width]`, `width >= 1` |
| dtype | `float32` or `float64` |
| `dimension_names` | `["cell", "component"]` or `["gene", "component"]` |
| chunk shape | `[C, width]` (cells) or `[G, width]` (genes), aligned to blocks; not sharded |
| codecs | `[bytes(little), zstd(L)]` |
| `fill_value` | `0` |

- Dense: every element is stored (subject to the all-fill-chunk rule of §4.4). There
  are no nulls; `NaN` is a value.
- Row `i` of a cells embedding is cell `i`; row `j` of a genes embedding is gene `j`
  (for example PCA loadings, anndata `varm`).

## 7. Graphs

Graph `<graph>` is a Zarr group `graphs/<graph>/` with attributes
`{"kyto": {"kind": "csr", "axis": "cells"}}`: an `n_cells x n_cells` sparse matrix in
plain (untiled) CSR. v0 has cell graphs only (`axis` is always `"cells"`).

| Array | Shape | Dtype | Chunks |
| --- | --- | --- | --- |
| `indptr` | `[n_cells + 1]` | `uint64` | `[65536]`, not sharded |
| `indices` | `[nnz]` | `uint32` | sharded like layer `indices` (§4.4), with the graph's own `K` and `S` |
| `data` | `[nnz]` | `float32` or `float64` | sharded like layer `data` (§4.4), with the graph's own `K` and `S` |

- Row `i` holds entries `indptr[i]..indptr[i+1]`; within a row `indices` are strictly
  increasing and `< n_cells`. `indptr[0] == 0`, `indptr[n_cells] == nnz`.
- A stored entry is an **edge**; its weight MAY be 0 (for example a distance of 0 to a
  duplicate cell). Absence means no edge, read as 0 when densified. This differs from
  layers (§4.6) on purpose; see §13.2.
- Self loops are allowed. Symmetry is not required.
- Graphs are untiled because they are read whole (6.7 to 6.9).

## 8. `schema.json`

One JSON object summarising the store so it can be inspected without reading the
matrix (8.3). It is the input of `AtlasSchema` (3.1). Keys, in this order:

```json
{
  "n_cells": 2700,
  "n_genes": 32738,
  "tiling": {"cell_block_size": 4096, "gene_block_size": 1024},
  "cells": {
    "columns": [
      {"name": "cell_id", "dtype": "String"},
      {"name": "louvain", "dtype": "Enum", "n_categories": 3, "ordered": false},
      {"name": "n_genes", "dtype": "UInt32", "lineage": {"layer": "counts", "node": "n3"}}
    ]
  },
  "genes": {
    "columns": [
      {"name": "gene_id", "dtype": "String"},
      {"name": "symbol", "dtype": "Enum", "n_categories": 32738, "ordered": false}
    ]
  },
  "layers": [{"name": "counts", "dtype": "uint32", "nnz": 2286884}],
  "embeddings": [{"name": "X_pca", "axis": "cells", "width": 50, "dtype": "float32"}],
  "graphs": [{"name": "connectivities", "axis": "cells", "dtype": "float32", "nnz": 37134}]
}
```

- `columns` lists every Parquet column, in Parquet order, ID column first. `dtype` uses
  the Polars dtype names of §5.3 (array dtypes below use Zarr names). `n_categories` and
  `ordered` appear exactly when `dtype` is `"Enum"`.
- `lineage` is optional and lives only here. It marks a stored derived column with the
  layer and plan node it came from. Node IDs are the stable, deterministic IDs of the
  plan JSON (4.3), so lineage does not break 8.7. Its use (3.8): a cell column with
  `lineage` was computed over every gene of the store, so a plan that changes the gene
  axis (`filter_genes`) refuses to read it as stored (`KY-E301`) and reads it
  recomputed on request (`cell_column(name).recompute()`, which needs `lineage`:
  `KY-E302`). Readers do not interpret `layer` and `node` yet; the executor (5.x) uses
  them to recompute. Importers write no `lineage`.
- JSON integers (`nnz`, counts) can exceed `2^53`; readers MUST parse them as `u64`,
  not as floating point.
- `layers`, `embeddings` and `graphs` list every member of the matching group, sorted by
  name (byte order). Array dtypes use Zarr names.
- Every value here MUST agree with the member it describes (Parquet schema, `zarr.json`
  shapes and dtypes). A reader that finds a mismatch raises a coded storage error.
- Unknown keys: see §11.

## 9. `provenance.json`

How the store was made (task 8.5 fills it in; 4.3 defines the plan JSON). v0 keys:

```json
{
  "kyto_version": "0.1.0",
  "dependencies": {"polars": "1.x.y", "zarrs": "0.x.y", "zstd": "1.x.y"},
  "plan": null,
  "source": {"kind": "h5ad", "path": "pbmc3k.h5ad", "sha256": "..."},
  "seeds": {"n7": 0},
  "device": "cpu",
  "run": {"started_at": "2026-11-07T10:00:00Z", "finished_at": "2026-11-07T10:00:05Z", "timings": {"n7": 1.25}}
}
```

| Key | Required | Meaning |
| --- | --- | --- |
| `kyto_version` | yes | kyto package version that wrote the store |
| `dependencies` | yes | versions of libraries that affect the bytes or the numbers, name to version string |
| `plan` | yes | the plan JSON (4.3) that produced the store, or `null` before 4.3 exists or for a plain import |
| `source` | yes | for an import: input kind, path as given, and the SHA-256 of the input file (computed while streaming it); otherwise `null` |
| `seeds` | yes | plan node ID to seed, for every seeded node (`{}` if none) |
| `device` | yes | `"cpu"` or a device name (9.x) |
| `run` | yes | timestamps (RFC 3339, UTC) and per-node timings in seconds |

`run` and `source.path` are the only parts that vary between identical runs; 8.7
excludes them when comparing stores (§10.3). Unknown keys: see §11.

## 10. Writing a store

### 10.1 Atomic write

1. The writer creates a temporary directory next to the target, in the same parent
   directory (so the rename stays on one filesystem):
   `<parent>/.<target name>.kyto-tmp-<random>`.
2. It writes the whole store there (order in §10.2), then fsyncs every file and
   directory in it, so a power loss after step 3 cannot expose truncated files.
3. It renames the temporary directory to the target path and fsyncs the parent
   directory. Only then does the target exist. If the target already exists the writer
   fails before step 1, unless the caller asked to overwrite. Overwrite renames the old
   store aside to `<parent>/.<target name>.kyto-old-<random>`, renames the new one in,
   then deletes the old one (never rename onto an existing directory: Windows refuses,
   POSIX only allows it when empty). 2.5 implements this.
4. On any failure it removes the temporary directory (best effort). A leftover
   `.kyto-tmp-` directory never has a target name, is never a valid store, and is safe
   to delete.
5. **Stale temporaries** (a killed writer runs no cleanup). `<random>` is 16 lowercase
   hex digits. While it writes, a writer holds an OS file lock on
   `<parent>/.<target name>.kyto-tmp-<random>.lock`, created before the directory and
   deleted after the rename. The OS releases the lock when the process dies, even by
   SIGKILL. A writer starting on a target first deletes every
   `.<target name>.kyto-tmp-<random>` directory of that target whose lock it can take,
   and lock files left without a directory; directories whose lock is held belong to a
   live writer and are kept. `.kyto-old-` directories are never deleted automatically:
   after a kill between the two renames of an overwrite, the old store there is the only
   copy.

Implemented by `kyto_store::StoreWriter` (2.5). If the target exists, it fails with
`KY-E118`; with overwrite it replaces only a kyto store (a directory whose root
`zarr.json` has `attributes.kyto.format_version`), never another file or directory.
Writes to one target from two processes at once are safe (each has its own temporary
directory) but only the first rename succeeds. A symbolic link at the target is not
followed: overwrite replaces the link itself with the new store and leaves what it
pointed at untouched.

A killed write therefore leaves the target path absent (2.5 Done when). Object stores
(2.8) have no atomic rename; there the root `zarr.json`, written last, is the commit
marker (§3).

### 10.2 Write order (streaming)

- **Before the first CellBlock** the writer MUST know the final gene axis (`n_genes` and
  gene order), because it fixes `n_gene_blocks` and every tile. A plan whose gene set is
  known only after a pass over the data (for example `filter_genes(min_cells=3)`) runs
  that pass first (5.4); the planner guarantees this, the writer checks it.
- Matrix rows arrive as CellBlocks of any size, each carrying the same rows for every
  layer being written. The writer re-blocks them into cell blocks of exactly `C` rows
  (the last may be shorter) and, per cell block and per layer, writes the `indptr` shard
  and appends that block's tiles (gene blocks in increasing order) to `indices` and
  `data`, flushing whole shards of `S` entries as they fill. The last shard is flushed
  at the end, full shape (§4.4).
- `cells.parquet` is written in row groups as cell blocks complete when its schema,
  including every Enum category list, is known before the first row group. Otherwise
  (for example a cluster label computed at the end) the writer buffers the cell table,
  which is small next to the matrix, and writes it at the end.
- Shapes (`n_cells`, `nnz`) are known only at the end. In this order, at the end: each
  array's `zarr.json`, `tile_offsets`, the embeddings and graphs, the `zarr.json` of
  every layer and graph group and of `layers/`, `embeddings/` and `graphs/`,
  `genes.parquet` (and `cells.parquet` if buffered), `provenance.json`, then
  `schema.json` with final counts.
- The root `zarr.json` is written **last**.
- Writer memory (2.5): per layer, fewer than `C` buffered rows, fewer than `S`
  unwritten entries and `tile_offsets` (8 bytes per tile) between pushes; during a push,
  a copy of the pushed rows and one `indptr` shard. Never a whole layer. `StoreWriter`
  takes the cell table whole at the end (the buffered path above).

### 10.3 Determinism (R11, 8.7)

The same input, plan and seed, written by the same kyto version with the same
`dependencies` (zstd and Parquet writer versions change bytes), MUST give
byte-identical files for every member except the `run` object and `source.path` of
`provenance.json`, on any thread count. This follows from: fixed
tile order (§4.3), canonical CSR (§4.6), no all-fill chunks (§4.4), fixed codec
parameters, JSON keys in the order this spec lists them, member lists sorted by name,
and no timestamps outside `run`.

## 11. Versioning and compatibility

- Writers write the `format_version` they implement (§3).
- A **minor** bump (`0.1` to `0.2`) adds only optional members, optional JSON keys or new
  allowed values. A **major** bump changes the meaning of an existing member or adds a
  required one.
- **While MAJOR is 0, there is no compatibility promise.** A reader accepts exactly the
  `0.x` versions it implements and rejects every other version with a coded storage
  error that names the store's version and the supported ones.
- **From 1.0 on** (task 10.6 states the release policy): a reader that implements `1.m`
  reads any `1.x`. For `x > m` it ignores members and JSON keys it does not know. It
  rejects a different major.
- Readers never modify a store, so reading never upgrades a version.

## 12. Differences from anndata's on-disk Zarr format

anndata stores `X`, `layers`, `obs`, `var`, `obsm`, `varm`, `obsp`, `varp`, `uns` and
`raw` as Zarr nodes with `encoding-type` attributes: dense matrices as arrays, sparse
ones (`csr_matrix`, `csc_matrix`) as groups, dataframes as groups. CSR may hold
unsorted or duplicate indices; categoricals use code `-1` for missing; nullable
integers and booleans carry a mask. kyto maps them as follows.

| anndata | kyto v0 | Note |
| --- | --- | --- |
| `X` | a named layer | No unnamed matrix (R4). `convert` (2.6) names it by its `layer` argument, default `counts`. |
| `layers/*` | `layers/<name>` | Tiled canonical CSR only (§4); dense, CSC, unsorted and duplicate-index inputs are converted on import (duplicates summed, as scipy does). |
| `obs`, `var` (dataframe groups, `_index`) | `cells.parquet`, `genes.parquet` with `cell_id`, `gene_id` | IDs must be unique (§5.2); anndata only warns. |
| categoricals (`codes` + `categories`, `ordered`) | Polars `Enum` + `ordered` in `schema.json` | Category order and unused categories kept; code `-1` becomes null (§5.4). Plain string columns also become `Enum`. |
| nullable masks in `obs`, `var` | Polars nulls | Masked entries become null (§5.5). |
| `NaN` in `obs`, `var` float columns | Polars nulls | anndata's missing float (no nullable float encoding); `NaN` in `X` and `layers` stays a value (§5.5). |
| `obsm`, `varm` | `embeddings/<name>` with `axis` | Dense `float32`/`float64` arrays only; sparse or dataframe entries are not stored. |
| `obsp` | `graphs/<name>` | Cell graphs only, stored as CSR even when the source is dense; stored zeros kept (§7). |
| `varp`, `uns`, `raw` | not stored | v0 gap. |
| any value dtype | layers `float32` or `uint32` | 5.1 CellBlock types. |
| explicit zeros allowed in sparse `X` | forbidden (§4.6) | nnz can be smaller than anndata's. |

Semantic differences are rows in [deviations.md](../deviations.md).

## 13. Decisions and open questions

### 13.1 Decisions (user, Nov 7, 2026, task 2.1)

| Question | Options considered | Decision |
| --- | --- | --- |
| Tile layout | flat CSR arrays + `tile_offsets` with regular chunks; same with the rectilinear chunk grid (one chunk per tile, a Zarr extension); one variable-length blob per tile | flat CSR arrays + `tile_offsets` (§4.2): any Zarr v3 reader can open it |
| Gene order | fixed table order; writer-chosen permutation | fixed table order (§4.1) |
| String columns | Polars `Enum`; `Categorical`; dictionary-encoded `String` | `Enum` (§5.4) |
| Explicit zeros in layers | forbidden (canonical); allowed | forbidden (§4.6) |

### 13.2 Open questions

- **Graph zeros.** §7 allows stored zero-weight edges, unlike layers, because an edge
  with distance 0 is information. Revisit with 6.6 (neighbours) against scanpy's
  `distances`.
- **`n_cells <= 2^32 - 1`.** Set by `u32` graph indices and Polars' default index. Raise
  it in a later version if an atlas needs more.
- **Dense layers, `varp`, `uns`, modalities.** Not in v0; each needs a minor version.
