<!-- GENERATED from errors.toml by scripts/error_registry.py. Do not edit. -->

# Error codes

Every kyto error has a stable `KY-Exxx` code (R10). Its message says what was
wrong and where, its hint how to fix it, and its `docs_url` links to its
entry on this page. Codes are never reused or renumbered.

Every error is a `kyto.KytoError` and an instance of a builtin exception, so
`except ValueError` still catches a `ResolutionError`. Catch one code with
`except kyto.KytoError as error` and test `error.code`; the structured fields
below are attributes of the error.

```text
KY-E101: the genes table has 2 IDs that occur more than once, first: 'A', 'B'
hint: IDs on the genes axis must be unique (exact byte comparison); ...
docs: https://github.com/datavil/kyto/blob/main/docs/errors.md#ky-e101
```

Ranges:

- `KY-E1xx`: storage
- `KY-E2xx`: names, axes and expressions (plan build)
- `KY-E3xx`: lineage
- `KY-E4xx`: memory
- `KY-E5xx`: devices
- `KY-E6xx`: not implemented yet (a documented method whose task is not done)

## Exception classes

| Class | Builtin base | Raised when | Codes |
| --- | --- | --- | --- |
| `DuplicateIdError` | `ValueError` | Cell or gene IDs that must be unique are not. | [KY-E101](#ky-e101) |
| `ResolutionError` | `ValueError` | A name or symbol does not resolve to exactly one feature. | [KY-E201](#ky-e201), [KY-E204](#ky-e204), [KY-E208](#ky-e208), [KY-E210](#ky-e210), [KY-E211](#ky-e211), [KY-E214](#ky-e214) |
| `AxisError` | `TypeError` | An expression mixes the cells and genes axes (R5). | [KY-E202](#ky-e202) |
| `RemovedFeatureError` | `ValueError` | A feature removed earlier in the plan is read again (R9). | [KY-E203](#ky-e203) |
| `StaleColumnError` | `ValueError` | A derived column is read after its inputs changed upstream. | [KY-E301](#ky-e301) |
| `LineageError` | `ValueError` | A column is asked to be recomputed but records no lineage to recompute it from. | [KY-E302](#ky-e302), [KY-E303](#ky-e303), [KY-E304](#ky-e304) |
| `MemoryLimitError` | `MemoryError` | A plan cannot run within its memory limit. | [KY-E401](#ky-e401), [KY-E402](#ky-e402) |
| `DeviceError` | `RuntimeError` | The requested device cannot run part of the plan. | [KY-E501](#ky-e501), [KY-E502](#ky-e502), [KY-E504](#ky-e504) |
| `StoreInputError` | `ValueError` | Data, names, parameters or selections given to a store writer or reader break the store format (docs/format.md). | [KY-E102](#ky-e102), [KY-E103](#ky-e103), [KY-E104](#ky-e104), [KY-E105](#ky-e105), [KY-E106](#ky-e106), [KY-E111](#ky-e111), [KY-E112](#ky-e112), [KY-E114](#ky-e114), [KY-E117](#ky-e117), [KY-E118](#ky-e118), [KY-E119](#ky-e119), [KY-E123](#ky-e123), [KY-E124](#ky-e124), [KY-E125](#ky-e125) |
| `CorruptStoreError` | `ValueError` | A store on disk breaks the store format (docs/format.md). | [KY-E107](#ky-e107), [KY-E113](#ky-e113), [KY-E116](#ky-e116), [KY-E122](#ky-e122), [KY-E126](#ky-e126), [KY-E127](#ky-e127) |
| `FormatVersionError` | `ValueError` | A store has a format_version this kyto does not read (format.md section 11). | [KY-E115](#ky-e115) |
| `StoreIOError` | `RuntimeError` | The file system or the Zarr layer failed while reading or writing a store. | [KY-E110](#ky-e110) |
| `MissingMemberError` | `LookupError` | A layer, embedding, graph or modality is not in the store. | [KY-E109](#ky-e109) |
| `LayerDtypeError` | `TypeError` | A layer is read with a value dtype other than the one it stores. | [KY-E108](#ky-e108) |
| `ExpressionError` | `TypeError` | An expression is given an operand, or used in a context, that it does not support. | [KY-E205](#ky-e205), [KY-E206](#ky-e206), [KY-E207](#ky-e207), [KY-E209](#ky-e209), [KY-E212](#ky-e212), [KY-E213](#ky-e213) |
| `InputFileError` | `ValueError` | An input file given to convert has a format, encoding or value kyto cannot import. | [KY-E120](#ky-e120), [KY-E121](#ky-e121) |
| `PlanError` | `ValueError` | A plan method is given an argument outside what it accepts, or cannot extend this plan. | [KY-E215](#ky-e215), [KY-E216](#ky-e216), [KY-E217](#ky-e217), [KY-E218](#ky-e218), [KY-E219](#ky-e219) |
| `RunCancelledError` | `RuntimeError` | A run was cancelled before it finished (Ctrl-C, or its progress callback raised). | [KY-E503](#ky-e503) |
| `NotImplementedYetError` | `NotImplementedError` | A documented method is not implemented yet in this kyto version; its task in the implementation plan adds it. | [KY-E601](#ky-e601) |

## KY-E101

IDs written to a store are not unique.

- Class: `DuplicateIdError`
- Message: `the {axis} table has {n_duplicated} IDs that occur more than once, first: {ids}`
- Hint: `IDs on the {axis} axis must be unique (exact byte comparison); drop or rename the duplicates before writing, or import with on_duplicate_ids='suffix' to make them unique`
- Fields:
    - `axis` (str): The axis, `cells` or `genes`. Example: `genes`.
    - `n_duplicated` (int): How many distinct IDs occur more than once. Example: `12`.
    - `ids` (list[str]): The first duplicated IDs, in the order they repeat (at most 5). Example: `'ENSG01', 'ENSG02'`.

## KY-E102

A layer, embedding or graph name breaks the naming rules of format.md section 2.1.

- Class: `StoreInputError`
- Message: `invalid {kind} name '{name}': {rule}`
- Hint: `use 1 to 64 characters from A-Z a-z 0-9 _ . - that do not start with . or __, do not end with . and are not a Windows device name such as CON`
- Fields:
    - `kind` (str): The member kind: `layer`, `embedding`, `graph` or `modality`. Example: `layer`.
    - `name` (str): The name as given. Example: `log1p/counts`.
    - `rule` (str): The rule it breaks. Example: `character '/' is not allowed`.

## KY-E103

A member with the same name, compared case-insensitively, is already in the store.

- Class: `StoreInputError`
- Message: `{kind} '{name}' already exists in the store as '{existing}'`
- Hint: `choose another name; names are compared case-insensitively because macOS and Windows file systems are`
- Fields:
    - `kind` (str): The member kind: `layer`, `embedding`, `graph` or `modality`. Example: `layer`.
    - `name` (str): The name as given. Example: `Counts`.
    - `existing` (str): The name already in the store. Example: `counts`.

## KY-E104

A store writer parameter (tile or chunk size, compression level) is out of range.

- Class: `StoreInputError`
- Message: `invalid {parameter} = {value}`
- Hint: `{parameter} must be {accepted}`
- Fields:
    - `parameter` (str): The parameter name. Example: `nnz_shard_size`.
    - `value` (str): The value given. Example: `1000`.
    - `accepted` (str): What the parameter accepts. Example: `a positive multiple of nnz_chunk_size (256)`.

## KY-E105

A size is above a limit of the store format (format.md section 4.5).

- Class: `StoreInputError`
- Message: `{quantity} is {value}, above the format limit of {limit}`
- Hint: `{advice} to stay within the limits of format.md section 4.5`
- Fields:
    - `quantity` (str): What is too large. Example: `cell_block_size * gene_block_size`.
    - `value` (int): Its value. Example: `8589934592`.
    - `limit` (int): The largest value allowed. Example: `4294967295`.
    - `advice` (str): How to get under the limit. Example: `use smaller cell_block_size or gene_block_size`.

## KY-E106

A sparse matrix given to a store writer is not valid CSR of the stated shape.

- Class: `StoreInputError`
- Message: `invalid sparse matrix for layer '{layer}': {problem}`
- Hint: `pass CSR with indptr of length n_cells + 1 starting at 0 and non-decreasing, and sorted, unique gene indices below n_genes in each row (format.md section 4.3)`
- Fields:
    - `layer` (str): The layer being written. Example: `counts`.
    - `problem` (str): What is wrong and where. Example: `row 4: gene index 7 follows 9; indices must be strictly increasing`.

## KY-E107

A stored layer breaks the tiled sparse format (format.md section 4).

- Class: `CorruptStoreError`
- Message: `layer '{layer}' is corrupt: {problem}`
- Hint: `the store breaks docs/format.md section 4; rewrite the layer with kyto, or report a bug if kyto wrote it`
- Fields:
    - `layer` (str): The layer being read. Example: `counts`.
    - `problem` (str): The invariant that fails and where. Example: `tile (0, 2): indptr ends at 5 but tile_offsets gives 7 entries`.

## KY-E108

A layer is read with a value dtype other than the one it stores.

- Class: `LayerDtypeError`
- Message: `layer '{layer}' stores {stored} values, not {requested}`
- Hint: `read layer '{layer}' as {stored}`
- Fields:
    - `layer` (str): The layer being read. Example: `counts`.
    - `stored` (str): The Zarr dtype of its data array. Example: `uint32`.
    - `requested` (str): The dtype asked for. Example: `float32`.

## KY-E109

A layer, embedding, graph or modality is not in the store.

- Class: `MissingMemberError`
- Message: `no {kind} named '{name}' in the store at '{store}'{suggestions}`
- Hint: `check the name against the store's schema; names are case-sensitive when read`
- Fields:
    - `kind` (str): The member kind: `layer`, `embedding`, `graph` or `modality`. Example: `layer`.
    - `name` (str): The name asked for. Example: `count`.
    - `store` (str): The store path. Example: `pbmc.kyto`.
    - `suggestions` (suggestions): Up to three members of that kind whose names are closest to `name` (Jaro-Winkler, task 8.2), best first; empty when none is close. Example: `'counts'`.

## KY-E110

The file system or the Zarr layer failed while reading or writing a store.

- Class: `StoreIOError`
- Message: `could not {operation} '{path}': {detail}`
- Hint: `check that the path exists, that you may read and write it, and that the disk is not full`
- Fields:
    - `operation` (str): What was being done. Example: `write array`.
    - `path` (str): The store path or node. Example: `pbmc.kyto/layers/counts/data`.
    - `detail` (str): The underlying error. Example: `No space left on device (os error 28)`.

## KY-E111

The ID column of a cell or gene table holds null or empty IDs.

- Class: `StoreInputError`
- Message: `the {axis} table has {n_invalid} {problem} IDs, first at rows {rows}`
- Hint: `every {axis} ID must be a non-null, non-empty string; fill or drop those rows before writing`
- Fields:
    - `axis` (str): The axis, `cells` or `genes`. Example: `cells`.
    - `n_invalid` (int): How many IDs are invalid. Example: `2`.
    - `problem` (str): What is wrong with them: `null` or `empty`. Example: `null`.
    - `rows` (list[str]): The first rows (0-based) with such an ID (at most 5). Example: `'3', '17'`.

## KY-E112

A column of a cell or gene table breaks format.md section 5 (name, dtype or position).

- Class: `StoreInputError`
- Message: `column '{column}' of the {axis} table {problem}`
- Hint: `{accepted} (format.md section 5)`
- Fields:
    - `axis` (str): The axis, `cells` or `genes`. Example: `cells`.
    - `column` (str): The column name. Example: `sample`.
    - `problem` (str): What is wrong. Example: `has dtype Categorical, which v0 stores do not allow`.
    - `accepted` (str): What would be accepted. Example: `cast it to String or to an Enum with fixed categories`.

## KY-E113

A stored cell or gene table breaks format.md section 5.

- Class: `CorruptStoreError`
- Message: `table '{path}' is corrupt: {problem}`
- Hint: `the store breaks docs/format.md section 5; rewrite the table with kyto, or report a bug if kyto wrote it`
- Fields:
    - `path` (str): The table file. Example: `pbmc.kyto/cells.parquet`.
    - `problem` (str): The invariant that fails. Example: `the first column is 'barcode', not 'cell_id'`.

## KY-E114

A cell or gene selection given to the tile reader is out of range, unsorted or repeats a position.

- Class: `StoreInputError`
- Message: `invalid {axis} selection: {problem}`
- Hint: `select {axis} by sorted, unique table positions below {n_axis}; the plan canonicalises other orders and repeats (format.md section 4.8)`
- Fields:
    - `axis` (str): The axis, `cells` or `genes`. Example: `genes`.
    - `problem` (str): What is wrong and where in the selection. Example: `position 2 is gene 7, which does not follow gene 9`.
    - `n_axis` (int): The axis length of the store. Example: `2000`.

## KY-E115

A store has a format_version this kyto does not read (format.md section 11).

- Class: `FormatVersionError`
- Message: `the store at '{store}' has format_version '{found}', but this kyto reads only {supported}`
- Hint: `open the store with the kyto version that wrote it, or rewrite it with this one`
- Fields:
    - `store` (str): The store path. Example: `pbmc.kyto`.
    - `found` (str): The store's format_version. Example: `0.2`.
    - `supported` (list[str]): The format versions this kyto reads. Example: `'0.1'`.

## KY-E116

A path is not a kyto store: its root zarr.json is missing or has no format_version (format.md section 3).

- Class: `CorruptStoreError`
- Message: `'{store}' is not a kyto store: {problem}`
- Hint: `check the path; a kyto store has a root zarr.json with attributes.kyto.format_version, written last, so an interrupted write leaves none`
- Fields:
    - `store` (str): The store path. Example: `pbmc.kyto`.
    - `problem` (str): What is missing or wrong. Example: `no root zarr.json`.

## KY-E117

A CellBlock is not valid CSR, or does not fit the store or layer it is written to (format.md sections 4.8 and 10.2).

- Class: `StoreInputError`
- Message: `invalid CellBlock ({context}): {problem}`
- Hint: `build CellBlocks with indptr starting at 0, non-decreasing and ending at nnz, and sorted, unique gene positions below n_genes in each row; a store writer also needs one block per layer, in layer order, over all genes of the store, with the layer's dtype and the same rows in every layer`
- Fields:
    - `context` (str): Where the block was given: the constructor, or the writer layer. Example: `StoreWriter.push, layer 'counts'`.
    - `problem` (str): What is wrong and where (row, position). Example: `the block has 2000 genes but the store has 32738`.

## KY-E118

A store writer's target path already exists (format.md section 10.1).

- Class: `StoreInputError`
- Message: `cannot write a store to '{path}': {problem}`
- Hint: `choose another path, remove the existing one first, or write with overwrite=True to replace an existing kyto store (it is replaced only after the new store is complete)`
- Fields:
    - `path` (str): The target path. Example: `pbmc.kyto`.
    - `problem` (str): What is there. Example: `a kyto store already exists there`.

## KY-E119

A cell or gene table given to a store writer does not have one row per matrix row or column (format.md section 5.1).

- Class: `StoreInputError`
- Message: `the {axis} table has {n_rows} rows, but the matrix has {n_axis} {axis}`
- Hint: `pass one {axis} table row per matrix {axis} entry, in matrix order`
- Fields:
    - `axis` (str): The axis, `cells` or `genes`. Example: `cells`.
    - `n_rows` (int): Rows in the table. Example: `2699`.
    - `n_axis` (int): Length of the matrix axis. Example: `2700`.

## KY-E120

An input file given to convert has a format or an encoding kyto does not import.

- Class: `InputFileError`
- Message: `cannot import '{path}': {problem}`
- Hint: `{accepted} (see the kyto.convert docstring for the inputs it reads)`
- Fields:
    - `path` (str): The input path, or the member inside it. Example: `pbmc.h5ad:/X`.
    - `problem` (str): What is not supported. Example: `encoding-type 'awkward-array' is not supported`.
    - `accepted` (str): What would be imported, and how to get there. Example: `store X as an array, csr_matrix or csc_matrix (anndata >= 0.8 writes these)`.

## KY-E121

An input file given to convert holds a value or a structure that breaks its own format or cannot be stored.

- Class: `InputFileError`
- Message: `invalid input '{path}' at {location}: {problem}`
- Hint: `{advice}, then convert again`
- Fields:
    - `path` (str): The input path, or the member inside it. Example: `pbmc.h5ad:/X`.
    - `location` (str): Where in the input: row, column, line or entry. Example: `cell 4, gene 17`.
    - `problem` (str): What is wrong. Example: `integer value -3 is negative`.
    - `advice` (str): How to fix it. Example: `integer matrices become uint32 layers; cast X to float32 in anndata before converting to keep negative values`.

## KY-E122

A store's schema.json is missing, is not JSON, or breaks format.md section 8.

- Class: `CorruptStoreError`
- Message: `schema.json of the store at '{store}' is invalid: {problem}`
- Hint: `the store breaks docs/format.md section 8; write the store again with kyto, or report a bug if kyto wrote it`
- Fields:
    - `store` (str): The store path. Example: `pbmc.kyto`.
    - `problem` (str): What is wrong, with the line and column of the key. Example: ``it does not match the format (unknown variant `int64`, expected `float32` or `uint32` at line 7 column 40)``.

## KY-E123

An export's target path already exists (sink_h5ad, task 2.7).

- Class: `StoreInputError`
- Message: `cannot write '{path}': {problem}`
- Hint: `choose another path, remove the existing one first, or pass overwrite=True to replace an existing file (it is replaced only after the new file is complete; a directory is never replaced)`
- Fields:
    - `path` (str): The target path. Example: `pbmc.h5ad`.
    - `problem` (str): What is there. Example: `a file already exists there`.

## KY-E124

A store holds a name or value the export format cannot hold (sink_h5ad, task 2.7).

- Class: `StoreInputError`
- Message: `cannot export {location} to '{path}': {problem}`
- Hint: `{advice}, then export again`
- Fields:
    - `location` (str): What cannot be exported: column, layer or value. Example: `cells column 'a/b'`.
    - `path` (str): The target path. Example: `pbmc.h5ad`.
    - `problem` (str): Why the format cannot hold it. Example: `h5ad names cannot contain '/'`.
    - `advice` (str): How to fix it. Example: `rename the column without '/'`.

## KY-E125

A remote store URL or one of its storage options cannot be used to open an object store (task 2.8).

- Class: `StoreInputError`
- Message: `cannot open remote store '{url}': {problem}`
- Hint: `{advice}, then open the store again`
- Fields:
    - `url` (str): The store URL, with any password removed. Example: `s3://bucket/pbmc.kyto`.
    - `problem` (str): What is wrong with the URL or the options. Example: `unknown storage option 'aws_acess_key_id'`.
    - `advice` (str): How to fix it. Example: `use an option key object_store accepts for s3, such as aws_access_key_id, aws_secret_access_key, aws_region, aws_endpoint or allow_http`.

## KY-E126

A stored embedding or graph breaks the store format (format.md sections 6 and 7), or disagrees with schema.json.

- Class: `CorruptStoreError`
- Message: `{kind} '{name}' is corrupt: {problem}`
- Hint: `the store breaks docs/format.md sections 6 and 7; write the {kind} again with kyto, or report a bug if kyto wrote it`
- Fields:
    - `kind` (str): `embedding` or `graph`. Example: `graph`.
    - `name` (str): The member being read. Example: `connectivities`.
    - `problem` (str): The invariant that fails and where. Example: `row 4: index 7 follows 9; indices must be strictly increasing`.

## KY-E127

A store was changed on disk after a plan scanned it.

- Class: `CorruptStoreError`
- Message: `the store at '{store}' changed since it was scanned: {problem}`
- Hint: `scan the store again and rebuild the plan; a plan answers from the store as it was when scanned`
- Fields:
    - `store` (str): The store path. Example: `pbmc.kyto`.
    - `problem` (str): What differs from the scan's snapshot. Example: `its schema.json is not the one the plan read`.

## KY-E201

A feature name matches more than one feature.

- Class: `ResolutionError`
- Message: `ambiguous name '{name}': it matches {candidates}`
- Hint: `pass one full name instead, one of {candidates}; where two share a full name, use cell_column(...), gene_expression(..., layer=...) or cell_embedding(...)`
- Fields:
    - `name` (str): The name as written to feature(...). Example: `CD3E`.
    - `candidates` (list[str]): The full names it matches (docs/feature_names.md), each followed by its kind when two are written alike. Example: `'meta:CD3E', 'log1p:CD3E'`.

## KY-E202

A cell expression and a gene expression are combined.

- Class: `AxisError`
- Message: `'{operation}' mixes a cell expression ({cell_expression}) with a gene expression ({gene_expression})`
- Hint: `combine expressions of one axis only: cell_column(...) with cell_column(...), gene_column(...) with gene_column(...)`
- Fields:
    - `operation` (str): The operator or method that combined them. Example: `&`.
    - `cell_expression` (str): The cell expression. Example: `cell_column('n_genes') > 200`.
    - `gene_expression` (str): The gene expression. Example: `gene_column('highly_variable')`.

## KY-E203

A gene removed by a filter is read later in the plan.

- Class: `RemovedFeatureError`
- Message: `gene '{gene}' was removed by '{removed_by}' earlier in the plan`
- Hint: `read '{gene}' before '{removed_by}', or change the filter to keep it`
- Fields:
    - `gene` (str): The gene that was read. Example: `MALAT1`.
    - `removed_by` (str): The plan node that removed it. Example: `filter_genes`.

## KY-E204

A gene symbol matches more than one gene ID.

- Class: `ResolutionError`
- Message: `symbol '{symbol}' matches more than one gene: {gene_ids}`
- Hint: `select the gene by ID with by='id', one of {gene_ids}`
- Fields:
    - `symbol` (str): The symbol as written. Example: `TBCE`.
    - `gene_ids` (list[str]): The gene IDs that carry this symbol. Example: `'ENSG00000284770', 'ENSG00000285053'`.

## KY-E205

An expression operation is given an operand of a dtype or type it does not accept.

- Class: `ExpressionError`
- Message: `'{operation}' needs {accepted}, got {got}: {operand}`
- Hint: `pass {accepted} to '{operation}', or use an operation that accepts {got}`
- Fields:
    - `operation` (str): The operator or method, such as `+`, `&` or `str.starts_with`. Example: `str.starts_with`.
    - `accepted` (str): What the operation accepts. Example: `a string expression`.
    - `got` (str): What it was given. Example: `a numeric expression`.
    - `operand` (str): The operand, written as code. Example: `cell_column('n_genes') + 1`.

## KY-E206

An expression is used where Python needs True or False.

- Class: `ExpressionError`
- Message: `the truth value of an expression is ambiguous: {expression}`
- Hint: `combine conditions with &, | and ~ instead of and, or and not, and write a < b < c as (a < b) & (b < c); an expression is evaluated only when the plan runs`
- Fields:
    - `expression` (str): The expression, written as code. Example: `cell_column('n_genes') > 200`.

## KY-E207

An expression tree is deeper than kyto builds.

- Class: `ExpressionError`
- Message: `'{operation}' would make an expression {depth} levels deep; the limit is {limit}`
- Hint: `build long chains as a balanced tree, combining pairs and then pairs of pairs, instead of adding one term at a time`
- Fields:
    - `operation` (str): The operator or method that would exceed the limit. Example: `+`.
    - `depth` (int): The depth the result would have. Example: `1001`.
    - `limit` (int): The deepest expression kyto builds. Example: `1000`.

## KY-E208

A feature name cannot name any feature.

- Class: `ResolutionError`
- Message: `invalid feature name '{name}' in feature(...): {problem}`
- Hint: `write a short name such as 'CD3E' or 'cluster', or a full name '<prefix>:<name>' such as 'log1p:CD3E' or 'meta:cluster'`
- Fields:
    - `name` (str): The name as written. Example: `log1p:`.
    - `problem` (str): What is wrong with it. Example: `nothing follows 'log1p:'`.

## KY-E209

An embedding component is not a non-negative int.

- Class: `ExpressionError`
- Message: `'{constructor}' needs a non-negative int component (0-based), got {got}: {component}`
- Hint: `pass the component as an int from 0, such as {constructor}('{name}', 0) for the first one`
- Fields:
    - `constructor` (str): The constructor called. Example: `cell_embedding`.
    - `name` (str): The embedding name passed. Example: `pca`.
    - `got` (str): What the component was. Example: `a negative int`.
    - `component` (str): The component, written as code. Example: `-1`.

## KY-E210

A name does not match any column, gene, cell, embedding component or feature of the plan.

- Class: `ResolutionError`
- Message: `unknown {kind} '{name}' in {expression}: {problem}{suggestions}`
- Hint: `names are matched exactly, case included; check '{name}' against the store's schema, and read a gene by its symbol with gene_expression(symbol, by='symbol')`
- Fields:
    - `kind` (str): What the name should name: `cell column`, `gene column`, `gene`, `gene symbol`, `cell`, `cell embedding`, `gene embedding`, `embedding component` or `feature`. Example: `feature`.
    - `name` (str): The name as written. Example: `CD3`.
    - `expression` (str): The expression it was written in, as code. Example: `feature('CD3')`.
    - `problem` (str): Why nothing matches. Example: `it is not a column of the cell table or a gene ID`.
    - `suggestions` (suggestions): Up to three names of the same kind closest to `name` (Jaro-Winkler, task 8.2), best first, written as `name` was; a gene symbol offered for a gene ID or a feature name reads `CD3E (gene symbol)`. Empty when none is close. Example: `'CD3E (gene symbol)', 'CD3D (gene symbol)', 'CD3G (gene symbol)'`.

## KY-E211

A gene or cell is read from the default layer, but the plan has none.

- Class: `ResolutionError`
- Message: `{expression} reads the default layer, but the plan has none`
- Hint: `set one with .with_default_layer(...) before resolving, or name the layer; the store has {layers}`
- Fields:
    - `expression` (str): The expression that reads it, as code. Example: `feature('CD3E')`.
    - `layers` (list[str]): The store's layers. Example: `'counts', 'log1p'`.

## KY-E212

An expression is compared with None, which is not a test for missing values.

- Class: `ExpressionError`
- Message: `'{operation}' with None does not test for missing values: {expression}`
- Hint: `write {suggestion} instead; missing values are null (R9)`
- Fields:
    - `operation` (str): The comparison, `==` or `!=`. Example: `==`.
    - `expression` (str): The comparison as written, as code. Example: `cell_column('cell_type') == None`.
    - `suggestion` (str): The null test to write instead, as code. Example: `cell_column('cell_type').is_null()`.

## KY-E213

`.recompute()` is called on an expression that is not a plain column read.

- Class: `ExpressionError`
- Message: `'.recompute()' needs a plain column read such as {example}, got {expression}`
- Hint: `call .recompute() on the column read itself, then build on the result: write {example} first`
- Fields:
    - `expression` (str): The expression it was called on, as code. Example: `cell_column('n_genes') + 1`.
    - `example` (str): A plain column read of the same axis, as code. Example: `cell_column('n_genes').recompute()`.

## KY-E214

An argument that picks how a name is looked up is given a value it does not accept.

- Class: `ResolutionError`
- Message: `{function}(..., {parameter}={value}) is not accepted`
- Hint: `{parameter} takes one of {accepted}`
- Fields:
    - `function` (str): The function or method called. Example: `gene_expression`.
    - `parameter` (str): The argument name. Example: `by`.
    - `value` (str): The value given, as code. Example: `'name'`.
    - `accepted` (list[str]): The values it accepts. Example: `'id', 'symbol'`.

## KY-E215

A plan method is given an argument value outside what it accepts.

- Class: `PlanError`
- Message: `{function}(..., {parameter}={value}) is invalid: {problem}`
- Hint: `pass {accepted} as {parameter}`
- Fields:
    - `function` (str): The plan method called. Example: `pca`.
    - `parameter` (str): The argument name. Example: `n_components`.
    - `value` (str): The value given, as code. Example: `0`.
    - `problem` (str): What is wrong with it. Example: `a PCA needs at least one component`.
    - `accepted` (str): What the argument accepts. Example: `an int from 1 to 2000`.

## KY-E216

A plan that ends in a sink is extended with another step.

- Class: `PlanError`
- Message: `'{method}' cannot follow '{sink}': a plan ends at its sink`
- Hint: `add '{method}' before '{sink}', or build a new plan from the store that '{sink}' writes`
- Fields:
    - `method` (str): The plan method called. Example: `log1p`.
    - `sink` (str): The sink that ends the plan. Example: `sink_zarr`.

## KY-E217

Text given to from_json is not a plan JSON, or breaks its format.

- Class: `PlanError`
- Message: `the plan JSON is invalid at {location}: {problem}`
- Hint: `pass the text Atlas.to_json() wrote, unchanged; to change a plan, build it with plan methods and write it again`
- Fields:
    - `location` (str): Where: the line and column, or the node and expression entry. Example: `node 2 (filter_cells), predicate entry 3`.
    - `problem` (str): What is wrong there. Example: `entry 3 reads entry 5, which is not written before it`.

## KY-E218

A plan JSON has a version this kyto does not read.

- Class: `PlanError`
- Message: `the plan JSON has version {found}, but this kyto reads only version {supported}`
- Hint: `load it with the kyto version that wrote it, or build the plan again with this one and write it with Atlas.to_json()`
- Fields:
    - `found` (int): The version in the text. Example: `2`.
    - `supported` (int): The version this kyto reads and writes. Example: `1`.

## KY-E219

peek() is called on a plan whose cell table cannot be read from metadata alone.

- Class: `PlanError`
- Message: `peek() reads metadata only, but this plan's cell table needs '{member}', which '{step}' reads`
- Hint: `run the plan for its cell table with cells.to_polars() or select(...), or call peek() on the plan before '{step}'`
- Fields:
    - `step` (str): The plan method whose step reads data. Example: `with_cells`.
    - `member` (str): What it reads, as kind:name. Example: `layer:counts`.

## KY-E301

A derived column is read after an upstream step changed its inputs.

- Class: `StaleColumnError`
- Message: `column '{column}' was computed before '{changed_by}' changed its inputs`
- Hint: `call .recompute() on '{column}' after '{changed_by}', or read it before that step`
- Fields:
    - `column` (str): The derived column. Example: `n_genes`.
    - `changed_by` (str): The plan node that changed its inputs. Example: `filter_genes`.

## KY-E302

A column without lineage is asked to be recomputed.

- Class: `LineageError`
- Message: `{expression} asks to recompute {kind} '{column}', but the store records no lineage for it`
- Hint: `only derived columns written by kyto record their lineage (schema.json); read '{column}' as stored, or compute it again from the matrix`
- Fields:
    - `expression` (str): The expression, as code. Example: `cell_column('n_genes').recompute()`.
    - `kind` (str): `cell column` or `gene column`. Example: `cell column`.
    - `column` (str): The column. Example: `n_genes`.

## KY-E303

A column with lineage is asked to be recomputed, but the store does not hold the plan that computed it.

- Class: `LineageError`
- Message: `{expression} asks to recompute {kind} '{column}', but {problem}`
- Hint: `only stores written by a kyto sink hold the plan that computed a column (provenance.json); read '{column}' as stored, or compute it again from the matrix`
- Fields:
    - `expression` (str): The expression, as code. Example: `cell_column('n_genes').recompute()`.
    - `kind` (str): `cell column` or `gene column`. Example: `cell column`.
    - `column` (str): The column. Example: `n_genes`.
    - `problem` (str): Why the recipe is missing. Example: `provenance.json records no plan (lineage names node 3)`.

## KY-E304

A store's provenance is asked for the plan that wrote it, but the store holds no such plan.

- Class: `LineageError`
- Message: `provenance() cannot rebuild the plan that wrote store '{store}': {problem}`
- Hint: `only stores written by sink_zarr hold the plan that made them; a store from convert records its input file instead (metadata()['source'])`
- Fields:
    - `store` (str): The store's path. Example: `pbmc.kyto`.
    - `problem` (str): Why there is no plan. Example: `provenance.json records no plan (the store was not written by a kyto sink)`.

## KY-E401

A plan needs more memory than its limit, even with the smallest blocks.

- Class: `MemoryLimitError`
- Message: `the plan needs at least {required_bytes} bytes but memory_limit is {limit_bytes} bytes`
- Hint: `raise memory_limit to at least {required_bytes} bytes, or select fewer cells or genes before this step`
- Fields:
    - `required_bytes` (int): The smallest peak memory the plan can run in, in bytes. Example: `6000000000`.
    - `limit_bytes` (int): The memory limit given, in bytes. Example: `4000000000`.

## KY-E402

A dense array is larger than the max_bytes its caller accepts (R8), or cannot be allocated.

- Class: `MemoryLimitError`
- Message: `'{method}' cannot allocate a dense {n_rows} x {n_columns} array of {required_bytes} bytes: {problem}`
- Hint: `select fewer cells or genes first, take a sparse copy with to_scipy(), or raise max_bytes if the memory is there`
- Fields:
    - `method` (str): The terminal called, as Class.method. Example: `Matrix.to_dense`.
    - `n_rows` (int): Rows of the dense array. Example: `2700`.
    - `n_columns` (int): Columns of the dense array. Example: `32738`.
    - `required_bytes` (int): Bytes of the dense array. Example: `353570400`.
    - `problem` (str): Why: max_bytes is smaller, or the system has no memory for it. Example: `max_bytes is 1000000`.

## KY-E501

The requested device has no kernel for an operation.

- Class: `DeviceError`
- Message: `no {device} kernel for '{operation}'`
- Hint: `pass device='auto' to fall back to the CPU for this step, or device='cpu'`
- Fields:
    - `operation` (str): The operation without a kernel. Example: `leiden`.
    - `device` (str): The device requested. Example: `cuda`.

## KY-E502

The CPU executor cannot start the threads it was asked for.

- Class: `DeviceError`
- Message: `could not start {threads} executor threads: {problem}`
- Hint: `pass fewer threads, or raise the process's limit on threads`
- Fields:
    - `threads` (int): The number of threads asked for. Example: `64`.
    - `problem` (str): What the operating system said. Example: `Resource temporarily unavailable (os error 35)`.

## KY-E503

A run was cancelled before it finished.

- Class: `RunCancelledError`
- Message: `the run was cancelled after {done} of {total} cell rows were streamed`
- Hint: `run the plan again to compute it; a cancelled run returns and writes nothing`
- Fields:
    - `done` (int): Cell rows streamed before the cancel, summed over the run's passes. Example: `4096`.
    - `total` (int): Cell rows the whole run streams, summed over its passes. Example: `16384`.

## KY-E504

The device a plan asks for is not on this machine, or not in this build.

- Class: `DeviceError`
- Message: `the plan asks for device '{device}', but {problem} (found: {found})`
- Hint: `pass on(device='auto') to use the GPU found, if any, and the CPU for the rest, or on(device='cpu'); explain() shows each step's device`
- Fields:
    - `device` (str): The device asked for with on(device=...). Example: `cuda`.
    - `problem` (str): Why it is missing: no such GPU or driver, or a build without its feature. Example: `this kyto was built without the 'cuda' feature`.
    - `found` (str): The devices found, as explain() names them. Example: `cpu, metal`.

## KY-E601

A documented method is called before the task that implements it is done.

- Class: `NotImplementedYetError`
- Message: `'{method}' is not implemented yet: it arrives with task {task} of the implementation plan`
- Hint: `inspect the data without reading it through names(), schema() and estimate() until then`
- Fields:
    - `method` (str): The method called, as Class.method. Example: `Matrix.to_scipy`.
    - `task` (str): The task of the implementation plan that implements it. Example: `7.5`.
