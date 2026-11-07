# duplicate_symbols (task 3.9)

A 10x matrix directory (Cell Ranger v3 layout, uncompressed): 4 genes x 3 cells.
`features.tsv` gives two gene IDs the symbol `TBCE` (ENSG00000284770 and
ENSG00000285053), as the 10x GRCh38-2020-A reference does. Owned by
`tests/test_gene_identity.py`; `gene_expression("TBCE", by="symbol")` on it must raise
`KY-E204` listing both IDs.
