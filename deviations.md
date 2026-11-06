# Deviations from scanpy and anndata

Every place where kyto gives different results or semantics than scanpy or anndata.
Rules in [AGENTS.md](AGENTS.md) ("Record every deviation"). Add a row in the same PR
that introduces the deviation. Keep rows short; long reasoning goes in the topic
`memory/` entry. API shape differences required by the design rules (R1-R12, for
example no `.X` and no in-place mutation) are not listed here.

Kind: `chosen` (deliberate), `gap` (not implemented yet or known difference to fix),
`inherited` (kyto follows scanpy where another reference differs).

| Topic | scanpy / anndata | Other reference | kyto | Kind | Why | Decided |
|---|---|---|---|---|---|---|
