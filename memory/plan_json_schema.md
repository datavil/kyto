# The plan JSON Schema is checked, not trusted (task 8.4)

**Works:** the published schema (`crates/kyto-plan/src/plan_json.schema.json`) is also
the runtime shape check of `validate_plan` (`jsonschema` crate, no default features), so
it cannot silently drift from what the validator enforces. Its drift from what `to_json`
writes is caught because 4.3's corpus and random chains (`tests/plan_json.rs`
`assert_round_trip`) must have no diagnostics, and every 4.3 refusal must give the same
first code from `validate_plan` as `from_json` raises (`refused`).

**Rules that keep "no diagnostics exactly when `from_json` reads it":** the schema may be
laxer than serde (the replay catches it, e.g. `1.0` for an integer), never stricter on
text serde reads. So `Option` keys are not `required` (serde reads them missing as
`None`); stricter checks are fine only where the replay also refuses (block size 0,
modalities, but only on the scan, which replays first). A stricter check on a later
node (an empty expression list was one) hides an earlier step's replay error, so the
first code differs from `from_json`'s. Known gaps: serde reads a JSON array in place of
an object (`["kyto-plan", 1, [...]]`) and an internally tagged variant by number
(`{"op": 0}` is `column`); the schema refuses both, `to_json` writes neither.
`anyOf` puts the error on the parent and dumps the whole value in the message: use
`type: [..., "null"]` with `items` instead.

**Messages:** use `if: {required: [tag], properties: {tag: {const}}} then: {$ref}` per
variant, never `oneOf` (one useless "not valid under any" error). Without `required`
in the `if`, a missing tag makes every `then` apply. Check the method of a node in one
place only (`scan_node`/`later_node`), or an unknown method is reported twice.
