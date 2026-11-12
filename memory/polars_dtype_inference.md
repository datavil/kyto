# Ask Polars for result dtypes; do not re-derive its rules (task 4.2)

**Breaks:** 4.1's `column_dtype` copied Polars' numeric rules by hand. Against
`collect_schema()` it was wrong in 140 of 560 column pairs (`i8 + u8` is `Int16`,
`i16 + f32` is `Float32`, `2 / f32` is `Float64`). A hand fix was still wrong for
literals: Polars' dynamic literals widen when the value does not fit (`u8 + -1` is
`Int16`, `i32 + 3_000_000_000` is `Int64`), and literal-only subtrees stay literals
(`1 + 1` is `Int32`). The bug-hunter found these.

**Works:** `column_dtype` lowers the expression with `ExprNode::to_polars_with`. Each
leaf (column, matrix read, embedding component) becomes an empty stand-in column of
its declared dtype (`Enum` stands in as `String`). Then it reads the dtype from
`lazy().select(..).collect_schema()`. Plain reads keep their own dtype (`Enum`
included). `crates/kyto-plan/tests/it/column_dtype_polars.rs` guards it, so a Polars
upgrade that changes the rules shows up there.
