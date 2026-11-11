# Close-match suggestions: Jaro-Winkler on digit runs (task 8.2)

Plain character Jaro-Winkler ties `CD3E` with `CD34`, `CD38`, `CD36`, `CD37`, `CD33`
for the query `CD3` (all 0.94). On pbmc3k the gene table is in chromosome order, so a
table-order tie-break puts `CD34` (chr1) first and the CD3 complex (chr11) never makes
the top 3. Scoring every run of ASCII digits as one symbol fixes it: `CD34` drops to
0.82 because `34` is a different number than `3`, not "3 plus a letter".

- Score case-insensitively (exact matching is case-sensitive, so a case typo is the
  most useful suggestion); keep ties in candidate (table/schema) order for R11.
- The registry renders suggestions through a field type `suggestions`, not a
  `list[str]` in the template: an empty list must render as nothing, and Rust
  (`kyto_error::did_you_mean`) and the generated Python `_text` must agree.
- On an ENSG-keyed store (10x/mtx imports) a short name is never close to an ID;
  suggest the symbols too, labelled `CD3E (gene symbol)`.
