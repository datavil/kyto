# Property tests: fixed seeds, scaled cases (task 10.1)

**Works:** one configuration for every proptest, `tests/property_cases.rs`, shared as a
module (`#[path = "../../../../tests/property_cases.rs"] mod property_cases;` in each
crate's `tests/it/main.rs`, `#[cfg(test)]` in kyto-exec's `lib.rs`), not a crate.
`crate::property_cases::config(cases)` sets `RngSeed::Fixed` (proptest's default is a
random seed per run), no failure file, and multiplies the cases by
`KYTO_PROPERTY_SCALE`; `KYTO_PROPERTY_SEED` draws other cases. The hypothesis suite
(`tests/test_properties.py`) reads the same two variables (`derandomize=True` unless a
seed is given, `database=None` so nothing is written into the repo).

**Breaks:** a fixed seed fixes the run time too. 5.6's thread proptest drew a 1-thread,
1-cell-block case with delayed reads (0 to 3 ms per tile read): 25 to 50 s an output in
debug, 126 s in all. Delays only reorder work on more than one thread, and blocks of at
least 4 cells keep the case under 15 s. Time a new proptest under the fixed seed (and a
couple of others) before committing.

**Also:**
- Ruff's `B` prefix selects `BLE` too: catching `BaseException` (a PyO3 panic is one)
  needs `# noqa: BLE001`.
- Random chains need staged drawing (an optional step per stage of a typical analysis,
  then a few steps in any order): steps drawn uniformly almost never reach
  `neighbors`, clustering or UMAP, which need a PCA first. Check with
  `pytest --hypothesis-show-statistics` (the suite labels steps and error codes with
  `event`).
- 10x Matrix Market converts with 0 cells, 0 genes or both; the property covers them.
