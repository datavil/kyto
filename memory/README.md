# Memory index

Critical lessons: approaches that work, or that break things. One file per lesson.
Add a line here for every new file. Format: `- [Title](file.md): one-line hook`.

Lessons carried over from ggrust (same stack: Rust + PyO3 + maturin + uv) are marked
`(from ggrust)`. Re-check them once the kyto build exists; delete any that do not apply.

- [One build path: uv](build_single_path.md): (from ggrust) never `uv run --no-sync` after `maturin develop`; it tests a stale extension (false green).
- [Never let a failing check reach a commit](check_before_commit.md): (from ggrust) don't pipe the check into grep and then `;` commit; gate on its exit status.
- [Tests need their own seeded generator](test_seed_isolation.md): (from ggrust) a shared RNG drawn at call time makes results depend on test order.
