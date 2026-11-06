# Tests must not share a random generator at call time

(from ggrust; matters for R11 and the 1.5 fixtures)

**Broke (in ggrust):** a test case that drew from a module-level RNG *when called* got
data that depended on which cases ran before it. Running one test with `-k` gave a
different result than the full run.

**Rule:** module-level data may use a shared RNG (drawn once at import). Inside a test
or fixture builder, use its own generator (`np.random.default_rng(<own seed>)`, or a
seeded Rust RNG). Run order-sensitive suites both whole and with `-k` to confirm.
