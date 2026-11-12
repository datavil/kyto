//! Task 6.3: `seurat_v3`'s loess against `skmisc.loess` (what scanpy calls).
//!
//! Python writes seeded inputs (smooth curves with noise, heavily tied predictors as
//! gene means of counts are, small and large `n`) and skmisc's fitted values
//! (`loess(x, y, span=0.3, degree=2)`); kyto's [`loess_fitted`] must give the same
//! values to 1e-12 relative (measured about 1e-14). Skipped without skmisc (exit 77).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

use kyto_exec::loess_fitted;

const WRITE_CASES: &str = "\
import sys
try:
    from skmisc.loess import loess
except ImportError:
    sys.exit(77)
import numpy as np
out = sys.argv[1]
cases = []
for seed, n in [(0, 40), (1, 200), (2, 1000), (3, 5000)]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    cases.append((x, np.sin(2 * x) + x**2 + rng.normal(size=n) * 0.3))
for seed, n in [(4, 60), (5, 500), (6, 4000)]:
    rng = np.random.default_rng(seed)
    x = np.log10(rng.integers(1, 40, size=n) / 2700.0)
    cases.append((x, 0.9 * x + rng.normal(size=n) * 0.2))
for index, (x, y) in enumerate(cases):
    model = loess(x, y, span=0.3, degree=2)
    model.fit()
    np.stack([x, y, model.outputs.fitted_values]).astype('<f8').tofile(f'{out}/{index}')
print(len(cases))
";

fn read_f64(path: &Path) -> Vec<f64> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect()
}

#[test]
fn loess_matches_skmisc() {
    let work = tempfile::tempdir().unwrap();
    let output = Command::new(std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned()))
        .args(["-c", WRITE_CASES])
        .arg(work.path())
        .output()
        .expect("run python (use `uv run cargo test`, or set PYTHON)");
    if output.status.code() == Some(77) {
        eprintln!("skipped: scikit-misc is not installed on this platform");
        return;
    }
    assert!(output.status.success(), "{output:?}");
    let cases: usize = String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    for case in 0..cases {
        let values = read_f64(&work.path().join(case.to_string()));
        let n = values.len() / 3;
        let (x, rest) = values.split_at(n);
        let (y, skmisc) = rest.split_at(n);
        let kyto = loess_fitted(x, y, 0.3);
        let worst = kyto
            .iter()
            .zip(skmisc)
            .map(|(kyto, skmisc)| (kyto - skmisc).abs() / skmisc.abs().max(1.0))
            .fold(0.0_f64, f64::max);
        eprintln!("case {case} (n = {n}): worst relative difference {worst:e}");
        assert!(worst < 1e-12, "case {case}: {worst:e}");
    }
}
