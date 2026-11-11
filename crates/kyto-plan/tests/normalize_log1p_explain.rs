//! Task 6.2 Done when (part): `explain()` shows `normalize_total -> log1p` as one fused
//! pass, with `target_sum` (one pass over the store) and with the median (one
//! statistics pass, then the fused apply pass). The parity with scanpy is
//! `kyto-exec`'s `tests/normalize_log1p_parity.rs`.
//!
//! The snapshots are `tests/snapshots/normalize_log1p/<name>.txt`. To update them on
//! purpose (say so in the commit message), run
//! `KYTO_UPDATE_SNAPSHOTS=1 uv run cargo test -p kyto-plan --test normalize_log1p_explain`
//! and read the diff.
#![allow(clippy::unwrap_used)]

mod support;

use std::path::PathBuf;
use std::sync::Arc;

use kyto_plan::{PhysicalOptions, Plan};
use support::scan;

/// `scan -> normalize_total(target_sum) -> log1p`, as scanpy's tutorial runs them.
fn normalize_log1p(target_sum: Option<f64>) -> Arc<Plan> {
    scan()
        .normalize_total(target_sum, Some("counts"), "normalized")
        .unwrap()
        .log1p(Some("normalized"), "log1p")
        .unwrap()
}

/// Compare `actual` with the snapshot `name`, or write it when `KYTO_UPDATE_SNAPSHOTS`
/// is set.
fn check_snapshot(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots/normalize_log1p")
        .join(format!("{name}.txt"));
    if std::env::var_os("KYTO_UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, format!("{actual}\n")).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "no snapshot {} ({error}); write it with KYTO_UPDATE_SNAPSHOTS=1",
            path.display()
        )
    });
    assert_eq!(
        actual,
        expected.strip_suffix('\n').unwrap_or(&expected),
        "{name} differs from its snapshot"
    );
}

#[test]
fn explain_shows_one_fused_pass() {
    for (name, target_sum, passes) in [("target_sum", Some(1e4), 1), ("median", None, 2)] {
        let plan = normalize_log1p(target_sum);
        let text = plan.explain();
        check_snapshot(name, &text);
        let optimized = text.split("optimized plan").nth(1).unwrap();
        // One fused node holds both maps; neither runs on its own.
        assert_eq!(optimized.matches("FUSED_ELEMENTWISE").count(), 1, "{text}");
        let fused = optimized
            .lines()
            .find(|line| line.contains("FUSED_ELEMENTWISE"))
            .unwrap();
        assert!(
            fused.contains("NORMALIZE_TOTAL") && fused.contains("LOG1P"),
            "{fused}"
        );
        assert_eq!(
            optimized
                .lines()
                .filter(|line| line.trim_start().starts_with('#'))
                .count(),
            2,
            "the scan and the fused pass only:\n{text}"
        );
        // Passes over the store: the fused pass, after a statistics pass for the median.
        let physical = plan.physical(PhysicalOptions::default());
        assert_eq!(physical.estimate().passes(), passes, "{text}");
    }
}
