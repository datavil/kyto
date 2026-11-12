//! Task 2.5 Done-when check: killing a write midway leaves the target path absent
//! (format.md §10.1).
//!
//! The test re-runs this test binary as a child that streams a store forever and never
//! finishes. Once the child has written shards, the parent kills it with
//! `Child::kill` (SIGKILL on Unix, TerminateProcess on Windows: no destructor runs) and
//! checks that the target does not exist and that only the child's temporary directory
//! and lock file are left. A new write to the same target then deletes them and
//! succeeds.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use kyto_store::{CellBlock, LayerSpec, LayerWriteOptions, StoreWriter, Tiling, ValueDtype};
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

/// Set in the child: the target path to write.
const TARGET_VARIABLE: &str = "KYTO_KILL_TEST_TARGET";
const N_GENES: u64 = 50;
const TILING: Tiling = Tiling {
    cell_block_size: 8,
    gene_block_size: 16,
};
const OPTIONS: LayerWriteOptions = LayerWriteOptions {
    nnz_chunk_size: 64,
    nnz_shard_size: 256,
    zstd_level: 3,
};

fn table(id_column: &str, prefix: &str, n: u64) -> DataFrame {
    let ids: Vec<String> = (0..n).map(|row| format!("{prefix}{row}")).collect();
    DataFrame::new(
        n as usize,
        vec![Series::new(id_column.into(), ids).into_column()],
    )
    .unwrap()
}

/// Ten dense rows with values from `seed`.
fn block(seed: u32) -> CellBlock {
    let rows = 10_u64;
    let indptr = (0..=rows).map(|row| row * N_GENES).collect();
    let indices = (0..rows)
        .flat_map(|_| 0..N_GENES as u32)
        .collect::<Vec<_>>();
    let data = (0..indices.len() as u32)
        .map(|at| 1 + seed.wrapping_add(at) % 97)
        .collect();
    CellBlock::from_u32(indptr, indices, data, N_GENES).unwrap()
}

fn spec() -> Vec<LayerSpec> {
    vec![LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: OPTIONS,
    }]
}

/// The child: write blocks forever, announcing on stdout once shards are on disk.
/// Ignored, so a normal test run skips it; without the variable it does nothing.
#[test]
#[ignore = "run only as the child of killing_a_write_midway_leaves_the_target_absent"]
fn child_writes_until_killed() {
    let Some(target) = std::env::var_os(TARGET_VARIABLE) else {
        return;
    };
    let target = PathBuf::from(target);
    let genes = table("gene_id", "g", N_GENES);
    let mut writer = StoreWriter::create(&target, &genes, &spec(), TILING, false).unwrap();
    let mut seed = 0_u32;
    loop {
        writer.push(&[block(seed)]).unwrap();
        if seed == 20 {
            // 210 rows of 50 genes: 41 shards of 256 entries are written by now.
            println!("midway");
        }
        seed = seed.wrapping_add(1);
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Count files below `dir`.
fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            if path.is_dir() { count_files(&path) } else { 1 }
        })
        .sum()
}

#[test]
fn killing_a_write_midway_leaves_the_target_absent() {
    use std::io::{BufRead, BufReader};

    let parent = tempfile::tempdir().unwrap();
    let target = parent.path().join("killed.kyto");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "store_writer_kill::child_writes_until_killed",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(TARGET_VARIABLE, &target)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut lines = BufReader::new(stdout).lines();
    loop {
        assert!(Instant::now() < deadline, "the child never got midway");
        match lines.next() {
            Some(line) => {
                if line.unwrap().contains("midway") {
                    break;
                }
            }
            None => panic!("the child exited before getting midway"),
        }
    }
    // Midway: the child is still writing, its temporary directory holds shards.
    let temporaries: Vec<String> = names(parent.path())
        .into_iter()
        .filter(|name| !name.ends_with(".lock"))
        .collect();
    assert_eq!(temporaries.len(), 1, "{temporaries:?}");
    assert!(temporaries[0].starts_with(".killed.kyto.kyto-tmp-"));
    assert!(count_files(&parent.path().join(&temporaries[0])) > 10);
    assert!(!target.exists());

    child.kill().unwrap(); // SIGKILL / TerminateProcess
    let status = child.wait().unwrap();
    assert!(!status.success());

    // Done when: the target path is absent. Only the temporary and its lock are left.
    assert!(target.symlink_metadata().is_err(), "the target exists");
    let lock = format!("{}.lock", temporaries[0]);
    assert_eq!(names(parent.path()), [temporaries[0].clone(), lock]);

    // The next write to the same target deletes the stale temporary and succeeds.
    let genes = table("gene_id", "g", N_GENES);
    let mut writer = StoreWriter::create(&target, &genes, &spec(), TILING, false).unwrap();
    writer.push(&[block(0)]).unwrap();
    writer.finish(&table("cell_id", "c", 10), None).unwrap();
    assert_eq!(names(parent.path()), ["killed.kyto"]);
}
