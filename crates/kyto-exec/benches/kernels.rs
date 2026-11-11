//! Criterion benches of the 5.3 kernels (`cargo bench -p kyto-exec --bench kernels`).
//!
//! One block shaped like pbmc3k's counts (4096 cells, 200 to 1500 values per cell, about
//! 850 on average, small counts) as `u32` and as `f32` (normalized). Each kernel runs
//! dispatched (`Kernels::detected()`), on pulp's scalar fallback (`Kernels::scalar()`,
//! still vectorized by the compiler for the target's baseline), and as the 5.2 scalar
//! stand-in it replaces (a left-to-right `f64` sum, `f32::ln_1p`), single-threaded.
//! Throughput is in values per second.

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use kyto_exec::Kernels;

const ROWS: usize = 4096;

struct Block {
    indptr: Vec<u64>,
    counts: Vec<u32>,
    normalized: Vec<f32>,
}

/// A seeded xorshift block (no RNG crate needed for a bench input).
fn block() -> Block {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut indptr = vec![0_u64];
    let mut counts = Vec::new();
    for _ in 0..ROWS {
        let len = 200 + (next() % 1300) as usize;
        for _ in 0..len {
            // Mostly 1s and 2s, sometimes large, as UMI counts.
            let draw = next();
            counts.push(
                1 + (draw % 3) as u32
                    + if draw % 50 == 0 {
                        (draw >> 40) as u32 % 500
                    } else {
                        0
                    },
            );
        }
        indptr.push(counts.len() as u64);
    }
    let normalized = counts.iter().map(|&count| count as f32 * 3.7).collect();
    Block {
        indptr,
        counts,
        normalized,
    }
}

fn kernels(criterion: &mut Criterion) {
    let block = block();
    let nnz = block.counts.len();
    let detected = Kernels::detected();
    let scalar = Kernels::scalar();
    let label = detected.instruction_set();
    let normalized_rows: Vec<&[f32]> = block
        .indptr
        .windows(2)
        .map(|bounds| &block.normalized[bounds[0] as usize..bounds[1] as usize])
        .collect();
    let factors: Vec<f64> = (0..ROWS).map(|row| 1e4 / (row as f64 + 1000.0)).collect();
    let mut sums = Vec::with_capacity(ROWS);
    let mut counts = Vec::with_capacity(ROWS);
    let mut out = vec![0.0_f32; nnz];

    let mut group = criterion.benchmark_group("row_sums_f32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench.iter(|| {
                kernels.row_sums(&block.indptr, black_box(&block.normalized), &mut sums);
            });
        });
    }
    group.bench_function("stand_in_5_2", |bench| {
        bench.iter(|| {
            sums.clear();
            sums.extend(normalized_rows.iter().map(|row| {
                black_box(*row)
                    .iter()
                    .map(|&value| f64::from(value))
                    .sum::<f64>()
            }));
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("row_sums_u32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench.iter(|| kernels.row_sums(&block.indptr, black_box(&block.counts), &mut sums));
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("row_sums_of_squares_f32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench.iter(|| {
                kernels.row_sums_of_squares(&block.indptr, black_box(&block.normalized), &mut sums);
            });
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("row_nnz_f32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench
                .iter(|| kernels.row_nnz(&block.indptr, black_box(&block.normalized), &mut counts));
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("scale_rows_u32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench.iter(|| {
                kernels.scale_rows(&block.indptr, black_box(&block.counts), &factors, &mut out);
            });
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("log1p_f32");
    group.throughput(Throughput::Elements(nnz as u64));
    for (name, kernels) in [(label, detected), ("pulp_scalar", scalar)] {
        group.bench_function(name, |bench| {
            bench.iter(|| kernels.log1p(black_box(&block.normalized), &mut out));
        });
    }
    group.bench_function("stand_in_5_2", |bench| {
        bench.iter(|| {
            for (out, &value) in out.iter_mut().zip(black_box(&block.normalized)) {
                *out = value.ln_1p();
            }
        });
    });
    group.finish();
}

criterion_group!(benches, kernels);
criterion_main!(benches);
