//! Task 2.9 Done when: the combined store streams with memory bounded by one block
//! per source. Two stores combined at 4 and at 16 cell blocks each: the peak heap of
//! `concat` grows with the cell tables only, not with the matrix (which grows by
//! about 24 blocks of entries), and stays within a few blocks of entries above the
//! tables. Its own test binary: it needs the global allocator to itself (AGENTS.md,
//! Agent-friendliness; memory/measuring_peak_memory.md).

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use std::path::Path;

use kyto_store::{
    CellBlock, ConcatOptions, ConvertOptions, Join, LayerSpec, LayerWriteOptions, StoreWriter,
    Tiling, ValueDtype, concat,
};
use peak_alloc::PeakAlloc;
use polars::prelude::{DataFrame, IntoColumn, NamedFrom, Series};

#[global_allocator]
static ALLOCATOR: PeakAlloc = PeakAlloc;

const CELLS_PER_BLOCK: u64 = 256;
const N_GENES: u64 = 2000;
const TILING: Tiling = Tiling {
    cell_block_size: CELLS_PER_BLOCK,
    gene_block_size: 1024,
};
/// Small shards, so the writer's own buffer (fewer than `S` entries) stays small.
const LAYER_OPTIONS: LayerWriteOptions = LayerWriteOptions {
    nnz_chunk_size: 4096,
    nnz_shard_size: 16384,
    zstd_level: 3,
};

/// Whether cell `cell` has gene `gene` (about 1 in 10).
fn stored(cell: u64, gene: u64) -> bool {
    (cell * 7 + gene * 13).is_multiple_of(10)
}

/// Write a store of `n_blocks` cell blocks, cells `<prefix><i>`, one uint32 layer.
fn write_store(path: &Path, prefix: &str, n_blocks: u64) {
    let genes: Vec<String> = (0..N_GENES).map(|gene| format!("G{gene:05}")).collect();
    let gene_table = DataFrame::new(
        genes.len(),
        vec![Series::new("gene_id".into(), genes).into_column()],
    )
    .unwrap();
    let spec = LayerSpec {
        name: "counts".to_owned(),
        dtype: ValueDtype::U32,
        options: LAYER_OPTIONS,
    };
    let mut writer = StoreWriter::create(path, &gene_table, &[spec], TILING, false).unwrap();
    for block in 0..n_blocks {
        let mut indptr = vec![0_u64];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for cell in block * CELLS_PER_BLOCK..(block + 1) * CELLS_PER_BLOCK {
            for gene in (0..N_GENES).filter(|gene| stored(cell, *gene)) {
                indices.push(gene as u32);
                data.push(((cell + gene) % 5 + 1) as u32);
            }
            indptr.push(indices.len() as u64);
        }
        let block = CellBlock::from_u32(indptr, indices, data, N_GENES).unwrap();
        writer.push(&[block]).unwrap();
    }
    let n_cells = (n_blocks * CELLS_PER_BLOCK) as usize;
    let cells: Vec<String> = (0..n_cells).map(|cell| format!("{prefix}{cell}")).collect();
    let cell_table = DataFrame::new(
        n_cells,
        vec![Series::new("cell_id".into(), cells).into_column()],
    )
    .unwrap();
    writer.finish(&cell_table, None).unwrap();
}

/// Peak heap bytes of combining two stores of `n_blocks` cell blocks each.
fn concat_peak(folder: &Path, n_blocks: u64) -> usize {
    let (x, y) = (
        folder.join(format!("x{n_blocks}.kyto")),
        folder.join(format!("y{n_blocks}.kyto")),
    );
    write_store(&x, "x", n_blocks);
    write_store(&y, "y", n_blocks);
    let options = ConcatOptions {
        join: Join::Outer,
        convert: ConvertOptions {
            tiling: TILING,
            layer_options: LAYER_OPTIONS,
            ..ConvertOptions::default()
        },
        ..ConcatOptions::default()
    };
    let target = folder.join(format!("both{n_blocks}.kyto"));
    // Run on one thread so decode buffers do not scale with the machine's cores.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let before = ALLOCATOR.current_usage();
    ALLOCATOR.reset_peak_usage();
    pool.install(|| concat(&[x, y], &target, &options)).unwrap();
    ALLOCATOR.peak_usage() - before
}

#[test]
fn concat_streams_one_block_at_a_time() {
    let folder = tempfile::tempdir().unwrap();
    // Warm up: the first run pays one-time costs (Polars, zarrs and rayon statics).
    concat_peak(folder.path(), 1);
    let small = concat_peak(folder.path(), 4);
    let large = concat_peak(folder.path(), 16);
    // One cell block of entries: indices and values, 8 bytes an entry.
    let block_bytes = (CELLS_PER_BLOCK * N_GENES / 10 * 8) as usize;
    eprintln!(
        "concat peak: {:.2} MB at 4 blocks a source, {:.2} MB at 16; a block holds {:.2} MB",
        small as f64 / 1e6,
        large as f64 / 1e6,
        block_bytes as f64 / 1e6
    );
    // 24 more blocks of entries; 6144 more cells in the tables (well under a block).
    assert!(
        large < small + 2 * block_bytes,
        "peak grew from {small} to {large} bytes with the matrix"
    );
    assert!(
        small < 12 * block_bytes,
        "peak {small} bytes is above 12 blocks of {block_bytes} bytes"
    );
}
