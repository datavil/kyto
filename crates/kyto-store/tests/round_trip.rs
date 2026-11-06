//! Task 2.2 Done-when check: a property-based round trip through a tiled layer is
//! exact, over random shapes (including 0 cells, 0 genes, a single cell and shapes on
//! tile boundaries), sparsity (empty tiles, all-zero rows and columns), tile and chunk
//! sizes, and both value dtypes. Stored zeros are dropped on write (format.md §4.6),
//! so the expected matrix is the input without them.

// A test crate: helpers outside `#[test]` fns may unwrap too (clippy.toml allows it in tests).
#![allow(clippy::unwrap_used)]

use kyto_store::{CsrMatrix, LayerValue, LayerWriteOptions, Tiling, read_layer, write_layer};
use proptest::prelude::*;

/// A dense matrix, row-major, `None` for an absent entry.
#[derive(Debug, Clone)]
struct Dense<T> {
    n_cells: u64,
    n_genes: u64,
    cells: Vec<Option<T>>,
}

impl<T: LayerValue> Dense<T> {
    /// CSR with every present entry, zeros included.
    fn to_csr(&self) -> CsrMatrix<T> {
        self.csr(|_| true)
    }

    /// CSR without stored zeros: what a reader must return.
    fn to_canonical_csr(&self) -> CsrMatrix<T> {
        self.csr(|value| !value.is_zero())
    }

    fn csr(&self, keep: impl Fn(T) -> bool) -> CsrMatrix<T> {
        let mut matrix = CsrMatrix {
            n_cells: self.n_cells,
            n_genes: self.n_genes,
            indptr: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        };
        let width = self.n_genes as usize;
        for cell in 0..self.n_cells as usize {
            for gene in 0..width {
                if let Some(value) = self.cells[cell * width + gene]
                    && keep(value)
                {
                    matrix.indices.push(gene as u32);
                    matrix.data.push(value);
                }
            }
            matrix.indptr.push(matrix.indices.len() as u64);
        }
        matrix
    }
}

/// Shapes up to 3 tiles past a few boundaries, densities from empty to full.
fn dense<T: std::fmt::Debug + Clone>(
    values: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (Dense<T>, Tiling, LayerWriteOptions)> {
    let tiling = (1_u64..=6, 1_u64..=6).prop_map(|(cell_block_size, gene_block_size)| Tiling {
        cell_block_size,
        gene_block_size,
    });
    let options = (1_u64..=5, 1_u64..=4, 1_i32..=3).prop_map(|(chunk, shards, zstd_level)| {
        LayerWriteOptions {
            nnz_chunk_size: chunk,
            nnz_shard_size: chunk * shards,
            zstd_level,
        }
    });
    let density = prop_oneof![Just(0.0), Just(1.0), 0.0_f64..=1.0];
    (tiling, options, density).prop_flat_map(move |(tiling, options, density)| {
        // Lengths near multiples of the block sizes hit edge blocks and exact fits.
        let cells = 0..=tiling.cell_block_size * 3 + 1;
        let genes = 0..=tiling.gene_block_size * 3 + 1;
        let values = values.clone();
        (cells, genes).prop_flat_map(move |(n_cells, n_genes)| {
            // An entry is present when its draw is below the density (never at 0, always at 1).
            let entry = (0.0_f64..1.0, values.clone());
            prop::collection::vec(entry, (n_cells * n_genes) as usize).prop_map(move |draws| {
                let cells = draws
                    .into_iter()
                    .map(|(draw, value)| (draw < density).then_some(value))
                    .collect();
                (
                    Dense {
                        n_cells,
                        n_genes,
                        cells,
                    },
                    tiling,
                    options,
                )
            })
        })
    })
}

fn round_trip<T: LayerValue>(
    dense: &Dense<T>,
    tiling: Tiling,
    options: &LayerWriteOptions,
) -> CsrMatrix<T> {
    let store = tempfile::tempdir().unwrap();
    write_layer(store.path(), "counts", &dense.to_csr(), tiling, options).unwrap();
    read_layer(store.path(), "counts", dense.n_cells, dense.n_genes, tiling).unwrap()
}

/// f32 values: zeros of both signs, NaN, infinities, subnormals and ordinary values.
fn f32_values() -> impl Strategy<Value = f32> + Clone {
    prop_oneof![
        Just(0.0_f32),
        Just(-0.0_f32),
        Just(f32::NAN),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(f32::MIN_POSITIVE / 2.0),
        any::<f32>(),
    ]
}

fn u32_values() -> impl Strategy<Value = u32> + Clone {
    prop_oneof![Just(0_u32), Just(u32::MAX), 1_u32..=5, any::<u32>()]
}

proptest! {
    // No failure file: integration tests have no lib.rs for proptest to place it by.
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn f32_layer_round_trip_is_exact((dense, tiling, options) in dense(f32_values())) {
        let expected = dense.to_canonical_csr();
        let found = round_trip(&dense, tiling, &options);
        prop_assert_eq!(&found.indptr, &expected.indptr);
        prop_assert_eq!(&found.indices, &expected.indices);
        // Bitwise, so NaN compares equal to itself and -0.0 would differ from 0.0.
        let bits = |data: &[f32]| data.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        prop_assert_eq!(bits(&found.data), bits(&expected.data));
        prop_assert_eq!((found.n_cells, found.n_genes), (dense.n_cells, dense.n_genes));
    }

    #[test]
    fn u32_layer_round_trip_is_exact((dense, tiling, options) in dense(u32_values())) {
        let expected = dense.to_canonical_csr();
        prop_assert_eq!(round_trip(&dense, tiling, &options), expected);
    }
}

#[test]
fn edge_shapes_round_trip() {
    let tiling = Tiling {
        cell_block_size: 3,
        gene_block_size: 2,
    };
    let options = LayerWriteOptions {
        nnz_chunk_size: 2,
        nnz_shard_size: 4,
        zstd_level: 3,
    };
    // (cells, genes): empty, no genes, no cells, one cell, exact tile multiples.
    for (n_cells, n_genes) in [(0, 0), (5, 0), (0, 5), (1, 1), (1, 7), (6, 4), (7, 5)] {
        let cells = (0..n_cells * n_genes)
            .map(|position| (position % 3 != 1).then_some(position as u32 + 1))
            .collect();
        let dense = Dense {
            n_cells,
            n_genes,
            cells,
        };
        let found = round_trip(&dense, tiling, &options);
        assert_eq!(found, dense.to_canonical_csr(), "{n_cells} x {n_genes}");
    }
}
