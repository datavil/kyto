//! Per-cell QC metrics (task 6.1): computes a resolved [`QcMetric`] leaf on each
//! streamed block, so the values always follow the rows and genes being read.
//!
//! Does: turn a gene-set predicate into a mask over the store's genes
//! ([`GeneSet::read`], a metadata read of `genes.parquet`), and compute each cell's
//! `total_counts`, `genes_detected`, `total_counts_in` or `percent_counts_in` from one
//! layer's [`CellBlock`] ([`CellQc::evaluate`]), as `scanpy.pp.calculate_qc_metrics`
//! defines them per cell. Sums are `f64` through the 5.3 row-sum kernel (same bits on
//! every ISA, R11); a gene-set sum is the same kernel over the row with the other genes'
//! values set to zero, so a gene set of every gene gives the total's exact bits.
//! On a GPU (task 9.10, [`CellQc::on_gpu`]) the row sums and nnz run through 9.2's
//! kernels ([`DeviceKernels`]): counts give the CPU's bits (integer sums), `float32`
//! sums are within about 1e-11 relative (a compensated `float32` pair); a row whose
//! GPU sum is not finite (NaN, an infinity, or a pair that overflows `float32`), or is
//! 0 while the row holds a nonzero value (subnormals a GPU flushed to zero, bug-hunter
//! 9.10), is summed again on the CPU, so such rows keep the CPU's value. The gene-set mask, the
//! percentage and everything after stay on the host.
//! Lives in kyto-exec since task 5.8, which computes the metric inside the stream
//! (`with_cells`, `filter_cells`) with the gene set taken over the genes present at that
//! step; kyto-algo re-exports it.
//! Does not: pick the layer or check the predicate (the resolver does, at plan build),
//! or compute scanpy's per-gene metrics, `log1p_` variants or
//! `pct_counts_in_top_N_genes` (deviations.md).

use std::path::Path;

use kyto_core::{GeneExpr, QcMetric};
use kyto_error::KytoError;
use kyto_gpu::{DeviceKernels, RowValues};
use kyto_store::{CellBlock, CellValues, TableAxis, read_table};
use polars::prelude::{ChunkExpandAtIndex, DataFrame, IntoLazy};

use crate::kernels::{KernelValue, Kernels};

/// The genes of a gene set: one flag per gene of the store, in table order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneSet {
    mask: Vec<bool>,
}

impl GeneSet {
    /// The genes whose flag in `mask` is set (one per gene, in table order).
    pub fn from_mask(mask: Vec<bool>) -> Self {
        Self { mask }
    }

    /// The genes of the store at `store` where the resolved metadata predicate
    /// `genes` is true (null counts as false, as a filter keeps only true rows, R9).
    /// Reads `genes.parquet` only (R1).
    ///
    /// # Errors
    /// - `KY-E110` or `KY-E113` if the gene table cannot be read
    ///   ([`kyto_store::read_table`]).
    /// - `KY-E113` if the predicate does not evaluate to one boolean per gene (an
    ///   unresolved predicate, or a table that changed since the plan was built).
    pub fn read(store: &Path, genes: &GeneExpr) -> Result<Self, KytoError> {
        let table = read_table(store, TableAxis::Genes)?;
        let path = store.join("genes.parquet").display().to_string();
        Self::evaluate(&table, genes, &path)
    }

    /// The genes of the gene table `table` where `genes` is true; `path` names the
    /// table in errors.
    ///
    /// # Errors
    /// `KY-E113` if `genes` does not lower to Polars or does not give one boolean per
    /// row of `table`.
    pub fn evaluate(table: &DataFrame, genes: &GeneExpr, path: &str) -> Result<Self, KytoError> {
        let corrupt = |problem: String| KytoError::CorruptTable {
            path: path.to_owned(),
            problem,
        };
        let predicate = genes.to_polars().ok_or_else(|| {
            corrupt(format!(
                "the gene set {genes} reads more than gene-table columns"
            ))
        })?;
        let frame = table
            .clone()
            .lazy()
            .select([predicate.alias("genes")])
            .collect()
            .map_err(|error| corrupt(format!("the gene set {genes} fails: {error}")))?;
        let flags = frame
            .column("genes")
            .and_then(|column| column.bool())
            .map_err(|error| corrupt(format!("the gene set {genes} is not boolean: {error}")))?;
        // A gene set without a column (a literal) gives one value: it holds for every gene.
        let flags = if flags.len() == 1 && table.height() != 1 {
            flags.new_from_index(0, table.height())
        } else {
            flags.clone()
        };
        if flags.len() != table.height() {
            return Err(corrupt(format!(
                "the gene set {genes} gives {} values for {} genes",
                flags.len(),
                table.height()
            )));
        }
        Ok(Self::from_mask(
            flags.iter().map(|flag| flag == Some(true)).collect(),
        ))
    }

    /// One flag per gene of the store, in table order.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The number of genes in the set.
    pub fn len(&self) -> usize {
        self.mask.iter().filter(|&&flag| flag).count()
    }

    /// Whether the set holds no gene.
    pub fn is_empty(&self) -> bool {
        !self.mask.contains(&true)
    }
}

/// The values of a QC metric for the rows of one block.
#[derive(Debug, Clone, PartialEq)]
pub enum QcValues {
    /// Sums and percentages (`total_counts`, `total_counts_in`, `percent_counts_in`).
    Float64(Vec<f64>),
    /// Counts of genes (`genes_detected`).
    UInt32(Vec<u32>),
}

/// What [`CellQc::evaluate`] computes, with its gene set read.
#[derive(Debug, Clone, PartialEq)]
enum Computation {
    TotalCounts,
    GenesDetected,
    TotalCountsIn(GeneSet),
    PercentCountsIn(GeneSet),
}

/// A per-cell QC metric ready to compute on blocks of one layer.
#[derive(Debug, Clone)]
pub struct CellQc {
    computation: Computation,
    kernels: Kernels,
    /// The GPU its row sums and nnz run on (task 9.10), `None` for the CPU.
    gpu: Option<DeviceKernels>,
    /// The metric's name (`total_counts`, ...), as GPU errors name the step.
    operation: &'static str,
}

impl CellQc {
    /// `metric` (resolved), with the gene set of a gene-set metric read from the store
    /// at `store` ([`GeneSet::read`]).
    ///
    /// # Errors
    /// Those of [`GeneSet::read`].
    pub fn read(store: &Path, metric: &QcMetric) -> Result<Self, KytoError> {
        let gene_set = metric
            .genes()
            .map(|genes| GeneSet::read(store, genes))
            .transpose()?;
        Ok(Self::new(metric, gene_set))
    }

    /// `metric` with the gene set `gene_set` (used by the gene-set metrics, ignored by
    /// the others; a gene-set metric without one has an empty set).
    pub fn new(metric: &QcMetric, gene_set: Option<GeneSet>) -> Self {
        let gene_set = || {
            gene_set
                .clone()
                .unwrap_or_else(|| GeneSet::from_mask(Vec::new()))
        };
        let computation = match metric {
            QcMetric::TotalCounts => Computation::TotalCounts,
            QcMetric::GenesDetected => Computation::GenesDetected,
            QcMetric::TotalCountsIn { .. } => Computation::TotalCountsIn(gene_set()),
            QcMetric::PercentCountsIn { .. } => Computation::PercentCountsIn(gene_set()),
        };
        Self {
            computation,
            kernels: Kernels::detected(),
            gpu: None,
            operation: metric.function(),
        }
    }

    /// The same metric with its row sums and nnz on the GPU of `gpu` (`None`: the
    /// CPU), as the plan placed its step (task 9.10).
    #[must_use]
    pub fn on_gpu(self, gpu: Option<DeviceKernels>) -> Self {
        Self { gpu, ..self }
    }

    /// The same metric computed with `kernels` (for checking that every ISA gives the
    /// same bits).
    #[must_use]
    pub fn with_kernels(self, kernels: Kernels) -> Self {
        Self { kernels, ..self }
    }

    /// The metric of each row of `block`, in row order. The block's gene positions
    /// index the gene set.
    ///
    /// # Errors
    /// - `KY-E117` if a gene-set metric's set does not have one flag per gene of the
    ///   block.
    /// - `KY-E505` if a GPU kernel or copy fails (on a GPU).
    pub fn evaluate(&self, block: &CellBlock) -> Result<QcValues, KytoError> {
        let indptr = block.indptr();
        match &self.computation {
            Computation::TotalCounts => Ok(QcValues::Float64(self.sums(block, None)?)),
            Computation::GenesDetected => {
                let mut counts = Vec::new();
                match (block.data(), &self.gpu) {
                    (CellValues::F32(values), None) => {
                        self.kernels.row_nnz(indptr, values, &mut counts);
                    }
                    (CellValues::U32(values), None) => {
                        self.kernels.row_nnz(indptr, values, &mut counts);
                    }
                    (CellValues::F32(values), Some(gpu)) => {
                        counts = gpu.row_nnz(indptr, RowValues::F32(values), self.operation)?;
                    }
                    (CellValues::U32(values), Some(gpu)) => {
                        counts = gpu.row_nnz(indptr, RowValues::U32(values), self.operation)?;
                    }
                }
                // A row has at most n_genes values, and n_genes fits u32 (format.md §4.3).
                Ok(QcValues::UInt32(
                    counts
                        .into_iter()
                        .map(|count| u32::try_from(count).unwrap_or(u32::MAX))
                        .collect(),
                ))
            }
            Computation::TotalCountsIn(gene_set) => {
                Ok(QcValues::Float64(self.sums(block, Some(gene_set))?))
            }
            Computation::PercentCountsIn(gene_set) => {
                let totals = self.sums(block, None)?;
                let in_set = self.sums(block, Some(gene_set))?;
                // scanpy: total_counts_<set> / total_counts * 100 (0 / 0 is NaN).
                Ok(QcValues::Float64(
                    in_set
                        .iter()
                        .zip(&totals)
                        .map(|(part, total)| part / total * 100.0)
                        .collect(),
                ))
            }
        }
    }

    /// Each row's sum over every gene, or over the genes of `gene_set`.
    fn sums(&self, block: &CellBlock, gene_set: Option<&GeneSet>) -> Result<Vec<f64>, KytoError> {
        let indptr = block.indptr();
        match (block.data(), gene_set) {
            (CellValues::F32(values), None) => {
                self.row_sums(indptr, values, RowValues::F32(values))
            }
            (CellValues::U32(values), None) => {
                self.row_sums(indptr, values, RowValues::U32(values))
            }
            (CellValues::F32(values), Some(set)) => {
                let masked = masked(block, values, set, 0.0)?;
                self.row_sums(indptr, &masked, RowValues::F32(&masked))
            }
            (CellValues::U32(values), Some(set)) => {
                let masked = masked(block, values, set, 0)?;
                self.row_sums(indptr, &masked, RowValues::U32(&masked))
            }
        }
    }

    /// Each row's sum of `values` (the same values as `on_device`), on the GPU when the
    /// metric has one, else with the CPU kernel. A row whose GPU sum is not finite is
    /// summed again on the CPU (module docs).
    fn row_sums<T: KernelValue>(
        &self,
        indptr: &[u64],
        values: &[T],
        on_device: RowValues<'_>,
    ) -> Result<Vec<f64>, KytoError> {
        let Some(gpu) = &self.gpu else {
            let mut sums = Vec::new();
            self.kernels.row_sums(indptr, values, &mut sums);
            return Ok(sums);
        };
        let mut sums = gpu.row_sums(indptr, on_device, self.operation)?;
        let mut row_sum = Vec::new();
        for (sum, range) in sums.iter_mut().zip(indptr.windows(2)) {
            // A checked CellBlock's row pointer starts at 0, ascends and ends at its nnz.
            let row = &values[range[0] as usize..range[1] as usize];
            // Not finite, or 0 from nonzero values (a GPU flushes subnormals to zero):
            // the CPU's sum.
            let flushed = *sum == 0.0 && row.iter().any(|value| value.to_f64() != 0.0);
            if !sum.is_finite() || flushed {
                self.kernels
                    .row_sums(&[0, row.len() as u64], row, &mut row_sum);
                *sum = row_sum[0];
            }
        }
        Ok(sums)
    }
}

/// `values` with every value of a gene outside `set` replaced by `zero`, so the row-sum
/// kernel keeps each value's lane.
fn masked<T: KernelValue>(
    block: &CellBlock,
    values: &[T],
    set: &GeneSet,
    zero: T,
) -> Result<Vec<T>, KytoError> {
    let n_genes = block.n_genes();
    if u64::try_from(set.mask.len()).ok() != Some(n_genes) {
        return Err(KytoError::InvalidCellBlock {
            context: "QC gene set".to_owned(),
            problem: format!(
                "the block has {n_genes} genes but the gene set has {} flags",
                set.mask.len()
            ),
        });
    }
    // Gene positions are below n_genes (a checked CellBlock), so indexing is in range.
    Ok(values
        .iter()
        .zip(block.indices())
        .map(
            |(&value, &gene)| {
                if set.mask[gene as usize] { value } else { zero }
            },
        )
        .collect())
}
