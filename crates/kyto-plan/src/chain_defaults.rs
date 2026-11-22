//! Defaults that follow the chain (task 7.10): what a step reads and writes when it is
//! called without `layer=`, `into=` or an embedding.
//!
//! Does: name the output of a step called without `into=` after the step
//! (`normalize_total` -> `normalized`, `pca` -> `X_pca`, ...) and refuse that name with
//! `KY-E220` when the plan already has it; give the layer a matrix step reads without
//! `layer=` ([`Plan::chain_layer`]: the layer the latest `normalize_total`, `log1p` or
//! `scale` wrote, or the default layer set after it) and the cell embedding
//! `neighbors` searches without one ([`Plan::chain_embedding`]). Each is resolved once,
//! when the step is added, from the plan alone, and stored in full on the node (R6),
//! so it is no hidden state (R7: decided by the user, Nov 22, 2026).
//! Does not: change how short names in expressions resolve (the default layer,
//! `Plan::resolve`), or check an explicit `into=` (each step does, as before).

use kyto_core::Axis;
use kyto_error::KytoError;

use crate::cross_cell::loadings_name;
use crate::plan::Plan;

/// A step that writes a member named after itself when called without `into=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStep {
    /// `normalize_total`, into the layer `normalized`.
    NormalizeTotal,
    /// `log1p`, into the layer `log1p`.
    Log1p,
    /// `scale`, into the layer `scaled`.
    Scale,
    /// `pca`, into the cell embedding `X_pca` (and the gene embedding `X_pca_loadings`).
    Pca,
    /// `umap`, into the cell embedding `X_umap`.
    Umap,
    /// `leiden`, into the cell column `leiden`.
    Leiden,
    /// `louvain`, into the cell column `louvain`.
    Louvain,
}

/// What a [`ChainStep`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Written {
    Layer,
    Embedding,
    CellColumn,
}

impl ChainStep {
    /// The plan method, as messages name it.
    pub fn method(self) -> &'static str {
        match self {
            Self::NormalizeTotal => "normalize_total",
            Self::Log1p => "log1p",
            Self::Scale => "scale",
            Self::Pca => "pca",
            Self::Umap => "umap",
            Self::Leiden => "leiden",
            Self::Louvain => "louvain",
        }
    }

    /// The name the step writes without `into=` (scanpy's key where it has one).
    pub fn default_into(self) -> &'static str {
        match self {
            Self::NormalizeTotal => "normalized",
            Self::Log1p => "log1p",
            Self::Scale => "scaled",
            Self::Pca => "X_pca",
            Self::Umap => "X_umap",
            Self::Leiden => "leiden",
            Self::Louvain => "louvain",
        }
    }

    fn written(self) -> Written {
        match self {
            Self::NormalizeTotal | Self::Log1p | Self::Scale => Written::Layer,
            Self::Pca | Self::Umap => Written::Embedding,
            Self::Leiden | Self::Louvain => Written::CellColumn,
        }
    }
}

impl Plan {
    /// The name `step` writes: `into` as given, or the step's default
    /// ([`ChainStep::default_into`]) when `into` is `None`. An explicit name is
    /// returned unchecked (the step checks it, and an existing name is replaced).
    ///
    /// # Errors
    /// `KY-E216` when `into` is `None` and this plan ends in a sink; `KY-E220` when `into` is `None` and the plan already has a member of that kind
    /// with the default name, compared ignoring case (for `pca`, also its
    /// `<into>_loadings` gene embedding).
    pub fn into_or_default(
        &self,
        step: ChainStep,
        into: Option<&str>,
    ) -> Result<String, KytoError> {
        if let Some(into) = into {
            return Ok(into.to_owned());
        }
        self.check_open(step.method())?;
        let name = step.default_into();
        let schema = self.schema();
        let (kind, existing): (&str, Vec<&str>) = match step.written() {
            Written::Layer => (
                "layer",
                schema.layers.iter().map(|m| m.name.as_str()).collect(),
            ),
            Written::Embedding => (
                "embedding",
                schema.embeddings.iter().map(|m| m.name.as_str()).collect(),
            ),
            Written::CellColumn => (
                "cell column",
                (schema.cells.columns.iter())
                    .map(|column| column.name.as_str())
                    .collect(),
            ),
        };
        let mut names = vec![name.to_owned()];
        if step == ChainStep::Pca {
            names.push(loadings_name(name));
        }
        for written in &names {
            if let Some(other) = existing
                .iter()
                .find(|other| other.eq_ignore_ascii_case(written))
            {
                return Err(KytoError::DefaultNameTaken {
                    method: step.method().to_owned(),
                    kind: kind.to_owned(),
                    name: written.clone(),
                    existing: (*other).to_owned(),
                });
            }
        }
        Ok(name.to_owned())
    }

    /// The layer a matrix step called without `layer=` reads: the layer the latest
    /// `normalize_total`, `log1p` or `scale` of this plan wrote, or the default layer
    /// when a [`Plan::with_default_layer`] comes after it (the nearest wins), or `None`
    /// with neither.
    pub fn chain_layer(&self) -> Option<&str> {
        self.state().chain_layer.as_deref()
    }

    /// The cell embedding `method` (`neighbors`) searches: `embedding` as given (the
    /// step checks it), or [`Plan::chain_embedding`] when it is `None`.
    ///
    /// # Errors
    /// `KY-E216` when `embedding` is `None` and this plan ends in a sink; `KY-E215`
    /// when it is `None` and no `pca` of the plan wrote a cell embedding and the plan
    /// has no `X_pca`.
    pub fn embedding_or_default(
        &self,
        method: &str,
        embedding: Option<&str>,
    ) -> Result<String, KytoError> {
        if let Some(embedding) = embedding {
            return Ok(embedding.to_owned());
        }
        self.check_open(method)?;
        self.chain_embedding().map(str::to_owned).ok_or_else(|| {
            let embeddings: Vec<String> = (self.schema().embeddings.iter())
                .filter(|member| member.axis == Axis::Cells)
                .map(|member| format!("'{}'", member.name))
                .collect();
            let accepted = if embeddings.is_empty() {
                "the name of a cell embedding; the plan has none, so add one with .pca() first"
                    .to_owned()
            } else {
                format!(
                    "the name of a cell embedding of the plan: {}",
                    embeddings.join(", ")
                )
            };
            Self::invalid_argument(
                method,
                "embedding",
                "None",
                "no pca step of the plan wrote a cell embedding, and the plan has no 'X_pca'",
                &accepted,
            )
        })
    }

    /// The cell embedding `neighbors` searches when given none (scanpy's `use_rep`):
    /// the one the latest `pca` of this plan wrote, else `X_pca` if the plan has it,
    /// else `None`.
    pub fn chain_embedding(&self) -> Option<&str> {
        match self.state().chain_embedding.as_deref() {
            Some(embedding) => Some(embedding),
            None => self
                .schema()
                .embeddings
                .iter()
                .find(|member| member.axis == Axis::Cells && member.name == "X_pca")
                .map(|member| member.name.as_str()),
        }
    }
}
