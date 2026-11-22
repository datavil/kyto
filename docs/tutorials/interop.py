# %% [markdown]
# # scvi-tools and Harmony on a kyto atlas
#
# kyto computes no deep-learning latent space and no batch correction of its own. It
# hands its data to the tools that do, and takes their outputs back with
# `Atlas.with_embedding` (task 6.11). The exports are the terminals: `to_anndata`
# (what scvi-tools reads), `Embedding.to_numpy` or `numpy.from_dlpack(embedding)`
# (what Harmony reads; `torch.from_dlpack(embedding)` for PyTorch), all copies the
# caller owns. Back in kyto, the embedding is a plan step like any other: `neighbors`,
# `leiden`, `umap` and the sinks read it by name.
#
# This file is a notebook in the percent format (`# %%` cells, for VS Code or Jupyter
# with jupytext) and a script:
#
# ```bash
# uv sync --group interop   # scvi-tools and harmonypy (PyTorch: large)
# uv run python docs/tutorials/interop.py <10x folder> <output folder>
# ```
#
# The 10x folder is pbmc3k's `filtered_gene_bc_matrices/hg19` (the archive
# `tests/fixtures/pbmc.py` fetches). The output folder (replaced) receives the store
# `pbmc3k_integrated.kyto` with both embeddings and their graphs, clusters and layouts.
# `tests/test_interop.py` runs it and compares each tool's output with the same tool
# run on scanpy's AnnData ([interop.md](interop.md)).

# %%
import shutil
import sys
from pathlib import Path

# Not dependencies of kyto: the `interop` group (pyright may run without them).
import harmonypy  # pyright: ignore[reportMissingImports]
import kyto as ky
import numpy as np
import scvi  # pyright: ignore[reportMissingImports]
from kyto import gene_column, qc

# scvi-tools and harmonypy ship without complete type information.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false

if len(sys.argv) != 3:
    raise SystemExit("usage: interop.py <10x folder> <output folder>")
source, output = Path(sys.argv[1]), Path(sys.argv[2])
if output.exists() and any(output.iterdir()) and not (output / "pbmc3k.kyto").exists():
    raise SystemExit(f"{output} is not empty and not an earlier run's output folder")
shutil.rmtree(output, ignore_errors=True)
output.mkdir(parents=True)
# Few epochs keep the example short; scvi's default for 2700 cells is 400.
MAX_EPOCHS = 30

# %% [markdown]
# ## The atlas
#
# pbmc3k's counts, cells with at least 200 genes, genes in at least 3 cells, and the
# 1200 most variable genes by scanpy's `seurat_v3` flavour (on counts, as scvi-tools
# recommends). Nothing runs yet: each step adds to the plan.

# %%
ky.convert(source, output / "pbmc3k.kyto", layer="counts")
counts = (
    ky.scan(output / "pbmc3k.kyto")
    .with_default_layer("counts")
    .filter(qc.genes_detected() >= 200)
    .filter(qc.cells_detected() >= 3)
)
variable = counts.highly_variable_genes(n_top_genes=1200, flavor="seurat_v3").filter(
    gene_column("highly_variable")
)
print(variable)

# %% [markdown]
# ## scvi-tools: export, train, store the latent space back
#
# `to_anndata` runs the plan and returns its output as a new `AnnData` (counts as `X`,
# sparse). scvi-tools trains on it; its latent space has one row per cell, in the
# plan's cell order (the order of every kyto export), which is what `with_embedding`
# takes. The values are copied into the plan once, bit for bit.

# %%
adata = variable.to_anndata()
scvi.settings.seed = 0
scvi.model.SCVI.setup_anndata(adata)
model = scvi.model.SCVI(adata, n_latent=10)
model.train(max_epochs=MAX_EPOCHS, accelerator="cpu", enable_progress_bar=False)
latent = model.get_latent_representation()
assert isinstance(latent, np.ndarray)  # one array per cell (not a mean and a variance)
print(latent.shape, latent.dtype)

with_scvi = (
    variable.with_embedding("X_scVI", latent)
    .neighbors("X_scVI", into="scvi")
    .leiden(graph="scvi_connectivities", seed=0, into="leiden_scvi")
    .umap(graph="scvi_connectivities", seed=0, into="X_umap_scvi")
)
print(with_scvi)

# %% [markdown]
# ## Harmony: export PCA, correct, store it back
#
# Harmony corrects a PCA embedding for a batch. pbmc3k is one sample, so the batch here
# is a stand-in (alternate cells), to show the hand-off; with real samples it is a cell
# column such as `concat`'s `sample`. `Embedding.to_numpy` runs the plan and returns
# the embedding as a new array; `numpy.from_dlpack(atlas.embedding("X_pca"))` does the
# same through DLPack.

# %%
pca = (
    with_scvi.normalize_total(1e4, layer="counts").log1p().pca(n_components=30, seed=0)
)
components = pca.embedding("X_pca").to_numpy()
batch = np.array(["a", "b"])[np.arange(components.shape[0]) % 2]
harmony = harmonypy.run_harmony(
    components, {"batch": batch}, "batch", random_state=0, verbose=False
)
corrected = np.asarray(harmony.Z_corr, dtype=np.float32)
# Cells x components (harmonypy returns components x cells before 0.1).
if corrected.shape != components.shape:
    corrected = corrected.T

integrated = (
    pca.with_embedding("X_pca_harmony", corrected)
    .neighbors("X_pca_harmony", into="harmony")
    .leiden(graph="harmony_connectivities", seed=0, into="leiden_harmony")
)

# %% [markdown]
# ## Write it
#
# One run writes the store: both embeddings as given, their graphs, clusters and the
# scvi layout. The values read back are the ones handed in.

# %%
integrated.sink_zarr(output / "pbmc3k_integrated.kyto")
stored = ky.scan(output / "pbmc3k_integrated.kyto")
assert np.array_equal(stored.embedding("X_scVI").to_numpy(), latent)
assert np.array_equal(stored.embedding("X_pca_harmony").to_numpy(), corrected)
clusters = stored.cells.to_polars()
print(clusters.group_by("leiden_scvi").len().sort("leiden_scvi"))
print(
    f"{clusters.height} cells, scvi latent {latent.shape[1]}-d, stored back bit for bit"
)
