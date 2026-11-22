# scvi-tools and Harmony on a kyto atlas

kyto has no deep-learning latent space and no batch correction of its own. It hands its
data to the tools that have them and takes their outputs back with
`Atlas.with_embedding` (task 6.11).

| File | What it is |
|---|---|
| [interop.py](interop.py) | the example: a notebook in the percent format (`# %%` cells) that also runs as a script. It trains scvi-tools on a kyto atlas and stores the latent space back, then corrects kyto's PCA with Harmony and stores that back too |
| [`tests/test_interop.py`](../../tests/test_interop.py) | runs the notebook, checks every round trip bit for bit, and compares each tool's output with the same tool run on scanpy's AnnData |

```bash
uv sync --group interop   # scvi-tools and harmonypy (PyTorch: large)
uv run python docs/tutorials/interop.py <pbmc3k 10x folder> <output folder>
```

## Out and back

| Direction | How | Copies |
|---|---|---|
| counts or a layer to scvi-tools | `atlas.to_anndata()` (sparse, as stored) | one, the `AnnData` the caller owns |
| a dense matrix | `atlas.layer(name).to_dense(max_bytes=...)`, `Matrix.to_torch(device=, max_bytes=)` | one array, within `max_bytes` (R8) |
| an embedding to Harmony or PyTorch | `atlas.embedding(name).to_numpy()`, `numpy.from_dlpack(embedding)`, `torch.from_dlpack(embedding)`, `Embedding.to_torch(device=)` | one array; DLPack and `to_torch` on the CPU share it |
| a tool's output back into the plan | `atlas.with_embedding(name, values)` | one, into the plan, bit for bit |

`with_embedding` takes a 2-D `float32` or `float64` NumPy array, or any object with
`__dlpack__` on the CPU (a torch tensor), of shape `(n_cells, n_components)`: row `r`
is the `r`-th cell of the plan, in `cell_ids()` order, the order of every export above.
It refuses other dtypes (cast first), NaN and infinities, and a row count that is not
the plan's: when the plan is built if the number of cells is known, else when it runs
(after a filter on cells). The result is an embedding like any other: `neighbors`,
`leiden`, `umap`, `embedding(...)`, the sinks and `to_anndata` read it by name. The plan
JSON holds the values, so a plan written with `to_json` and read back with `from_json`
gives the same bits.

## Results against scanpy

Measured Nov 22, 2026, macOS 27 arm64 (Apple M4), scvi-tools 1.5.2, harmonypy 2.1.0,
scanpy 1.12.5, PyTorch 2.15.0 on the CPU. pbmc3k, cells with at least 200 genes, genes
in at least 3 cells, the 1200 `seurat_v3` genes (kyto and scanpy pick the same).

| Check | Agreement | Tolerance |
|---|---|---|
| The `AnnData` kyto exports against scanpy's | same cells, genes and counts | exact |
| scvi-tools latent space (seed 0, 10 dimensions, 10 epochs), trained on each | equal bit for bit | exact |
| Values stored back and read again (NumPy, DLPack, torch, sinks, plan JSON) | equal bit for bit | exact |
| Harmony on kyto's PCA against Harmony on scanpy's PCA (30 components, two stand-in batches) | median per-cell relative difference 1.1e-3 (worst cell 7.5e-2); nearest-neighbour overlap 0.991 | `interop.harmony_rows` 5e-3, `interop.harmony_knn` 0.97 |

Harmony's difference is not the hook's: Harmony on the same input is the same in both,
and kyto's PCA matches scanpy's within the PCA tolerances (leading components 5e-5),
but its trailing components differ more (a flat spectrum, `pca.all_angle`), and
Harmony uses all of them. scanpy's own `harmony_integrate` (1.12.5) transposes
harmonypy's output, which harmonypy 2 already returns cells first, so it fails; the
test calls harmonypy on scanpy's `X_pca` itself (see [deviations](../../deviations.md)).
