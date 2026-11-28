#!/usr/bin/env bash
# Task 12.1: every timed run behind performance.md's "Where kyto stands" table.
#
# Usage: measure_12_1.sh DATA PHASE   (PHASE: 50k | 1m | profile)
#
# DATA is a folder on a disk with room (about 30 GB for 1M cells), never /tmp or the
# repo. Run from the repo root with the release extension built with the Metal
# kernels and symbols (profiles need them):
#   MATURIN_PEP517_ARGS="--features wgpu --strip false" \
#   CARGO_PROFILE_RELEASE_DEBUG=line-tables-only uv sync --reinstall-package kyto
# Runs call .venv/bin/python directly: `uv run` would rebuild the extension without
# the features above. Nothing else should run; each run prints the load average first.
set -euo pipefail

DATA=$1
PHASE=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
W="$ROOT/benchmarks/scanpy_workflow"
RAW="$DATA/raw"
mkdir -p "$RAW"

load() { echo "load $(sysctl -n vm.loadavg) at $(date +%H:%M:%S): $*" | tee -a "$RAW/load.log"; }

if [[ $PHASE == 50k ]]; then
  S="$DATA/suite/50k"
  load "suite 50k (kyto exact cpu, scanpy warm and cold)"
  "$PY" "$ROOT/benchmarks/suite/suite.py" --sizes 50k --threads 1,4,10 --repeats 3 \
    --no-kernels --data-dir "$DATA/suite" --output "$RAW/suite-50k" || true
  load "kyto approximate"
  "$PY" "$W/kyto_steps.py" "$S/store.kyto" "$S/work/approx.kyto" --threads 1,4,10 \
    --repeats 3 --search approximate > "$RAW/kyto-50k-approximate.jsonl"
  load "kyto metal (device auto, exact)"
  "$PY" "$W/kyto_steps.py" "$S/store.kyto" "$S/work/metal.kyto" --threads 1,4,10 \
    --repeats 3 --device auto --with-write > "$RAW/kyto-50k-metal.jsonl"
  load "kyto sink_h5ad"
  "$PY" "$W/kyto_steps.py" "$S/store.kyto" "$S/work/out.h5ad" --threads 1,4,10 \
    --repeats 3 --only-write --sink h5ad > "$RAW/kyto-50k-sink-h5ad.jsonl"
  load "read / convert"
  "$PY" "$W/extra_steps.py" read "$S/counts.h5ad" "$S/work" > "$RAW/extra-50k-read.jsonl"
  load "rank_genes_groups"
  "$PY" "$W/extra_steps.py" rank "$S/work/kyto_out.kyto" "$S/work/scanpy_out.h5ad" \
    > "$RAW/extra-50k-rank.jsonl"
  load "metacells"
  "$PY" "$W/extra_steps.py" metacells "$S/work/kyto_out.kyto" > "$RAW/extra-50k-metacells.jsonl"
  "$PY" "$W/compare_labels.py" "$S/work/scanpy_out.h5ad" "$S/work/kyto_out.kyto" \
    > "$RAW/labels-50k.txt"
elif [[ $PHASE == 1m ]]; then
  M="$DATA/1m"
  mkdir -p "$M"
  if [[ ! -f "$M/counts.h5ad" ]]; then
    "$PY" "$W/make_data.py" 1000000 "$M/counts.h5ad" --stream
  fi
  load "convert 1M"
  "$PY" "$W/extra_steps.py" read "$M/counts.h5ad" "$M" --repeats 1 --kyto-only \
    > "$RAW/extra-1m-read.jsonl"
  for threads in 1 4 10; do
    load "kyto 1M approximate, $threads threads"
    "$PY" "$W/run_kyto_timed.py" "$M/convert.kyto" "$M/out-approx.kyto" \
      --threads "$threads" --search approximate >> "$RAW/kyto-1m-approximate.jsonl"
  done
  load "kyto 1M exact, 10 threads"
  "$PY" "$W/run_kyto_timed.py" "$M/convert.kyto" "$M/out-exact.kyto" --threads 10 \
    --search exact >> "$RAW/kyto-1m-exact.jsonl"
  load "kyto 1M metal (auto, exact), 10 threads"
  "$PY" "$W/guard.py" --timeout 5400 -- "$PY" "$W/run_kyto_timed.py" "$M/convert.kyto" \
    "$M/out-metal.kyto" --threads 10 --search exact --device auto >> "$RAW/kyto-1m-metal.jsonl"
  load "kyto 1M rank_genes_groups and metacells, 10 threads"
  "$PY" "$W/extra_steps.py" rank "$M/out-approx.kyto" none --threads 10 --repeats 1 \
    --kyto-only > "$RAW/extra-1m-rank.jsonl"
  "$PY" "$W/extra_steps.py" metacells "$M/out-approx.kyto" --threads 10 --repeats 1 \
    > "$RAW/extra-1m-metacells.jsonl"
  load "scanpy 1M, 10 threads, under the watchdog"
  "$PY" "$W/guard.py" --timeout 3600 -- "$PY" "$W/run_scanpy.py" "$M/counts.h5ad" \
    "$M/scanpy_out.h5ad" --threads 10 --warmup 50000 > "$RAW/scanpy-1m.json"
elif [[ $PHASE == profile ]]; then
  S="$DATA/suite/50k"
  P="$DATA/profiles"
  mkdir -p "$P"
  rec() { local name=$1; shift; load "profile $name"
    ~/.cargo/bin/samply record --save-only --unstable-presymbolicate -o "$P/$name.json.gz" -- "$@" > /dev/null; }
  for threads in 1 10; do
    rec "workflow-exact-$threads" "$PY" "$W/kyto_steps.py" --child write "$threads" exact \
      "$S/store.kyto" "$S/work/profile.kyto" cpu zarr
    rec "workflow-approximate-$threads" "$PY" "$W/kyto_steps.py" --child umap "$threads" \
      approximate "$S/store.kyto" "$S/work/profile.kyto" cpu zarr
    rec "workflow-metal-$threads" "$PY" "$W/kyto_steps.py" --child umap "$threads" exact \
      "$S/store.kyto" "$S/work/profile.kyto" auto zarr
    rec "sink-h5ad-$threads" "$PY" "$W/kyto_steps.py" --child write "$threads" exact \
      "$S/store.kyto" "$S/work/profile.h5ad" cpu h5ad
    rec "rank-ttest-$threads" "$PY" "$W/extra_steps.py" --child kyto_rank:t-test "$threads" \
      "$S/work/kyto_out.kyto"
    rec "rank-wilcoxon-$threads" "$PY" "$W/extra_steps.py" --child kyto_rank:wilcoxon \
      "$threads" "$S/work/kyto_out.kyto"
    rec "metacells-$threads" "$PY" "$W/extra_steps.py" --child kyto_metacells "$threads" \
      "$S/work/kyto_out.kyto"
  done
  rec convert "$PY" "$W/extra_steps.py" --child kyto_convert 0 "$S/counts.h5ad" \
    "$S/work/profile-convert.kyto"
  for profile in "$P"/*.json.gz; do
    "$PY" "$W/hotspots.py" "$profile" --top 6 \
      --stage 'umap=kyto_exec::umap|kyto_algo::umap' \
      --stage 'neighbors=neighbors|hnsw|knn' \
      --stage 'leiden=leiden|cluster_graph|connectivit' \
      --stage 'pca=pca' \
      --stage 'hvg=hvg|loess' \
      --stage 'rank_genes=rank_genes|wilcoxon|t_test' \
      --stage 'metacells=metacells|emeans' \
      --stage 'sink=sink|store_writer|h5ad|hdf5|H5' \
      --stage 'import=import_convert' \
      --stage 'stream=pipeline|block_stream|cell_steps|reader|zarrs|decode' \
      > "${profile%.json.gz}.hotspots.json"
  done
fi
load "done $PHASE"
