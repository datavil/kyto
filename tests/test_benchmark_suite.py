"""The benchmark suite's regression check and table (task 10.3, benchmarks/suite).

The suite itself runs only on purpose (``benchmarks/suite/suite.py``; the gate runs
its ``--smoke`` mode); these tests pin what counts as a regression and what the table
shows, on hand-made results.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from benchmark_report import (
    THRESHOLDS,
    Change,
    comparable,
    compare,
    kyto_numbers,
    markdown,
    missing,
)

SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "suite"


def _run(seconds: float, peak_mb: float) -> dict[str, dict[str, float]]:
    return {
        "seconds": {"pca": seconds, "total": seconds * 3},
        "peak_mb": {"pca": peak_mb, "total": peak_mb},
    }


def _result() -> dict[str, Any]:
    return {
        "machine": {"platform": "darwin-arm64", "cpu": "Apple M4", "cpu_count": 10},
        "settings": {"repeats": 3},
        "workflow": {
            "pbmc3k": {
                "shape": {"n_cells": 2700, "n_genes": 32738, "nnz": 2286884},
                "threads": {
                    "1": {"kyto": _run(1.0, 300.0), "scanpy": _run(2.0, 900.0)},
                    "4": {"kyto": _run(0.5, 400.0), "scanpy": None},
                },
            }
        },
        "kernels": {"log1p_f32/neon": 1000.0},
        "notices": ["rapids-singlecell: skipped, needs CUDA."],
    }


@pytest.mark.parametrize(
    ("metric", "baseline", "value", "regressed"),
    [
        ("seconds", 1.0, 1.25, False),  # at the ratio, not past it
        ("seconds", 1.0, 1.30, True),
        ("seconds", 0.02, 0.10, False),  # 5x, but under the absolute 0.1 s
        ("seconds", 0.30, 0.45, True),
        ("seconds", -0.05, 0.04, False),  # noise around a tiny step
        ("seconds", -0.05, 0.20, True),  # no ratio: the absolute limit decides
        ("seconds", 2.0, 1.0, False),  # faster
        ("peak_mb", 300.0, 340.0, False),  # 1.13x
        ("peak_mb", 300.0, 350.0, True),
        ("peak_mb", 100.0, 120.0, False),  # 1.2x, but only 20 MB
        ("kernel_ns", 1000.0, 1300.0, True),
        ("kernel_ns", 1000.0, 1200.0, False),
    ],
)
def test_change_regressed_needs_both_limits(
    metric: str, baseline: float, value: float, regressed: bool
) -> None:
    assert Change("key", metric, baseline, value).regressed is regressed


def test_thresholds_are_the_documented_ones() -> None:
    readme = (SUITE / "README.md").read_text()
    for threshold, text in [
        (THRESHOLDS["seconds"], "1.25x and more than 0.1 s"),
        (THRESHOLDS["peak_mb"], "1.15x and more than 25 MB"),
        (THRESHOLDS["kernel_ns"], "1.25x"),
    ]:
        assert text in readme, (threshold, text)


def test_kyto_numbers_flattens_kyto_only() -> None:
    numbers = kyto_numbers(_result())
    assert numbers["pbmc3k/1t/seconds/pca"] == ("seconds", 1.0)
    assert numbers["pbmc3k/4t/peak_mb/total"] == ("peak_mb", 400.0)
    assert numbers["kernels/log1p_f32/neon"] == ("kernel_ns", 1000.0)
    assert not any("scanpy" in key for key in numbers)
    # Seconds: one through-step time per step, no total (it equals write's).
    assert len(numbers) == 2 * (1 + 2) + 1


def test_kyto_seconds_are_through_each_step() -> None:
    # A step is a difference of two prefix medians: noise can make it negative.
    result = _result()
    run = {
        "seconds": {"hvg": 0.5, "pca": -0.2, "neighbors": 1.0, "total": 1.3},
        "peak_mb": {"pca": 300.0},
    }
    result["workflow"]["pbmc3k"]["threads"] = {"1": {"kyto": run}}
    numbers = kyto_numbers(result)
    assert numbers["pbmc3k/1t/seconds/hvg"] == ("seconds", 0.5)
    assert numbers["pbmc3k/1t/seconds/pca"] == ("seconds", 0.3)
    assert numbers["pbmc3k/1t/seconds/neighbors"] == ("seconds", 1.3)
    assert "pbmc3k/1t/seconds/total" not in numbers
    # A step that "grew" from -0.2 to 0.1 s while its prefix's median stayed the
    # same is not a regression: the through-step times are compared.
    other = copy.deepcopy(result)
    other_run = other["workflow"]["pbmc3k"]["threads"]["1"]["kyto"]
    other_run["seconds"].update(hvg=0.2, pca=0.1)
    changes = compare(other, result)
    assert not any(change.regressed for change in changes)


def test_comparable_refuses_another_machine() -> None:
    result = _result()
    assert comparable(result, copy.deepcopy(result)) is None
    other = copy.deepcopy(result)
    other["machine"]["cpu_count"] = 4
    reason = comparable(result, other)
    assert reason is not None and "cpu_count" in reason


def test_compare_pairs_keys_in_both_and_flags_regressions() -> None:
    baseline = _result()
    result = copy.deepcopy(baseline)
    result["workflow"]["pbmc3k"]["threads"]["1"]["kyto"] = _run(1.5, 300.0)
    result["workflow"]["10k"] = copy.deepcopy(result["workflow"]["pbmc3k"])
    changes = compare(result, baseline)
    assert {change.key for change in changes} == set(kyto_numbers(baseline))
    regressed = {change.key for change in changes if change.regressed}
    assert regressed == {"pbmc3k/1t/seconds/pca"}
    assert missing(result, baseline) == []


def test_missing_flags_numbers_that_vanished_from_what_ran() -> None:
    baseline = _result()
    result = copy.deepcopy(baseline)
    del result["workflow"]["pbmc3k"]["threads"]["1"]["kyto"]["seconds"]["pca"]
    result["kernels"] = {"other_bench": 5.0}
    assert missing(result, baseline) == [
        "pbmc3k/1t/seconds/pca",
        "kernels/log1p_f32/neon",
    ]
    # Sizes, thread counts or kernels this run did not measure are not missing.
    partial = copy.deepcopy(baseline)
    del partial["workflow"]["pbmc3k"]["threads"]["4"]
    partial["kernels"] = {}
    assert missing(partial, baseline) == []
    assert "- MISSING pbmc3k/1t/seconds/pca" in markdown(
        result, compare(result, baseline), missing(result, baseline)
    )


def test_markdown_shows_ratios_deltas_and_regressions() -> None:
    baseline = _result()
    result = copy.deepcopy(baseline)
    result["workflow"]["pbmc3k"]["threads"]["1"]["kyto"] = _run(1.5, 300.0)
    table = markdown(result, compare(result, baseline))
    assert "> rapids-singlecell: skipped, needs CUDA." in table
    assert "## pbmc3k: 2,700 cells x 32,738 genes, 1 thread(s)" in table
    assert "| PCA | 2.00 | 1.50 | 0.75 | **+50%** | 900 | 300 | +0% |" in table
    assert "rapids s" not in table
    assert "- REGRESSION pbmc3k/1t/seconds/pca: 1.000 -> 1.500" in table
    assert "| log1p_f32/neon | 1.0 | +0% |" in table


def test_markdown_adds_rapids_columns_when_rapids_ran() -> None:
    result = _result()
    result["workflow"]["pbmc3k"]["threads"]["1"]["rapids"] = _run(0.25, 2000.0)
    table = markdown(result, None)
    assert "| PCA | 2.00 | 1.00 | 0.50 |  | 900 | 300 |  | 0.25 | 2,000 |" in table
    assert "Compared" not in table


def test_stored_baselines_are_full_results() -> None:
    baselines = sorted(SUITE.glob("baseline-*.json"))
    assert baselines
    for path in baselines:
        baseline = json.loads(path.read_text())
        assert path.name == f"baseline-{baseline['machine']['platform']}.json"
        numbers = kyto_numbers(baseline)
        assert any(key.startswith("pbmc3k/") for key in numbers)
        assert any(key.startswith("kernels/") for key in numbers)
