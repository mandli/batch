"""Tests for batch.analysis.

parse_timing is pure and always tested.  plot_performance is guarded by
importorskip so the suite still runs without matplotlib/numpy installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.analysis import parse_timing, plot_performance

# A synthetic GeoClaw timing.txt exercising every field parse_timing reads.
TIMING_TXT = """\
 Timing summary
 Level      Wall        CPU        Cells
     1     10.0        80.0       1000.0
     2     20.0       160.0       2000.0
total       30.0       240.0       3000.0

 stepgrid            25.0        200.0
 BC/ghost cells       3.0         24.0
 Regridding           1.0          8.0
 Output (valout)      1.0          8.0

 Total time:         30.5        244.0
 Using 8 threads
"""


def write_timing(job_dir: Path, text: str = TIMING_TXT) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "timing.txt").write_text(text)
    return job_dir


# ---------------------------------------------------------------------------
# parse_timing
# ---------------------------------------------------------------------------


class TestParseTiming:
    def test_missing_file_returns_none(self, tmp_path):
        assert parse_timing(tmp_path) is None

    def test_levels(self, tmp_path):
        t = parse_timing(write_timing(tmp_path / "job"))
        assert len(t["levels"]) == 2
        assert t["levels"][0] == {
            "level": 1,
            "wall": 10.0,
            "cpu": 80.0,
            "cells": 1000.0,
        }
        assert t["levels"][1]["level"] == 2

    def test_total_integration(self, tmp_path):
        t = parse_timing(write_timing(tmp_path / "job"))
        assert t["total_integration"] == {
            "wall": 30.0,
            "cpu": 240.0,
            "cells": 3000.0,
        }

    def test_components(self, tmp_path):
        t = parse_timing(write_timing(tmp_path / "job"))
        assert t["components"]["stepgrid"] == {"wall": 25.0, "cpu": 200.0}
        assert t["components"]["bc"] == {"wall": 3.0, "cpu": 24.0}
        assert t["components"]["regrid"] == {"wall": 1.0, "cpu": 8.0}
        assert t["components"]["output"] == {"wall": 1.0, "cpu": 8.0}

    def test_total_and_threads(self, tmp_path):
        t = parse_timing(write_timing(tmp_path / "job"))
        assert t["total"] == {"wall": 30.5, "cpu": 244.0}
        assert t["n_threads"] == 8

    def test_threads_default_when_absent(self, tmp_path):
        # A file with only a total line: n_threads falls back to 1.
        t = parse_timing(write_timing(tmp_path / "job", " Total time:  5.0  5.0\n"))
        assert t["n_threads"] == 1
        assert t["total"] == {"wall": 5.0, "cpu": 5.0}

    def test_accepts_str_path(self, tmp_path):
        job = write_timing(tmp_path / "job")
        assert parse_timing(str(job)) is not None


# ---------------------------------------------------------------------------
# plot_performance (needs matplotlib + numpy)
# ---------------------------------------------------------------------------


class TestPlotPerformance:
    def test_writes_png_and_returns_path(self, tmp_path):
        pytest.importorskip("matplotlib")
        pytest.importorskip("numpy")
        dirs = [write_timing(tmp_path / f"job{i}") for i in range(3)]
        out = plot_performance(dirs, out_path=tmp_path / "perf.png")
        assert out == tmp_path / "perf.png"
        assert out.exists() and out.stat().st_size > 0

    def test_default_labels_from_dir_names(self, tmp_path):
        pytest.importorskip("matplotlib")
        dirs = [write_timing(tmp_path / "runA"), write_timing(tmp_path / "runB")]
        out = plot_performance(dirs, out_path=tmp_path / "p.png")
        assert out is not None

    def test_none_when_no_timing_data(self, tmp_path):
        pytest.importorskip("matplotlib")
        empty = [tmp_path / "a", tmp_path / "b"]
        for d in empty:
            d.mkdir()
        assert plot_performance(empty, out_path=tmp_path / "p.png") is None

    def test_label_count_mismatch_raises(self, tmp_path):
        pytest.importorskip("matplotlib")
        dirs = [write_timing(tmp_path / "j0"), write_timing(tmp_path / "j1")]
        with pytest.raises(ValueError, match="match"):
            plot_performance(dirs, labels=["only-one"], out_path=tmp_path / "p.png")
