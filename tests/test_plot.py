"""Tests for batch.plot.plot_job and batch.plot._plot_inprocess."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from batch.job import JobResult
from batch.plot import _plot_inprocess, plot_job
from tests.conftest import MockJob


class TestPlotJob:
    def test_plot_job_returns_true_on_success(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        mock_proc = MagicMock(returncode=0)
        with patch("batch.plot.subprocess.run", return_value=mock_proc):
            assert plot_job(result) is True
        assert "--- plotclaw ---" in job_paths.log.read_text()

    def test_plot_job_returns_false_on_nonzero_returncode(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        mock_proc = MagicMock(returncode=1)
        with patch("batch.plot.subprocess.run", return_value=mock_proc):
            assert plot_job(result) is False

    def test_plot_job_resolves_relative_setplot_against_job_dir(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        (job_paths.job / "setplot.py").write_text("# dummy\n")
        captured = {}

        def capture_run(args, **kwargs):
            captured["args"] = args
            return MagicMock(returncode=0)

        with patch("batch.plot.subprocess.run", side_effect=capture_run):
            plot_job(result, setplot="setplot.py")

        assert captured["args"][-1] == str(job_paths.job / "setplot.py")

    def test_plot_job_passes_absolute_setplot_unchanged(self, job_paths, tmp_path):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        abs_path = tmp_path / "custom_setplot.py"
        abs_path.touch()
        captured = {}

        def capture_run(args, **kwargs):
            captured["args"] = args
            return MagicMock(returncode=0)

        with patch("batch.plot.subprocess.run", side_effect=capture_run):
            plot_job(result, setplot=abs_path)

        assert captured["args"][-1] == str(abs_path.resolve())

    def test_plot_job_output_appended_to_log(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        with patch("batch.plot.subprocess.run", return_value=MagicMock(returncode=0)):
            plot_job(result)
        assert "--- plotclaw ---" in job_paths.log.read_text()

    def test_plot_job_callable_setplot_uses_inprocess_fallback(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        def setplot_fn():
            pass
        with patch("batch.plot._plot_inprocess", return_value=True) as mock_inproc:
            with patch("batch.plot.subprocess.run") as mock_run:
                plot_job(result, setplot=setplot_fn)
        mock_inproc.assert_called_once()
        mock_run.assert_not_called()


class TestPlotInprocess:
    def test_plot_inprocess_returns_false_when_visclaw_not_importable(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        with patch.dict(sys.modules, {
            "clawpack": MagicMock(),
            "clawpack.visclaw": MagicMock(),
            "clawpack.visclaw.plotclaw": None,
        }):
            assert _plot_inprocess(result, "setplot.py", "ascii") is False

    def test_plot_inprocess_returns_false_on_exception(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        mock_module = MagicMock()
        mock_module.plotclaw = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, {
            "clawpack": MagicMock(),
            "clawpack.visclaw": MagicMock(),
            "clawpack.visclaw.plotclaw": mock_module,
        }):
            assert _plot_inprocess(result, "setplot.py", "ascii") is False

    def test_plot_inprocess_returns_true_on_success(self, job_paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=job_paths, returncode=0)
        mock_module = MagicMock()
        mock_module.plotclaw = MagicMock()
        with patch.dict(sys.modules, {
            "clawpack": MagicMock(),
            "clawpack.visclaw": MagicMock(),
            "clawpack.visclaw.plotclaw": mock_module,
        }):
            assert _plot_inprocess(result, "setplot.py", "ascii") is True
