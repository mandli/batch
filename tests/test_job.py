"""Tests for batch.job — Job, JobPaths, JobResult, ClobberPolicy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batch.executors.local import ParallelExecutor, SerialExecutor
from batch.job import ClobberPolicy, Job, JobPaths, JobResult
from tests.conftest import MockJob


class TestJob:
    def test_default_attributes(self):
        job = Job()
        assert job.prefix is None
        assert job.executable == "xgeoclaw"
        assert job.setplot == "setplot"
        assert job.restart is False
        assert job.paths is None
        assert job.rundata is None

    def test_repr(self):
        job = MockJob(prefix="test_001")
        assert "MockJob" in repr(job)
        assert "test_001" in repr(job)

    def test_write_data_objects_calls_rundata_write(self, tmp_path):
        job = Job()
        job.prefix = "job_001"
        job.rundata = MagicMock()
        job.write_data_objects(tmp_path)
        job.rundata.write.assert_called_once_with(out_dir=tmp_path)

    def test_write_data_objects_raises_without_rundata(self, tmp_path):
        job = Job()
        job.prefix = "job_001"
        with pytest.raises(ValueError, match="rundata is not set"):
            job.write_data_objects(tmp_path)

    def test_build_is_noop_by_default(self, job_paths):
        """build() must not raise or produce side effects by default."""
        job = MockJob()
        job.build(job_paths)  # should not raise

    def test_mock_job_write_creates_data_file(self, tmp_path):
        job = MockJob(prefix="abc")
        job.write_data_objects(tmp_path)
        assert (tmp_path / "claw.data").exists()
        assert len(job._write_calls) == 1
        assert job._write_calls[0] == tmp_path


class TestJobPaths:
    def test_fields_are_paths(self, tmp_path):
        paths = JobPaths(
            job=tmp_path / "job",
            plots=tmp_path / "job" / "plots",
            log=tmp_path / "job" / "job_log.txt",
        )
        assert isinstance(paths.job, Path)
        assert isinstance(paths.plots, Path)
        assert isinstance(paths.log, Path)


class TestJobResult:
    def test_success_true_when_returncode_zero(self, job_paths):
        job = MockJob()
        result = JobResult(job=job, paths=job_paths, returncode=0)
        assert result.success is True

    def test_success_false_when_nonzero(self, job_paths):
        job = MockJob()
        result = JobResult(job=job, paths=job_paths, returncode=1)
        assert result.success is False

    def test_success_false_when_none(self, job_paths):
        job = MockJob()
        result = JobResult(job=job, paths=job_paths, returncode=None)
        assert result.success is False

    def test_pending_true_when_returncode_none(self, job_paths):
        job = MockJob()
        result = JobResult(job=job, paths=job_paths, returncode=None)
        assert result.pending is True

    def test_pending_false_when_returncode_set(self, job_paths):
        job = MockJob()
        result = JobResult(job=job, paths=job_paths, returncode=0)
        assert result.pending is False


class TestClobberPolicy:
    def test_all_values_present(self):
        assert ClobberPolicy.OVERWRITE.value == "overwrite"
        assert ClobberPolicy.ERROR.value == "error"
        assert ClobberPolicy.SKIP.value == "skip"


class TestPostRun:
    def test_post_run_is_noop_by_default(self, job_paths):
        job = Job()
        job.prefix = "job_001"
        result = JobResult(job=job, paths=job_paths, returncode=0)
        job.post_run(result)  # must not raise

    def test_post_run_called_on_success_serial(self, job_paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("batch.executors.local.subprocess.run", return_value=mock_proc):
            with patch.object(job, "post_run") as mock_post_run:
                result = executor.submit(job, job_paths)
        mock_post_run.assert_called_once_with(result)

    def test_post_run_not_called_on_failure_serial(self, job_paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("batch.executors.local.subprocess.run", return_value=mock_proc):
            with patch.object(job, "post_run") as mock_post_run:
                executor.submit(job, job_paths)
        mock_post_run.assert_not_called()

    def test_post_run_called_on_success_parallel(self, job_paths):
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 0
        result = JobResult(job=job, paths=job_paths, returncode=None)
        log_fh = MagicMock()
        executor._active = [(proc, result, log_fh, None)]
        with patch.object(job, "post_run") as mock_post_run:
            executor._drain()
        mock_post_run.assert_called_once_with(result)

    def test_post_run_not_called_on_failure_parallel(self, job_paths):
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 1
        result = JobResult(job=job, paths=job_paths, returncode=None)
        log_fh = MagicMock()
        executor._active = [(proc, result, log_fh, None)]
        with patch.object(job, "post_run") as mock_post_run:
            executor._drain()
        mock_post_run.assert_not_called()

    def test_post_run_exception_does_not_propagate(self, job_paths):
        class ExplodingJob(MockJob):
            def post_run(self, result):
                raise RuntimeError("boom")

        job = ExplodingJob(prefix="job_001")
        executor = SerialExecutor()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("batch.executors.local.subprocess.run", return_value=mock_proc):
            result = executor.submit(job, job_paths)  # must not raise
        assert result.returncode == 0
