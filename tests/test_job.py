"""Tests for batch.job — Job, JobPaths, JobResult, ClobberPolicy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
