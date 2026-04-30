"""Tests for BatchController: path layout, clobber policies, setup, run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from batch.controller import BatchController
from batch.job import ClobberPolicy, Job, JobPaths, JobResult
from tests.conftest import MockJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stub_executor(returncode: int = 0):
    """Return a mock Executor whose submit() returns a successful JobResult."""
    executor = MagicMock()

    def _submit(job, paths):
        result = JobResult(job=job, paths=paths, returncode=returncode)
        job.paths = paths
        return result

    executor.submit.side_effect = _submit
    executor.wait_all.side_effect = lambda results: results
    return executor


# ---------------------------------------------------------------------------
# _make_paths
# ---------------------------------------------------------------------------

class TestMakePaths:
    def test_paths_under_base(self, tmp_path):
        ctrl = BatchController(base_path=tmp_path)
        job = MockJob(prefix="run_001")
        paths = ctrl._make_paths(job)
        assert paths.job == tmp_path / "run_001"
        assert paths.plots == tmp_path / "run_001" / "plots"
        assert paths.log == tmp_path / "run_001" / "run_001_log.txt"

    def test_paths_with_experiment(self, tmp_path):
        ctrl = BatchController(base_path=tmp_path, experiment="hurricane_ike")
        job = MockJob(prefix="run_001")
        paths = ctrl._make_paths(job)
        assert paths.job == tmp_path / "hurricane_ike" / "run_001"

    def test_raises_without_prefix(self, tmp_path):
        ctrl = BatchController(base_path=tmp_path)
        job = MockJob()
        job.prefix = None
        with pytest.raises(ValueError, match="no prefix"):
            ctrl._make_paths(job)


# ---------------------------------------------------------------------------
# _setup_job_dir — clobber policies
# ---------------------------------------------------------------------------

class TestSetupJobDir:
    def test_overwrite_creates_directory(self, tmp_path):
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.OVERWRITE)
        job = MockJob(prefix="job_001")
        paths = ctrl._make_paths(job)
        result = ctrl._setup_job_dir(job, paths)
        assert result is True
        assert paths.job.is_dir()

    def test_overwrite_removes_data_files(self, tmp_path):
        job = MockJob(prefix="job_001")
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.OVERWRITE)
        paths = ctrl._make_paths(job)
        paths.job.mkdir(parents=True)
        stale = paths.job / "claw.data"
        stale.write_text("old data\n")
        keeper = paths.job / "fort.q0001"
        keeper.write_text("output\n")

        ctrl._setup_job_dir(job, paths)

        assert not stale.exists(), "stale .data file should have been removed"
        assert keeper.exists(), "fort.* output should be untouched"

    def test_overwrite_keeps_data_files_on_restart(self, tmp_path):
        job = MockJob(prefix="job_001")
        job.restart = True
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.OVERWRITE)
        paths = ctrl._make_paths(job)
        paths.job.mkdir(parents=True)
        data_file = paths.job / "claw.data"
        data_file.write_text("restart data\n")

        ctrl._setup_job_dir(job, paths)

        assert data_file.exists(), ".data file must be preserved for restart"

    def test_error_policy_raises_on_existing_dir(self, tmp_path):
        job = MockJob(prefix="job_001")
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.ERROR)
        paths = ctrl._make_paths(job)
        paths.job.mkdir(parents=True)

        with pytest.raises(FileExistsError, match="already exists"):
            ctrl._setup_job_dir(job, paths)

    def test_error_policy_passes_for_new_dir(self, tmp_path):
        job = MockJob(prefix="job_001")
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.ERROR)
        paths = ctrl._make_paths(job)
        result = ctrl._setup_job_dir(job, paths)
        assert result is True

    def test_skip_policy_returns_false_on_existing_dir(self, tmp_path):
        job = MockJob(prefix="job_001")
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.SKIP)
        paths = ctrl._make_paths(job)
        paths.job.mkdir(parents=True)

        result = ctrl._setup_job_dir(job, paths)
        assert result is False

    def test_skip_policy_returns_true_for_new_dir(self, tmp_path):
        job = MockJob(prefix="job_001")
        ctrl = BatchController(base_path=tmp_path, clobber=ClobberPolicy.SKIP)
        paths = ctrl._make_paths(job)
        result = ctrl._setup_job_dir(job, paths)
        assert result is True


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_creates_directories_and_data_files(self, tmp_path, three_jobs):
        ctrl = BatchController(
            jobs=three_jobs,
            base_path=tmp_path,
            executor=make_stub_executor(),
        )
        paths_list = ctrl.setup()
        assert len(paths_list) == 3
        for paths in paths_list:
            assert paths.job.is_dir()
            assert (paths.job / "claw.data").exists()

    def test_setup_assigns_paths_to_job(self, tmp_path, mock_job):
        ctrl = BatchController(
            jobs=[mock_job],
            base_path=tmp_path,
            executor=make_stub_executor(),
        )
        ctrl.setup()
        assert mock_job.paths is not None
        assert mock_job.paths.job == tmp_path / "job_001"

    def test_setup_writes_log_header(self, tmp_path, mock_job):
        ctrl = BatchController(
            jobs=[mock_job],
            base_path=tmp_path,
            executor=make_stub_executor(),
        )
        ctrl.setup()
        log = mock_job.paths.log.read_text()
        assert "Started" in log

    def test_setup_skips_existing_dirs_under_skip_policy(self, tmp_path):
        job_a = MockJob(prefix="job_000")
        job_b = MockJob(prefix="job_001")
        ctrl = BatchController(
            jobs=[job_a, job_b],
            base_path=tmp_path,
            clobber=ClobberPolicy.SKIP,
        )
        # Pre-create job_a directory
        (tmp_path / "job_000").mkdir(parents=True)
        paths_list = ctrl.setup()
        # Only job_b should have been set up
        assert len(paths_list) == 1
        assert paths_list[0].job == tmp_path / "job_001"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_calls_submit_for_each_job(self, tmp_path, three_jobs):
        executor = make_stub_executor()
        ctrl = BatchController(
            jobs=three_jobs,
            base_path=tmp_path,
            executor=executor,
        )
        results = ctrl.run()
        assert executor.submit.call_count == 3
        assert len(results) == 3

    def test_run_calls_wait_all_when_wait_true(self, tmp_path, mock_job):
        executor = make_stub_executor()
        ctrl = BatchController(
            jobs=[mock_job],
            base_path=tmp_path,
            executor=executor,
        )
        ctrl.run(wait=True)
        executor.wait_all.assert_called_once()

    def test_run_skips_wait_all_when_wait_false(self, tmp_path, mock_job):
        executor = make_stub_executor()
        ctrl = BatchController(
            jobs=[mock_job],
            base_path=tmp_path,
            executor=executor,
        )
        ctrl.run(wait=False)
        executor.wait_all.assert_not_called()

    def test_run_calls_build_on_each_job(self, tmp_path):
        job = MockJob(prefix="job_001")
        job.build = MagicMock()
        ctrl = BatchController(
            jobs=[job],
            base_path=tmp_path,
            executor=make_stub_executor(),
        )
        ctrl.run()
        job.build.assert_called_once()

    def test_run_returns_results_with_correct_prefix(self, tmp_path, three_jobs):
        executor = make_stub_executor()
        ctrl = BatchController(
            jobs=three_jobs,
            base_path=tmp_path,
            executor=executor,
        )
        results = ctrl.run()
        prefixes = {r.job.prefix for r in results}
        assert prefixes == {j.prefix for j in three_jobs}

    def test_run_skips_jobs_under_skip_policy(self, tmp_path):
        job_a = MockJob(prefix="job_000")
        job_b = MockJob(prefix="job_001")
        executor = make_stub_executor()
        ctrl = BatchController(
            jobs=[job_a, job_b],
            base_path=tmp_path,
            clobber=ClobberPolicy.SKIP,
            executor=executor,
        )
        (tmp_path / "job_000").mkdir(parents=True)
        results = ctrl.run()
        # Only job_b submitted
        assert len(results) == 1
        assert results[0].job.prefix == "job_001"

    def test_run_logs_warning_on_failures(self, tmp_path, mock_job, caplog):
        import logging
        executor = make_stub_executor(returncode=1)
        ctrl = BatchController(
            jobs=[mock_job],
            base_path=tmp_path,
            executor=executor,
        )
        with caplog.at_level(logging.WARNING, logger="batch.controller"):
            ctrl.run()
        assert any("failed" in rec.message for rec in caplog.records)

    def test_base_path_falls_back_to_env_var(self, tmp_path, mock_job, monkeypatch):
        monkeypatch.setenv("OUTPUT_PATH", str(tmp_path))
        ctrl = BatchController(jobs=[mock_job], executor=make_stub_executor())
        assert ctrl.base_path == tmp_path
