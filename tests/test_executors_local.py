"""Tests for SerialExecutor and ParallelExecutor.

No Clawpack installation is required.  subprocess.run / subprocess.Popen are
patched to avoid actually launching xgeoclaw.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batch.executors.local import (
    ParallelExecutor,
    SerialExecutor,
    _build_run_args,
)
from batch.job import JobPaths, JobResult
from tests.conftest import MockJob


@pytest.fixture
def paths(tmp_path: Path) -> JobPaths:
    job_dir = tmp_path / "job_001"
    job_dir.mkdir()
    return JobPaths(
        job=job_dir,
        plots=job_dir / "plots",
        log=job_dir / "job_001_log.txt",
    )


# ---------------------------------------------------------------------------
# _build_run_args
# ---------------------------------------------------------------------------


class TestBuildRunArgs:
    def test_uses_sys_executable(self, paths):
        job = MockJob(prefix="job_001")
        args = _build_run_args(job, paths)
        assert args[0] == sys.executable

    def test_invokes_runclaw_module(self, paths):
        job = MockJob(prefix="job_001")
        args = _build_run_args(job, paths)
        assert args[1:3] == ["-m", "clawpack.clawutil.runclaw"]

    def test_overwrite_flag_when_not_restarting(self, paths):
        # args: [python, -m, runclaw, exe, outdir, overwrite, restart, rundir, verbose]
        #         0       1   2        3    4       5          6        7       8
        job = MockJob(prefix="job_001")
        job.restart = False
        args = _build_run_args(job, paths)
        assert args[5] == "T"  # overwrite
        assert args[6] == "F"  # restart

    def test_restart_flags_when_restarting(self, paths):
        job = MockJob(prefix="job_001")
        job.restart = True
        args = _build_run_args(job, paths)
        assert args[5] == "F"  # overwrite
        assert args[6] == "T"  # restart

    def test_outdir_and_rundir_are_same(self, paths):
        job = MockJob(prefix="job_001")
        args = _build_run_args(job, paths)
        outdir = args[4]
        rundir = args[7]
        assert outdir == rundir == str(paths.job)


# ---------------------------------------------------------------------------
# SerialExecutor
# ---------------------------------------------------------------------------


class TestSerialExecutor:
    def test_submit_returns_success_result(self, paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("batch.executors.local.subprocess.run", return_value=mock_result):
            result = executor.submit(job, paths)
        assert result.returncode == 0
        assert result.job is job

    def test_submit_returns_failure_returncode(self, paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor()
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("batch.executors.local.subprocess.run", return_value=mock_result):
            result = executor.submit(job, paths)
        assert result.returncode == 1

    def test_submit_writes_to_log(self, paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("batch.executors.local.subprocess.run", return_value=mock_result):
            executor.submit(job, paths)
        assert paths.log.exists()

    def test_wait_all_is_identity(self, paths):
        job = MockJob(prefix="job_001")
        result = JobResult(job=job, paths=paths, returncode=0)
        executor = SerialExecutor()
        returned = executor.wait_all([result])
        assert returned == [result] or returned == [result]

    def test_extra_args_appended(self, paths):
        job = MockJob(prefix="job_001")
        executor = SerialExecutor(extra_args=["--extra", "flag"])
        captured_args = []
        mock_result = MagicMock()
        mock_result.returncode = 0

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return mock_result

        with patch("batch.executors.local.subprocess.run", side_effect=fake_run):
            executor.submit(job, paths)
        assert "--extra" in captured_args
        assert "flag" in captured_args


# ---------------------------------------------------------------------------
# ParallelExecutor._drain — the core correctness test
# ---------------------------------------------------------------------------


class TestParallelExecutorDrain:
    """Test _drain without actually spawning processes."""

    def _make_mock_proc(self, poll_return):
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = poll_return
        return proc

    def test_drain_removes_completed_processes(self, paths):
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")

        done_proc = self._make_mock_proc(poll_return=0)
        running_proc = self._make_mock_proc(poll_return=None)
        done_result = JobResult(job=job, paths=paths, returncode=None)
        running_result = JobResult(job=job, paths=paths, returncode=None)
        log_fh = MagicMock()

        executor._active = [
            (done_proc, done_result, log_fh, None),
            (running_proc, running_result, log_fh, None),
        ]
        executor._drain()

        assert len(executor._active) == 1
        assert executor._active[0][1] is running_result

    def test_drain_sets_returncode_on_completed(self, paths):
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")

        proc = self._make_mock_proc(poll_return=0)
        result = JobResult(job=job, paths=paths, returncode=None)
        log_fh = MagicMock()
        executor._active = [(proc, result, log_fh, None)]

        executor._drain()

        assert result.returncode == 0

    def test_drain_closes_log_handle_on_completion(self, paths):
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")

        proc = self._make_mock_proc(poll_return=0)
        result = JobResult(job=job, paths=paths, returncode=None)
        log_fh = MagicMock()
        executor._active = [(proc, result, log_fh, None)]

        executor._drain()

        log_fh.close.assert_called_once()

    def test_drain_returns_slot_to_free_pool_on_completion(self, paths):
        """A completed job's affinity slot should be returned to the pool."""
        executor = ParallelExecutor(max_workers=4)
        job = MockJob(prefix="job_001")

        proc = self._make_mock_proc(poll_return=0)
        result = JobResult(job=job, paths=paths, returncode=None)
        log_fh = MagicMock()
        # Simulate a job that was handed slot 2 (removed from the free pool).
        executor._free_slots.remove(2)
        executor._active = [(proc, result, log_fh, 2)]

        executor._drain()

        assert 2 in executor._free_slots

    def test_drain_does_not_skip_consecutive_completed_processes(self, paths):
        """Regression test: the original list-modify-while-iterating bug
        caused every other completed process to be silently skipped."""
        executor = ParallelExecutor(max_workers=8)
        job = MockJob(prefix="job_001")
        results = []
        log_fh = MagicMock()

        for _ in range(6):
            proc = self._make_mock_proc(poll_return=0)
            result = JobResult(job=job, paths=paths, returncode=None)
            results.append(result)
            executor._active.append((proc, result, log_fh, None))

        executor._drain()

        assert executor._active == [], "all completed processes should be drained"
        assert all(r.returncode == 0 for r in results), (
            "all results should have their returncode set"
        )
