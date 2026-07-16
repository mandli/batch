"""Tests for batch.cli.

All tests avoid a real scheduler and solver: executor construction is checked by
type, packed dispatch is patched, and execute() is driven through the
dry_run / setup-only paths so nothing is actually submitted or run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from batch.cli import (
    ResultSummary,
    add_execution_args,
    clobber_from_args,
    execute,
    executor_from_args,
    report_results,
)
from batch.executors.local import ParallelExecutor
from batch.executors.pbs import PBSExecutor
from batch.executors.slurm import SLURMExecutor
from batch.job import ClobberPolicy, JobPaths, JobResult
from tests.conftest import MockJob


def parse(argv, *, packed=True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_execution_args(parser, packed=packed)
    return parser.parse_args(argv)


def make_result(prefix, returncode, job_id=None, tmp_path=None) -> JobResult:
    job = MockJob(prefix=prefix)
    base = tmp_path or Path("/tmp")
    paths = JobPaths(
        job=base / prefix, plots=base / prefix / "plots", log=base / prefix / "log.txt"
    )
    return JobResult(job=job, paths=paths, returncode=returncode, job_id=job_id)


# ---------------------------------------------------------------------------
# add_execution_args
# ---------------------------------------------------------------------------


class TestAddExecutionArgs:
    def test_defaults(self):
        args = parse([])
        assert args.scheduler == "local"
        assert args.resume is False
        assert args.setup_only is False
        assert args.max_workers >= 1
        assert args.queue == "main"

    def test_overrides(self):
        args = parse(
            [
                "--scheduler",
                "pbs",
                "--resume",
                "--account",
                "NCAR0001",
                "--walltime",
                "06:00:00",
                "--omp-num-threads",
                "16",
                "--modules",
                "ncarenv",
                "conda",
            ]
        )
        assert args.scheduler == "pbs"
        assert args.resume is True
        assert args.account == "NCAR0001"
        assert args.walltime == "06:00:00"
        assert args.omp_num_threads == 16
        assert args.modules == ["ncarenv", "conda"]

    def test_packed_flags_present(self):
        args = parse(
            ["--nodes", "4", "--node-cpus", "128", "--shard", "2/4", "--pin-cpus"]
        )
        assert args.nodes == 4
        assert args.node_cpus == 128
        assert args.shard == "2/4"
        assert args.pin_cpus is True

    def test_packed_flags_absent_when_disabled(self):
        parser = argparse.ArgumentParser()
        add_execution_args(parser, packed=False)
        with pytest.raises(SystemExit):
            parser.parse_args(["--nodes", "4"])

    def test_project_can_add_own_flags(self):
        parser = argparse.ArgumentParser()
        add_execution_args(parser)
        parser.add_argument("--storms-path")
        args = parser.parse_args(["--storms-path", "/data", "--scheduler", "slurm"])
        assert args.storms_path == "/data"
        assert args.scheduler == "slurm"


# ---------------------------------------------------------------------------
# clobber_from_args
# ---------------------------------------------------------------------------


class TestClobberFromArgs:
    def test_resume_gives_skip(self):
        assert clobber_from_args(parse(["--resume"])) is ClobberPolicy.SKIP

    def test_no_resume_gives_overwrite(self):
        assert clobber_from_args(parse([])) is ClobberPolicy.OVERWRITE


# ---------------------------------------------------------------------------
# executor_from_args
# ---------------------------------------------------------------------------


class TestExecutorFromArgs:
    def test_local_returns_parallel_executor(self):
        ex = executor_from_args(parse(["--scheduler", "local", "--max-workers", "3"]))
        assert isinstance(ex, ParallelExecutor)
        assert ex.max_workers == 3
        assert ex.cpu_affinity is False

    def test_local_pin_cpus_enables_affinity(self):
        ex = executor_from_args(
            parse(
                [
                    "--scheduler",
                    "local",
                    "--max-workers",
                    "4",
                    "--pin-cpus",
                    "--node-cpus",
                    "128",
                ]
            )
        )
        assert ex.cpu_affinity is True
        assert ex.total_cpus == 128

    def test_omp_threads_in_env(self):
        ex = executor_from_args(
            parse(["--scheduler", "local", "--omp-num-threads", "8"])
        )
        assert ex.env["OMP_NUM_THREADS"] == "8"

    def test_extra_env_merged(self):
        ex = executor_from_args(parse(["--scheduler", "local"]), env={"FOO": "bar"})
        assert ex.env["FOO"] == "bar"
        assert "OMP_NUM_THREADS" in ex.env

    def test_pbs_returns_pbs_executor_with_threads(self):
        ex = executor_from_args(
            parse(["--scheduler", "pbs", "--omp-num-threads", "64"])
        )
        assert isinstance(ex, PBSExecutor)
        assert ex.default_resources.ompthreads == 64
        assert ex.default_resources.ncpus == 64

    def test_setup_only_sets_scheduler_dry_run(self):
        ex = executor_from_args(parse(["--scheduler", "pbs", "--setup-only"]))
        assert ex.dry_run is True

    def test_slurm_returns_slurm_executor(self):
        ex = executor_from_args(parse(["--scheduler", "slurm", "--queue", "gpu"]))
        assert isinstance(ex, SLURMExecutor)
        assert ex.default_resources.partition == "gpu"

    def test_packed_scheduler_raises(self):
        with pytest.raises(ValueError, match="packed"):
            executor_from_args(parse(["--scheduler", "pbs-packed"]))


# ---------------------------------------------------------------------------
# report_results
# ---------------------------------------------------------------------------


class TestReportResults:
    def test_counts(self):
        results = [
            make_result("a", 0),
            make_result("b", 1),
            make_result("c", 0),
            make_result("d", None, job_id="123"),
        ]
        summary = report_results(results, echo=False)
        assert summary == ResultSummary(
            n_total=4,
            n_ok=2,
            n_failed=1,
            n_pending=1,
            failures=summary.failures,
        )
        assert [r.job.prefix for r in summary.failures] == ["b"]

    def test_all_pending_reports_submitted(self, capsys):
        results = [
            make_result("a", None, job_id="1"),
            make_result("b", None, job_id="2"),
        ]
        summary = report_results(results, echo=True)
        assert summary.n_pending == 2
        out = capsys.readouterr().out
        assert "Submitted 2 job(s)" in out
        assert "job 1" in out and "job 2" in out

    def test_failures_printed(self, capsys, tmp_path):
        report_results([make_result("boom", 1, tmp_path=tmp_path)], echo=True)
        out = capsys.readouterr().out
        assert "FAILED: boom" in out


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_packed_requires_inner_command(self):
        with pytest.raises(ValueError, match="inner_command"):
            execute(parse(["--scheduler", "pbs-packed"]), [], experiment="e")

    def test_packed_dispatches_to_submit_packed(self, tmp_path):
        args = parse(["--scheduler", "pbs-packed", "--nodes", "3", "--setup-only"])
        inner = lambda i, n: ["python", "d.py", "--shard", f"{i}/{n}"]  # noqa: E731
        with patch("batch.cli.submit_packed") as mock_sp:
            out = execute(
                args, [], experiment="e", inner_command=inner, script_dir=tmp_path
            )
        assert out == []
        mock_sp.assert_called_once()
        kwargs = mock_sp.call_args.kwargs
        assert kwargs["n_nodes"] == 3
        assert kwargs["scheduler"] == "pbs"
        assert kwargs["dry_run"] is True

    def test_slurm_packed_maps_to_slurm(self, tmp_path):
        args = parse(["--scheduler", "slurm-packed", "--nodes", "2"])
        with patch("batch.cli.submit_packed") as mock_sp:
            execute(
                args,
                [],
                experiment="e",
                inner_command=lambda i, n: ["x"],
                script_dir=tmp_path,
            )
        assert mock_sp.call_args.kwargs["scheduler"] == "slurm"

    def test_local_setup_only_writes_data_and_returns_empty(self, tmp_path):
        jobs = [MockJob(prefix=f"j{i}") for i in range(3)]
        out = execute(
            parse(["--scheduler", "local", "--setup-only"]),
            jobs,
            experiment="exp",
            base_path=tmp_path,
        )
        assert out == []
        # setup() wrote each job's directory + data file.
        for i in range(3):
            assert (tmp_path / "exp" / f"j{i}" / "claw.data").exists()

    def test_local_setup_only_honors_shard(self, tmp_path):
        jobs = [MockJob(prefix=f"j{i}") for i in range(4)]
        execute(
            parse(["--scheduler", "local", "--setup-only", "--shard", "1/2"]),
            jobs,
            experiment="exp",
            base_path=tmp_path,
        )
        # Shard 1/2 of [j0,j1,j2,j3] is [j0, j2].
        exp = tmp_path / "exp"
        assert (exp / "j0").exists() and (exp / "j2").exists()
        assert not (exp / "j1").exists() and not (exp / "j3").exists()

    def test_pbs_setup_only_writes_scripts(self, tmp_path):
        jobs = [MockJob(prefix=f"j{i}") for i in range(2)]
        out = execute(
            parse(["--scheduler", "pbs", "--setup-only"]),
            jobs,
            experiment="exp",
            base_path=tmp_path,
        )
        # dry_run executor: one pending result + a submission script per job.
        assert len(out) == 2
        assert all(r.pending for r in out)
        for i in range(2):
            assert (tmp_path / "exp" / f"j{i}" / "j{}_run.sh".format(i)).exists()
