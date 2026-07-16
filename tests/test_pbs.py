"""Tests for PBSResources and render_pbs_script.

render_pbs_script is a pure function so all tests run without a cluster.
PBSExecutor submission is tested via dry_run=True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.executors.pbs import PBSExecutor, PBSResources, render_pbs_script
from batch.job import JobPaths
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


@pytest.fixture
def minimal_resources() -> PBSResources:
    return PBSResources(queue="main", nodes=1, walltime="02:00:00")


# ---------------------------------------------------------------------------
# render_pbs_script — directive correctness
# ---------------------------------------------------------------------------


class TestRenderPBSScript:
    def test_starts_with_shebang(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert script.startswith("#!/bin/bash")

    def test_contains_job_name_directive(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "#PBS -N job_001" in script

    def test_log_path_in_directives(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert str(paths.log) in script

    def test_join_output_directive_present(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "#PBS -j oe" in script

    def test_queue_in_directives(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(queue="develop")
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -q develop" in script

    def test_walltime_in_directives(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(walltime="12:30:00")
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -l walltime=12:30:00" in script

    def test_select_chunk_composed(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(nodes=2, ncpus=128, mpiprocs=1, ompthreads=64)
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -l select=2:ncpus=128:mpiprocs=1:ompthreads=64" in script

    def test_mem_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        minimal_resources.mem = ""
        script = render_pbs_script(job, paths, minimal_resources)
        assert ":mem=" not in script

    def test_mem_present_in_chunk_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(mem="235GB")
        script = render_pbs_script(job, paths, resources)
        assert ":mem=235GB" in script

    def test_account_present_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(account="NCAR0001")
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -A NCAR0001" in script

    def test_account_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "#PBS -A" not in script

    def test_email_directives_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(email="user@example.com", mail_points="abe")
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -M user@example.com" in script
        assert "#PBS -m abe" in script

    def test_email_directives_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "#PBS -M" not in script

    def test_module_load_lines_present(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(modules=["ncarenv/23.09", "conda"])
        script = render_pbs_script(job, paths, resources)
        assert "module load ncarenv/23.09" in script
        assert "module load conda" in script

    def test_env_vars_exported(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(env_vars={"OMP_NUM_THREADS": "128"})
        script = render_pbs_script(job, paths, resources)
        assert "export OMP_NUM_THREADS=128" in script

    def test_extra_directives_appended(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(
            extra_directives=["#PBS -l job_priority=premium", "#PBS -r y"]
        )
        script = render_pbs_script(job, paths, resources)
        assert "#PBS -l job_priority=premium" in script
        assert "#PBS -r y" in script

    def test_plot_line_absent_by_default(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "clawpack.visclaw.plotclaw" not in script

    def test_plot_line_present_when_enabled(self, paths):
        job = MockJob(prefix="job_001")
        resources = PBSResources(plot=True, setplot="/path/to/setplot.py")
        script = render_pbs_script(job, paths, resources)
        assert "clawpack.visclaw.plotclaw" in script
        assert "/path/to/setplot.py" in script

    def test_plot_line_falls_back_to_job_setplot(self, paths):
        job = MockJob(prefix="job_001")
        job.setplot = "/job/setplot.py"
        resources = PBSResources(plot=True)  # no setplot on resources
        script = render_pbs_script(job, paths, resources)
        assert "/job/setplot.py" in script

    def test_script_ends_with_newline(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert script.endswith("\n")

    def test_run_command_present(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_pbs_script(job, paths, minimal_resources)
        assert "clawpack.clawutil.runclaw" in script

    def test_solver_precedes_plot(self, paths):
        """plotclaw must run after the solver, not before."""
        job = MockJob(prefix="job_001")
        resources = PBSResources(plot=True, setplot="/path/to/setplot.py")
        script = render_pbs_script(job, paths, resources)
        assert script.index("runclaw") < script.index("plotclaw")

    def test_per_job_resource_override(self, paths):
        """pbs_resources on the job should override executor defaults."""
        job = MockJob(prefix="job_001")
        job.pbs_resources = PBSResources(queue="preempt", walltime="04:00:00")

        executor = PBSExecutor(
            default_resources=PBSResources(queue="main", walltime="01:00:00"),
            dry_run=True,
        )
        executor.submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "#PBS -q preempt" in script
        assert "#PBS -l walltime=04:00:00" in script


# ---------------------------------------------------------------------------
# PBSExecutor dry_run
# ---------------------------------------------------------------------------


class TestPBSExecutorDryRun:
    def test_dry_run_writes_script_file(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        executor = PBSExecutor(default_resources=minimal_resources, dry_run=True)
        executor.submit(job, paths)
        assert (paths.job / "job_001_run.sh").exists()

    def test_dry_run_returns_dry_run_job_id(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        executor = PBSExecutor(default_resources=minimal_resources, dry_run=True)
        result = executor.submit(job, paths)
        assert result.job_id == "dry-run"
        assert result.returncode is None

    def test_dry_run_does_not_call_qsub(self, paths, minimal_resources):
        from unittest.mock import patch

        job = MockJob(prefix="job_001")
        executor = PBSExecutor(default_resources=minimal_resources, dry_run=True)
        with patch("batch.executors.pbs.subprocess.run") as mock_run:
            executor.submit(job, paths)
        mock_run.assert_not_called()

    def test_wait_all_skips_dry_run_jobs(self, paths, minimal_resources):
        """wait_all should not poll qstat for dry-run job IDs."""
        from unittest.mock import patch

        job = MockJob(prefix="job_001")
        executor = PBSExecutor(default_resources=minimal_resources, dry_run=True)
        result = executor.submit(job, paths)
        with patch("batch.executors.pbs.subprocess.run") as mock_run:
            executor.wait_all([result])
        mock_run.assert_not_called()
