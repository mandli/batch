"""Tests for SLURMResources and render_slurm_script.

render_slurm_script is a pure function so all tests run without a cluster.
SLURMExecutor submission is tested via dry_run=True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.executors.slurm import SLURMExecutor, SLURMResources, render_slurm_script
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
def minimal_resources() -> SLURMResources:
    return SLURMResources(partition="main", nodes=1, time="02:00:00")


# ---------------------------------------------------------------------------
# render_slurm_script — directive correctness
# ---------------------------------------------------------------------------


class TestRenderSlurmScript:
    def test_starts_with_shebang(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert script.startswith("#!/bin/bash")

    def test_contains_job_name_directive(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert "#SBATCH -J job_001" in script

    def test_log_path_in_directives(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert str(paths.log) in script

    def test_partition_in_directives(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(partition="preempt")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH -p preempt" in script

    def test_walltime_in_directives(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(time="12:30:00")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH -t 12:30:00" in script

    def test_cpus_per_task_in_directives(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(cpus_per_task=16)
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH --cpus-per-task=16" in script

    def test_memory_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        minimal_resources.memory = ""
        script = render_slurm_script(job, paths, minimal_resources)
        assert "--mem=" not in script

    def test_memory_present_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(memory="8G")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH --mem=8G" in script

    def test_account_present_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(account="NCAR0001")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH -A NCAR0001" in script

    def test_account_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert "#SBATCH -A" not in script

    def test_constraint_present_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(constraint="cpu")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH --constraint=cpu" in script

    def test_email_directives_when_set(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(email="user@example.com", mail_type="END,FAIL")
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH --mail-user=user@example.com" in script
        assert "#SBATCH --mail-type=END,FAIL" in script

    def test_email_directives_absent_when_empty(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert "--mail-user" not in script

    def test_module_load_lines_present(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(modules=["ncarenv/23.09", "python/3.11.4"])
        script = render_slurm_script(job, paths, resources)
        assert "module load ncarenv/23.09" in script
        assert "module load python/3.11.4" in script

    def test_env_vars_exported(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(env_vars={"OMP_NUM_THREADS": "8"})
        script = render_slurm_script(job, paths, resources)
        assert "export OMP_NUM_THREADS=8" in script

    def test_extra_directives_appended(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(
            extra_directives=["#SBATCH --gres=gpu:1", "#SBATCH --licenses=scratch:1"]
        )
        script = render_slurm_script(job, paths, resources)
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --licenses=scratch:1" in script

    def test_plot_line_absent_by_default(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert "clawpack.visclaw.plotclaw" not in script

    def test_plot_line_present_when_enabled(self, paths):
        job = MockJob(prefix="job_001")
        resources = SLURMResources(plot=True, setplot="/path/to/setplot.py")
        script = render_slurm_script(job, paths, resources)
        assert "clawpack.visclaw.plotclaw" in script
        assert "/path/to/setplot.py" in script

    def test_plot_line_falls_back_to_job_setplot(self, paths):
        job = MockJob(prefix="job_001")
        job.setplot = "/job/setplot.py"
        resources = SLURMResources(plot=True)  # no setplot on resources
        script = render_slurm_script(job, paths, resources)
        assert "/job/setplot.py" in script

    def test_solver_precedes_plot(self, paths):
        """plotclaw must run after the solver, not before."""
        job = MockJob(prefix="job_001")
        resources = SLURMResources(plot=True, setplot="/path/to/setplot.py")
        script = render_slurm_script(job, paths, resources)
        assert script.index("runclaw") < script.index("plotclaw")

    def test_script_ends_with_newline(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        assert script.endswith("\n")

    def test_run_command_is_last_non_empty_line(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        script = render_slurm_script(job, paths, minimal_resources)
        non_empty = [ln for ln in script.splitlines() if ln.strip()]
        last_line = non_empty[-1]
        # Should invoke runclaw
        assert "clawpack.clawutil.runclaw" in last_line

    def test_per_job_resource_override(self, paths):
        """slurm_resources on the job should override executor defaults."""
        job = MockJob(prefix="job_001")
        job.slurm_resources = SLURMResources(partition="gpu", time="04:00:00")

        executor = SLURMExecutor(
            default_resources=SLURMResources(partition="main", time="01:00:00"),
            dry_run=True,
        )
        executor.submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "#SBATCH -p gpu" in script
        assert "#SBATCH -t 04:00:00" in script


# ---------------------------------------------------------------------------
# SLURMExecutor dry_run
# ---------------------------------------------------------------------------


class TestSLURMExecutorDryRun:
    def test_dry_run_writes_script_file(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        executor = SLURMExecutor(default_resources=minimal_resources, dry_run=True)
        executor.submit(job, paths)
        assert (paths.job / "job_001_run.sh").exists()

    def test_dry_run_returns_dry_run_job_id(self, paths, minimal_resources):
        job = MockJob(prefix="job_001")
        executor = SLURMExecutor(default_resources=minimal_resources, dry_run=True)
        result = executor.submit(job, paths)
        assert result.job_id == "dry-run"
        assert result.returncode is None

    def test_dry_run_does_not_call_sbatch(self, paths, minimal_resources):
        from unittest.mock import patch

        job = MockJob(prefix="job_001")
        executor = SLURMExecutor(default_resources=minimal_resources, dry_run=True)
        with patch("batch.executors.slurm.subprocess.run") as mock_run:
            executor.submit(job, paths)
        mock_run.assert_not_called()

    def test_wait_all_skips_dry_run_jobs(self, paths, minimal_resources):
        """wait_all should not poll squeue for dry-run job IDs."""
        from unittest.mock import patch

        job = MockJob(prefix="job_001")
        executor = SLURMExecutor(default_resources=minimal_resources, dry_run=True)
        result = executor.submit(job, paths)
        with patch("batch.executors.slurm.subprocess.run") as mock_run:
            executor.wait_all([result])
        mock_run.assert_not_called()
