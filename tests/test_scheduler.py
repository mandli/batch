"""Tests for the Scheduler abstraction and SchedulerExecutor.

Everything here runs without a live scheduler: the render methods are pure and
submission is exercised via ``dry_run`` / patched ``subprocess.run``.  The
headline test is :class:`TestBodyParity`, which asserts the emitted script body
below the header + normalize blocks is byte-identical across PBS and SLURM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.executors.scheduler import SchedulerExecutor, render_job_script
from batch.job import JobPaths
from batch.scheduler import (
    JobRequest,
    PBSScheduler,
    SlurmScheduler,
    get_scheduler,
)
from tests.conftest import MockJob


@pytest.fixture
def request_full() -> JobRequest:
    """A normalized request exercising the optional directives."""
    return JobRequest(
        name="job_001",
        log_path="/scratch/job_001/job_001_log.txt",
        queue="main",
        account="NCAR0001",
        walltime="02:00:00",
        nodes=1,
        cpus_per_node=128,
        tasks_per_node=1,
        ompthreads=128,
        exclusive=True,
        email="user@example.com",
    )


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
# Directive rendering — golden header blocks
# ---------------------------------------------------------------------------


class TestPBSDirectives:
    def test_golden_header(self, request_full):
        assert PBSScheduler().directives(request_full) == [
            "#PBS -N job_001",
            "#PBS -o /scratch/job_001/job_001_log.txt",
            "#PBS -j oe",
            "#PBS -q main",
            "#PBS -l select=1:ncpus=128:mpiprocs=1:ompthreads=128",
            "#PBS -l walltime=02:00:00",
            "#PBS -A NCAR0001",
            "#PBS -M user@example.com",
            "#PBS -m abe",
        ]

    def test_mem_appends_to_chunk(self, request_full):
        request_full.mem = "235GB"
        assert (
            "#PBS -l select=1:ncpus=128:mpiprocs=1:ompthreads=128:mem=235GB"
            in PBSScheduler().directives(request_full)
        )

    def test_account_and_queue_omitted_when_empty(self):
        req = JobRequest(name="j", log_path="/l", queue="", account="")
        lines = PBSScheduler().directives(req)
        assert not any(line.startswith("#PBS -A") for line in lines)
        assert not any(line.startswith("#PBS -q") for line in lines)

    def test_array_directive(self):
        req = JobRequest(name="j", log_path="/l", array="1-16:2")
        assert "#PBS -J 1-16:2" in PBSScheduler().directives(req)

    def test_exclusive_is_noop(self):
        req = JobRequest(name="j", log_path="/l", exclusive=True)
        lines = PBSScheduler().directives(req)
        assert not any("exclusive" in line for line in lines)

    def test_depend_directive(self):
        req = JobRequest(name="j", log_path="/l", depend=["111", "222"])
        assert "#PBS -W depend=afterok:111:222" in PBSScheduler().directives(req)

    def test_extra_directives_appended(self):
        req = JobRequest(
            name="j", log_path="/l", extra_directives=["#PBS -l job_priority=premium"]
        )
        assert "#PBS -l job_priority=premium" in PBSScheduler().directives(req)


class TestSlurmDirectives:
    def test_golden_header(self, request_full):
        assert SlurmScheduler().directives(request_full) == [
            "#SBATCH --job-name=job_001",
            "#SBATCH --output=/scratch/job_001/job_001_log.txt",
            "#SBATCH --error=/scratch/job_001/job_001_log.txt",
            "#SBATCH --partition=main",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=128",
            "#SBATCH --time=02:00:00",
            "#SBATCH --exclusive",
            "#SBATCH --account=NCAR0001",
            "#SBATCH --mail-user=user@example.com",
            "#SBATCH --mail-type=END,FAIL",
        ]

    def test_cpus_per_task_from_cores_single_rank(self):
        # Pure-OpenMP job reserving 8 cores: one task gets all 8.
        req = JobRequest(name="j", log_path="/l", tasks_per_node=1, cpus_per_node=8)
        lines = SlurmScheduler().directives(req)
        assert "#SBATCH --cpus-per-task=8" in lines

    def test_cpus_per_task_splits_node_across_ranks(self):
        # Exclusive 128-core node with 4 ranks: 32 cores per task.
        req = JobRequest(name="j", log_path="/l", tasks_per_node=4, cpus_per_node=128)
        lines = SlurmScheduler().directives(req)
        assert "#SBATCH --cpus-per-task=32" in lines

    def test_constraint_and_mem(self):
        req = JobRequest(name="j", log_path="/l", constraint="cpu", mem="235G")
        lines = SlurmScheduler().directives(req)
        assert "#SBATCH --constraint=cpu" in lines
        assert "#SBATCH --mem=235G" in lines

    def test_array_directive(self):
        req = JobRequest(name="j", log_path="/l", array="1-16:2")
        assert "#SBATCH --array=1-16:2" in SlurmScheduler().directives(req)

    def test_depend_directive(self):
        req = JobRequest(name="j", log_path="/l", depend=["111", "222"])
        assert "#SBATCH --dependency=afterok:111:222" in SlurmScheduler().directives(
            req
        )


# ---------------------------------------------------------------------------
# normalize_env — the BATCH_* contract
# ---------------------------------------------------------------------------


class TestNormalizeEnv:
    def test_pbs_exports_batch_contract(self):
        env = "\n".join(PBSScheduler().normalize_env())
        assert 'export BATCH_JOB_ID="$PBS_JOBID"' in env
        assert 'export BATCH_SUBMIT_DIR="$PBS_O_WORKDIR"' in env
        assert 'export BATCH_NODEFILE="$PBS_NODEFILE"' in env
        assert "BATCH_NNODES=" in env
        assert 'export BATCH_ARRAY_INDEX="${PBS_ARRAY_INDEX:-}"' in env

    def test_slurm_exports_batch_contract(self):
        env = "\n".join(SlurmScheduler().normalize_env())
        assert 'export BATCH_JOB_ID="$SLURM_JOB_ID"' in env
        assert 'export BATCH_SUBMIT_DIR="$SLURM_SUBMIT_DIR"' in env
        assert 'export BATCH_ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-}"' in env

    def test_slurm_nodefile_is_a_real_file(self):
        """SLURM must materialize a nodefile so packer node handling matches PBS."""
        env = "\n".join(SlurmScheduler().normalize_env())
        assert 'export BATCH_NODEFILE="$(mktemp)"' in env
        assert (
            'scontrol show hostnames "$SLURM_JOB_NODELIST" > "$BATCH_NODEFILE"' in env
        )

    def test_both_export_the_same_variable_set(self):
        def names(sched):
            return {
                line.split("=", 1)[0].removeprefix("export ").strip()
                for line in sched.normalize_env()
                if line.startswith("export ")
            }

        assert (
            names(PBSScheduler())
            == names(SlurmScheduler())
            == {
                "BATCH_JOB_ID",
                "BATCH_SUBMIT_DIR",
                "BATCH_NODEFILE",
                "BATCH_NNODES",
                "BATCH_ARRAY_INDEX",
            }
        )


# ---------------------------------------------------------------------------
# submit_argv / parse_job_id / poll_argv
# ---------------------------------------------------------------------------


class TestSubmitAndParse:
    def test_pbs_submit_argv(self):
        assert PBSScheduler().submit_argv("/s/run.sh") == ["qsub", "/s/run.sh"]

    def test_slurm_submit_argv_is_parsable(self):
        assert SlurmScheduler().submit_argv("/s/run.sh") == [
            "sbatch",
            "--parsable",
            "/s/run.sh",
        ]

    def test_pbs_parses_jobid_server_form(self):
        assert PBSScheduler().parse_job_id("1473351.desched1\n") == "1473351.desched1"

    def test_slurm_parses_bare_id(self):
        assert SlurmScheduler().parse_job_id("1473351\n") == "1473351"

    def test_slurm_strips_cluster_suffix(self):
        assert SlurmScheduler().parse_job_id("1473351;cluster\n") == "1473351"

    def test_pbs_poll_argv(self):
        assert PBSScheduler().poll_argv("123") == ["qstat", "123"]

    def test_slurm_poll_argv(self):
        assert SlurmScheduler().poll_argv("123") == [
            "squeue",
            "--job",
            "123",
            "--noheader",
        ]


class TestDependFlag:
    def test_empty(self):
        assert PBSScheduler().depend_flag([]) == ""
        assert SlurmScheduler().depend_flag([]) == ""

    def test_afterok_chain(self):
        assert PBSScheduler().depend_flag(["1", "2"]) == "afterok:1:2"
        assert SlurmScheduler().depend_flag(["1", "2"]) == "afterok:1:2"


class TestRegistry:
    def test_get_scheduler(self):
        assert isinstance(get_scheduler("pbs"), PBSScheduler)
        assert isinstance(get_scheduler("slurm"), SlurmScheduler)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown scheduler"):
            get_scheduler("lsf")


# ---------------------------------------------------------------------------
# Script rendering — env_file body correctness
# ---------------------------------------------------------------------------


class TestRenderJobScript:
    def _render(self, scheduler, request_full, **kw):
        kw.setdefault("env_file", "/etc/batch/derecho.zsh")
        kw.setdefault("python", "/venv/bin/python")
        kw.setdefault("workdir", "/scratch/job_001")
        return render_job_script(
            scheduler, request_full, ["/venv/bin/python", "run"], **kw
        )

    def test_shebang_zsh_no_login_flag(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert script.startswith("#!/bin/zsh\n")
        assert "zsh -l" not in script
        assert "#!/bin/zsh -l" not in script

    def test_sources_env_file(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert "source /etc/batch/derecho.zsh" in script

    def test_sets_strict_mode(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert "set -euo pipefail" in script

    def test_fail_fast_import_check(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert '/venv/bin/python -c "import batch"' in script
        assert "batch import failed on $(hostname)" in script

    def test_launches_via_absolute_python(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        # exec line uses the absolute interpreter, not a bare "python".
        assert "exec /venv/bin/python run" in script

    def test_execs_the_run(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert "\nexec " in script

    def test_cd_into_workdir(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert "cd /scratch/job_001" in script

    def test_paths_are_quoted(self, request_full):
        script = self._render(
            PBSScheduler(), request_full, env_file="/has space/env.zsh"
        )
        assert "source '/has space/env.zsh'" in script

    def test_optional_modules_and_env(self, request_full):
        script = self._render(
            PBSScheduler(),
            request_full,
            modules=["ncarenv/23.09"],
            env_vars={"OMP_NUM_THREADS": "128"},
        )
        assert "module load ncarenv/23.09" in script
        assert "export OMP_NUM_THREADS=128" in script

    def test_import_check_precedes_run(self, request_full):
        script = self._render(PBSScheduler(), request_full)
        assert script.index("import batch") < script.index("\nexec ")

    def test_ends_with_newline(self, request_full):
        assert self._render(PBSScheduler(), request_full).endswith("\n")

    def test_plot_command_runs_after_solver_without_exec(self, request_full):
        script = render_job_script(
            PBSScheduler(),
            request_full,
            ["/venv/bin/python", "-m", "clawpack.clawutil.runclaw", "x"],
            env_file="/e",
            python="/venv/bin/python",
            workdir="/w",
            plot_command=[
                "/venv/bin/python",
                "-m",
                "clawpack.visclaw.plotclaw",
                "/s.py",
            ],
        )
        # With a plot step the run is NOT exec'd (solver then plot, both plain).
        assert "exec " not in script
        assert "clawpack.visclaw.plotclaw" in script
        assert script.index("runclaw") < script.index("plotclaw")


# ---------------------------------------------------------------------------
# Parity: the body below header + normalize must be byte-identical
# ---------------------------------------------------------------------------


class TestBodyParity:
    """The whole point of the abstraction: identical body, per-backend header."""

    def _body_after_normalize(self, scheduler, request):
        """Return the script text below the header + normalize_env blocks."""
        script = render_job_script(
            scheduler,
            request,
            ["/venv/bin/python", "-m", "batch.pack", "--foo"],
            env_file="/etc/batch/env.zsh",
            python="/venv/bin/python",
            workdir="/scratch/job",
            env_vars={"OMP_NUM_THREADS": "128"},
        )
        lines = script.splitlines()
        # Drop every directive line and every normalize_env export/expansion;
        # what remains is the scheduler-agnostic body.
        normalize = set(scheduler.normalize_env())
        body = [
            ln
            for ln in lines
            if not ln.startswith(("#PBS", "#SBATCH")) and ln not in normalize
        ]
        return "\n".join(body)

    def test_body_is_byte_identical(self, request_full):
        pbs_body = self._body_after_normalize(PBSScheduler(), request_full)
        slurm_body = self._body_after_normalize(SlurmScheduler(), request_full)
        assert pbs_body == slurm_body

    def test_body_contains_the_run(self, request_full):
        pbs_body = self._body_after_normalize(PBSScheduler(), request_full)
        assert "exec /venv/bin/python -m batch.pack --foo" in pbs_body


# ---------------------------------------------------------------------------
# SchedulerExecutor — dry-run submission
# ---------------------------------------------------------------------------


class TestSchedulerExecutor:
    def _executor(self, scheduler, **kw):
        kw.setdefault("dry_run", True)
        return SchedulerExecutor(
            scheduler,
            env_file="/etc/batch/env.zsh",
            default_request=JobRequest(name="", log_path=""),
            python="/venv/bin/python",
            **kw,
        )

    def test_dry_run_writes_script(self, paths):
        job = MockJob(prefix="job_001")
        self._executor(PBSScheduler()).submit(job, paths)
        assert (paths.job / "job_001_run.sh").exists()

    def test_dry_run_returns_dry_run_id(self, paths):
        job = MockJob(prefix="job_001")
        result = self._executor(SlurmScheduler()).submit(job, paths)
        assert result.job_id == "dry-run"
        assert result.returncode is None

    def test_dry_run_does_not_submit(self, paths):
        from unittest.mock import patch

        job = MockJob(prefix="job_001")
        with patch("batch.executors.scheduler.subprocess.run") as mock_run:
            self._executor(PBSScheduler()).submit(job, paths)
        mock_run.assert_not_called()

    def test_name_and_log_filled_from_job(self, paths):
        job = MockJob(prefix="job_001")
        self._executor(PBSScheduler()).submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "#PBS -N job_001" in script
        assert str(paths.log) in script

    def test_per_job_request_override(self, paths):
        job = MockJob(prefix="job_001")
        job.job_request = JobRequest(name="", log_path="", queue="preempt")
        self._executor(PBSScheduler()).submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "#PBS -q preempt" in script

    def test_missing_request_raises(self, paths):
        job = MockJob(prefix="job_001")
        executor = SchedulerExecutor(
            PBSScheduler(), env_file="/e", python="/p", dry_run=True
        )
        with pytest.raises(ValueError, match="No JobRequest"):
            executor.submit(job, paths)

    def test_plot_appends_plotclaw(self, paths):
        job = MockJob(prefix="job_001")
        job.setplot = "/job/setplot.py"
        self._executor(PBSScheduler(), plot=True).submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "clawpack.visclaw.plotclaw" in script
        assert "/job/setplot.py" in script

    def test_no_plot_by_default(self, paths):
        job = MockJob(prefix="job_001")
        self._executor(SlurmScheduler()).submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "plotclaw" not in script

    def test_run_launched_via_absolute_python(self, paths):
        job = MockJob(prefix="job_001")
        self._executor(PBSScheduler()).submit(job, paths)
        script = (paths.job / "job_001_run.sh").read_text()
        assert "exec /venv/bin/python -m clawpack.clawutil.runclaw" in script

    def test_submit_parses_job_id(self, paths):
        from unittest.mock import MagicMock, patch

        job = MockJob(prefix="job_001")
        executor = self._executor(SlurmScheduler(), dry_run=False)
        fake = MagicMock(stdout="9988;cluster\n", returncode=0)
        with patch("batch.executors.scheduler.subprocess.run", return_value=fake):
            result = executor.submit(job, paths)
        assert result.job_id == "9988"
