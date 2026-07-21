"""Tests for batch.packed.

Packed wrappers are now rendered through the shared, scheduler-agnostic
:func:`batch.executors.scheduler.render_job_script` with an injected
:class:`~batch.scheduler.Scheduler`, so there is no per-scheduler renderer to
test in isolation.  Rendering is exercised via ``submit_packed(dry_run=True)``
(which writes the wrappers) and submission via a patched ``subprocess.run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.packed import PackedResources, submit_packed

ENV_FILE = "/etc/batch/env.zsh"
PYTHON = "/venv/bin/python"


def inner(shard_i: int, n_shards: int) -> list[str]:
    """A representative per-node inner command."""
    return [
        PYTHON,
        "driver.py",
        "--scheduler",
        "local",
        "--shard",
        f"{shard_i}/{n_shards}",
        "--pin-cpus",
    ]


@pytest.fixture
def resources() -> PackedResources:
    return PackedResources(
        queue="main",
        walltime="12:00:00",
        account="NCAR0001",
        node_cpus=128,
        modules=["ncarenv/23.09", "conda"],
    )


def _submit(resources, scheduler, tmp_path, **kw):
    kw.setdefault("env_file", ENV_FILE)
    kw.setdefault("python", PYTHON)
    kw.setdefault("dry_run", True)
    return submit_packed(kw.pop("n", 1), inner, resources, scheduler, tmp_path, **kw)


def _script(resources, scheduler, tmp_path, **kw):
    _submit(resources, scheduler, tmp_path, n=1, **kw)
    return (tmp_path / "pack_1of1_run.sh").read_text()


# ---------------------------------------------------------------------------
# Rendered wrapper content — exclusive node, env_file body, inner command
# ---------------------------------------------------------------------------


class TestPBSWrapperContent:
    def test_exclusive_single_node_chunk(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert "#PBS -l select=1:ncpus=128:mpiprocs=1:ompthreads=1" in script

    def test_name_log_and_join(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert "#PBS -N pack_1of1" in script
        assert "#PBS -o " in script and "pack_1of1_log.txt" in script
        assert "#PBS -j oe" in script

    def test_queue_walltime_account(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert "#PBS -q main" in script
        assert "#PBS -l walltime=12:00:00" in script
        assert "#PBS -A NCAR0001" in script

    def test_account_absent_when_empty(self, resources, tmp_path):
        resources.account = ""
        assert "#PBS -A" not in _script(resources, "pbs", tmp_path)

    def test_sources_env_file_and_normalizes(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert f"source {ENV_FILE}" in script
        assert 'export BATCH_NODEFILE="$PBS_NODEFILE"' in script

    def test_modules_loaded(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert "module load ncarenv/23.09" in script
        assert "module load conda" in script

    def test_inner_command_execd(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path)
        assert (
            f"exec {PYTHON} driver.py --scheduler local --shard 1/1 --pin-cpus"
            in script
        )

    def test_cd_workdir(self, resources, tmp_path):
        script = _script(resources, "pbs", tmp_path, workdir="/home/me/proj")
        assert "cd /home/me/proj" in script

    def test_extra_directives_appended(self, resources, tmp_path):
        resources.extra_directives = ["#PBS -l job_priority=premium"]
        assert "#PBS -l job_priority=premium" in _script(resources, "pbs", tmp_path)


class TestSlurmWrapperContent:
    def test_exclusive_single_node(self, resources, tmp_path):
        script = _script(resources, "slurm", tmp_path)
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --exclusive" in script
        assert "#SBATCH --cpus-per-task=128" in script

    def test_partition_walltime_account(self, resources, tmp_path):
        script = _script(resources, "slurm", tmp_path)
        assert "#SBATCH --partition=main" in script
        assert "#SBATCH --time=12:00:00" in script
        assert "#SBATCH --account=NCAR0001" in script

    def test_slurm_nodefile_materialized(self, resources, tmp_path):
        script = _script(resources, "slurm", tmp_path)
        assert 'export BATCH_NODEFILE="$(mktemp)"' in script
        assert "scontrol show hostnames" in script

    def test_inner_command_execd(self, resources, tmp_path):
        script = _script(resources, "slurm", tmp_path)
        assert f"exec {PYTHON} driver.py --scheduler local" in script


# ---------------------------------------------------------------------------
# submit_packed behavior
# ---------------------------------------------------------------------------


class TestSubmitPacked:
    def test_dry_run_writes_n_scripts(self, resources, tmp_path: Path):
        out = _submit(resources, "pbs", tmp_path, n=3, name_prefix="pack")
        assert len(out) == 3
        for i in range(1, 4):
            assert (tmp_path / f"pack_{i}of3_run.sh").exists()
        assert all(str(tmp_path) in p for p in out)

    def test_dry_run_does_not_submit(self, resources, tmp_path: Path):
        from unittest.mock import patch

        with patch("batch.packed.subprocess.run") as mock_run:
            _submit(resources, "pbs", tmp_path, n=2)
        mock_run.assert_not_called()

    def test_inner_command_receives_shard_indices(self, resources, tmp_path: Path):
        seen = []

        def spy(i, n):
            seen.append((i, n))
            return inner(i, n)

        submit_packed(
            3,
            spy,
            resources,
            "slurm",
            tmp_path,
            env_file=ENV_FILE,
            python=PYTHON,
            dry_run=True,
        )
        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_pbs_calls_qsub_per_node(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="123.desched\n")
            out = _submit(resources, "pbs", tmp_path, n=2, dry_run=False)
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].args[0][0] == "qsub"
        assert out == ["123.desched", "123.desched"]

    def test_slurm_calls_sbatch_parsable(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="456;cluster\n")
            out = _submit(resources, "slurm", tmp_path, n=1, dry_run=False)
        argv = mock_run.call_args_list[0].args[0]
        assert argv[:2] == ["sbatch", "--parsable"]
        assert out == ["456"]  # cluster suffix stripped

    def test_submit_failure_raises_system_exit(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="qsub: bad account\n"
            )
            with pytest.raises(SystemExit, match="bad account"):
                _submit(resources, "pbs", tmp_path, n=3, dry_run=False)
        # Stops on the first failure rather than firing all three.
        assert mock_run.call_count == 1

    def test_invalid_n_nodes_raises(self, resources, tmp_path: Path):
        with pytest.raises(ValueError):
            _submit(resources, "pbs", tmp_path, n=0)

    def test_unknown_scheduler_raises(self, resources, tmp_path: Path):
        with pytest.raises(ValueError):
            _submit(resources, "lsf", tmp_path, n=1)
