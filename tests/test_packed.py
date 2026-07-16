"""Tests for batch.packed.

The wrapper renderers are pure functions, so all rendering tests run without a
cluster.  submit_packed is exercised with dry_run=True and by patching
subprocess.run, mirroring the SLURM/PBS executor dry-run tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch.packed import (
    PackedResources,
    render_packed_pbs_wrapper,
    render_packed_slurm_wrapper,
    submit_packed,
)


def inner(shard_i: int, n_shards: int) -> list[str]:
    """A representative per-node inner command."""
    return [
        "python",
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


# ---------------------------------------------------------------------------
# render_packed_pbs_wrapper
# ---------------------------------------------------------------------------


class TestRenderPackedPBSWrapper:
    def _render(self, resources, **kw):
        kw.setdefault("name", "pack_1of4")
        kw.setdefault("log_path", "/scratch/pack_1of4_log.txt")
        return render_packed_pbs_wrapper(1, 4, inner(1, 4), resources, **kw)

    def test_starts_with_shebang(self, resources):
        assert self._render(resources).startswith("#!/bin/bash")

    def test_exclusive_single_node_chunk(self, resources):
        script = self._render(resources)
        assert "#PBS -l select=1:ncpus=128:mpiprocs=1:ompthreads=1" in script

    def test_name_and_log_and_join(self, resources):
        script = self._render(resources)
        assert "#PBS -N pack_1of4" in script
        assert "#PBS -o /scratch/pack_1of4_log.txt" in script
        assert "#PBS -j oe" in script

    def test_queue_and_walltime(self, resources):
        script = self._render(resources)
        assert "#PBS -q main" in script
        assert "#PBS -l walltime=12:00:00" in script

    def test_account_present_when_set(self, resources):
        assert "#PBS -A NCAR0001" in self._render(resources)

    def test_account_absent_when_empty(self, resources):
        resources.account = ""
        assert "#PBS -A" not in self._render(resources)

    def test_modules_loaded(self, resources):
        script = self._render(resources)
        assert "module load ncarenv/23.09" in script
        assert "module load conda" in script

    def test_inner_command_present(self, resources):
        script = self._render(resources)
        assert "python driver.py --scheduler local --shard 1/4 --pin-cpus" in script

    def test_cd_workdir_present_when_given(self, resources):
        script = self._render(resources, workdir="/home/me/proj")
        assert "cd /home/me/proj" in script

    def test_no_cd_when_workdir_none(self, resources):
        assert "\ncd " not in self._render(resources)

    def test_extra_directives_appended(self, resources):
        resources.extra_directives = ["#PBS -l job_priority=premium"]
        assert "#PBS -l job_priority=premium" in self._render(resources)

    def test_ends_with_newline(self, resources):
        assert self._render(resources).endswith("\n")


# ---------------------------------------------------------------------------
# render_packed_slurm_wrapper
# ---------------------------------------------------------------------------


class TestRenderPackedSlurmWrapper:
    def _render(self, resources, **kw):
        kw.setdefault("name", "pack_1of4")
        kw.setdefault("log_path", "/scratch/pack_1of4_log.txt")
        return render_packed_slurm_wrapper(1, 4, inner(1, 4), resources, **kw)

    def test_exclusive_single_node(self, resources):
        script = self._render(resources)
        assert "#SBATCH -N 1" in script
        assert "#SBATCH --exclusive" in script
        assert "#SBATCH --cpus-per-task=128" in script

    def test_partition_and_walltime(self, resources):
        script = self._render(resources)
        assert "#SBATCH -p main" in script
        assert "#SBATCH -t 12:00:00" in script

    def test_account_present_when_set(self, resources):
        assert "#SBATCH -A NCAR0001" in self._render(resources)

    def test_account_absent_when_empty(self, resources):
        resources.account = ""
        assert "#SBATCH -A" not in self._render(resources)

    def test_inner_command_present(self, resources):
        script = self._render(resources)
        assert "python driver.py --scheduler local --shard 1/4 --pin-cpus" in script

    def test_ends_with_newline(self, resources):
        assert self._render(resources).endswith("\n")


# ---------------------------------------------------------------------------
# submit_packed
# ---------------------------------------------------------------------------


class TestSubmitPacked:
    def test_dry_run_writes_n_scripts(self, resources, tmp_path: Path):
        out = submit_packed(
            3, inner, resources, "pbs", tmp_path, dry_run=True, name_prefix="pack"
        )
        assert len(out) == 3
        for i in range(1, 4):
            assert (tmp_path / f"pack_{i}of3_run.sh").exists()
        # Returned values are the script paths under dry_run.
        assert all(str(tmp_path) in p for p in out)

    def test_dry_run_does_not_submit(self, resources, tmp_path: Path):
        from unittest.mock import patch

        with patch("batch.packed.subprocess.run") as mock_run:
            submit_packed(2, inner, resources, "pbs", tmp_path, dry_run=True)
        mock_run.assert_not_called()

    def test_inner_command_receives_shard_indices(self, resources, tmp_path: Path):
        seen = []

        def spy(i, n):
            seen.append((i, n))
            return inner(i, n)

        submit_packed(3, spy, resources, "slurm", tmp_path, dry_run=True)
        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_pbs_calls_qsub_per_node(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="123.desched\n")
            out = submit_packed(2, inner, resources, "pbs", tmp_path)
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].args[0][0] == "qsub"
        assert out == ["123.desched", "123.desched"]

    def test_slurm_calls_sbatch(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="456\n")
            submit_packed(1, inner, resources, "slurm", tmp_path)
        assert mock_run.call_args_list[0].args[0][0] == "sbatch"

    def test_submit_failure_raises_system_exit(self, resources, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        with patch("batch.packed.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="qsub: bad account\n"
            )
            with pytest.raises(SystemExit, match="bad account"):
                submit_packed(3, inner, resources, "pbs", tmp_path)
        # Stops on the first failure rather than firing all three.
        assert mock_run.call_count == 1

    def test_invalid_n_nodes_raises(self, resources, tmp_path: Path):
        with pytest.raises(ValueError):
            submit_packed(0, inner, resources, "pbs", tmp_path, dry_run=True)

    def test_unknown_scheduler_raises(self, resources, tmp_path: Path):
        with pytest.raises(ValueError):
            submit_packed(1, inner, resources, "lsf", tmp_path, dry_run=True)
