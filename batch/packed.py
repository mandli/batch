"""Packed submission: fan a sweep across several exclusive nodes.

A single GeoClaw run rarely saturates a modern HPC node, so instead of one
scheduler job per run (see :mod:`batch.executors.pbs` /
:mod:`batch.executors.slurm`) you can *pack* many runs onto each of ``n`` nodes.
Each node requests one exclusive node and, on the compute node, runs a disjoint
shard of the job list through the local :class:`~batch.executors.local.ParallelExecutor`
with ``cpu_affinity=True`` (see :func:`batch.sweep.shard_jobs`).

Because packing re-runs the *caller's* driver script on the compute node — the
job-definition code has to be present there to build/plot each run — the inner
per-node command is supplied by the caller rather than owned here.  This module
provides the reusable outer layer: rendering the per-node wrapper script and
submitting the wrappers with ``qsub`` / ``sbatch``.

The renderers are pure functions (no filesystem, no subprocess), so they are
unit-testable without a cluster, mirroring ``render_pbs_script`` /
``render_slurm_script``.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Scheduler = Literal["pbs", "slurm"]

# (shard_i, n_shards) -> argv list for the per-node re-invocation of the driver.
InnerCommand = Callable[[int, int], Sequence[str]]


@dataclass
class PackedResources:
    """Resource request for one exclusive node in a packed submission.

    Scheduler-agnostic: the same object renders to ``#PBS`` or ``#SBATCH``
    directives depending on the ``scheduler`` argument to :func:`submit_packed`.

    Parameters
    ----------
    queue:
        Queue / partition name (PBS ``-q`` or SLURM ``-p``).
    walltime:
        Walltime limit in ``HH:MM:SS`` format.
    account:
        Allocation project code (``-A``).  The directive is omitted when empty.
    node_cpus:
        Cores to request on the (exclusive) node.  On Derecho CPU nodes, 128.
    modules:
        Module names to ``module load`` in the wrapper before the inner command.
    extra_directives:
        Raw ``#PBS`` / ``#SBATCH`` lines appended after the standard directives.
    """

    queue: str = "main"
    walltime: str = "12:00:00"
    account: str = ""
    node_cpus: int = 128
    modules: list[str] = field(default_factory=list)
    extra_directives: list[str] = field(default_factory=list)


def _wrapper_body(
    resources: PackedResources,
    inner: Sequence[str],
    workdir: Path | str | None,
) -> list[str]:
    """Shared script body: module loads, optional cd, and the inner command."""
    lines: list[str] = [""]
    if resources.modules:
        lines.extend(f"module load {m}" for m in resources.modules)
        lines.append("")
    if workdir is not None:
        lines.append(f"cd {workdir}")
    lines.append(" ".join(str(a) for a in inner))
    lines.append("")  # trailing newline
    return lines


def render_packed_pbs_wrapper(
    shard_i: int,
    n_shards: int,
    inner_command: Sequence[str],
    resources: PackedResources,
    *,
    name: str,
    log_path: Path | str,
    workdir: Path | str | None = None,
) -> str:
    """Render a PBS wrapper that packs one shard onto one exclusive node.

    The wrapper requests a single node with ``node_cpus`` cores
    (``mpiprocs=1:ompthreads=1`` — the packing is done by the local pool inside
    the inner command, not by MPI) and runs *inner_command* on the compute node.

    Parameters
    ----------
    shard_i, n_shards:
        This node's 1-based shard index and the total shard count (used only for
        the default job name / logging; the caller's *inner_command* is
        responsible for actually selecting the shard).
    inner_command:
        The per-node command to run on the compute node, as an argv sequence.
    resources:
        Node resource request.
    name:
        PBS job name (``-N``).
    log_path:
        Joined stdout/stderr log path (``-o`` with ``-j oe``).
    workdir:
        Directory to ``cd`` into before running the inner command.  When None,
        no ``cd`` is emitted (the scheduler's default working directory is used).

    Returns
    -------
    str
        Complete wrapper script text.
    """
    chunk = f"1:ncpus={resources.node_cpus}:mpiprocs=1:ompthreads=1"
    directives = [
        f"#PBS -N {name}",
        f"#PBS -o {log_path}",
        "#PBS -j oe",
        f"#PBS -q {resources.queue}",
        f"#PBS -l select={chunk}",
        f"#PBS -l walltime={resources.walltime}",
    ]
    if resources.account:
        directives.append(f"#PBS -A {resources.account}")
    directives.extend(resources.extra_directives)

    lines = ["#!/bin/bash"] + directives
    lines.extend(_wrapper_body(resources, inner_command, workdir))
    return "\n".join(lines)


def render_packed_slurm_wrapper(
    shard_i: int,
    n_shards: int,
    inner_command: Sequence[str],
    resources: PackedResources,
    *,
    name: str,
    log_path: Path | str,
    workdir: Path | str | None = None,
) -> str:
    """Render a SLURM wrapper that packs one shard onto one exclusive node.

    The SLURM analogue of :func:`render_packed_pbs_wrapper`: requests one
    exclusive node (``-N 1 --exclusive``) and runs *inner_command* on it.

    Parameters
    ----------
    shard_i, n_shards, inner_command, resources, name, log_path, workdir:
        See :func:`render_packed_pbs_wrapper`.

    Returns
    -------
    str
        Complete wrapper script text.
    """
    directives = [
        f"#SBATCH -J {name}",
        f"#SBATCH -o {log_path}",
        f"#SBATCH -e {log_path}",
        f"#SBATCH -p {resources.queue}",
        "#SBATCH -N 1",
        "#SBATCH --exclusive",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={resources.node_cpus}",
        f"#SBATCH -t {resources.walltime}",
    ]
    if resources.account:
        directives.append(f"#SBATCH -A {resources.account}")
    directives.extend(resources.extra_directives)

    lines = ["#!/bin/bash"] + directives
    lines.extend(_wrapper_body(resources, inner_command, workdir))
    return "\n".join(lines)


_RENDERERS = {
    "pbs": render_packed_pbs_wrapper,
    "slurm": render_packed_slurm_wrapper,
}
_SUBMIT_CMD = {"pbs": "qsub", "slurm": "sbatch"}


def submit_packed(
    n_nodes: int,
    inner_command: InnerCommand,
    resources: PackedResources,
    scheduler: Scheduler,
    script_dir: Path | str,
    *,
    dry_run: bool = False,
    name_prefix: str = "pack",
    workdir: Path | str | None = None,
) -> list[str]:
    """Render one wrapper per node and (unless *dry_run*) submit them.

    Submit-and-exit: each node's wrapper self-packs its shard via the caller's
    local-mode re-invocation.  Nothing is waited on here — poll the queue
    yourself, and run any cross-run post-processing once the jobs finish.

    Parameters
    ----------
    n_nodes:
        Number of exclusive nodes to fan the sweep over (one submission each).
    inner_command:
        Callable ``(shard_i, n_shards) -> argv`` returning the per-node command
        to run on the compute node.  Typically re-invokes the driver script in
        ``--scheduler local --shard i/n --pin-cpus`` mode.
    resources:
        Per-node resource request.
    scheduler:
        ``"pbs"`` or ``"slurm"``.
    script_dir:
        Directory to write the generated wrapper scripts into (created if
        missing).
    dry_run:
        When True, write the wrapper scripts but do not call ``qsub`` / ``sbatch``.
    name_prefix:
        Prefix for job names and script filenames (``<prefix>_<i>of<n>``).
    workdir:
        Directory each wrapper ``cd``s into before the inner command.

    Returns
    -------
    list[str]
        Submitted scheduler job IDs, or — under *dry_run* — the paths of the
        wrapper scripts written.

    Raises
    ------
    ValueError
        If ``n_nodes < 1`` or *scheduler* is unknown.
    SystemExit
        If ``qsub`` / ``sbatch`` rejects a submission (the message is surfaced
        and remaining nodes are not fired at the same wall).
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1; got {n_nodes}")
    if scheduler not in _RENDERERS:
        raise ValueError(
            f"scheduler must be one of {sorted(_RENDERERS)}; got {scheduler!r}"
        )

    render = _RENDERERS[scheduler]
    submit_cmd = _SUBMIT_CMD[scheduler]
    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    for i in range(1, n_nodes + 1):
        name = f"{name_prefix}_{i}of{n_nodes}"
        script_path = script_dir / f"{name}_run.sh"
        script = render(
            i,
            n_nodes,
            inner_command(i, n_nodes),
            resources,
            name=name,
            log_path=script_dir / f"{name}_log.txt",
            workdir=workdir,
        )
        script_path.write_text(script)
        logger.debug("Wrote packed wrapper: %s", script_path)

        if dry_run:
            logger.info("[dry-run] Would submit: %s", script_path)
            out.append(str(script_path))
            continue

        proc = subprocess.run(
            [submit_cmd, str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Surface the scheduler's own rejection (account/queue/walltime, etc.)
            # and stop rather than firing the remaining nodes at the same wall.
            msg = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"({submit_cmd} produced no output)"
            )
            raise SystemExit(
                f"{submit_cmd} failed for {script_path} "
                f"(exit {proc.returncode}):\n{msg}"
            )
        job_id = proc.stdout.strip()
        logger.info("Submitted packed shard %d/%d -> job %s", i, n_nodes, job_id)
        out.append(job_id)

    return out
