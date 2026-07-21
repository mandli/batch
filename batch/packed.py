"""Packed submission: fan a sweep across several exclusive nodes.

A single GeoClaw run rarely saturates a modern HPC node, so instead of one
scheduler job per run (see :class:`~batch.executors.scheduler.SchedulerExecutor`)
you can *pack* many runs onto each of ``n`` nodes.  Each node requests one
exclusive node and, on the compute node, runs a disjoint shard of the job list
through the local :class:`~batch.executors.local.ParallelExecutor` with
``cpu_affinity=True`` (see :func:`batch.sweep.shard_jobs`).

Because packing re-runs the *caller's* driver script on the compute node — the
job-definition code has to be present there to build/plot each run — the inner
per-node command is supplied by the caller rather than owned here.  This module
provides the reusable outer layer: it renders the same scheduler-agnostic
wrapper that :class:`~batch.executors.scheduler.SchedulerExecutor` uses (header
from the injected :class:`~batch.scheduler.Scheduler`, ``env_file`` sourcing, and
the ``BATCH_*`` normalization), then submits the wrappers.  All PBS/SLURM
difference lives in the :class:`~batch.scheduler.Scheduler`, so there is no
per-scheduler code here.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from batch.executors.scheduler import render_job_script
from batch.scheduler import JobRequest, get_scheduler

logger = logging.getLogger(__name__)

SchedulerName = Literal["pbs", "slurm"]

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
        Queue / partition name.
    walltime:
        Walltime limit in ``HH:MM:SS`` format.
    account:
        Allocation project code.  Omitted from the directives when empty.
    node_cpus:
        Cores to request on the (exclusive) node.  On Derecho CPU nodes, 128.
    modules:
        Module names to ``module load`` in the wrapper (after the env_file is
        sourced) before the inner command.
    extra_directives:
        Raw ``#PBS`` / ``#SBATCH`` lines appended after the standard directives.
    """

    queue: str = "main"
    walltime: str = "12:00:00"
    account: str = ""
    node_cpus: int = 128
    modules: list[str] = field(default_factory=list)
    extra_directives: list[str] = field(default_factory=list)

    def to_request(self, name: str, log_path: str) -> JobRequest:
        """Normalize into a full-exclusive-node :class:`JobRequest`."""
        return JobRequest(
            name=name,
            log_path=log_path,
            queue=self.queue,
            account=self.account,
            walltime=self.walltime,
            nodes=1,
            cpus_per_node=self.node_cpus,
            tasks_per_node=1,
            ompthreads=1,  # PBS hint only; the local pool sets OMP per slot
            exclusive=True,
            extra_directives=self.extra_directives,
        )


def submit_packed(
    n_nodes: int,
    inner_command: InnerCommand,
    resources: PackedResources,
    scheduler: SchedulerName,
    script_dir: Path | str,
    *,
    env_file: str | Path,
    python: str | Path,
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
        the wrapper ``exec``s on the compute node.  Typically re-invokes the
        driver script in ``--scheduler local --shard i/n --pin-cpus`` mode.
    resources:
        Per-node resource request.
    scheduler:
        ``"pbs"`` or ``"slurm"``.
    script_dir:
        Directory to write the generated wrapper scripts into (created if
        missing).
    env_file:
        Per-machine env file each wrapper sources on the compute node.
    python:
        Absolute venv python used for the wrapper's ``import batch`` check.
    dry_run:
        When True, write the wrapper scripts but do not submit them.
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

    backend = get_scheduler(scheduler)  # raises ValueError on an unknown name
    # Expand ~ before the template quotes these (see SchedulerExecutor).
    env_file = str(Path(env_file).expanduser())
    python = str(Path(python).expanduser())
    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    for i in range(1, n_nodes + 1):
        name = f"{name_prefix}_{i}of{n_nodes}"
        script_path = script_dir / f"{name}_run.sh"
        request = resources.to_request(
            name=name, log_path=str(script_dir / f"{name}_log.txt")
        )
        script = render_job_script(
            backend,
            request,
            list(inner_command(i, n_nodes)),
            env_file=env_file,
            python=python,
            workdir=workdir if workdir is not None else script_dir,
            modules=resources.modules,
        )
        script_path.write_text(script)
        logger.debug("Wrote packed wrapper: %s", script_path)

        if dry_run:
            logger.info("[dry-run] Would submit: %s", script_path)
            out.append(str(script_path))
            continue

        proc = subprocess.run(
            backend.submit_argv(str(script_path)),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Surface the scheduler's own rejection (account/queue/walltime, etc.)
            # and stop rather than firing the remaining nodes at the same wall.
            submit_cmd = backend.submit_argv(str(script_path))[0]
            msg = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"({submit_cmd} produced no output)"
            )
            raise SystemExit(
                f"{submit_cmd} failed for {script_path} "
                f"(exit {proc.returncode}):\n{msg}"
            )
        job_id = backend.parse_job_id(proc.stdout)
        logger.info("Submitted packed shard %d/%d -> job %s", i, n_nodes, job_id)
        out.append(job_id)

    return out
