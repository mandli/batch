"""SLURM executor: submit jobs to a SLURM scheduler via sbatch."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field

from batch.executors.local import _build_run_args
from batch.job import Job, JobPaths, JobResult

logger = logging.getLogger(__name__)


@dataclass
class SLURMResources:
    """SLURM resource request for a single job.

    Maps directly to ``#SBATCH`` directives.  Attach an instance to
    ``job.slurm_resources`` to override the executor's defaults on a per-job
    basis.

    Parameters
    ----------
    partition:
        SLURM partition (queue) name.
    nodes:
        Number of nodes to request.
    ntasks_per_node:
        MPI tasks per node.  For pure-OpenMP GeoClaw runs this should be 1.
    cpus_per_task:
        CPUs (hardware threads) per task.  Set this to ``OMP_NUM_THREADS``.
    time:
        Walltime limit in ``HH:MM:SS`` format.
    memory:
        Memory per node, e.g. ``"4G"``.  Empty string uses the partition
        default.
    account:
        Allocation account (``-A``).  Required on most HPC allocations.
    constraint:
        Node feature constraint, e.g. ``"cpu"`` on Derecho or ``"knl"`` on
        older Stampede partitions.
    modules:
        List of module names to load (``module load <name>``).
    env_vars:
        Environment variables to export in the job script.  The canonical use
        is ``{"OMP_NUM_THREADS": "8"}``.
    email:
        Email address for job notifications.  Empty string disables mail.
    mail_type:
        Comma-separated SLURM mail event types.  Default ``"END,FAIL"``.
    extra_directives:
        Raw ``#SBATCH`` lines appended after the standard directives.  Use for
        anything not covered above (GRES, licenses, heterogeneous jobs, etc.).
    """

    partition: str = "main"
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    time: str = "01:00:00"
    memory: str = ""
    account: str = ""
    constraint: str = ""
    modules: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    email: str = ""
    mail_type: str = "END,FAIL"
    extra_directives: list[str] = field(default_factory=list)


def render_slurm_script(
    job: Job,
    paths: JobPaths,
    resources: SLURMResources,
) -> str:
    """Generate a self-contained sbatch script for one job.

    This is a pure function — it does not touch the filesystem or call any
    external processes, which makes it straightforward to unit-test without
    a cluster.

    Parameters
    ----------
    job:
        The job being submitted.
    paths:
        Pre-computed filesystem layout.
    resources:
        SLURM resource requests.

    Returns
    -------
    str
        Complete bash script text, ready to write to a ``.sh`` file.
    """
    run_cmd = " ".join(str(a) for a in _build_run_args(job, paths))

    # Standard directives — always present
    directives = [
        f"#SBATCH -J {job.prefix}",
        f"#SBATCH -o {paths.log}",
        f"#SBATCH -e {paths.log}",
        f"#SBATCH -p {resources.partition}",
        f"#SBATCH -N {resources.nodes}",
        f"#SBATCH --ntasks-per-node={resources.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH -t {resources.time}",
    ]

    # Optional directives
    if resources.memory:
        directives.append(f"#SBATCH --mem={resources.memory}")
    if resources.account:
        directives.append(f"#SBATCH -A {resources.account}")
    if resources.constraint:
        directives.append(f"#SBATCH --constraint={resources.constraint}")
    if resources.email:
        directives.append(f"#SBATCH --mail-user={resources.email}")
        directives.append(f"#SBATCH --mail-type={resources.mail_type}")
    directives.extend(resources.extra_directives)

    lines: list[str] = ["#!/bin/bash"] + directives + [""]

    if resources.modules:
        lines.extend(f"module load {m}" for m in resources.modules)
        lines.append("")

    if resources.env_vars:
        lines.extend(f"export {k}={v}" for k, v in resources.env_vars.items())
        lines.append("")

    lines.append(run_cmd)
    lines.append("")  # ensure trailing newline

    return "\n".join(lines)


class SLURMExecutor:
    """Submit jobs to SLURM via ``sbatch``.

    ``submit`` returns immediately after queuing; ``wait_all`` polls
    ``squeue`` until all submitted jobs leave the queue.

    Per-job resource overrides are supported by attaching a
    :class:`SLURMResources` instance as ``job.slurm_resources``.  Jobs
    without that attribute use ``default_resources``.

    Parameters
    ----------
    default_resources:
        Resource defaults applied to every job that does not carry its own
        ``slurm_resources`` attribute.
    dry_run:
        If True, write the submission script but do not call ``sbatch``.
        Useful for inspecting what would be submitted.
    poll_interval:
        Seconds between ``squeue`` polls in ``wait_all``.  Default 30.0.
    """

    def __init__(
        self,
        default_resources: SLURMResources | None = None,
        dry_run: bool = False,
        poll_interval: float = 30.0,
    ) -> None:
        self.default_resources = default_resources or SLURMResources()
        self.dry_run = dry_run
        self.poll_interval = poll_interval

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        resources: SLURMResources = getattr(
            job, "slurm_resources", self.default_resources
        )
        script = render_slurm_script(job, paths, resources)

        script_path = paths.job / f"{job.prefix}_run.sh"
        script_path.write_text(script)
        logger.debug("Wrote submission script: %s", script_path)

        if self.dry_run:
            logger.info("[dry-run] Would submit: %s", script_path)
            return JobResult(job=job, paths=paths, returncode=None, job_id="dry-run")

        proc = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # --parsable output: "<jobid>" or "<jobid>;<cluster>"
        job_id = proc.stdout.strip().split(";")[0]
        logger.info("Submitted job %s → SLURM job ID %s", job.prefix, job_id)
        return JobResult(job=job, paths=paths, returncode=None, job_id=job_id)

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """Poll squeue until all submitted jobs leave the queue."""
        pending = {r.job_id: r for r in results if r.job_id and r.job_id != "dry-run"}
        while pending:
            time.sleep(self.poll_interval)
            completed = []
            for job_id in list(pending):
                proc = subprocess.run(
                    ["squeue", "--job", job_id, "--noheader"],
                    capture_output=True,
                    text=True,
                )
                if not proc.stdout.strip():
                    # Job no longer in queue — finished (success or failure).
                    # squeue exit code is non-zero for unknown job IDs on some
                    # clusters so we key on empty stdout rather than returncode.
                    pending[job_id].returncode = 0
                    logger.info(
                        "Job %s (SLURM %s) left the queue",
                        pending[job_id].job.prefix,
                        job_id,
                    )
                    completed.append(job_id)
            for job_id in completed:
                del pending[job_id]
            if pending:
                logger.info("%d job(s) still in queue", len(pending))
        return results
