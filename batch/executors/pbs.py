"""PBS executor: submit jobs to a PBS Pro scheduler via qsub.

Targets PBS Pro as deployed on NCAR Derecho, but the directives are standard
PBS and work on other PBS Pro sites with site-appropriate ``queue`` / ``account``
values.  Mirrors :mod:`batch.executors.slurm` in structure: a resource dataclass,
a pure script-rendering function, and an executor that submits and polls.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field

from batch.executors.local import _build_run_args
from batch.job import Job, JobPaths, JobResult

logger = logging.getLogger(__name__)


@dataclass
class PBSResources:
    """PBS Pro resource request for a single job.

    Maps directly to ``#PBS`` directives.  Attach an instance to
    ``job.pbs_resources`` to override the executor's defaults on a per-job
    basis.

    Parameters
    ----------
    queue:
        PBS queue name (``-q``).  On Derecho, e.g. ``"main"``, ``"develop"``,
        or ``"preempt"``.
    nodes:
        Number of node chunks to request (the ``select=`` count).
    ncpus:
        CPUs (cores) per chunk.  Derecho CPU nodes have 128 cores; request the
        whole node for a pure-OpenMP run.
    mpiprocs:
        MPI ranks per chunk.  For pure-OpenMP GeoClaw runs this should be 1.
    ompthreads:
        OpenMP threads per chunk.  Set this equal to ``OMP_NUM_THREADS``.
    walltime:
        Walltime limit in ``HH:MM:SS`` format (``-l walltime=``).
    account:
        Allocation project code (``-A``).  Required on Derecho; the directive
        is omitted when this is empty.
    mem:
        Memory per chunk, e.g. ``"235GB"``.  Empty string uses the queue
        default; when set it is appended to the ``select`` chunk as ``:mem=``.
    modules:
        List of module names to load (``module load <name>``).
    env_vars:
        Environment variables to export in the job script.  The canonical use
        is ``{"OMP_NUM_THREADS": "128"}``.
    email:
        Email address for job notifications (``-M``).  Empty string disables mail.
    mail_points:
        PBS mail event points (``-m``).  Default ``"abe"`` (abort/begin/end).
    plot:
        When True, append a ``plotclaw`` invocation after the solver so the job
        produces its VisClaw frame plots on the compute node.  Requires a
        setplot path (see ``setplot``).
    setplot:
        Path to the setplot file used by the appended ``plotclaw`` call.  When
        empty, falls back to ``job.setplot``.
    extra_directives:
        Raw ``#PBS`` lines appended after the standard directives.  Use for
        anything not covered above.
    """

    queue: str = "main"
    nodes: int = 1
    ncpus: int = 128
    mpiprocs: int = 1
    ompthreads: int = 128
    walltime: str = "12:00:00"
    account: str = ""
    mem: str = ""
    modules: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    email: str = ""
    mail_points: str = "abe"
    plot: bool = False
    setplot: str = ""
    extra_directives: list[str] = field(default_factory=list)


def render_pbs_script(
    job: Job,
    paths: JobPaths,
    resources: PBSResources,
) -> str:
    """Generate a self-contained qsub script for one job.

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
        PBS resource requests.

    Returns
    -------
    str
        Complete bash script text, ready to write to a ``.sh`` file.
    """
    run_cmd = " ".join(str(a) for a in _build_run_args(job, paths))

    # select chunk: nodes:ncpus=..:mpiprocs=..:ompthreads=..[:mem=..]
    chunk = (
        f"{resources.nodes}:ncpus={resources.ncpus}"
        f":mpiprocs={resources.mpiprocs}:ompthreads={resources.ompthreads}"
    )
    if resources.mem:
        chunk += f":mem={resources.mem}"

    # Standard directives — always present
    directives = [
        f"#PBS -N {job.prefix}",
        f"#PBS -o {paths.log}",
        "#PBS -j oe",  # join stdout and stderr into the -o file
        f"#PBS -q {resources.queue}",
        f"#PBS -l select={chunk}",
        f"#PBS -l walltime={resources.walltime}",
    ]

    # Optional directives
    if resources.account:
        directives.append(f"#PBS -A {resources.account}")
    if resources.email:
        directives.append(f"#PBS -M {resources.email}")
        directives.append(f"#PBS -m {resources.mail_points}")
    directives.extend(resources.extra_directives)

    lines: list[str] = ["#!/bin/bash"] + directives + [""]

    if resources.modules:
        lines.extend(f"module load {m}" for m in resources.modules)
        lines.append("")

    if resources.env_vars:
        lines.extend(f"export {k}={v}" for k, v in resources.env_vars.items())
        lines.append("")

    lines.append(run_cmd)

    # Optional compute-node plotting: run plotclaw after the solver so each job
    # produces its VisClaw frames without a login-node process (mirrors the
    # invocation in batch.plot.plot_job).
    if resources.plot:
        setplot = resources.setplot or str(job.setplot)
        plot_cmd = " ".join(
            str(a)
            for a in [
                sys.executable,
                "-m",
                "clawpack.visclaw.plotclaw",
                str(paths.job),
                str(paths.plots),
                setplot,
            ]
        )
        lines.append(plot_cmd)

    lines.append("")  # ensure trailing newline

    return "\n".join(lines)


class PBSExecutor:
    """Submit jobs to PBS Pro via ``qsub``.

    ``submit`` returns immediately after queuing; ``wait_all`` polls ``qstat``
    until all submitted jobs leave the queue.

    Per-job resource overrides are supported by attaching a
    :class:`PBSResources` instance as ``job.pbs_resources``.  Jobs without that
    attribute use ``default_resources``.

    Parameters
    ----------
    default_resources:
        Resource defaults applied to every job that does not carry its own
        ``pbs_resources`` attribute.
    dry_run:
        If True, write the submission script but do not call ``qsub``.
        Useful for inspecting what would be submitted.
    poll_interval:
        Seconds between ``qstat`` polls in ``wait_all``.  Default 30.0.
    """

    def __init__(
        self,
        default_resources: PBSResources | None = None,
        dry_run: bool = False,
        poll_interval: float = 30.0,
    ) -> None:
        self.default_resources = default_resources or PBSResources()
        self.dry_run = dry_run
        self.poll_interval = poll_interval

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        resources: PBSResources = getattr(
            job, "pbs_resources", self.default_resources
        )
        script = render_pbs_script(job, paths, resources)

        script_path = paths.job / f"{job.prefix}_run.sh"
        script_path.write_text(script)
        logger.debug("Wrote submission script: %s", script_path)

        if self.dry_run:
            logger.info("[dry-run] Would submit: %s", script_path)
            return JobResult(job=job, paths=paths, returncode=None, job_id="dry-run")

        proc = subprocess.run(
            ["qsub", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # qsub prints the job ID, e.g. "1234567.desched1"
        job_id = proc.stdout.strip()
        logger.info("Submitted job %s → PBS job ID %s", job.prefix, job_id)
        return JobResult(job=job, paths=paths, returncode=None, job_id=job_id)

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """Poll qstat until all submitted jobs leave the queue."""
        pending = {r.job_id: r for r in results if r.job_id and r.job_id != "dry-run"}
        while pending:
            time.sleep(self.poll_interval)
            completed = []
            for job_id in list(pending):
                proc = subprocess.run(
                    ["qstat", job_id],
                    capture_output=True,
                    text=True,
                )
                # A finished/purged job makes qstat return non-zero ("Unknown
                # Job Id") with empty stdout.  Key on either signal rather than
                # on returncode alone, since transient states differ by site.
                if proc.returncode != 0 or not proc.stdout.strip():
                    pending[job_id].returncode = 0
                    try:
                        pending[job_id].job.post_run(pending[job_id])
                    except Exception:
                        logger.exception(
                            "post_run failed for job %s",
                            pending[job_id].job.prefix,
                        )
                    logger.info(
                        "Job %s (PBS %s) left the queue",
                        pending[job_id].job.prefix,
                        job_id,
                    )
                    completed.append(job_id)
            for job_id in completed:
                del pending[job_id]
            if pending:
                logger.info("%d job(s) still in queue", len(pending))
        return results
