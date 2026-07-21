"""Shared command-line surface for batch driver scripts.

Every HPC driver ends up re-implementing the same scheduler/resource argparse
block, the same ``--resume`` → :class:`~batch.job.ClobberPolicy` mapping, and the
same success/failure report.  This module factors that out **without** owning
your ``main()``: you keep your own :class:`argparse.ArgumentParser`, add your
domain flags to it, and build the job list yourself.  The pieces are layered so
you can use as much or as little as you want:

- :func:`add_execution_args` — add the shared scheduler/resource flag group to
  your parser.
- :func:`executor_from_args` — build the right executor from the parsed args.
- :func:`clobber_from_args` — ``--resume`` → ``ClobberPolicy``.
- :func:`report_results` — print a run summary and return a
  :class:`ResultSummary`.
- :func:`execute` — the thin dispatch that ties the above together and handles
  local / scheduler / packed submission in one call.

Drop to the factories whenever you want the scheduler switch explicit in your
own code; reach for :func:`execute` when you just want it handled.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from batch.controller import BatchController
from batch.executors import Executor
from batch.executors.local import ParallelExecutor
from batch.executors.scheduler import SchedulerExecutor
from batch.job import ClobberPolicy, Job, JobResult
from batch.packed import PackedResources, submit_packed
from batch.scheduler import JobRequest, get_scheduler
from batch.sweep import parse_shard_spec, shard_jobs

logger = logging.getLogger(__name__)

SCHEDULERS = ["local", "pbs", "slurm", "pbs-packed", "slurm-packed"]


def add_execution_args(parser: argparse.ArgumentParser, *, packed: bool = True) -> None:
    """Add the shared execution/scheduler flag group to *parser*.

    The parser stays yours — add your project-specific flags to it before or
    after calling this.  The added flags are consumed by
    :func:`executor_from_args`, :func:`clobber_from_args`, and :func:`execute`.

    Parameters
    ----------
    parser:
        The argument parser to extend.
    packed:
        When True (default), also add the packing flags (``--nodes``,
        ``--node-cpus``, ``--shard``, ``--pin-cpus``).  Set False for a driver
        that never packs.

    Flags added
    -----------
    ``--scheduler`` (local/pbs/slurm/pbs-packed/slurm-packed), ``--setup-only``,
    ``--resume``, ``--max-workers`` (``$BATCH_MAX_JOBS`` or 4),
    ``--omp-num-threads`` (``$OMP_NUM_THREADS`` or 1), ``--account``
    (``$BATCH_ACCOUNT``), ``--queue``, ``--walltime``, ``--modules``
    (``$BATCH_MODULES``), and — with ``packed=True`` — ``--nodes``,
    ``--node-cpus``, ``--shard`` (``I/N``), ``--pin-cpus``.
    """
    g = parser.add_argument_group("execution")
    g.add_argument(
        "--scheduler",
        choices=SCHEDULERS,
        default="local",
        help="Execution backend (default: local).",
    )
    g.add_argument(
        "--setup-only",
        action="store_true",
        help="Write .data files (local) or submission scripts (scheduler) "
        "without running or submitting anything.",
    )
    g.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output directory already exists.",
    )
    g.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("BATCH_MAX_JOBS", 4)),
        help="Local only: max concurrent jobs (default: $BATCH_MAX_JOBS or 4).",
    )
    g.add_argument(
        "--omp-num-threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", 1)),
        help="OpenMP threads per job (default: $OMP_NUM_THREADS or 1). On a "
        "scheduler this also sets the per-job core request.",
    )
    g.add_argument(
        "--account",
        default=os.environ.get("BATCH_ACCOUNT", ""),
        help="Scheduler allocation/project code, PBS or SLURM alike "
        "(default: $BATCH_ACCOUNT).",
    )
    g.add_argument(
        "--queue",
        default="main",
        help="Scheduler queue / partition name (default: main).",
    )
    g.add_argument(
        "--walltime",
        default="12:00:00",
        help="Scheduler walltime limit HH:MM:SS (default: 12:00:00).",
    )
    g.add_argument(
        "--modules",
        nargs="*",
        default=os.environ.get("BATCH_MODULES", "").split(),
        help="Modules to 'module load' in the job script "
        "(default: $BATCH_MODULES, space-separated).",
    )
    g.add_argument(
        "--env-file",
        default=os.environ.get("BATCH_ENV_FILE", ""),
        help="Path to the per-machine env file the job sources on the compute "
        "node (default: $BATCH_ENV_FILE). Required for the pbs/slurm backends; "
        "must leave the run modules, venv python, and 'import batch' working.",
    )
    g.add_argument(
        "--python",
        default=sys.executable,
        help="Absolute path to the venv python used to launch the run on the "
        "compute node (default: the submitting interpreter).",
    )
    if packed:
        p = parser.add_argument_group("packing (pbs-packed / slurm-packed)")
        p.add_argument(
            "--nodes",
            type=int,
            default=1,
            help="Packed only: number of exclusive nodes to fan over (default: 1).",
        )
        p.add_argument(
            "--node-cpus",
            type=int,
            default=128,
            help="Packed / local pinning: cores per node (default: 128).",
        )
        p.add_argument(
            "--shard",
            default="",
            metavar="I/N",
            help="Run only shard I of N of the job list (round-robin). "
            "Packed wrappers set this per node.",
        )
        p.add_argument(
            "--pin-cpus",
            action="store_true",
            help="Local only: pin each concurrent job to a disjoint core range "
            "(numactl). Needed when packing jobs on a node.",
        )


def clobber_from_args(args: argparse.Namespace) -> ClobberPolicy:
    """Map ``--resume`` to a :class:`ClobberPolicy`.

    ``SKIP`` when resuming (skip finished job dirs), else ``OVERWRITE``.
    """
    return ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE


def executor_from_args(
    args: argparse.Namespace,
    *,
    env: dict[str, str] | None = None,
    plot: bool = False,
    setplot: str = "",
) -> Executor:
    """Build the executor selected by ``--scheduler``.

    Handles the three *executor* backends — ``local``, ``pbs``, ``slurm``.  The
    ``*-packed`` schedulers do **not** map to an executor (packing submits
    wrapper jobs, it is not a per-job backend); use :func:`execute` or
    :func:`batch.submit_packed` for those.

    The ``pbs`` and ``slurm`` backends are the *same* :class:`SchedulerExecutor`
    with a different injected :class:`~batch.scheduler.Scheduler`; the resource
    request is normalized into a single :class:`~batch.scheduler.JobRequest`.

    Parameters
    ----------
    args:
        Parsed args carrying the flags from :func:`add_execution_args`.
    env:
        Extra environment variables merged into each job's environment, on top
        of the ``OMP_NUM_THREADS`` derived from ``--omp-num-threads``.
    plot:
        When True, request compute-node self-plotting on the scheduler backends.
        Ignored for the ``local`` backend, which plots via each job's
        ``post_run`` hook instead.
    setplot:
        setplot path for the compute-node ``plotclaw`` call (paired with
        *plot*); empty falls back to ``job.setplot``.

    Returns
    -------
    Executor
        A ``ParallelExecutor`` or a ``SchedulerExecutor``.

    Raises
    ------
    ValueError
        If ``--scheduler`` is a ``*-packed`` value, or a scheduler backend is
        selected without an ``--env-file``, or the value is unsupported here.
    """
    run_env = {"OMP_NUM_THREADS": str(args.omp_num_threads)}
    if env:
        run_env.update(env)
    threads = args.omp_num_threads

    if args.scheduler == "local":
        pin = getattr(args, "pin_cpus", False)
        return ParallelExecutor(
            max_workers=args.max_workers,
            env=run_env,
            cpu_affinity=pin,
            total_cpus=getattr(args, "node_cpus", None) if pin else None,
        )
    if args.scheduler in ("pbs", "slurm"):
        if not args.env_file:
            raise ValueError(
                f"--scheduler {args.scheduler} requires --env-file (or "
                "$BATCH_ENV_FILE): the job sources it on the compute node to "
                "load modules and make 'import batch' work."
            )
        # One exclusive-node-per-job request, normalized once for both backends.
        request = JobRequest(
            name="",  # filled in per job at submit time
            log_path="",
            queue=args.queue,
            account=args.account,
            walltime=args.walltime,
            nodes=1,
            cpus_per_node=threads,
            tasks_per_node=1,
            ompthreads=threads,
        )
        return SchedulerExecutor(
            get_scheduler(args.scheduler),
            env_file=args.env_file,
            default_request=request,
            python=args.python,
            modules=args.modules,
            env_vars=run_env,
            plot=plot,
            setplot=setplot,
            dry_run=args.setup_only,
        )
    raise ValueError(
        f"executor_from_args does not handle --scheduler={args.scheduler!r}; "
        "packed schedulers submit wrappers via execute()/submit_packed()."
    )


@dataclass
class ResultSummary:
    """Aggregate outcome of a batch run.

    ``n_ok`` + ``n_failed`` + ``n_pending`` == ``n_total``.  Pending jobs are
    scheduler submissions whose result is not yet known (``run(wait=False)``).
    """

    n_total: int
    n_ok: int
    n_failed: int
    n_pending: int
    failures: list[JobResult] = field(default_factory=list)


def report_results(results: list[JobResult], *, echo: bool = True) -> ResultSummary:
    """Summarize *results*; when *echo*, print the summary and any failures.

    Returns the :class:`ResultSummary` so callers can act on it without parsing
    stdout.  When every result is still pending (a submit-and-exit scheduler
    run), the printed form reports what was *submitted* rather than a
    success/failure tally.
    """
    n_ok = sum(1 for r in results if r.success)
    n_pending = sum(1 for r in results if r.pending)
    failures = [r for r in results if not r.success and not r.pending]
    summary = ResultSummary(
        n_total=len(results),
        n_ok=n_ok,
        n_failed=len(failures),
        n_pending=n_pending,
        failures=failures,
    )
    if echo:
        if n_pending and not n_ok and not failures:
            print(f"Submitted {n_pending} job(s).")
            for r in results:
                if r.job_id:
                    print(f"  {r.job.prefix}  ->  job {r.job_id}")
        else:
            print(
                f"\nCompleted: {n_ok}/{len(results)} successful, "
                f"{len(failures)} failed."
            )
            for r in failures:
                print(f"  FAILED: {r.job.prefix}  (see {r.paths.log})")
    return summary


def execute(
    args: argparse.Namespace,
    jobs: list[Job],
    *,
    experiment: str = "",
    inner_command=None,
    env: dict[str, str] | None = None,
    wait: bool | None = None,
    base_path: Path | str | None = None,
    script_dir: Path | str | None = None,
    workdir: Path | str | None = None,
    plot: bool = False,
    setplot: str = "",
) -> list[JobResult]:
    """Run *jobs* according to ``--scheduler``, tying the factories together.

    Dispatch:

    - ``*-packed`` → build a per-node :class:`PackedResources` from the args and
      call :func:`batch.submit_packed`.  Requires *inner_command*; returns an
      empty list (the per-run work happens on the compute nodes).
    - otherwise → apply ``--shard`` (if set) via :func:`shard_jobs`, build a
      :class:`BatchController` with :func:`executor_from_args` /
      :func:`clobber_from_args`, then ``setup()`` (``--setup-only`` + local) or
      ``run()`` and :func:`report_results`.

    Parameters
    ----------
    args:
        Parsed args from a parser extended by :func:`add_execution_args`.
    jobs:
        The full job list.  Sharding is applied here for a ``--shard`` run.
    experiment:
        Experiment subdirectory for the controller.
    inner_command:
        ``(shard_i, n_shards) -> argv`` callable, required for a ``*-packed``
        scheduler (the per-node re-invocation of your driver in local mode).
    env:
        Extra environment variables for each job (merged with ``OMP_NUM_THREADS``).
    wait:
        Override the blocking behavior.  Default: ``True`` for ``local``,
        ``False`` for a scheduler (submit-and-exit).
    base_path:
        Root output directory passed to the controller (else ``$OUTPUT_PATH``).
    script_dir:
        Where packed wrappers are written (default: ``<cwd>/_pack_scripts``).
    workdir:
        Directory each packed wrapper ``cd``s into before its inner command.
    plot:
        Request compute-node self-plotting on the ``pbs`` / ``slurm`` backends
        (forwarded to :func:`executor_from_args`).  Ignored for ``local``.
    setplot:
        setplot path for the compute-node ``plotclaw`` call (paired with *plot*).

    Returns
    -------
    list[JobResult]
        Per-job results for the local/scheduler paths (empty for packed and for
        ``--setup-only`` local).
    """
    scheduler = args.scheduler

    if scheduler.endswith("-packed"):
        if inner_command is None:
            raise ValueError(
                f"--scheduler {scheduler} requires an inner_command "
                "(the per-node re-invocation of your driver)."
            )
        base = scheduler.split("-")[0]
        if not args.env_file:
            raise ValueError(
                f"--scheduler {scheduler} requires --env-file (or "
                "$BATCH_ENV_FILE): each node's wrapper sources it before "
                "re-invoking your driver."
            )
        resources = PackedResources(
            queue=args.queue,
            walltime=args.walltime,
            account=args.account,
            node_cpus=getattr(args, "node_cpus", 128),
            modules=args.modules,
        )
        sdir = (
            Path(script_dir) if script_dir is not None else Path.cwd() / "_pack_scripts"
        )
        submit_packed(
            n_nodes=getattr(args, "nodes", 1),
            inner_command=inner_command,
            resources=resources,
            scheduler=base,
            script_dir=sdir,
            env_file=args.env_file,
            python=args.python,
            dry_run=args.setup_only,
            workdir=workdir,
        )
        return []

    # local / pbs / slurm
    shard = getattr(args, "shard", "")
    if shard:
        i, n = parse_shard_spec(shard)
        if n > 1:
            jobs = shard_jobs(jobs, i, n)
            logger.info("Shard %d/%d: running %d of %d jobs.", i, n, len(jobs), n)

    ctrl = BatchController(
        jobs=jobs,
        executor=executor_from_args(args, env=env, plot=plot, setplot=setplot),
        base_path=base_path,
        experiment=experiment,
        clobber=clobber_from_args(args),
    )

    if args.setup_only and scheduler == "local":
        paths = ctrl.setup()
        logger.info("Setup complete for %d job(s).", len(paths))
        return []

    should_wait = (scheduler == "local") if wait is None else wait
    results = ctrl.run(wait=should_wait)
    report_results(results)
    return results
