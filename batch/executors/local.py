"""Local executors: serial and parallel subprocess-based runners."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

from batch.job import Job, JobPaths, JobResult

logger = logging.getLogger(__name__)


def _build_run_args(job: Job, paths: JobPaths) -> list[str]:
    """Build the argument list for invoking runclaw.

    Uses ``python -m clawpack.clawutil.runclaw`` so the invocation works
    wherever clawpack is importable without requiring ``$CLAW`` to be set.
    The runclaw positional interface is::

        runclaw.py  <executable>  <outdir>  <overwrite>  <restart>  <rundir>  <verbose>

    With the flattened directory layout, ``outdir`` and ``rundir`` are both
    ``paths.job``.
    """
    return [
        sys.executable,
        "-m",
        "clawpack.clawutil.runclaw",
        str(job.executable),
        str(paths.job),  # outdir
        "F" if job.restart else "T",  # overwrite
        "T" if job.restart else "F",  # restart
        str(paths.job),  # rundir (same directory)
        "True",  # verbose
    ]


class SerialExecutor:
    """Run jobs one at a time, blocking until each finishes.

    This is the simplest executor and the right choice for interactive or
    debugging runs.  The calling process blocks until every job completes,
    so ``wait_all`` is a no-op.

    Parameters
    ----------
    extra_args:
        Additional arguments appended to the runclaw invocation.  Rarely
        needed but provided as an escape hatch.
    """

    def __init__(self, extra_args: list[str] | None = None) -> None:
        self.extra_args = extra_args or []

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        """Run the job synchronously and return its result."""
        args = _build_run_args(job, paths) + self.extra_args
        logger.info("Running job %s: %s", job.prefix, " ".join(args))
        with open(paths.log, "a") as log:
            proc = subprocess.run(args, stdout=log, stderr=log)
        if proc.returncode != 0:
            logger.error("Job %s failed (returncode=%d)", job.prefix, proc.returncode)
        return JobResult(job=job, paths=paths, returncode=proc.returncode)

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """No-op — all jobs already completed in ``submit``."""
        return results


class ParallelExecutor:
    """Run up to *max_workers* jobs concurrently as subprocesses.

    Jobs are submitted as soon as a slot is free.  ``wait_all`` drains the
    remaining queue before returning.

    Parameters
    ----------
    max_workers:
        Maximum number of simultaneous subprocesses.  Defaults to the value
        of the ``BATCH_MAX_JOBS`` environment variable, or 4 if that is not
        set.  Set this to the number of independent jobs you want in flight
        at once, not to the number of OpenMP threads per job.
    poll_interval:
        Seconds between queue drain checks.  Default 5.0.
    extra_args:
        Additional arguments appended to every runclaw invocation.
    """

    def __init__(
        self,
        max_workers: int = int(os.environ.get("BATCH_MAX_JOBS", 4)),
        poll_interval: float = 5.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.poll_interval = poll_interval
        self.extra_args = extra_args or []
        # Each entry: (Popen, JobResult, open log file handle)
        self._active: list[tuple[subprocess.Popen, JobResult, object]] = []

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        """Start the job, blocking only if the worker pool is full."""
        self._drain()
        while len(self._active) >= self.max_workers:
            time.sleep(self.poll_interval)
            self._drain()

        args = _build_run_args(job, paths) + self.extra_args
        log_fh = open(paths.log, "a")
        proc = subprocess.Popen(args, stdout=log_fh, stderr=log_fh)
        result = JobResult(job=job, paths=paths, returncode=None)
        self._active.append((proc, result, log_fh))
        logger.info("Started job %s (pid=%d)", job.prefix, proc.pid)
        return result

    def _drain(self) -> None:
        """Harvest completed processes.

        Rebuilds ``_active`` via list comprehension to avoid the
        modify-while-iterating pitfall of the original implementation.
        """
        still_running: list[tuple[subprocess.Popen, JobResult, object]] = []
        for proc, result, log_fh in self._active:
            rc = proc.poll()
            if rc is not None:
                result.returncode = rc
                log_fh.close()
                if rc != 0:
                    logger.error(
                        "Job %s exited with returncode=%d",
                        result.job.prefix,
                        rc,
                    )
                else:
                    logger.info("Job %s complete", result.job.prefix)
            else:
                still_running.append((proc, result, log_fh))
        self._active = still_running

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """Block until all in-flight jobs finish."""
        while self._active:
            time.sleep(self.poll_interval)
            self._drain()
            if self._active:
                logger.info("%d job(s) still running", len(self._active))
        return results
