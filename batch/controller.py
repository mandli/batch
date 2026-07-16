"""BatchController: orchestrates path setup, data writing, and job dispatch."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from batch.executors import Executor
from batch.job import ClobberPolicy, Job, JobPaths, JobResult

logger = logging.getLogger(__name__)


class BatchController:
    """Orchestrate setup and submission for a list of jobs.

    The controller is responsible for:

    1. Computing the canonical directory layout for each job.
    2. Creating directories and applying the clobber policy.
    3. Writing a log-file header.
    4. Calling ``job.write_data_objects()`` to produce ``.data`` files.
    5. Calling ``job.build()`` to compile the executable (no-op by default).
    6. Dispatching to the executor.

    None of this is scheduler-specific; all scheduler differences live in the
    :mod:`batch.executors` implementations.

    Parameters
    ----------
    jobs:
        Jobs to run.  May also be added later by appending to ``self.jobs``.
    executor:
        Backend that actually runs or queues jobs.  Defaults to
        ``ParallelExecutor(max_workers=4)``.
    base_path:
        Root output directory.  Falls back to the ``OUTPUT_PATH`` environment
        variable, then the current working directory.
    experiment:
        Subdirectory under *base_path* grouping all jobs in this batch.
        Typically the name of the experiment or storm (e.g. ``"hurricane_ike"``).
        Leave empty to write directly under *base_path*.
    clobber:
        Policy for pre-existing job directories.  See :class:`ClobberPolicy`.
    """

    def __init__(
        self,
        jobs: Sequence[Job] | None = None,
        executor: Executor | None = None,
        base_path: Path | str | None = None,
        experiment: str = "",
        clobber: ClobberPolicy = ClobberPolicy.OVERWRITE,
    ) -> None:
        self.jobs: list[Job] = list(jobs) if jobs else []

        if executor is None:
            from batch.executors.local import ParallelExecutor

            executor = ParallelExecutor()
        self.executor: Executor = executor

        if base_path is None:
            base_path = os.environ.get("OUTPUT_PATH", os.getcwd())
        self.base_path = Path(base_path).expanduser().resolve()
        self.experiment = experiment
        self.clobber = clobber

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _root(self) -> Path:
        """Base path with optional experiment subdirectory."""
        if self.experiment:
            return self.base_path / self.experiment
        return self.base_path

    def _make_paths(self, job: Job) -> JobPaths:
        """Compute the canonical directory layout for *job*."""
        if not job.prefix:
            raise ValueError(
                f"Job {job!r} has no prefix set. "
                "Assign job.prefix before adding it to the controller."
            )
        job_dir = self._root() / job.prefix
        return JobPaths(
            job=job_dir,
            plots=job_dir / "plots",
            log=job_dir / f"{job.prefix}_log.txt",
        )

    def _setup_job_dir(self, job: Job, paths: JobPaths) -> bool:
        """Create the job directory, applying the clobber policy.

        Returns
        -------
        bool
            True if the job should proceed; False if it should be skipped
            (``ClobberPolicy.SKIP`` with an existing directory).

        Raises
        ------
        FileExistsError
            When ``ClobberPolicy.ERROR`` and the directory already exists.
        """
        if paths.job.exists():
            if self.clobber is ClobberPolicy.ERROR:
                raise FileExistsError(
                    f"Job directory already exists: {paths.job}\n"
                    "Use ClobberPolicy.OVERWRITE to allow re-running or "
                    "ClobberPolicy.SKIP to resume a partial batch."
                )
            if self.clobber is ClobberPolicy.SKIP:
                logger.info("Skipping job %s (directory exists)", job.prefix)
                return False
            # OVERWRITE: remove stale .data files unless restarting
            if not job.restart:
                for f in paths.job.glob("*.data"):
                    f.unlink()
                    logger.debug("Removed stale data file: %s", f)
        else:
            paths.job.mkdir(parents=True, exist_ok=True)
        return True

    @staticmethod
    def _write_log_header(paths: JobPaths) -> None:
        with open(paths.log, "w") as fh:
            fh.write(f"Started {datetime.now().isoformat()}\n")
            fh.write("-" * 60 + "\n")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> list[JobPaths]:
        """Write ``.data`` files for all jobs without running them.

        Useful for staging a batch before submission, or for inspecting
        what would be written.

        Returns
        -------
        list[JobPaths]
            Paths for every job that was set up (skipped jobs are omitted).
        """
        all_paths: list[JobPaths] = []
        for job in self.jobs:
            paths = self._make_paths(job)
            if not self._setup_job_dir(job, paths):
                continue
            self._write_log_header(paths)
            job.write_data_objects(paths.job)
            job.paths = paths
            all_paths.append(paths)
            logger.info("Setup complete for job %s → %s", job.prefix, paths.job)
        return all_paths

    def run(self, wait: bool = True) -> list[JobResult]:
        """Set up, optionally build, and submit all jobs.

        Parameters
        ----------
        wait:
            If True (default), block until all jobs complete.  Set False for
            SLURM/PBS backends if you want to return immediately after
            submission and check results later.

        Returns
        -------
        list[JobResult]
            One result per submitted job.  Skipped jobs are omitted.
            ``result.returncode`` is None for scheduler-submitted jobs when
            ``wait=False``.
        """
        results: list[JobResult] = []
        for job in self.jobs:
            paths = self._make_paths(job)
            if not self._setup_job_dir(job, paths):
                continue

            self._write_log_header(paths)
            job.write_data_objects(paths.job)
            job.build(paths)
            job.paths = paths

            result = self.executor.submit(job, paths)
            results.append(result)

        if wait:
            self.executor.wait_all(results)

        failures = [
            r for r in results if r.returncode is not None and r.returncode != 0
        ]
        if failures:
            logger.warning(
                "%d job(s) failed: %s",
                len(failures),
                [r.job.prefix for r in failures],
            )
        return results
