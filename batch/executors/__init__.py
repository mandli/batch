"""Executor protocol and re-exports for batch execution backends."""

from __future__ import annotations

from typing import Protocol

from batch.job import Job, JobPaths, JobResult


class Executor(Protocol):
    """Interface that all execution backends must satisfy.

    :class:`~batch.executors.local.SerialExecutor`,
    :class:`~batch.executors.local.ParallelExecutor`, and
    :class:`~batch.executors.scheduler.SchedulerExecutor` (parametrized by a
    PBS or SLURM :class:`~batch.scheduler.Scheduler`) all implement this
    protocol, as does any custom executor the caller provides.
    """

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        """Start or queue one job and return its result object."""
        ...

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """Block until every result in *results* has a final returncode."""
        ...
