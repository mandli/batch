"""batch — utilities for running Clawpack/GeoClaw batch jobs.

Public API
----------
The most commonly used names are re-exported here for convenience::

    from batch import Job, BatchController, ClobberPolicy
    from batch import SerialExecutor, ParallelExecutor
    from batch import SchedulerExecutor, JobRequest
    from batch import PBSScheduler, SlurmScheduler, get_scheduler
    from batch import PackedResources, submit_packed
    from batch import add_execution_args, executor_from_args, execute, report_results
    from batch import parse_timing, plot_performance
    from batch.sweep import product_sweep, zip_sweep, shard_jobs

PBS and SLURM are the same :class:`SchedulerExecutor` parametrized by an
injected :class:`~batch.scheduler.Scheduler` (``PBSScheduler`` /
``SlurmScheduler``) and a per-machine ``env_file``.  See
``docs/env_file.example.zsh`` for the env_file contract.
"""

from batch.analysis import parse_timing, plot_performance
from batch.cli import (
    ResultSummary,
    add_execution_args,
    clobber_from_args,
    execute,
    executor_from_args,
    report_results,
)
from batch.controller import BatchController
from batch.executors.local import ParallelExecutor, SerialExecutor
from batch.executors.scheduler import SchedulerExecutor, render_job_script
from batch.job import ClobberPolicy, Job, JobPaths, JobResult
from batch.packed import PackedResources, submit_packed
from batch.plot import plot_job
from batch.scheduler import (
    JobRequest,
    PBSScheduler,
    Scheduler,
    SlurmScheduler,
    get_scheduler,
)
from batch.sweep import shard_jobs

__version__ = "2.0.0"

__all__ = [
    "Job",
    "JobPaths",
    "JobResult",
    "ClobberPolicy",
    "BatchController",
    "SerialExecutor",
    "ParallelExecutor",
    "SchedulerExecutor",
    "render_job_script",
    "Scheduler",
    "PBSScheduler",
    "SlurmScheduler",
    "JobRequest",
    "get_scheduler",
    "PackedResources",
    "submit_packed",
    "shard_jobs",
    "plot_job",
    "add_execution_args",
    "executor_from_args",
    "clobber_from_args",
    "execute",
    "report_results",
    "ResultSummary",
    "parse_timing",
    "plot_performance",
]
