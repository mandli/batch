"""batch — utilities for running Clawpack/GeoClaw batch jobs.

Public API
----------
The most commonly used names are re-exported here for convenience::

    from batch import Job, BatchController, ClobberPolicy
    from batch import SerialExecutor, ParallelExecutor
    from batch import SLURMExecutor, SLURMResources
    from batch import PBSExecutor, PBSResources
    from batch import PackedResources, submit_packed
    from batch.sweep import product_sweep, zip_sweep, shard_jobs
"""

from batch.controller import BatchController
from batch.executors.local import ParallelExecutor, SerialExecutor
from batch.executors.pbs import PBSExecutor, PBSResources
from batch.executors.slurm import SLURMExecutor, SLURMResources
from batch.job import ClobberPolicy, Job, JobPaths, JobResult
from batch.packed import PackedResources, submit_packed
from batch.plot import plot_job
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
    "SLURMExecutor",
    "SLURMResources",
    "PBSExecutor",
    "PBSResources",
    "PackedResources",
    "submit_packed",
    "shard_jobs",
    "plot_job",
]
