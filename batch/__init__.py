"""batch — utilities for running Clawpack/GeoClaw batch jobs.

Public API
----------
The most commonly used names are re-exported here for convenience::

    from batch import Job, BatchController, ClobberPolicy
    from batch import SerialExecutor, ParallelExecutor
    from batch import SLURMExecutor, SLURMResources
    from batch.sweep import product_sweep, zip_sweep
"""

from batch.job import ClobberPolicy, Job, JobPaths, JobResult
from batch.controller import BatchController
from batch.executors.local import ParallelExecutor, SerialExecutor
from batch.executors.slurm import SLURMExecutor, SLURMResources

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
]
