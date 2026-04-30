"""Core data types for batch job description and results."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ClobberPolicy(enum.Enum):
    """Controls behavior when a job's output directory already exists.

    OVERWRITE
        Remove stale ``.data`` files and re-run.  Existing output (``fort.*``)
        is left in place and will be overwritten by the solver.  This is the
        default and matches the original batch behavior.
    ERROR
        Raise ``FileExistsError`` immediately.  Use this when you want a hard
        guarantee that you are not accidentally stomping a previous run.
    SKIP
        Skip any job whose output directory already exists.  Together with a
        sentinel file produced by the solver this gives free resumability: run
        the same batch script again after a walltime kill and only the jobs
        that did not finish will be resubmitted.
    """

    OVERWRITE = "overwrite"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class JobPaths:
    """Filesystem layout for one job.

    All data files, solver output (``fort.*``), and the run log share the same
    root directory ``job``.  Plots are kept in a subdirectory so they can be
    tarred or discarded independently.
    """

    job: Path    # root directory — data files and fort.* output go here
    plots: Path  # plots subdirectory
    log: Path    # per-job log file


@dataclass
class JobResult:
    """Outcome record for a single submitted job."""

    job: Job
    paths: JobPaths
    returncode: int | None  # None until the job completes (scheduler backends)
    job_id: str | None = None  # scheduler job ID when submitted to a queue

    @property
    def success(self) -> bool:
        """True only when returncode is known and zero."""
        return self.returncode == 0

    @property
    def pending(self) -> bool:
        """True for scheduler-submitted jobs whose result is not yet known."""
        return self.returncode is None


class Job:
    """Base class for all Clawpack batch jobs.

    Subclass this to define a concrete simulation.  At minimum you must:

    - Set ``self.prefix`` — a unique string used to name the job directory.
    - Populate ``self.rundata`` with a ``ClawRunData`` object (e.g. from
      ``setrun.setrun()``).

    Optionally override:

    - ``write_data_objects(path)`` — if you need to write auxiliary files
      beyond what ``rundata.write()`` produces.
    - ``build(paths)`` — to compile the executable before submission.

    Attributes
    ----------
    prefix : str | None
        Unique identifier for this job.  Becomes the job directory name.
        Must be set before the job is submitted.
    executable : str | Path
        Name or path of the compiled binary.  A bare name (``"xgeoclaw"``) is
        resolved against the job directory after ``build()`` runs; an absolute
        path is used as-is.
    setplot : str
        Module name passed to ``plotclaw`` if plotting is requested.
    restart : bool
        If True, the controller will not clobber existing ``.data`` files and
        will pass the restart flag to ``runclaw``.
    paths : JobPaths | None
        Populated by ``BatchController`` before the job is submitted.
        Available for use in postprocessing.
    rundata : ClawRunData | None
        Clawpack run-data object.  Must be set by the subclass.
    """

    def __init__(self) -> None:
        self.prefix: str | None = None
        self.executable: str | Path = "xgeoclaw"
        self.setplot: str = "setplot"
        self.restart: bool = False
        self.paths: JobPaths | None = None
        self.rundata = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prefix={self.prefix!r})"

    def write_data_objects(self, path: Path) -> None:
        """Write Clawpack ``.data`` files into *path*.

        The default implementation calls ``self.rundata.write(out_dir=path)``.
        Override to write additional auxiliary files, calling ``super()`` first.

        Parameters
        ----------
        path:
            Destination directory.  Always ``paths.job`` — the same directory
            that will receive solver output.
        """
        if self.rundata is None:
            raise ValueError(
                f"Job {self.prefix!r}: rundata is not set. "
                "Assign a ClawRunData object before running."
            )
        self.rundata.write(out_dir=path)

    def build(self, paths: JobPaths) -> None:
        """Compile the executable before job submission.

        The default is a no-op.  Override when each job requires a fresh
        build — for example, a parameter that is compiled into the Fortran
        source rather than read from a data file.

        The compiled executable should be placed at ``paths.job / self.executable``
        (or ``self.executable`` should be updated to an absolute path) so that
        the executor can locate it.

        Parameters
        ----------
        paths:
            Paths object for this job, provided by the controller.
        """
        pass
