"""Scheduler backends: the only PBS/SLURM-specific code in the package.

``batch`` runs the *same* job-execution core on PBS Pro (NCAR Derecho) and on
SLURM clusters.  Everything that differs between the two schedulers is isolated
here behind :class:`Scheduler`, so the executor, the packer, the sentinel-free
polling loop, and the emitted script *body* are all scheduler-agnostic.

A :class:`Scheduler` owns exactly five things:

``directives(request)``
    The ``#PBS`` / ``#SBATCH`` header block, translated from a normalized
    :class:`JobRequest`.
``normalize_env()``
    Shell lines that export the ``BATCH_*`` contract (see below) from the
    scheduler's native environment variables.
``submit_argv(path)``
    The submit command, e.g. ``["qsub", path]`` vs
    ``["sbatch", "--parsable", path]``.
``parse_job_id(stdout)``
    Extract the job id from the submit command's stdout.
``depend_flag(ids)``
    An ``afterok`` dependency directive, for future job chaining.

plus one concession to keeping completion detection via polling rather than
sentinel files:

``poll_argv(job_id)``
    The ``qstat`` / ``squeue`` query used to test whether a job has left the
    queue.

The ``BATCH_*`` contract
------------------------
``normalize_env()`` maps the scheduler's native variables onto a fixed set the
rest of the package reads (and *only* reads — the batch path never redefines a
scheduler- or site-provided variable):

======================  ===================================================
``BATCH_JOB_ID``        the running job's id
``BATCH_SUBMIT_DIR``    the directory the job was submitted from
``BATCH_NODEFILE``      a real file, one line per node (see below)
``BATCH_NNODES``        number of nodes in the allocation
``BATCH_ARRAY_INDEX``   array task index, or empty for a non-array job
======================  ===================================================

``BATCH_NODEFILE`` is a genuine file on *both* schedulers: PBS already provides
one as ``$PBS_NODEFILE``; on SLURM :meth:`SlurmScheduler.normalize_env` expands
``scontrol show hostnames`` into a temp file so downstream node handling is
identical.

All render methods are pure (no filesystem, no subprocess), which keeps them
unit-testable without a live scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class JobRequest:
    """A scheduler-neutral resource request for one submission.

    The resource model is "request a full, exclusive node; the packer
    subdivides it."  Each concrete :class:`Scheduler` translates these fields
    into its native directives (see the table in
    :meth:`PBSScheduler.directives` / :meth:`SlurmScheduler.directives`).

    Parameters
    ----------
    name:
        Job name (PBS ``-N`` / SLURM ``--job-name``).
    log_path:
        Combined stdout/stderr log path.
    queue:
        Queue (PBS ``-q``) / partition (SLURM ``--partition``) name.  Empty
        omits the directive (use the site default).
    account:
        Allocation project code (PBS ``-A`` / SLURM ``--account``).  Empty omits
        the directive.
    walltime:
        Walltime limit in ``HH:MM:SS`` form.
    nodes:
        Node count (PBS ``select=`` count / SLURM ``--nodes``).
    cpus_per_node:
        Cores to reserve on the node (PBS ``ncpus``).  SLURM reserves them as
        ``--cpus-per-task = cpus_per_node // tasks_per_node``.  For a pure-OpenMP
        job this is the thread count; for a packed exclusive node it is the whole
        node (128 on a Derecho CPU node) for the local pool to subdivide.
    tasks_per_node:
        MPI ranks per node (PBS ``mpiprocs`` / SLURM ``--ntasks-per-node``).
        For a pure-OpenMP run packed by the local pool this is 1.
    ompthreads:
        OpenMP threads per rank (PBS ``ompthreads`` hint).  SLURM derives its
        per-task cores from ``cpus_per_node`` instead, so this is PBS-only.
    exclusive:
        Request the whole node exclusively.  On Derecho's ``main`` queue this
        is the default and PBS emits nothing; SLURM emits ``--exclusive``.
    mem:
        Memory per node (PBS ``:mem=`` chunk suffix / SLURM ``--mem``).  Empty
        uses the queue default.
    constraint:
        Node feature constraint (SLURM ``--constraint``).  Ignored by PBS.
    array:
        Array spec ``"start-end[:step]"`` (PBS ``-J`` / SLURM ``--array``).
        Empty means a non-array job.
    depend:
        Job ids this submission should run ``afterok`` of.  Rendered by
        :meth:`Scheduler.depend_flag`; empty means no dependency.
    email:
        Notification address.  Empty disables mail.
    extra_directives:
        Raw scheduler directive lines appended verbatim after the standard
        block (e.g. ``"#PBS -l job_priority=premium"``).
    """

    name: str
    log_path: str
    queue: str = ""
    account: str = ""
    walltime: str = "12:00:00"
    nodes: int = 1
    cpus_per_node: int = 128
    tasks_per_node: int = 1
    ompthreads: int = 1
    exclusive: bool = False
    mem: str = ""
    constraint: str = ""
    array: str = ""
    depend: list[str] = field(default_factory=list)
    email: str = ""
    extra_directives: list[str] = field(default_factory=list)


@runtime_checkable
class Scheduler(Protocol):
    """Interface every scheduler backend implements.

    Concrete implementations: :class:`PBSScheduler`, :class:`SlurmScheduler`.
    The methods are the *only* scheduler-specific code in the package; see the
    module docstring for the division of responsibilities.
    """

    #: Human-readable backend name (``"pbs"`` / ``"slurm"``), handy for logging.
    name: str

    def directives(self, request: JobRequest) -> list[str]:
        """Render the ``#PBS`` / ``#SBATCH`` header block for *request*."""
        ...

    def normalize_env(self) -> list[str]:
        """Shell lines exporting the ``BATCH_*`` contract (module docstring)."""
        ...

    def submit_argv(self, script_path: str) -> list[str]:
        """The argv that submits *script_path* to the scheduler."""
        ...

    def parse_job_id(self, stdout: str) -> str:
        """Extract the assigned job id from submit-command stdout."""
        ...

    def depend_flag(self, job_ids: list[str]) -> str:
        """An ``afterok`` dependency directive for *job_ids* (or ``""``)."""
        ...

    def poll_argv(self, job_id: str) -> list[str]:
        """The argv that queries whether *job_id* is still in the queue."""
        ...


class PBSScheduler:
    """PBS Pro backend, as deployed on NCAR Derecho.

    Directive mapping (from a :class:`JobRequest`):

    ==============  =============================================
    name            ``-N``
    account         ``-A``            (omitted when empty)
    queue           ``-q``            (omitted when empty)
    walltime        ``-l walltime=``
    node / cpus     ``-l select=<nodes>:ncpus=..:mpiprocs=..:ompthreads=..``
    memory          ``:mem=`` chunk suffix
    combined out    ``-o <log> -j oe``
    array           ``-J start-end[:step]``
    ==============  =============================================

    ``exclusive`` is a no-op: whole-node allocation is the default on Derecho's
    ``main`` queue.
    """

    name = "pbs"

    def directives(self, request: JobRequest) -> list[str]:
        chunk = (
            f"{request.nodes}:ncpus={request.cpus_per_node}"
            f":mpiprocs={request.tasks_per_node}:ompthreads={request.ompthreads}"
        )
        if request.mem:
            chunk += f":mem={request.mem}"

        lines = [
            f"#PBS -N {request.name}",
            f"#PBS -o {request.log_path}",
            "#PBS -j oe",  # join stdout+stderr into the -o file
        ]
        if request.queue:
            lines.append(f"#PBS -q {request.queue}")
        lines.append(f"#PBS -l select={chunk}")
        lines.append(f"#PBS -l walltime={request.walltime}")
        if request.account:
            lines.append(f"#PBS -A {request.account}")
        if request.array:
            lines.append(f"#PBS -J {request.array}")
        if request.email:
            lines.append(f"#PBS -M {request.email}")
            lines.append("#PBS -m abe")  # abort/begin/end
        if request.depend:
            lines.append(f"#PBS -W depend={self.depend_flag(request.depend)}")
        lines.extend(request.extra_directives)
        return lines

    def normalize_env(self) -> list[str]:
        # PBS already provides a nodefile; BATCH_NNODES is its unique line count.
        return [
            'export BATCH_JOB_ID="$PBS_JOBID"',
            'export BATCH_SUBMIT_DIR="$PBS_O_WORKDIR"',
            'export BATCH_NODEFILE="$PBS_NODEFILE"',
            'export BATCH_NNODES="$(sort -u "$PBS_NODEFILE" | wc -l | tr -d " ")"',
            'export BATCH_ARRAY_INDEX="${PBS_ARRAY_INDEX:-}"',
        ]

    def submit_argv(self, script_path: str) -> list[str]:
        return ["qsub", script_path]

    def parse_job_id(self, stdout: str) -> str:
        # qsub prints the full id, e.g. "1473351.desched1".
        return stdout.strip()

    def depend_flag(self, job_ids: list[str]) -> str:
        if not job_ids:
            return ""
        return "afterok:" + ":".join(job_ids)

    def poll_argv(self, job_id: str) -> list[str]:
        return ["qstat", job_id]


class SlurmScheduler:
    """SLURM backend.

    Directive mapping (from a :class:`JobRequest`):

    ==============  =============================================
    name            ``--job-name``
    account         ``--account``     (omitted when empty)
    partition       ``--partition``   (omitted when empty)
    walltime        ``--time=``
    node / cpus     ``--nodes=N --ntasks-per-node=.. --cpus-per-task=..``
    exclusive       ``--exclusive``
    memory          ``--mem``
    constraint      ``--constraint``
    combined out    ``--output`` and ``--error`` to the same file
    array           ``--array=start-end[:step]``
    ==============  =============================================
    """

    name = "slurm"

    def directives(self, request: JobRequest) -> list[str]:
        # SLURM reserves cores as cpus-per-task * ntasks-per-node, so spread the
        # requested cores across the ranks.  For a pure-OpenMP run (1 rank) this
        # is the whole reservation; for a packed exclusive node it is all the
        # node's cores handed to the single task for the local pool to subdivide.
        cpus_per_task = max(1, request.cpus_per_node // request.tasks_per_node)
        lines = [
            f"#SBATCH --job-name={request.name}",
            f"#SBATCH --output={request.log_path}",
            f"#SBATCH --error={request.log_path}",
        ]
        if request.queue:
            lines.append(f"#SBATCH --partition={request.queue}")
        lines.append(f"#SBATCH --nodes={request.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={request.tasks_per_node}")
        lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
        lines.append(f"#SBATCH --time={request.walltime}")
        if request.exclusive:
            lines.append("#SBATCH --exclusive")
        if request.account:
            lines.append(f"#SBATCH --account={request.account}")
        if request.mem:
            lines.append(f"#SBATCH --mem={request.mem}")
        if request.constraint:
            lines.append(f"#SBATCH --constraint={request.constraint}")
        if request.array:
            lines.append(f"#SBATCH --array={request.array}")
        if request.email:
            lines.append(f"#SBATCH --mail-user={request.email}")
            lines.append("#SBATCH --mail-type=END,FAIL")
        if request.depend:
            lines.append(f"#SBATCH --dependency={self.depend_flag(request.depend)}")
        lines.extend(request.extra_directives)
        return lines

    def normalize_env(self) -> list[str]:
        # SLURM has no nodefile; expand the nodelist into a real temp file so
        # the packer's node handling is byte-for-byte identical to PBS.
        return [
            'export BATCH_JOB_ID="$SLURM_JOB_ID"',
            'export BATCH_SUBMIT_DIR="$SLURM_SUBMIT_DIR"',
            'export BATCH_NODEFILE="$(mktemp)"',
            'scontrol show hostnames "$SLURM_JOB_NODELIST" > "$BATCH_NODEFILE"',
            'export BATCH_NNODES="$SLURM_NNODES"',
            'export BATCH_ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-}"',
        ]

    def submit_argv(self, script_path: str) -> list[str]:
        return ["sbatch", "--parsable", script_path]

    def parse_job_id(self, stdout: str) -> str:
        # --parsable prints "<jobid>" or "<jobid>;<cluster>".
        return stdout.strip().split(";")[0]

    def depend_flag(self, job_ids: list[str]) -> str:
        if not job_ids:
            return ""
        return "afterok:" + ":".join(job_ids)

    def poll_argv(self, job_id: str) -> list[str]:
        return ["squeue", "--job", job_id, "--noheader"]


#: Registry so string-keyed callers (CLI ``--scheduler``) can look up a backend.
SCHEDULERS: dict[str, type] = {
    "pbs": PBSScheduler,
    "slurm": SlurmScheduler,
}


def get_scheduler(name: str) -> Scheduler:
    """Instantiate the scheduler backend registered under *name*.

    Raises
    ------
    ValueError
        If *name* is not a known backend.
    """
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown scheduler {name!r}; choose from {sorted(SCHEDULERS)}"
        ) from None
