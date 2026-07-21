"""SchedulerExecutor: one executor, parametrized by an injected Scheduler.

This is the single per-job scheduler backend.  It renders a *scheduler-agnostic*
job script whose only per-backend lines are the header block and the
``BATCH_*`` normalization, both injected by the :class:`~batch.scheduler.Scheduler`
passed in.  The body below those two blocks is byte-for-byte identical across
PBS and SLURM (see ``tests/test_scheduler.py``), which is what "parity" means
here: same body, same ``BATCH_*`` contract, same packer, same polling loop.

The emitted script is deliberately independent of login shells.  It sources a
per-machine *env_file* (the one machine-specific artifact, whose path — not
contents — the package carries) and never relies on ``.zprofile`` / ``.bashrc``:

.. code-block:: zsh

    #!/bin/zsh
    <directives>
    set -euo pipefail
    source <env_file>
    <normalize_env>          # exports BATCH_*
    <module load ...>        # optional, back-compat with --modules
    export OMP_NUM_THREADS=.. # optional per-job env
    <python> -c "import batch" || { diagnostic; exit 1; }
    cd <workdir>
    exec <run command>

The packer / solver is launched through an absolute *python* (an executor arg),
immune to any PATH reordering a ``module load`` inside the env_file might do, and
``exec``'d so scheduler walltime and signals propagate to it directly.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

from batch.executors.local import _build_run_args
from batch.job import Job, JobPaths, JobResult
from batch.scheduler import JobRequest, Scheduler

logger = logging.getLogger(__name__)


def render_job_script(
    scheduler: Scheduler,
    request: JobRequest,
    run_command: list[str],
    *,
    env_file: str | Path,
    python: str | Path = sys.executable,
    workdir: str | Path,
    modules: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    plot_command: list[str] | None = None,
    shell: str = "/bin/zsh",
) -> str:
    """Render the scheduler-agnostic job script.

    Pure function — no filesystem, no subprocess — so the emitted text can be
    diffed in tests without a cluster.

    Parameters
    ----------
    scheduler:
        Backend supplying the directive block and ``BATCH_*`` normalization.
    request:
        Normalized resource request feeding ``scheduler.directives``.
    run_command:
        The argv to ``exec`` into on the compute node (the solver or the
        packer).  Interpolated with :func:`shlex.quote`.
    env_file:
        Path to the per-machine env file sourced at the top of the body.  Must
        leave the run modules loaded, *python* resolvable, and ``import batch``
        working — the package ships an annotated example, not the file itself.
    python:
        Absolute path to the venv python used for the import check and — when
        the caller builds *run_command* from it — the run itself.  Passing an
        absolute path makes the launch immune to module-load PATH reordering.
    workdir:
        Directory the body ``cd``s into before the ``exec``.
    modules:
        Optional module names ``module load``ed after the env_file is sourced.
        The env_file is expected to own machine setup; this stays only for
        back-compat with drivers that pass ``--modules``.
    env_vars:
        Optional per-job environment exports (canonically ``OMP_NUM_THREADS``).
    plot_command:
        Optional argv run *after* the solver for compute-node self-plotting.
        When given, the solver and plot run as two plain commands (matching the
        pre-refactor behavior); when absent, the run is ``exec``'d so scheduler
        walltime and signals propagate straight to it.
    shell:
        Interpreter for the shebang.  No login flag — the env_file is
        self-sufficient.

    Returns
    -------
    str
        Complete script text with a trailing newline.
    """
    q_env_file = shlex.quote(str(env_file))
    q_python = shlex.quote(str(python))
    q_workdir = shlex.quote(str(workdir))
    run = " ".join(shlex.quote(str(a)) for a in run_command)

    lines: list[str] = [f"#!{shell}"]
    lines.extend(scheduler.directives(request))
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append(f"source {q_env_file}")
    lines.append("")
    lines.extend(scheduler.normalize_env())
    lines.append("")

    if modules:
        lines.extend(f"module load {m}" for m in modules)
        lines.append("")
    if env_vars:
        lines.extend(f"export {k}={shlex.quote(str(v))}" for k, v in env_vars.items())
        lines.append("")

    # Fail fast with a diagnostic if the env_file did not make batch importable,
    # rather than dying obscurely deep inside the run.
    lines.append(
        f'{q_python} -c "import batch" || '
        f'{{ echo "batch import failed on $(hostname)" >&2; '
        f"{q_python} --version >&2; exit 1; }}"
    )
    lines.append("")
    lines.append(f"cd {q_workdir}")
    if plot_command:
        # Solver then plot: two plain commands (cannot exec the solver and still
        # plot afterward), matching the original per-job self-plotting behavior.
        lines.append(run)
        lines.append(" ".join(shlex.quote(str(a)) for a in plot_command))
    else:
        lines.append(f"exec {run}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


class SchedulerExecutor:
    """Submit jobs to a scheduler via an injected :class:`Scheduler` backend.

    Composition, not inheritance: PBS vs SLURM is a constructor argument, not a
    subclass.  ``submit`` writes a script and queues it; ``wait_all`` polls the
    scheduler until every submitted job leaves the queue.

    Parameters
    ----------
    scheduler:
        The backend (``PBSScheduler()`` / ``SlurmScheduler()``).
    env_file:
        Path to the per-machine env file the job sources.  Required — this is
        the one machine-specific input the package needs.
    default_request:
        Fallback :class:`JobRequest` template for jobs without a per-job
        override.  The per-job ``name`` / ``log_path`` are filled in at submit
        time from the job and its paths.
    python:
        Absolute venv python for the import check and the run launch.  Defaults
        to the submitting interpreter.
    modules, env_vars:
        Optional module loads / per-job env exports threaded into every script
        (see :func:`render_job_script`).
    plot:
        When True, append a ``plotclaw`` call so each job self-plots on the
        compute node (see :func:`render_job_script`).
    setplot:
        setplot path for the compute-node ``plotclaw`` call; empty falls back to
        ``job.setplot``.
    dry_run:
        Write the script but do not submit.
    poll_interval:
        Seconds between polls in ``wait_all``.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        env_file: str | Path,
        *,
        default_request: JobRequest | None = None,
        python: str | Path = sys.executable,
        modules: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        plot: bool = False,
        setplot: str = "",
        dry_run: bool = False,
        poll_interval: float = 30.0,
    ) -> None:
        self.scheduler = scheduler
        # Expand ~ before the template quotes these paths — shlex.quote would
        # otherwise defeat tilde expansion in the emitted `source` / launch lines.
        self.env_file = str(Path(env_file).expanduser())
        self.default_request = default_request
        self.python = str(Path(python).expanduser())
        self.modules = modules or []
        self.env_vars = env_vars or {}
        self.plot = plot
        self.setplot = setplot
        self.dry_run = dry_run
        self.poll_interval = poll_interval

    def _request_for(self, job: Job, paths: JobPaths) -> JobRequest:
        """Resolve the per-job request, filling in name/log from the job."""
        base = getattr(job, "job_request", None) or self.default_request
        if base is None:
            raise ValueError(
                f"No JobRequest for job {job.prefix!r}: pass default_request to "
                "SchedulerExecutor or attach job.job_request."
            )
        # Copy so the shared default is not mutated by per-job name/log.
        from dataclasses import replace

        return replace(base, name=job.prefix or base.name, log_path=str(paths.log))

    def submit(self, job: Job, paths: JobPaths) -> JobResult:
        request = self._request_for(job, paths)
        # Launch through the absolute venv python (self.python) rather than the
        # runclaw default, so a module-load PATH reorder cannot pick a different
        # interpreter on the compute node.
        run_command = [str(self.python)] + _build_run_args(job, paths)[1:]
        plot_command = None
        if self.plot:
            plot_command = [
                str(self.python),
                "-m",
                "clawpack.visclaw.plotclaw",
                str(paths.job),
                str(paths.plots),
                self.setplot or str(job.setplot),
            ]
        script = render_job_script(
            self.scheduler,
            request,
            run_command,
            env_file=self.env_file,
            python=self.python,
            workdir=paths.job,
            modules=self.modules,
            env_vars=self.env_vars,
            plot_command=plot_command,
        )

        script_path = paths.job / f"{job.prefix}_run.sh"
        script_path.write_text(script)
        logger.debug("Wrote submission script: %s", script_path)

        if self.dry_run:
            logger.info("[dry-run] Would submit: %s", script_path)
            return JobResult(job=job, paths=paths, returncode=None, job_id="dry-run")

        proc = subprocess.run(
            self.scheduler.submit_argv(str(script_path)),
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = self.scheduler.parse_job_id(proc.stdout)
        logger.info(
            "Submitted job %s → %s job ID %s", job.prefix, self.scheduler.name, job_id
        )
        return JobResult(job=job, paths=paths, returncode=None, job_id=job_id)

    def wait_all(self, results: list[JobResult]) -> list[JobResult]:
        """Poll the scheduler until all submitted jobs leave the queue."""
        pending = {r.job_id: r for r in results if r.job_id and r.job_id != "dry-run"}
        while pending:
            time.sleep(self.poll_interval)
            completed = []
            for job_id in list(pending):
                proc = subprocess.run(
                    self.scheduler.poll_argv(job_id),
                    capture_output=True,
                    text=True,
                )
                # A finished/purged job makes the poll return non-zero
                # ("Unknown Job Id") or empty stdout.  Key on either signal
                # rather than returncode alone, since transient states differ
                # by site.
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
                        "Job %s (%s %s) left the queue",
                        pending[job_id].job.prefix,
                        self.scheduler.name,
                        job_id,
                    )
                    completed.append(job_id)
            for job_id in completed:
                del pending[job_id]
            if pending:
                logger.info("%d job(s) still in queue", len(pending))
        return results
