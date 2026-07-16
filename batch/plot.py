"""Plotting utilities for post-run analysis."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from batch.job import JobResult

logger = logging.getLogger(__name__)


def _plot_inprocess(result: JobResult, setplot, format: str) -> bool:
    """In-process fallback used only when setplot is callable."""
    try:
        from clawpack.visclaw.plotclaw import plotclaw
    except ImportError:
        logger.warning(
            "clawpack.visclaw not importable; skipping plot for %s",
            result.job.prefix,
        )
        return False

    try:
        plotclaw(
            outdir=str(result.paths.job),
            plotdir=str(result.paths.plots),
            setplot=setplot,
            format=format,
        )
        logger.info("Plots written to %s", result.paths.plots)
        return True
    except Exception:
        logger.exception("Plotting failed for job %s", result.job.prefix)
        return False


def plot_job(
    result: JobResult,
    setplot: str | Path = "setplot.py",
    format: str = "ascii",
    verbose: bool = False,
) -> bool:
    """Run plotclaw on a completed job's output.

    Runs plotclaw as a subprocess, capturing all output (including C-level
    output from matplotlib) to the job's log file. A ``--- plotclaw ---``
    separator is written to the log before the subprocess call so solver
    and plotting output are visually distinct.

    Parameters
    ----------
    result:
        Completed job result. Uses result.paths.job as outdir and
        result.paths.plots as plotdir.
    setplot:
        File path (str or Path) or callable. A relative string is resolved
        against result.paths.job if that file exists; a Path is resolved to
        an absolute path. A callable cannot cross the subprocess boundary
        and triggers an in-process fallback with a logged warning.
    format:
        Clawpack output format, passed to plotclaw. Default 'ascii'.
    verbose:
        When True, log the full args list at INFO level before running.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    if callable(setplot):
        logger.warning(
            "setplot is callable; falling back to in-process plotting for %s "
            "(output will not be captured to log)",
            result.job.prefix,
        )
        return _plot_inprocess(result, setplot, format)

    if isinstance(setplot, Path):
        setplot_arg = str(setplot.resolve())
    else:
        candidate = result.paths.job / setplot
        setplot_arg = str(candidate) if candidate.exists() else str(setplot)

    args = [
        sys.executable,
        "-m",
        "clawpack.visclaw.plotclaw",
        str(result.paths.job),
        str(result.paths.plots),
        setplot_arg,
    ]

    if verbose:
        logger.info("plotclaw args: %s", args)

    with open(result.paths.log, "a") as log_fh:
        log_fh.write("\n--- plotclaw ---\n")
        log_fh.flush()
        proc = subprocess.run(args, stdout=log_fh, stderr=log_fh)

    if proc.returncode != 0:
        logger.warning(
            "plotclaw exited with returncode %d for job %s; see %s",
            proc.returncode,
            result.job.prefix,
            result.paths.log,
        )
        return False

    logger.info("Plots written to %s", result.paths.plots)
    return True
