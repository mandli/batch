"""Manning's n sensitivity ensemble — local parallel run.

Demonstrates:
- product_sweep to generate a Cartesian parameter grid
- ParallelExecutor for local multi-process execution
- ClobberPolicy.SKIP for free resumability

Usage
-----
From the example directory::

    python run_batch.py

or with a custom output path::

    OUTPUT_PATH=/scratch/myproject python run_batch.py

To do a dry run that only writes .data files::

    python run_batch.py --setup-only

To resume a partially-completed batch::

    python run_batch.py --resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Use non-interactive backend so plotting works in batch without a display
import matplotlib

matplotlib.use("Agg")

from manning_job import ManningJob

from batch import add_execution_args, execute
from batch.sweep import product_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def make_jobs() -> list[ManningJob]:
    """Define the parameter grid and return the job list."""
    return product_sweep(
        factory=lambda manning, max_level: ManningJob(
            manning=manning,
            max_level=max_level,
            setrun_path=Path(__file__).parent / "setrun.py",
        ),
        namer=lambda p: f"n{p['manning']:.3f}_l{p['max_level']}",
        manning=[0.020, 0.025, 0.030, 0.035],
        max_level=[4, 5],
    )


def plot_ensemble(results: list) -> None:
    """Plot surface elevation vs time for all successful jobs on one figure.

    Reads ``fort.gauge`` from each job's output directory.  Jobs without that
    file are skipped with a warning.  The figure is written next to the job
    directories as ``ensemble_comparison.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    logger = logging.getLogger(__name__)
    successful = [r for r in results if r.success]
    if not successful:
        logger.warning("No successful jobs to plot in plot_ensemble.")
        return

    fig, ax = plt.subplots()
    for r in successful:
        gauge_file = r.paths.job / "fort.gauge"
        if not gauge_file.exists():
            logger.warning("fort.gauge not found for %s, skipping.", r.job.prefix)
            continue
        try:
            data = np.loadtxt(gauge_file)
            # fort.gauge columns: gauge_num, level, time, q[0], q[1], q[2], eta
            ax.plot(data[:, 2], data[:, 6], label=r.job.prefix)
        except Exception as exc:
            logger.warning("Failed to load gauge data for %s: %s", r.job.prefix, exc)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Surface elevation (m)")
    ax.legend()

    out_path = successful[0].paths.job.parent / "ensemble_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Ensemble comparison written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Shared execution flags (--scheduler/--resume/--max-workers/…). This example
    # is local-only, so the packing flags are omitted (packed=False); pass a
    # scheduler flag to reuse the same driver on a cluster.
    add_execution_args(parser, packed=False)
    args = parser.parse_args()

    # execute() blocks for the local scheduler, prints the run summary via
    # report_results, and returns the results ([] under --setup-only).
    results = execute(args, make_jobs(), experiment="manning_sensitivity")

    if not args.setup_only:
        plot_ensemble(results)


if __name__ == "__main__":
    main()
