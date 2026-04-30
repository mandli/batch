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
import os
from pathlib import Path

from batch import BatchController, ClobberPolicy, ParallelExecutor
from batch.sweep import product_sweep
from manning_job import ManningJob

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Write .data files only; do not run the solver.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output directory already exists.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("BATCH_MAX_JOBS", 4)),
        help="Maximum concurrent jobs (default: $BATCH_MAX_JOBS or 4).",
    )
    args = parser.parse_args()

    jobs = make_jobs()

    clobber = ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE

    ctrl = BatchController(
        jobs=jobs,
        executor=ParallelExecutor(max_workers=args.max_workers),
        experiment="manning_sensitivity",
        clobber=clobber,
    )

    if args.setup_only:
        paths = ctrl.setup()
        print(f"Setup complete for {len(paths)} job(s).")
        return

    results = ctrl.run(wait=True)

    n_ok = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success and r.returncode is not None)
    print(f"\nCompleted: {n_ok}/{len(results)} successful, {n_fail} failed.")

    if n_fail:
        for r in results:
            if r.returncode is not None and r.returncode != 0:
                print(f"  FAILED: {r.job.prefix}  (see {r.paths.log})")


if __name__ == "__main__":
    main()
