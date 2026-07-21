"""Storm surge ensemble — SLURM submission example.

Demonstrates:
- Subclassing Job for storm-file-driven GeoClaw runs
- The ``--scheduler slurm`` backend: one ``SchedulerExecutor`` parametrized by a
  ``SlurmScheduler`` and a per-machine ``env_file``
- Building a job list directly (one job per storm number)

The PBS analogue is ``examples/derecho_ensemble/pbs_batch.py``; the same driver
runs on either scheduler by swapping ``--scheduler`` and the ``env_file``.

Directory layout produced::

    OUTPUT_PATH/
      storm_ensemble/
        00001/          ← one directory per storm
          00001_log.txt
          00001_run.sh
          *.data
          fort.*
          plots/
        00002/
          ...

The compute node needs an *env_file* that loads the run modules and makes
``import batch`` work; see the annotated template at
``docs/env_file.example.zsh`` and point ``--env-file`` (or ``$BATCH_ENV_FILE``)
at your copy.

Usage
-----
Inspect generated scripts without submitting::

    python run_batch.py --setup-only --env-file ~/cluster_env.zsh

Submit to SLURM (8 OpenMP threads per job)::

    python run_batch.py --env-file ~/cluster_env.zsh --omp-num-threads 8

Resume after partial completion::

    python run_batch.py --env-file ~/cluster_env.zsh --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from batch import Job, add_execution_args, execute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------


class StormJob(Job):
    """One GeoClaw storm-surge simulation driven by a single ``.storm`` file.

    Parameters
    ----------
    storm_num:
        Integer storm identifier.  Becomes the zero-padded prefix and is used
        to locate the storm file.
    storms_path:
        Directory containing ``.storm`` files named ``<storm_num>.storm``.
    setrun_path:
        Path to the base ``setrun.py``.
    """

    def __init__(
        self,
        storm_num: int,
        storms_path: Path = Path("."),
        setrun_path: Path | None = None,
    ) -> None:
        super().__init__()

        self.storm_num = storm_num
        self.prefix = str(storm_num).zfill(5)
        self.executable = "xgeoclaw"

        if setrun_path is None:
            setrun_path = Path(__file__).parent / "setrun.py"

        # Use clawutil's fullpath_import if available:
        # mod = clawutil.fullpath_import(setrun_path)
        spec = importlib.util.spec_from_file_location("setrun", setrun_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rundata = mod.setrun()

        storm_file = (Path(storms_path) / f"{storm_num}.storm").resolve()
        self.rundata.surge_data.storm_file = str(storm_file)

        # Resources are uniform, so they come from the CLI flags
        # (--queue/--account/--walltime/--omp-num-threads/--modules) via
        # execute().  Override per job without a subclass by attaching a
        # JobRequest (``from batch import JobRequest``) when a sweep is
        # heterogeneous.

    def __repr__(self) -> str:
        return f"StormJob(storm_num={self.storm_num}, prefix={self.prefix!r})"


# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------


def make_jobs(storms_path: Path, setrun_path: Path) -> list[StormJob]:
    """Build jobs for storms 1–100."""
    return [
        StormJob(storm_num=n, storms_path=storms_path, setrun_path=setrun_path)
        for n in range(1, 101)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Shared execution flags; default this driver to SLURM.  Resources come from
    # the flags (--queue/--account/--walltime/--omp-num-threads/--modules).
    add_execution_args(parser, packed=False)
    parser.set_defaults(scheduler="slurm")
    parser.add_argument(
        "--storms-path",
        type=Path,
        default=Path(__file__).parent / "storms",
        help="Directory containing .storm files.",
    )
    parser.add_argument(
        "--setrun",
        type=Path,
        default=Path(__file__).parent / "setrun.py",
        help="Path to setrun.py.",
    )
    args = parser.parse_args()

    jobs = make_jobs(storms_path=args.storms_path, setrun_path=args.setrun)

    # execute() submits and returns immediately for a scheduler (wait=False),
    # printing the submitted job IDs via report_results.
    execute(args, jobs, experiment="storm_ensemble")


if __name__ == "__main__":
    main()
