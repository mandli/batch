"""Storm surge ensemble — PBS / NCAR Derecho submission example.

Demonstrates:
- Subclassing Job for storm-file-driven GeoClaw runs
- The ``--scheduler pbs`` backend: one ``SchedulerExecutor`` parametrized by a
  ``PBSScheduler`` and a per-machine ``env_file``
- Compute-node self-plotting via ``execute(plot=True)``

This is the PBS analogue of ``examples/storm_surge/storm_batch.py`` — the same
driver runs on SLURM by passing ``--scheduler slurm``; only ``env_file`` and the
``--scheduler`` value change.

Directory layout produced::

    OUTPUT_PATH/
      derecho_storm_ensemble/
        00001/              ← one directory per storm
          00001_log.txt
          00001_run.sh      ← the generated qsub script
          *.data
          fort.*
          plots/            ← VisClaw frames (plot=True runs plotclaw on the node)
        00002/
          ...

The compute node needs an *env_file* that loads the run modules and makes
``import batch`` work; the package ships an annotated template at
``docs/env_file.example.zsh``.  Point ``--env-file`` (or ``$BATCH_ENV_FILE``) at
your copy.

Usage
-----
Write qsub scripts without submitting (needs no scheduler)::

    python pbs_batch.py --setup-only --env-file ~/derecho_env.zsh

Submit to Derecho (128 OpenMP threads per node)::

    python pbs_batch.py --account NCAR0001 --env-file ~/derecho_env.zsh \\
        --omp-num-threads 128 --modules ncarenv/23.09 conda

Resume after a partial / walltime-killed run::

    python pbs_batch.py --account NCAR0001 --env-file ~/derecho_env.zsh --resume

Constructing the jobs calls ``setrun()``, which requires a Clawpack install and a
``setrun.py`` in this directory; the module itself imports without Clawpack.
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
        self.setplot = "setplot.py"

        if setrun_path is None:
            setrun_path = Path(__file__).parent / "setrun.py"

        spec = importlib.util.spec_from_file_location("setrun", setrun_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rundata = mod.setrun()

        storm_file = (Path(storms_path) / f"{storm_num}.storm").resolve()
        self.rundata.surge_data.storm_file = str(storm_file)

        # Resources here are uniform, so they come from the CLI flags
        # (--queue/--account/--walltime/--omp-num-threads) via execute().  For a
        # *heterogeneous* sweep, override per job without a subclass by attaching
        # a JobRequest (``from batch import JobRequest``), e.g. a longer wall for
        # the big storms:
        #
        #     self.job_request = JobRequest(
        #         name="", log_path="",   # filled in at submit time
        #         queue="main", walltime="24:00:00",
        #         nodes=1, cpus_per_node=128, tasks_per_node=1, ompthreads=128,
        #     )

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
    # Shared execution flags; default this driver to PBS.  Resources come from
    # the flags (--queue/--account/--walltime/--omp-num-threads/--modules).
    add_execution_args(parser, packed=False)
    parser.set_defaults(scheduler="pbs")
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

    # execute() submits and returns immediately for PBS (wait=False), printing
    # the submitted job IDs via report_results.  plot=True appends a plotclaw
    # call so each job self-plots on the compute node.
    execute(
        args,
        jobs,
        experiment="derecho_storm_ensemble",
        plot=True,
        setplot="setplot.py",
    )


if __name__ == "__main__":
    main()
