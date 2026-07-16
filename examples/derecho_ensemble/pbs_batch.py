"""Storm surge ensemble — PBS / NCAR Derecho submission example.

Demonstrates:
- Subclassing Job for storm-file-driven GeoClaw runs
- Per-job PBSResources override (attached to the job, no scheduler subclass)
- PBSExecutor with dry_run for script inspection
- Compute-node self-plotting via PBSResources(plot=True)

This is the PBS analogue of ``examples/storm_surge/storm_batch.py`` — the Job
definition is identical in spirit; only the resource object and executor differ.

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

Usage
-----
Write qsub scripts without submitting (needs no scheduler)::

    python pbs_batch.py --setup-only

Submit to Derecho::

    python pbs_batch.py --account NCAR0001

Resume after a partial / walltime-killed run::

    python pbs_batch.py --account NCAR0001 --resume

Constructing the jobs calls ``setrun()``, which requires a Clawpack install and a
``setrun.py`` in this directory; the module itself imports without Clawpack.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from batch import Job, PBSResources, add_execution_args, execute

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
    account:
        Derecho project code (``#PBS -A``) applied to this job's resources.
    threads:
        OpenMP threads for this job; drives both the ``ompthreads`` request and
        the ``OMP_NUM_THREADS`` export.  Derecho CPU nodes have 128 cores.
    """

    def __init__(
        self,
        storm_num: int,
        storms_path: Path = Path("."),
        setrun_path: Path | None = None,
        account: str = "",
        threads: int = 128,
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

        # PBS resource request — attached directly to the job so the executor
        # applies it without needing a scheduler-specific Job subclass.  A pure
        # OpenMP GeoClaw run uses one MPI rank and the whole node's cores.
        self.pbs_resources = PBSResources(
            queue="main",
            nodes=1,
            ncpus=128,
            mpiprocs=1,
            ompthreads=threads,
            walltime="12:00:00",
            account=account,
            env_vars={"OMP_NUM_THREADS": str(threads)},
            modules=["ncarenv/23.09", "conda"],
            plot=True,  # self-plot on the compute node
            setplot="setplot.py",
        )

    def __repr__(self) -> str:
        return f"StormJob(storm_num={self.storm_num}, prefix={self.prefix!r})"


# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------


def make_jobs(storms_path: Path, setrun_path: Path, account: str) -> list[StormJob]:
    """Build jobs for storms 1–100."""
    return [
        StormJob(
            storm_num=n,
            storms_path=storms_path,
            setrun_path=setrun_path,
            account=account,
        )
        for n in range(1, 101)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Shared execution flags; default this driver to PBS. Each StormJob carries
    # its own PBSResources (with plot=True), which override the executor default.
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

    jobs = make_jobs(
        storms_path=args.storms_path,
        setrun_path=args.setrun,
        account=args.account,
    )

    # execute() submits and returns immediately for PBS (wait=False), printing
    # the submitted job IDs via report_results.
    execute(args, jobs, experiment="derecho_storm_ensemble")


if __name__ == "__main__":
    main()
