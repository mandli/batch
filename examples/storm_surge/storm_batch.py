"""Storm surge ensemble — SLURM submission example.

Demonstrates:
- Subclassing Job for storm-file-driven GeoClaw runs
- Per-job SLURMResources override
- SLURMExecutor with dry_run for script inspection
- zip_sweep for paired (storm_id, intensity) runs

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

Usage
-----
Dry run (inspect generated scripts without submitting)::

    python run_batch.py --dry-run

Submit to SLURM::

    python run_batch.py

Resume after partial completion::

    python run_batch.py --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from batch import BatchController, ClobberPolicy, Job, JobPaths
from batch import SLURMExecutor, SLURMResources
from batch.sweep import zip_sweep

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
    cpus:
        Number of OpenMP threads for this job; controls both the SLURM
        ``--cpus-per-task`` request and the ``OMP_NUM_THREADS`` export.
    """

    def __init__(
        self,
        storm_num: int,
        storms_path: Path = Path("."),
        setrun_path: Path | None = None,
        cpus: int = 8,
    ) -> None:
        super().__init__()

        self.storm_num = storm_num
        self.prefix = str(storm_num).zfill(5)
        self.executable = "xgeoclaw"

        if setrun_path is None:
            setrun_path = Path(__file__).parent / "setrun.py"

        spec = importlib.util.spec_from_file_location("setrun", setrun_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rundata = mod.setrun()

        storm_file = (Path(storms_path) / f"{storm_num}.storm").resolve()
        self.rundata.surge_data.storm_file = str(storm_file)

        # SLURM resource request — attached directly to the job so the
        # executor applies it without needing a subclass.
        self.slurm_resources = SLURMResources(
            partition="main",
            nodes=1,
            ntasks_per_node=1,
            cpus_per_task=cpus,
            time="06:00:00",
            account="",  # fill in your allocation
            env_vars={"OMP_NUM_THREADS": str(cpus)},
            modules=["ncarenv/23.09", "python/3.11.4"],
        )

    def __repr__(self) -> str:
        return f"StormJob(storm_num={self.storm_num}, prefix={self.prefix!r})"


# ---------------------------------------------------------------------------
# Run script
# ---------------------------------------------------------------------------

def make_jobs(storms_path: Path, setrun_path: Path) -> list[StormJob]:
    """Build jobs for storms 1–100."""
    return [
        StormJob(
            storm_num=n,
            storms_path=storms_path,
            setrun_path=setrun_path,
        )
        for n in range(1, 101)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write submission scripts but do not call sbatch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output directory already exists.",
    )
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

    executor = SLURMExecutor(
        default_resources=SLURMResources(
            partition="main",
            nodes=1,
            cpus_per_task=8,
            time="06:00:00",
        ),
        dry_run=args.dry_run,
    )

    ctrl = BatchController(
        jobs=jobs,
        executor=executor,
        experiment="storm_ensemble",
        clobber=ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE,
    )

    # For SLURM we do not wait — sbatch returns immediately
    results = ctrl.run(wait=False)

    if args.dry_run:
        print(f"Dry run: {len(results)} script(s) written, none submitted.")
    else:
        print(f"Submitted {len(results)} job(s) to SLURM.")
        for r in results:
            print(f"  {r.job.prefix}  →  SLURM job {r.job_id}")


if __name__ == "__main__":
    main()
