"""Example Job subclass for a Manning's n sensitivity study.

This module demonstrates the minimal pattern for defining a batch job:
subclass Job, populate rundata in __init__, override write_data_objects
if additional files are needed.

This example is designed to be self-contained: it does not require an actual
Clawpack installation to import, though running the batch will.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from batch import Job
from batch.plot import plot_job


class ManningJob(Job):
    """One GeoClaw run with a specific uniform Manning's n coefficient.

    Parameters
    ----------
    manning:
        Manning's roughness coefficient to apply uniformly over the domain.
    max_level:
        Maximum AMR refinement level.  Allows coarsening for quick sweeps or
        full-resolution runs without separate setrun files.
    setrun_path:
        Path to the ``setrun.py`` file that defines the base configuration.
        Defaults to the ``setrun.py`` in the same directory as this file.
    """

    def __init__(
        self,
        manning: float,
        max_level: int = 5,
        setrun_path: Path | None = None,
    ) -> None:
        super().__init__()

        self.manning = manning
        self.max_level = max_level

        # Prefix encodes the swept parameters for easy identification
        self.prefix = f"n{manning:.3f}_l{max_level}"
        self.executable = "xgeoclaw"

        # Load base configuration and apply parameter overrides
        if setrun_path is None:
            setrun_path = Path(__file__).parent / "setrun.py"

        spec = importlib.util.spec_from_file_location("setrun", setrun_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.rundata = mod.setrun()

        # Override the Manning coefficient
        self.rundata.geo_data.manning_coefficient = manning

        # Override the maximum refinement level
        self.rundata.amrdata.amr_levels_max = max_level

    def post_run(self, result) -> None:
        plot_job(result, setplot=Path(__file__).parent / "setplot.py")


    def __repr__(self) -> str:
        return (
            f"ManningJob(prefix={self.prefix!r}, "
            f"manning={self.manning}, max_level={self.max_level})"
        )
