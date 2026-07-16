"""Cross-run analysis for GeoClaw batches.

Two pieces, both driven off each job's on-disk output so they work whether the
runs happened locally or on a scheduler:

- :func:`parse_timing` — read the GeoClaw ``timing.txt`` a completed run leaves
  in its output directory into a structured dict.  Standard library only, so it
  is always available.
- :func:`plot_performance` — a three-panel wall-time / efficiency comparison
  across a set of runs.  Needs ``matplotlib`` and ``numpy``; install the optional
  extra with ``pip install clawpack-batch[analysis]``.  If they are not
  importable it logs a warning and returns ``None`` rather than raising, mirroring
  :func:`batch.plot.plot_job`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_timing(job_dir: Path | str) -> dict | None:
    """Parse the GeoClaw ``timing.txt`` in *job_dir*.

    Parameters
    ----------
    job_dir:
        A job's output directory (``paths.job``).  ``timing.txt`` is expected
        directly inside it.

    Returns
    -------
    dict | None
        ``None`` if ``timing.txt`` is absent.  Otherwise a dict with:

        ``levels``
            list of ``{level, wall, cpu, cells}`` per AMR level.
        ``total_integration``
            ``{wall, cpu, cells}`` for the integration total (if present).
        ``components``
            ``{stepgrid, bc, regrid, output}`` each ``{wall, cpu}`` (present
            keys only).
        ``total``
            ``{wall, cpu}`` overall (if present).
        ``n_threads``
            OpenMP thread count reported by the solver (1 if not found).
    """
    txt_path = Path(job_dir) / "timing.txt"
    if not txt_path.exists():
        return None

    text = txt_path.read_text()
    result: dict = {"levels": [], "components": {}}

    for m in re.finditer(
        r"^\s+(\d+)\s+([\d.E+]+)\s+([\d.E+]+)\s+([\d.E+]+)",
        text,
        re.MULTILINE,
    ):
        result["levels"].append(
            {
                "level": int(m.group(1)),
                "wall": float(m.group(2)),
                "cpu": float(m.group(3)),
                "cells": float(m.group(4)),
            }
        )

    m = re.search(r"^total\s+([\d.E+]+)\s+([\d.E+]+)\s+([\d.E+]+)", text, re.MULTILINE)
    if m:
        result["total_integration"] = {
            "wall": float(m.group(1)),
            "cpu": float(m.group(2)),
            "cells": float(m.group(3)),
        }

    for key, pattern in [
        ("stepgrid", r"stepgrid\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("bc", r"BC/ghost cells\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("regrid", r"Regridding\s+([\d.E+]+)\s+([\d.E+]+)"),
        ("output", r"Output \(valout\)\s+([\d.E+]+)\s+([\d.E+]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            result["components"][key] = {
                "wall": float(m.group(1)),
                "cpu": float(m.group(2)),
            }

    m = re.search(r"Total time:\s+([\d.E+]+)\s+([\d.E+]+)", text)
    if m:
        result["total"] = {"wall": float(m.group(1)), "cpu": float(m.group(2))}

    m = re.search(r"Using (\d+) thread", text)
    result["n_threads"] = int(m.group(1)) if m else 1

    return result


def plot_performance(
    job_dirs: Sequence[Path | str],
    labels: Sequence[str] | None = None,
    out_path: Path | str = "performance.png",
    *,
    title: str = "Performance Analysis",
    dpi: int = 150,
) -> Path | None:
    """Three-panel wall-time / efficiency comparison across *job_dirs*.

    Reads ``timing.txt`` from each directory via :func:`parse_timing`, skipping
    any that lack it.  The figure has three panels:

    - total wall time (bars) with CPU efficiency overlaid, where efficiency is
      ``cpu / (n_threads * wall)``;
    - wall time stacked by AMR level;
    - wall time stacked by solver component (step / BC / regrid / output).

    Parameters
    ----------
    job_dirs:
        Output directories of the runs to compare.
    labels:
        One label per directory (parallel to *job_dirs*).  Defaults to each
        directory's name.
    out_path:
        Where to write the PNG.
    title:
        Figure suptitle.
    dpi:
        Output resolution.

    Returns
    -------
    Path | None
        The path written, or ``None`` if matplotlib/numpy are unavailable or no
        directory had timing data.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning(
            "matplotlib/numpy not importable; skipping plot_performance. "
            "Install the optional extra with: pip install clawpack-batch[analysis]"
        )
        return None

    job_dirs = [Path(d) for d in job_dirs]
    if labels is None:
        labels = [d.name for d in job_dirs]
    if len(labels) != len(job_dirs):
        raise ValueError(
            f"labels ({len(labels)}) must match job_dirs ({len(job_dirs)})"
        )

    timings, used_labels = [], []
    for d, label in zip(job_dirs, labels):
        t = parse_timing(d)
        if t is None or "total" not in t:
            logger.warning("No timing data for %s; skipping.", d)
            continue
        timings.append(t)
        used_labels.append(label)

    if not timings:
        logger.warning("No timing data found in any job dir; no plot written.")
        return None

    n = len(timings)
    x = np.arange(n)
    tick_fs = max(4, 9 - n // 8)

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 0.9 + 4), 6))
    fig.suptitle(title, fontsize=13)

    # Total wall time (bars) + CPU efficiency (line on a twin axis).
    ax = axes[0]
    wall_min = [t["total"]["wall"] / 60 for t in timings]
    ax.bar(x, wall_min, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(used_labels, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Total Wall Time")

    ax2 = ax.twinx()
    eff = [
        t["total"]["cpu"] / (t.get("n_threads", 1) * t["total"]["wall"]) * 100
        if t["total"]["wall"] > 0
        else 0.0
        for t in timings
    ]
    ax2.plot(x, eff, "o-", color="orange", label="CPU efficiency")
    ax2.set_ylabel("CPU Efficiency (%)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(0, 110)

    # Stacked wall time by AMR level.
    ax = axes[1]
    max_lev = max(len(t["levels"]) for t in timings)
    bottoms = np.zeros(n)
    for li in range(max_lev):
        vals = np.array(
            [
                t["levels"][li]["wall"] / 60 if li < len(t["levels"]) else 0.0
                for t in timings
            ]
        )
        ax.bar(x, vals, bottom=bottoms, label=f"Level {li + 1}")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(used_labels, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Time by AMR Level")
    ax.legend(fontsize=8)

    # Stacked wall time by solver component.
    ax = axes[2]
    bottoms = np.zeros(n)
    for key, lbl in [
        ("stepgrid", "Step (PDE)"),
        ("bc", "BC/Ghost"),
        ("regrid", "Regrid"),
        ("output", "Output"),
    ]:
        vals = np.array(
            [t["components"].get(key, {}).get("wall", 0.0) / 60 for t in timings]
        )
        ax.bar(x, vals, bottom=bottoms, label=lbl)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(used_labels, rotation=90, fontsize=tick_fs)
    ax.set_ylabel("Wall Time (min)")
    ax.set_title("Time by Component")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Performance plot saved: %s", out_path)
    return out_path
