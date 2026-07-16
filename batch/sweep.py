"""Parameter sweep helpers for building job lists from parameter grids."""

from __future__ import annotations

import itertools
from typing import Any, Callable

from batch.job import Job


def product_sweep(
    factory: Callable[..., Job],
    namer: Callable[[dict[str, Any]], str],
    **param_grid: list[Any],
) -> list[Job]:
    """Build jobs from the Cartesian product of parameter lists.

    Parameters
    ----------
    factory:
        Callable that accepts keyword arguments drawn from *param_grid* and
        returns a configured :class:`~batch.job.Job`.  The ``prefix`` is set
        by *namer* after construction, so the factory does not need to set it.
    namer:
        Callable mapping a parameter dict to a prefix string.
    **param_grid:
        Keyword arguments where each value is a list of options.  All
        combinations are enumerated.

    Returns
    -------
    list[Job]
        One job per combination in the Cartesian product, in row-major order.

    Examples
    --------
    >>> jobs = product_sweep(
    ...     factory=lambda manning, level: MyJob(manning=manning, max_level=level),
    ...     namer=lambda p: f"n{p['manning']:.3f}_l{p['level']}",
    ...     manning=[0.020, 0.025, 0.030],
    ...     level=[4, 5],
    ... )
    >>> len(jobs)   # 3 × 2
    6
    """
    keys = list(param_grid.keys())
    jobs: list[Job] = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        job = factory(**params)
        job.prefix = namer(params)
        jobs.append(job)
    return jobs


def zip_sweep(
    factory: Callable[..., Job],
    namer: Callable[[dict[str, Any]], str],
    **param_grid: list[Any],
) -> list[Job]:
    """Build jobs by pairing parameter lists element-wise (like ``zip``).

    All parameter lists must have the same length.  This is useful when
    parameters are not independent — for example, paired storm tracks and
    intensities.

    Parameters
    ----------
    factory:
        Callable that accepts keyword arguments drawn from *param_grid*.
    namer:
        Callable mapping a parameter dict to a prefix string.
    **param_grid:
        Keyword arguments where each value is a list of options.  Lists must
        all have the same length.

    Returns
    -------
    list[Job]
        One job per index position.

    Raises
    ------
    ValueError
        If the parameter lists have different lengths.

    Examples
    --------
    >>> jobs = zip_sweep(
    ...     factory=lambda storm_id, intensity: StormJob(storm_id, intensity),
    ...     namer=lambda p: f"{p['storm_id']}_{p['intensity']}",
    ...     storm_id=["katrina", "ike", "harvey"],
    ...     intensity=["low", "mid", "high"],
    ... )
    >>> len(jobs)
    3
    """
    lengths = {k: len(v) for k, v in param_grid.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All parameter lists must have the same length. Got: {lengths}"
        )
    keys = list(param_grid.keys())
    jobs: list[Job] = []
    for combo in zip(*param_grid.values()):
        params = dict(zip(keys, combo))
        job = factory(**params)
        job.prefix = namer(params)
        jobs.append(job)
    return jobs
