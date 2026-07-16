"""Parameter sweep helpers for building job lists from parameter grids."""

from __future__ import annotations

import itertools
from typing import Any, Callable, TypeVar

from batch.job import Job

_J = TypeVar("_J", bound=Job)


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


def parse_shard_spec(spec: str) -> tuple[int, int]:
    """Parse a 1-based ``"I/N"`` shard spec into ``(i, n)``.

    An empty string means "no sharding" and returns ``(1, 1)``.  Used to split a
    job list across nodes: shard ``i`` of ``n`` (see :func:`shard_jobs`).

    Parameters
    ----------
    spec:
        Either an empty string or ``"I/N"`` with ``1 <= I <= N`` and ``N >= 1``.

    Returns
    -------
    tuple[int, int]
        The ``(i, n)`` pair.

    Raises
    ------
    ValueError
        If *spec* is malformed or out of range.

    Examples
    --------
    >>> parse_shard_spec("3/16")
    (3, 16)
    >>> parse_shard_spec("")
    (1, 1)
    """
    if not spec:
        return 1, 1
    try:
        i_str, n_str = spec.split("/")
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(f"shard spec must be I/N (e.g. 1/16); got {spec!r}") from None
    if n < 1 or not (1 <= i <= n):
        raise ValueError(
            f"shard spec I/N requires N >= 1 and 1 <= I <= N; got {spec!r}"
        )
    return i, n


def shard_jobs(jobs: list[_J], i: int, n: int) -> list[_J]:
    """Return shard *i* of *n* from *jobs*, round-robin (1-based).

    Splitting the job list this way lets each of *n* nodes run a disjoint slice
    of the same sweep: node ``i`` runs ``shard_jobs(jobs, i, n)``.  Round-robin
    (``jobs[i-1::n]``) rather than contiguous blocks keeps the per-shard work
    balanced when jobs are ordered by cost.

    The *n* shards are disjoint and their union is *jobs*, so no job is dropped
    or duplicated across nodes.

    Parameters
    ----------
    jobs:
        The full job list.
    i:
        1-based shard index, ``1 <= i <= n``.
    n:
        Number of shards, ``n >= 1``.

    Returns
    -------
    list[Job]
        The jobs assigned to shard *i*.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``i`` is out of ``[1, n]``.
    """
    if n < 1 or not (1 <= i <= n):
        raise ValueError(f"shard requires n >= 1 and 1 <= i <= n; got i={i}, n={n}")
    return jobs[i - 1 :: n]
