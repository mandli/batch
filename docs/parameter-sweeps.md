# Parameter sweeps

Rather than build a job list by hand, `batch.sweep` generates one from a
parameter grid. Two helpers cover the common cases, and both work with any
executor.

← Back to the [documentation index](index.md).

---

## How a sweep works

Both helpers take:

- `factory` — a callable that receives the swept parameters as keyword arguments
  and returns a configured `Job`. It does **not** need to set `prefix`.
- `namer` — a callable that maps a parameter dict to the job's `prefix` string.
  The helper sets `job.prefix = namer(params)` after construction.
- `**param_grid` — keyword arguments where each value is a list of options.

```python
from batch.sweep import product_sweep, zip_sweep
```

---

## Cartesian product — `product_sweep`

Enumerates *every combination* of the parameter lists (row-major order). Use it
when parameters are independent.

```python
jobs = product_sweep(
    factory=lambda manning, max_level: ManningJob(manning=manning, max_level=max_level),
    namer=lambda p: f"n{p['manning']:.3f}_l{p['max_level']}",
    manning=[0.020, 0.025, 0.030, 0.035],
    max_level=[4, 5],
)
# 8 jobs: 4 Manning values × 2 refinement levels
```

The `namer` builds a prefix from the parameters so each job gets a distinct,
self-describing output directory (`n0.020_l4`, `n0.020_l5`, …). Keep prefixes
filesystem-safe and unique — they become directory names.

---

## Paired lists — `zip_sweep`

Pairs the lists element-wise (like `zip`), producing one job per index. Use it
when the parameters are *not* independent — e.g. storm tracks paired with their
intensities. All lists must be the same length, or `zip_sweep` raises
`ValueError`.

```python
jobs = zip_sweep(
    factory=lambda storm_id, intensity: StormJob(storm_id, intensity),
    namer=lambda p: f"{p['storm_id']}_{p['intensity']}",
    storm_id=["katrina", "ike", "harvey"],
    intensity=["low", "mid", "high"],
)
# 3 jobs, one per (storm, intensity) pair — NOT 9
```

---

## Resuming a partial batch

A large ensemble on a cluster will sometimes be cut short — a walltime kill, a
node failure, or you simply added more parameters. `ClobberPolicy.SKIP` makes
re-running the same script cheap: any job whose output directory already exists
is skipped, so only unfinished jobs are (re)submitted.

```python
from batch import BatchController, ClobberPolicy

ctrl = BatchController(
    jobs=jobs,
    executor=executor,
    experiment="my_ensemble",
    clobber=ClobberPolicy.SKIP,
)
results = ctrl.run(wait=False)   # skipped jobs are omitted from results
```

The three policies:

| Policy | Behavior when the job directory exists |
|---|---|
| `OVERWRITE` (default) | Remove stale `.data` files and re-run. Existing `fort.*` is overwritten by the solver. |
| `ERROR` | Raise `FileExistsError` immediately — a hard guard against stomping a prior run. |
| `SKIP` | Skip the job entirely. Gives free resumability. |

> `SKIP` keys on directory *existence*, not on whether the run actually
> finished. For true resumability, have your solver drop a sentinel file on
> completion and only trust directories that contain it — or clear out the
> directories of jobs you know were interrupted before re-running. See the
> [completion-detection note](running-on-hpc.md#waittrue-vs-waitfalse-and-monitoring)
> for why "left the queue" is not the same as "succeeded".

A common pattern is a `--resume` flag that flips the policy (as in both bundled
examples):

```python
clobber = ClobberPolicy.SKIP if args.resume else ClobberPolicy.OVERWRITE
```

---

## Next steps

- Run the sweep on a cluster: [Running on HPC](running-on-hpc.md).
- Plot each member and compare across the ensemble: [Post-processing & analysis](postprocessing.md).

Runnable sweep: `examples/local_ensemble/run_batch.py` uses `product_sweep`.
`examples/storm_surge/storm_batch.py` builds its job list directly and shows the
CLI-driven scheduler submission and `--resume` patterns.
