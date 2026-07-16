# `batch` documentation

`batch` (`clawpack-batch`) runs [Clawpack](https://www.clawpack.org) / GeoClaw
simulations in batches — parameter sweeps and ensembles. It manages three
things so your application code doesn't have to:

1. the per-job directory layout,
2. Clawpack `.data` file generation, and
3. job submission.

The central idea is **pluggable execution backends**: the same job definition
runs locally (serial or parallel) or on a SLURM / PBS cluster without changing
any application code — you swap the *executor*, not the job.

> **v2 breaking change.** The v1 API (`batch.Job`, `batch.BatchController` with
> scheduler-specific subclasses) is preserved on the `v1.0.0` tag. See the
> [CHANGELOG](../CHANGELOG.md) for the full migration guide.

---

## Guides

Start here and read in order, or jump to the task you have in front of you:

| Guide | When you want to… |
|---|---|
| [Getting started](getting-started.md) | Define your first job and run it locally. |
| [Running on HPC](running-on-hpc.md) | Submit to SLURM or PBS/Derecho. |
| [Parameter sweeps](parameter-sweeps.md) | Generate many jobs from a parameter grid, and resume a killed batch. |
| [Post-processing & analysis](postprocessing.md) | Plot each job and analyze results across the whole ensemble. |
| [Extending batch](extending.md) | Compile per job, or write a custom executor. |
| [Troubleshooting](troubleshooting.md) | Diagnose common failures and tune oversubscription. |

---

## Core concepts

| Class / Function | Role |
|---|---|
| `Job` | Describes one simulation: `prefix`, `executable`, `rundata`, optional `build()` / `post_run()` overrides. |
| `BatchController` | Orchestrates directory setup, data writing, and dispatch. |
| `Executor` | Protocol implemented by `SerialExecutor`, `ParallelExecutor`, `SLURMExecutor`, `PBSExecutor`. |
| `JobPaths` | Typed paths for one job's directory, plots, and log. |
| `JobResult` | Return value from `run()`: `job`, `paths`, `returncode`, scheduler `job_id`; `.success` / `.pending` properties. |
| `ClobberPolicy` | Controls what happens when output already exists: `OVERWRITE`, `ERROR`, `SKIP`. |
| `plot_job` | Runs plotclaw after a job completes; handles missing visclaw gracefully. |
| `product_sweep` / `zip_sweep` | Build job lists from parameter grids (`batch.sweep`). |
| `parse_timing` / `plot_performance` | Parse a run's `timing.txt` and plot cross-run performance (`batch.analysis`). |

The workflow is always the same shape:

```
subclass Job  ──►  (optionally) product_sweep / zip_sweep  ──►  BatchController(jobs, executor)  ──►  ctrl.run()
```

All top-level names are importable directly from `batch`; the sweep helpers live
in `batch.sweep`:

```python
from batch import Job, BatchController, ClobberPolicy
from batch import SerialExecutor, ParallelExecutor
from batch import SLURMExecutor, SLURMResources
from batch import PBSExecutor, PBSResources
from batch import plot_job
from batch import parse_timing, plot_performance
from batch.sweep import product_sweep, zip_sweep
```

---

## Which executor do I want?

All four implement the same `Executor` interface, so switching backends means
changing one constructor argument to `BatchController`.

| Executor | Use when | Blocking? |
|---|---|---|
| `SerialExecutor` | Debugging or a single job; run one at a time. | Yes — `run()` blocks. |
| `ParallelExecutor` | A local multi-core workstation; run several jobs at once. | Yes, but jobs run concurrently up to `max_workers`. |
| `SLURMExecutor` | A SLURM cluster (`sbatch`/`squeue`). | Optional — `run(wait=False)` returns right after submission. |
| `PBSExecutor` | A PBS Pro cluster such as NCAR Derecho (`qsub`/`qstat`). | Optional — same submit-and-return semantics as SLURM. |

`ParallelExecutor` is the default if you don't pass an executor.

See [Running on HPC](running-on-hpc.md) for the scheduler backends and
[Troubleshooting](troubleshooting.md) for how `max_workers` interacts with
`OMP_NUM_THREADS`.
