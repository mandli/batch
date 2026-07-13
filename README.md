# batch

Utilities for running [Clawpack](https://www.clawpack.org) / GeoClaw batch jobs.

`batch` manages the directory layout, data file generation, and job submission
for parameter sweeps and ensemble simulations.  Execution backends are
pluggable: the same job definition runs locally (serial or parallel) or on
a SLURM cluster without changing any application code.

> **v2 breaking change.**  The v1 API (`batch.Job`, `batch.BatchController`
> with scheduler-specific subclasses) is preserved on the `v1.0.0` tag.
> See [CHANGELOG](CHANGELOG.md) for the full migration guide.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10.  Clawpack must be importable at runtime but is not
listed as a hard dependency (it is assumed to be present in the environment).

---

## Core concepts

| Class / Function | Role |
|---|---|
| `Job` | Describes one simulation: `prefix`, `executable`, `rundata`, optional `build()` / `post_run()` overrides |
| `BatchController` | Orchestrates directory setup, data writing, and dispatch |
| `Executor` | Protocol implemented by `SerialExecutor`, `ParallelExecutor`, `SLURMExecutor`, `PBSExecutor` |
| `JobPaths` | Typed paths for one job's directory, plots, and log |
| `JobResult` | Return value from `run()`: job, paths, returncode, scheduler job ID |
| `ClobberPolicy` | Controls what happens when output already exists: `OVERWRITE`, `ERROR`, `SKIP` |
| `plot_job` | Calls plotclaw in-process after job completion; handles missing visclaw gracefully |

---

## Quick start

### 1. Define a job

Subclass `Job`, populate `rundata`, and set `prefix`:

```python
from pathlib import Path
import importlib.util
from batch import Job

import clawpack.clawutil.util as clawutil

class MyGeoClawJob(Job):
    def __init__(self, manning: float) -> None:
        super().__init__()
        self.prefix = f"n{manning:.3f}"
        self.executable = "xgeoclaw"
        self.manning = manning

        # Load base configuration from a local setrun.py - requires clawpack
        setrun = clawutil.fullpath_import(setrun_path)
        self.rundata = setrun.setrun()

        # Apply parameter override
        self.rundata.geo_data.manning_coefficient = manning
```

### 2. Run locally

```python
from batch import BatchController, ParallelExecutor

jobs = [MyGeoClawJob(manning=n) for n in [0.020, 0.025, 0.030]]

ctrl = BatchController(
    jobs=jobs,
    executor=ParallelExecutor(max_workers=3),
    experiment="manning_sensitivity",
)
results = ctrl.run()

for r in results:
    status = "ok" if r.success else f"FAILED (rc={r.returncode})"
    print(f"  {r.job.prefix}  {status}  ->  {r.paths.job}")
```

Output layout:

```
OUTPUT_PATH/
  manning_sensitivity/
    n0.020/
      n0.020_log.txt
      *.data
      fort.*
      plots/
    n0.025/
      ...
```

### 3. Run on SLURM

```python
from batch import BatchController, SLURMExecutor, SLURMResources

executor = SLURMExecutor(
    default_resources=SLURMResources(
        partition="main",
        nodes=1,
        cpus_per_task=8,
        time="06:00:00",
        account="MY_ALLOCATION",
        env_vars={"OMP_NUM_THREADS": "8"},
        modules=["ncarenv/23.09", "python/3.11.4"],
    ),
)

ctrl = BatchController(
    jobs=jobs,
    executor=executor,
    experiment="manning_sensitivity",
)
# Returns immediately after sbatch submission; job IDs in results
results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  SLURM job {r.job_id}")
```

Per-job resource overrides are supported without subclassing — attach a
`SLURMResources` instance directly to the job:

```python
job.slurm_resources = SLURMResources(partition="gpu", time="12:00:00")
```

### 4. Run on PBS / Derecho

NCAR Derecho runs PBS Pro (`qsub`/`qstat`).  `PBSExecutor` is the PBS analogue
of `SLURMExecutor` — same submit-and-return semantics, same per-job override
mechanism (attach `job.pbs_resources`), same `dry_run` flag.

```python
from batch import BatchController, PBSExecutor, PBSResources

executor = PBSExecutor(
    default_resources=PBSResources(
        queue="main",
        nodes=1,
        ncpus=128,          # Derecho CPU nodes have 128 cores
        mpiprocs=1,         # pure-OpenMP GeoClaw → 1 MPI rank
        ompthreads=128,
        walltime="12:00:00",
        account="NCAR0001",  # your Derecho project code (#PBS -A)
        env_vars={"OMP_NUM_THREADS": "128"},
        modules=["ncarenv/23.09", "conda"],
    ),
)

ctrl = BatchController(
    jobs=jobs,
    executor=executor,
    experiment="storm_ensemble",
)
# Returns immediately after qsub submission; job IDs in results
results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  PBS job {r.job_id}")
```

`PBSResources.plot=True` (with a `setplot` path) appends a `plotclaw` call to
the generated script so each job produces its VisClaw frames on the compute
node, avoiding a long-lived login-node process.  Monitor the queue with
`qstat -u $USER`.

### 5. Inspect scripts without submitting

```python
# SLURMExecutor and PBSExecutor both accept dry_run=True
executor = SLURMExecutor(default_resources=resources, dry_run=True)
ctrl = BatchController(jobs=jobs, executor=executor, experiment="test")
ctrl.run(wait=False)
# Scripts written to each job directory; sbatch/qsub not called
```

---

## Parameter sweeps

### Cartesian product

```python
from batch.sweep import product_sweep

jobs = product_sweep(
    factory=lambda manning, level: MyGeoClawJob(manning, max_level=level),
    namer=lambda p: f"n{p['manning']:.3f}_l{p['level']}",
    manning=[0.020, 0.025, 0.030],
    level=[4, 5],
)
# 6 jobs: 3 Manning values x 2 refinement levels
```

### Paired sweep

```python
from batch.sweep import zip_sweep

jobs = zip_sweep(
    factory=lambda storm_id, intensity: StormJob(storm_id, intensity),
    namer=lambda p: f"{p['storm_id']}_{p['intensity']}",
    storm_id=["katrina", "ike", "harvey"],
    intensity=["low", "mid", "high"],
)
# 3 jobs: one per (storm, intensity) pair
```

---

## Resuming a partial batch

Use `ClobberPolicy.SKIP` to skip jobs whose output directory already exists.
Re-run the same script after a walltime kill and only unfinished jobs are
submitted:

```python
from batch import ClobberPolicy

ctrl = BatchController(
    jobs=jobs,
    executor=executor,
    experiment="my_ensemble",
    clobber=ClobberPolicy.SKIP,
)
```

---

## Per-job compilation

Override `build()` when a job requires compiling the executable before
submission.  The no-op default is used when all jobs share a pre-built binary.

```python
import shutil
import subprocess
from batch import Job, JobPaths

class CompiledJob(Job):
    def __init__(self, source_path):
        super().__init__()
        self.source_path = source_path

    def build(self, paths: JobPaths) -> None:
        subprocess.run(["make", ".exe"], cwd=self.source_path, check=True)
        shutil.move(self.source_path / self.executable, paths.job)
        self.executable = paths.job / self.executable
```

The controller calls `job.build(paths)` after writing data files and before
calling `executor.submit()`.  For SLURM this means compilation happens on
the login node, which is the correct behavior.

---

## Per-job postprocessing

Override `post_run(result)` to run plotting, data conversion, or any other
work immediately after a job completes successfully.  The default is a no-op.
`post_run` receives a `JobResult` giving access to `result.paths` and
`result.returncode`.  For `ParallelExecutor` it fires as each job is
harvested in `_drain`, so postprocessing for a finished job runs concurrently
with jobs still in flight.  For `SLURMExecutor` it fires as each job leaves
the queue in `wait_all`.  Exceptions raised inside `post_run` are logged and
swallowed — a failing postprocessing step never aborts the batch loop.

Use `plot_job` for the common case of running plotclaw after a job completes.
It runs plotclaw as a subprocess so all output — including C-level output from
matplotlib — is captured to the job's log file rather than the terminal; a
`--- plotclaw ---` separator is written to the log between solver and plotting
output.  It resolves relative setplot paths against the job directory and
returns `False` gracefully when visclaw is not installed rather than raising.

```python
from pathlib import Path
from batch import Job
from batch import plot_job

class MyJob(Job):
    def post_run(self, result) -> None:
        plot_job(result, setplot=Path(__file__).parent / "setplot.py")
```

For cross-run analysis, use the `results` list returned by `ctrl.run()`.
Each element is a `JobResult` with `.success`, `.paths`, and `.job`; filter to
`r.success` and iterate to load output files, compute statistics, or produce
comparison plots spanning the full ensemble.  This is the right place for
anything that needs data from more than one job at once.

```python
results = ctrl.run(wait=True)
successful = [r for r in results if r.success]
# load fort.gauge, compute metrics, write ensemble_comparison.png …
```

---

## Environment variables

| Variable | Effect |
|---|---|
| `OUTPUT_PATH` | Base directory for all job output (default: cwd) |
| `BATCH_MAX_JOBS` | Default `max_workers` for `ParallelExecutor` (default: 4) |
| `OMP_NUM_THREADS` | Number of threads to allow OpenMP (default: environment variable or 1). |

**Note:** The value `BATCH_MAX_JOBS` x `OMP_NUM_THREADS` should not exceed the physical core count or you may run into swapping/contention problems.  For example, a 16-core machine one could do `BATCH_MAX_JOBS = 2` and `OMP_NUM_THREADS=8`.  For the `SLURMExecutor` or any other HPC environment this will not be an issue and the maximum number of cores available should be used. 
---

## Examples

- [`examples/local_ensemble/`](examples/local_ensemble/) — Manning's n
  sensitivity sweep run locally with `ParallelExecutor`.
- [`examples/storm_surge/`](examples/storm_surge/) — 100-member storm
  ensemble submitted to SLURM.

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite has no dependency on an installed Clawpack or a running
scheduler.  All executor and scheduler behavior is tested via mocks.

Integration tests that exercise the actual solver are marked
`@pytest.mark.integration` and are skipped by default.

---

## License

MIT — see [LICENSE](LICENSE).
