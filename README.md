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

| Class | Role |
|---|---|
| `Job` | Describes one simulation: `prefix`, `executable`, `rundata`, optional `build()` override |
| `BatchController` | Orchestrates directory setup, data writing, and dispatch |
| `Executor` | Protocol implemented by `SerialExecutor`, `ParallelExecutor`, `SLURMExecutor` |
| `JobPaths` | Typed paths for one job's directory, plots, and log |
| `JobResult` | Return value from `run()`: job, paths, returncode, scheduler job ID |
| `ClobberPolicy` | Controls what happens when output already exists: `OVERWRITE`, `ERROR`, `SKIP` |

---

## Quick start

### 1. Define a job

Subclass `Job`, populate `rundata`, and set `prefix`:

```python
from pathlib import Path
import importlib.util
from batch import Job

class MyGeoClawJob(Job):
    def __init__(self, manning: float) -> None:
        super().__init__()
        self.prefix = f"n{manning:.3f}"
        self.executable = "xgeoclaw"
        self.manning = manning

        # Load base configuration from a local setrun.py
        spec = importlib.util.spec_from_file_location("setrun", "setrun.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rundata = mod.setrun()

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

### 4. Inspect scripts without submitting

```python
executor = SLURMExecutor(default_resources=resources, dry_run=True)
ctrl = BatchController(jobs=jobs, executor=executor, experiment="test")
ctrl.run(wait=False)
# Scripts written to each job directory; sbatch not called
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

## Environment variables

| Variable | Effect |
|---|---|
| `OUTPUT_PATH` | Base directory for all job output (default: cwd) |
| `BATCH_MAX_JOBS` | Default `max_workers` for `ParallelExecutor` (default: 4) |

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
