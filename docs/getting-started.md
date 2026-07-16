# Getting started

This guide walks through defining a single job and running it locally. By the
end you'll have run a batch and know where to find its output.

← Back to the [documentation index](index.md).

---

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10. Clawpack must be importable at runtime but is **not**
listed as a hard dependency — `batch` assumes it is already present in your
environment. Nothing else is required to run locally.

---

## 1. Define a job

A job is a subclass of `Job`. At minimum you must set two attributes in
`__init__`:

- `self.prefix` — a unique string used to name the job's output directory.
- `self.rundata` — a Clawpack `ClawRunData` object, usually produced by calling
  `setrun()` from a `setrun.py`.

```python
import importlib.util
from pathlib import Path

from batch import Job


class ManningJob(Job):
    """One GeoClaw run with a specific uniform Manning's n coefficient."""

    def __init__(self, manning: float, max_level: int = 5,
                 setrun_path: Path | None = None) -> None:
        super().__init__()

        self.manning = manning
        self.prefix = f"n{manning:.3f}_l{max_level}"   # names the output directory
        self.executable = "xgeoclaw"                   # the compiled solver binary

        # Load the base configuration from a setrun.py …
        if setrun_path is None:
            setrun_path = Path(__file__).parent / "setrun.py"
        spec = importlib.util.spec_from_file_location("setrun", setrun_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.rundata = mod.setrun()

        # … then apply the swept parameter overrides.
        self.rundata.geo_data.manning_coefficient = manning
        self.rundata.amrdata.amr_levels_max = max_level
```

A few attributes are available on every `Job` (all optional except `prefix` and
`rundata`):

| Attribute | Default | Meaning |
|---|---|---|
| `prefix` | `None` | Output directory name. **Must** be set before submission. |
| `executable` | `"xgeoclaw"` | Solver binary. A bare name is resolved against the job directory after `build()`; an absolute path is used as-is. |
| `setplot` | `"setplot"` | Module/path passed to plotclaw if plotting is requested. |
| `restart` | `False` | If `True`, existing `.data` files are not clobbered and the restart flag is passed to `runclaw`. |
| `rundata` | `None` | The `ClawRunData` object. Must be set by your subclass. |

> This example is written so it imports without Clawpack installed — the
> `setrun()` call is what actually needs Clawpack, and that only happens when you
> construct the job. See `examples/local_ensemble/manning_job.py` for the full
> version.

---

## 2. Run locally

Hand a list of jobs to a `BatchController` together with an executor, then call
`run()`. `ParallelExecutor` runs several jobs at once as separate subprocesses.

```python
from batch import BatchController, ParallelExecutor

jobs = [ManningJob(manning=n) for n in (0.020, 0.025, 0.030)]

ctrl = BatchController(
    jobs=jobs,
    executor=ParallelExecutor(max_workers=3),
    experiment="manning_sensitivity",   # groups all jobs under one subdirectory
)
results = ctrl.run()                    # wait=True by default: blocks until done

for r in results:
    status = "ok" if r.success else f"FAILED (rc={r.returncode})"
    print(f"  {r.job.prefix}  {status}  ->  {r.paths.job}")
```

`BatchController` takes:

| Argument | Default | Meaning |
|---|---|---|
| `jobs` | `[]` | The jobs to run. |
| `executor` | `ParallelExecutor()` | The backend. See [which executor?](index.md#which-executor-do-i-want). |
| `base_path` | `$OUTPUT_PATH`, else cwd | Root output directory. |
| `experiment` | `""` | Subdirectory grouping this batch (e.g. `"hurricane_ike"`). Empty writes directly under `base_path`. |
| `clobber` | `ClobberPolicy.OVERWRITE` | What to do when a job directory already exists. See [parameter sweeps](parameter-sweeps.md#resuming-a-partial-batch). |

`run(wait=True)` blocks until every job finishes. For the scheduler backends you
can pass `wait=False` to return right after submission — see
[Running on HPC](running-on-hpc.md).

If you only want to stage the `.data` files without running the solver, call
`ctrl.setup()` instead of `run()` — it returns the `JobPaths` it wrote.

---

## 3. Find your output

Everything for one job lives under `base_path/experiment/prefix/`. Data files,
solver output (`fort.*`), and the per-job log all share that directory; only
plots get a subdirectory.

```
OUTPUT_PATH/
  manning_sensitivity/          ← experiment
    n0.020_l5/                  ← prefix (one directory per job)
      n0.020_l5_log.txt         ← per-job log (solver + plotting output)
      *.data                    ← generated Clawpack data files
      fort.*                    ← solver output
      plots/                    ← VisClaw frames (if plotted)
    n0.025_l5/
      ...
```

Set the root with the `OUTPUT_PATH` environment variable, or pass `base_path=`
explicitly:

```bash
OUTPUT_PATH=/scratch/myproject python run_batch.py
```

---

## 4. Read the results

`run()` returns a list of `JobResult`, one per submitted job (skipped jobs are
omitted). Each carries:

- `r.job` — the job that was submitted,
- `r.paths` — its `JobPaths` (`.job`, `.plots`, `.log`),
- `r.returncode` — the process exit code (`None` until known),
- `r.job_id` — the scheduler job ID (`None` for local executors),
- `r.success` — `True` only when `returncode` is known and zero,
- `r.pending` — `True` for scheduler jobs whose result isn't known yet.

```python
successful = [r for r in results if r.success]
failed = [r for r in results if r.returncode not in (None, 0)]

for r in failed:
    print(f"  {r.job.prefix} failed — see {r.paths.log}")
```

---

## Next steps

- Generate many jobs from a grid: [Parameter sweeps](parameter-sweeps.md).
- Run on a cluster: [Running on HPC](running-on-hpc.md).
- Plot each job and analyze the ensemble: [Post-processing & analysis](postprocessing.md).

A complete, runnable version of this example lives in
`examples/local_ensemble/`.
