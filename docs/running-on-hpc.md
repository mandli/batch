# Running on HPC

`batch` submits to schedulers by swapping the executor — your `Job` subclass is
unchanged from the [local case](getting-started.md). Two scheduler backends ship
today:

- **`SLURMExecutor`** — submits via `sbatch --parsable`, polls `squeue`.
- **`PBSExecutor`** — submits via `qsub`, polls `qstat`; targets PBS Pro as
  deployed on NCAR Derecho.

Both share the same shape: a resource dataclass, submit-and-return semantics, a
per-job override mechanism, and a `dry_run` flag.

← Back to the [documentation index](index.md).

---

## SLURM

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

ctrl = BatchController(jobs=jobs, executor=executor, experiment="manning_sensitivity")

# sbatch returns immediately; pass wait=False so run() does too.
results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  SLURM job {r.job_id}")
```

`SLURMResources` maps directly to `#SBATCH` directives:

| Field | Default | `#SBATCH` |
|---|---|---|
| `partition` | `"main"` | `-p` |
| `nodes` | `1` | `-N` |
| `ntasks_per_node` | `1` | `--ntasks-per-node` (1 for pure-OpenMP GeoClaw) |
| `cpus_per_task` | `1` | `--cpus-per-task` (set to `OMP_NUM_THREADS`) |
| `time` | `"01:00:00"` | `-t` (walltime `HH:MM:SS`) |
| `memory` | `""` | `--mem` (empty → partition default) |
| `account` | `""` | `-A` (allocation; usually required) |
| `constraint` | `""` | `--constraint` |
| `modules` | `[]` | `module load <name>` lines |
| `env_vars` | `{}` | `export K=V` lines |
| `email` / `mail_type` | `""` / `"END,FAIL"` | `--mail-user` / `--mail-type` |
| `extra_directives` | `[]` | raw `#SBATCH` lines appended verbatim |

---

## PBS / Derecho

`PBSExecutor` is the PBS analogue of `SLURMExecutor` — same submit-and-return
semantics, same per-job override mechanism, same `dry_run` flag.

```python
from batch import BatchController, PBSExecutor, PBSResources

executor = PBSExecutor(
    default_resources=PBSResources(
        queue="main",
        nodes=1,
        ncpus=128,           # Derecho CPU nodes have 128 cores
        mpiprocs=1,          # pure-OpenMP GeoClaw → 1 MPI rank
        ompthreads=128,
        walltime="12:00:00",
        account="NCAR0001",  # your Derecho project code (#PBS -A)
        env_vars={"OMP_NUM_THREADS": "128"},
        modules=["ncarenv/23.09", "conda"],
    ),
)

ctrl = BatchController(jobs=jobs, executor=executor, experiment="storm_ensemble")

results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  PBS job {r.job_id}")
```

`PBSResources` maps directly to `#PBS` directives. The CPU-related fields combine
into a single `-l select=` chunk (`nodes:ncpus=..:mpiprocs=..:ompthreads=..`):

| Field | Default | `#PBS` |
|---|---|---|
| `queue` | `"main"` | `-q` |
| `nodes` | `1` | `select=` chunk count |
| `ncpus` | `128` | `:ncpus=` (whole Derecho node) |
| `mpiprocs` | `1` | `:mpiprocs=` (1 for pure-OpenMP) |
| `ompthreads` | `128` | `:ompthreads=` (set to `OMP_NUM_THREADS`) |
| `walltime` | `"12:00:00"` | `-l walltime=` |
| `account` | `""` | `-A` (Derecho project code; omitted when empty) |
| `mem` | `""` | appended to the select chunk as `:mem=` |
| `modules` | `[]` | `module load <name>` lines |
| `env_vars` | `{}` | `export K=V` lines |
| `email` / `mail_points` | `""` / `"abe"` | `-M` / `-m` |
| `plot` / `setplot` | `False` / `""` | append a compute-node `plotclaw` call (see below) |
| `extra_directives` | `[]` | raw `#PBS` lines appended verbatim |

### Self-plotting on the compute node

Set `PBSResources.plot=True` (with a `setplot` path) and `batch` appends a
`plotclaw` call to the generated script, so each job produces its VisClaw frames
on the compute node — avoiding a long-lived login-node plotting process. If
`setplot` is left empty it falls back to `job.setplot`.

```python
PBSResources(..., plot=True, setplot="setplot.py")
```

A complete PBS driver lives in `examples/derecho_ensemble/pbs_batch.py`.

---

## Per-job resource overrides (no subclassing)

Both executors read resources with `getattr(job, "<name>", default_resources)`,
so any job that carries its own resource object overrides the executor default —
no subclass needed. Attach `job.slurm_resources` or `job.pbs_resources`:

```python
# One heavier job in an otherwise-uniform batch:
job.slurm_resources = SLURMResources(partition="gpu", time="12:00:00")
# or, on Derecho:
job.pbs_resources = PBSResources(queue="preempt", walltime="24:00:00")
```

This is exactly what the storm-surge example does — each `StormJob` builds its
own `SLURMResources` in `__init__`.

---

## Inspect the script without submitting (`dry_run`)

Both scheduler executors accept `dry_run=True`. The submission script is written
to each job directory as `<prefix>_run.sh`, but `sbatch` / `qsub` is never
called. This is the safe way to check your directives before spending
allocation.

```python
executor = SLURMExecutor(default_resources=resources, dry_run=True)   # or PBSExecutor(...)
ctrl = BatchController(jobs=jobs, executor=executor, experiment="test")
ctrl.run(wait=False)
# Read the generated OUTPUT_PATH/test/<prefix>/<prefix>_run.sh files.
```

Dry-run results carry `job_id="dry-run"` and are skipped by `wait_all`.

---

## `wait=True` vs `wait=False`, and monitoring

- **`wait=False`** — `run()` returns right after all jobs are queued.
  `r.returncode` is `None` and `r.pending` is `True`; the real work continues in
  the scheduler after your script exits. This is the usual mode for a large
  ensemble.
- **`wait=True`** — `run()` blocks, polling the queue (`squeue` / `qstat`) every
  `poll_interval` seconds (default 30 s) until every submitted job leaves the
  queue, then sets `returncode=0` and fires each job's
  [`post_run` hook](postprocessing.md). Use this when a downstream step in the
  same script depends on the jobs finishing.

Monitor a `wait=False` batch yourself from the shell:

```bash
squeue -u $USER      # SLURM
qstat -u $USER       # PBS / Derecho
```

> **Note on completion detection.** When `wait=True`, both executors treat a job
> that has *left the queue* as complete with `returncode=0` — they do not inspect
> the solver's actual exit status (schedulers purge finished jobs). To confirm a
> run truly succeeded, check the job's log and `fort.*` output, or use
> [`ClobberPolicy.SKIP`](parameter-sweeps.md#resuming-a-partial-batch) with a
> solver sentinel file to drive resumption.

---

## Next steps

- Build the job list for a large ensemble: [Parameter sweeps](parameter-sweeps.md).
- Plot on the compute node and analyze afterward: [Post-processing & analysis](postprocessing.md).
- Compile a fresh binary per job: [Extending batch](extending.md#per-job-compilation).
