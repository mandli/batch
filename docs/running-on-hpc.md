# Running on HPC

`batch` submits to schedulers by swapping the executor — your `Job` subclass is
unchanged from the [local case](getting-started.md). PBS and SLURM are the *same*
`SchedulerExecutor`, parametrized by an injected **scheduler backend** and a
per-machine **env_file**:

- **`PBSScheduler`** — `qsub` / `qstat`; targets PBS Pro as deployed on NCAR
  Derecho.
- **`SlurmScheduler`** — `sbatch --parsable` / `squeue`.

Each backend owns only what differs between the two schedulers (the directive
header, the `BATCH_*` environment normalization, the submit/poll commands, and
job-id parsing). Everything else — the script body, the resource request, the
packer, the polling loop — is shared. The resource request is a single
normalized `JobRequest`; the emitted script is scheduler-agnostic.

← Back to the [documentation index](index.md).

---

## The env_file (read this first)

The generated job script runs with a **non-login** shell (`#!/bin/zsh`, no `-l`)
and does not read your `.zprofile` / `.bashrc`. Instead it `source`s a
per-machine *env_file* on the compute node. That file is the one
machine-specific artifact `batch` needs; the package carries only its **path**
(`--env-file`, or `$BATCH_ENV_FILE`), never its contents.

**Contract** — after the env_file is sourced, the shell must have (1) the pinned
run modules loaded, (2) the venv python resolvable, and (3) `SRC` / `CLAW` /
`PYTHONPATH` exported so that `python -c "import batch"` exits 0. The job script
runs exactly that import as a fail-fast check before the solver, printing the
hostname and python version if it fails.

An annotated, ready-to-adapt template ships at
[`docs/env_file.example.zsh`](env_file.example.zsh). Copy it **outside** the
package, edit the marked lines for your machine, and verify it in a clean shell:

```bash
env -i HOME="$HOME" bash --noprofile --norc -c \
    'source ~/my_env.zsh; python -c "import batch"'
```

---

## SLURM

```python
from batch import BatchController, SchedulerExecutor, SlurmScheduler, JobRequest

executor = SchedulerExecutor(
    SlurmScheduler(),
    env_file="~/cluster_env.zsh",           # sourced on the compute node
    python="/venv/bin/python",              # absolute venv python for the launch
    default_request=JobRequest(
        name="", log_path="",               # filled in per job at submit time
        queue="main",
        account="MY_ALLOCATION",
        walltime="06:00:00",
        nodes=1,
        cpus_per_node=8,                     # cores reserved → SLURM --cpus-per-task
        tasks_per_node=1,                    # 1 rank for pure-OpenMP GeoClaw
    ),
    modules=["ncarenv/23.09", "python/3.11.4"],
    env_vars={"OMP_NUM_THREADS": "8"},
)

ctrl = BatchController(jobs=jobs, executor=executor, experiment="manning_sensitivity")

# sbatch returns immediately; pass wait=False so run() does too.
results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  SLURM job {r.job_id}")
```

## PBS / Derecho

The only differences from the SLURM example are the injected backend and the
env_file:

```python
from batch import BatchController, SchedulerExecutor, PBSScheduler, JobRequest

executor = SchedulerExecutor(
    PBSScheduler(),
    env_file="~/derecho_env.zsh",
    python="/glade/work/me/venv/bin/python",
    default_request=JobRequest(
        name="", log_path="",
        queue="main",
        account="NCAR0001",                 # your Derecho project code (#PBS -A)
        walltime="12:00:00",
        nodes=1,
        cpus_per_node=128,                  # whole Derecho CPU node
        tasks_per_node=1,                   # pure-OpenMP → 1 MPI rank
        ompthreads=128,                     # PBS ompthreads hint
    ),
    modules=["ncarenv/23.09", "conda"],
    env_vars={"OMP_NUM_THREADS": "128"},
)

ctrl = BatchController(jobs=jobs, executor=executor, experiment="storm_ensemble")

results = ctrl.run(wait=False)
for r in results:
    print(f"  {r.job.prefix}  ->  PBS job {r.job_id}")
```

### The `JobRequest` fields

`JobRequest` is scheduler-neutral and holds **only** the resource directives.
Each backend translates it to `#PBS` / `#SBATCH` lines:

| Field | Default | PBS | SLURM |
|---|---|---|---|
| `name` | — | `-N` | `--job-name` (filled per job) |
| `log_path` | — | `-o` (+ `-j oe`) | `--output` / `--error` (filled per job) |
| `queue` | `""` | `-q` | `--partition` (omitted when empty) |
| `account` | `""` | `-A` | `--account` (omitted when empty) |
| `walltime` | `"12:00:00"` | `-l walltime=` | `--time=` |
| `nodes` | `1` | `select=` count | `--nodes` |
| `cpus_per_node` | `128` | `:ncpus=` | `--cpus-per-task = cpus_per_node // tasks_per_node` |
| `tasks_per_node` | `1` | `:mpiprocs=` | `--ntasks-per-node` |
| `ompthreads` | `1` | `:ompthreads=` | *(PBS-only hint)* |
| `exclusive` | `False` | *(no-op; Derecho main default)* | `--exclusive` |
| `mem` | `""` | `:mem=` chunk suffix | `--mem` |
| `constraint` | `""` | *(ignored)* | `--constraint` |
| `array` | `""` | `-J start-end[:step]` | `--array=start-end[:step]` |
| `depend` | `[]` | `-W depend=afterok:…` | `--dependency=afterok:…` |
| `email` | `""` | `-M` / `-m abe` | `--mail-user` / `--mail-type=END,FAIL` |
| `extra_directives` | `[]` | raw lines appended | raw lines appended |

Module loads (`modules`), per-job environment (`env_vars`, e.g.
`OMP_NUM_THREADS`), and compute-node plotting (`plot` / `setplot`) are
**executor**-level arguments, not part of `JobRequest`.

### Self-plotting on the compute node

Pass `plot=True` (with a `setplot` path) to `SchedulerExecutor` and `batch`
appends a `plotclaw` call after the solver, so each job produces its VisClaw
frames on the compute node — avoiding a long-lived login-node plotting process.
If `setplot` is empty it falls back to `job.setplot`.

```python
SchedulerExecutor(PBSScheduler(), env_file=..., plot=True, setplot="setplot.py")
```

A complete PBS driver lives in `examples/derecho_ensemble/pbs_batch.py`; the same
driver runs on SLURM by passing `--scheduler slurm` and a SLURM env_file.

---

## Packing multiple jobs per node

A single GeoClaw run rarely saturates a modern HPC node (a Derecho CPU node has
128 cores). When each job needs fewer threads than a whole node, you can *pack*
several jobs onto one node instead of burning a node per job. `ParallelExecutor`
does this directly with `cpu_affinity=True`:

```python
from batch import ParallelExecutor

# One exclusive 128-core node: 16 jobs at once, 8 OpenMP threads each.
executor = ParallelExecutor(
    max_workers=16,
    env={"OMP_NUM_THREADS": "8"},
    cpu_affinity=True,          # pin each job to a disjoint core range
    total_cpus=128,             # cores to partition (defaults to cores granted)
)
```

With `cpu_affinity=True` the pool splits `total_cpus` into `max_workers` equal
ranges and binds each running job to one via `numactl --physcpubind=<range>
--localalloc`, also exporting `OMP_PROC_BIND=close` / `OMP_PLACES=cores`. This
keeps co-scheduled OpenMP jobs from migrating across sockets or contending for
the same cores.

- Keep `max_workers × OMP_NUM_THREADS ≤ total_cpus`, or the pinned ranges
  oversubscribe (the executor logs a warning when threads exceed cores per slot).
- `total_cpus` defaults to the cores actually granted to the process
  (`sched_getaffinity`), which is the full node under an exclusive allocation, so
  you can usually omit it on a compute node.
- Requires `numactl` on `PATH` (Linux/HPC). Leave `cpu_affinity=False` on
  machines without it (e.g. macOS).

### Fanning a sweep across several packed nodes

To pack on more than one node at once, split the job list into shards — one node
per shard — and submit a wrapper per node. `batch.submit_packed` renders and
submits those wrappers through the same scheduler backend and env_file body used
above; each wrapper re-invokes *your own driver* on the compute node in local
mode over its shard. (It has to re-run your driver because your `Job`-definition
code — `build` / `post_run` — must be present on the node; `batch` supplies the
outer packing layer, not your job definitions.)

```python
import sys
from pathlib import Path
from batch import submit_packed, PackedResources, shard_jobs

# In your driver's local path, select this node's shard of the full sweep:
#   i, n = parse_shard_spec(args.shard)      # from batch.sweep
#   jobs = shard_jobs(all_jobs, i, n)
# and run them through ParallelExecutor(cpu_affinity=True) as above.

def inner(shard_i, n_shards):
    """Per-node command: re-run this driver in local mode over one shard."""
    return [
        sys.executable, __file__,
        "--scheduler", "local",
        "--shard", f"{shard_i}/{n_shards}",
        "--pin-cpus",
    ]

submit_packed(
    n_nodes=4,
    inner_command=inner,
    resources=PackedResources(
        queue="main", walltime="12:00:00", account="NCAR0001",
        node_cpus=128, modules=["ncarenv/23.09", "conda"],
    ),
    scheduler="pbs",                    # or "slurm"
    script_dir=Path("_pack_scripts"),
    env_file="~/derecho_env.zsh",       # sourced by each node's wrapper
    python=sys.executable,              # absolute venv python for the import check
    dry_run=False,                      # True writes wrappers without submitting
    workdir=Path(__file__).parent,
)
```

`submit_packed` returns the submitted job IDs (or, under `dry_run=True`, the paths
of the wrapper scripts it wrote so you can inspect the directives first). It is
submit-and-exit — poll the queue yourself and build any cross-run figures once
the jobs finish. `shard_jobs(jobs, i, n)` and `parse_shard_spec("i/n")` (both in
`batch.sweep`) give each node a disjoint, balanced slice of the sweep.

---

## Driving a batch from the command line

Every scheduler driver ends up re-implementing the same `--scheduler` /
`--account` / `--resume` / `--max-workers` argparse block and the same
arg→executor glue. `batch.cli` factors that out **without** owning your `main()`:
you keep your own parser and job list, and add only your domain flags.

```python
import argparse, sys
from batch import add_execution_args, execute

parser = argparse.ArgumentParser()
add_execution_args(parser)                       # shared scheduler/resource flags
parser.add_argument("--storms-path", ...)        # your domain flags
args = parser.parse_args()

jobs = make_jobs(args)                            # you build the job list

def inner(i, n):                                  # only needed for *-packed
    return [sys.executable, __file__, "--scheduler", "local",
            "--shard", f"{i}/{n}", "--pin-cpus"]

results = execute(args, jobs, experiment="storm_ensemble", inner_command=inner)
```

`add_execution_args` includes `--env-file` (default `$BATCH_ENV_FILE`) and
`--python` (default the submitting interpreter). `--env-file` is **required** for
the `pbs` / `slurm` / `*-packed` backends — they raise if it is missing, since
the job sources it on the compute node.

`execute` dispatches on `--scheduler`:

- `local` → `ParallelExecutor` (blocks); `--setup-only` writes `.data` only.
- `pbs` / `slurm` → a `SchedulerExecutor` with the matching backend,
  submit-and-exit (`wait=False`); `--setup-only` writes the submission scripts
  without submitting.
- `pbs-packed` / `slurm-packed` → `submit_packed` with the per-node
  `inner_command` (required); `--nodes` / `--node-cpus` size the packing, and the
  re-invoked local shard is selected by `--shard`.

For [compute-node self-plotting](#self-plotting-on-the-compute-node) on the
scheduler backends, pass `plot=`/`setplot=` to `execute` (or `executor_from_args`);
they set the executor's `plot` / `setplot`:

```python
execute(args, jobs, experiment="storm_ensemble",
        plot=not args.no_run_plots, setplot="setplot.py")
```

Prefer the switch explicit in your own code? Skip `execute` and use the factories
directly — they are what `execute` is built from:

```python
from batch import (BatchController, executor_from_args, clobber_from_args,
                   report_results)

ctrl = BatchController(jobs, executor_from_args(args),
                       experiment="storm_ensemble", clobber=clobber_from_args(args))
report_results(ctrl.run(wait=args.scheduler == "local"))
```

`report_results` prints a `Completed: X/Y successful, Z failed` summary (or, for a
submit-and-exit run, the submitted job IDs) and returns a `ResultSummary`
(`n_ok` / `n_failed` / `n_pending` / `failures`) so you can branch on the outcome.

---

## Per-job resource overrides (no subclassing)

`SchedulerExecutor` reads the request with `getattr(job, "job_request",
default_request)`, so any job that carries its own `JobRequest` overrides the
executor default — no subclass needed. Leave `name` / `log_path` empty; they are
filled in from the job at submit time:

```python
# One heavier job in an otherwise-uniform batch:
from batch import JobRequest
job.job_request = JobRequest(
    name="", log_path="",
    queue="preempt", walltime="24:00:00",
    nodes=1, cpus_per_node=128, tasks_per_node=1, ompthreads=128,
)
```

Note that a per-job `JobRequest` overrides only the directives; module loads,
`env_vars`, and `plot` stay at the executor level and apply to every job.

---

## Inspect the script without submitting (`dry_run`)

`SchedulerExecutor` accepts `dry_run=True`. The submission script is written to
each job directory as `<prefix>_run.sh`, but `sbatch` / `qsub` is never called.
This is the safe way to check your directives before spending allocation.

```python
executor = SchedulerExecutor(PBSScheduler(), env_file="~/env.zsh",
                             default_request=request, dry_run=True)
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

> **Note on completion detection.** When `wait=True`, `SchedulerExecutor` treats a
> job that has *left the queue* as complete with `returncode=0` — it does not
> inspect the solver's actual exit status (schedulers purge finished jobs). To
> confirm a run truly succeeded, check the job's log and `fort.*` output, or use
> [`ClobberPolicy.SKIP`](parameter-sweeps.md#resuming-a-partial-batch) with a
> solver sentinel file to drive resumption.

---

## Next steps

- Build the job list for a large ensemble: [Parameter sweeps](parameter-sweeps.md).
- Plot on the compute node and analyze afterward: [Post-processing & analysis](postprocessing.md).
- Compile a fresh binary per job: [Extending batch](extending.md#per-job-compilation).
