# Troubleshooting

Common failure modes and how to resolve them.

← Back to the [documentation index](index.md).

---

## Environment variables

`batch` reads these environment variables:

| Variable | Effect |
|---|---|
| `OUTPUT_PATH` | Base directory for all job output. Defaults to the current working directory. Overridden by an explicit `base_path=` on `BatchController`. |
| `BATCH_MAX_JOBS` | Default `max_workers` for `ParallelExecutor`. Defaults to 4. |
| `OMP_NUM_THREADS` | Threads each OpenMP solver process uses. Set it via the executor's `env=` (local) or `env_vars=` (scheduler). |

The `batch.cli` helpers add three more, used only as defaults for their
matching flags (any of which you can still pass explicitly):

| Variable | Flag default |
|---|---|
| `BATCH_ACCOUNT` | `--account` — scheduler allocation/project code (PBS or SLURM). |
| `BATCH_MODULES` | `--modules` — space-separated modules to `module load`. |
| `BATCH_MAX_JOBS` | `--max-workers` (as above). |

---

## Local runs are thrashing / swapping

The number of cores in use locally is roughly
`BATCH_MAX_JOBS × OMP_NUM_THREADS` — one factor is how many jobs run at once, the
other is how many threads each job spawns. If the product exceeds your physical
core count, the jobs oversubscribe the CPU and contend, and you may swap.

Pick the two so their product fits the machine. On a 16-core workstation:

```bash
BATCH_MAX_JOBS=2 OMP_NUM_THREADS=8 python run_batch.py   # 2 jobs × 8 threads = 16
```

or set them explicitly in code:

```python
ParallelExecutor(max_workers=2, env={"OMP_NUM_THREADS": "8"})
```

On a scheduler this is not a concern — request whole nodes and use the full core
count (`cpus_per_task` / `ompthreads`).

If you instead pack several smaller jobs onto one exclusive node with
`ParallelExecutor(cpu_affinity=True)`, the same product rule applies per node:
keep `max_workers × OMP_NUM_THREADS ≤ total_cpus`. Pinning binds each job to a
disjoint core range so co-scheduled jobs don't migrate or contend — see
[Packing multiple jobs per node](running-on-hpc.md#packing-multiple-jobs-per-node).

---

## A job was killed by walltime — how do I resume?

Re-run the same script with `ClobberPolicy.SKIP`; jobs whose output directory
already exists are skipped and only the rest are (re)submitted. See
[Resuming a partial batch](parameter-sweeps.md#resuming-a-partial-batch).

Because `SKIP` keys on directory *existence* rather than solver success, delete
(or otherwise exclude) the directories of jobs you know were interrupted before
resuming, or gate resumption on a solver-produced sentinel file.

---

## Plots didn't appear

- **`clawpack.visclaw` is not installed.** `plot_job` logs a warning and returns
  `False` instead of raising — check the return value and the job log. Install
  VisClaw to enable plotting.
- **Look in the log, not the terminal.** `plot_job` runs plotclaw as a subprocess
  and captures its output (including matplotlib C-level output) to the job log,
  after a `--- plotclaw ---` separator. Errors show up there.
- **A callable `setplot`.** Passing a callable triggers an in-process fallback
  (output is *not* captured to the log). Prefer a file path so output is logged
  and the compute-node path works.
- **No display on a compute node.** Use a non-interactive matplotlib backend in
  your driver: `import matplotlib; matplotlib.use("Agg")` before importing
  `pyplot`. See `examples/local_ensemble/run_batch.py`.

---

## `ModuleNotFoundError` / Clawpack not importable

`batch` does not depend on Clawpack — it assumes Clawpack is importable in your
environment at runtime. Jobs invoke the solver via
`python -m clawpack.clawutil.runclaw`, and plotting via
`python -m clawpack.visclaw.plotclaw`, using the **same** interpreter
(`sys.executable`) that launched `batch`. If those modules aren't found, activate
the environment that has Clawpack before running, and on a scheduler make sure
the job script loads the right modules / conda environment (via
`modules=` / `env_vars=` on the resource object).

---

## `FileExistsError` on an existing job directory

You're running with `ClobberPolicy.ERROR` (the deliberate hard guard) against a
directory that already exists. Switch to `OVERWRITE` to re-run in place, or
`SKIP` to resume a partial batch. See
[the clobber policies](parameter-sweeps.md#resuming-a-partial-batch).

---

## A scheduler job reports success but the run clearly failed

When `run(wait=True)`, the SLURM and PBS executors mark a job complete with
`returncode=0` once it *leaves the queue* — they do not inspect the solver's real
exit status, because schedulers purge finished jobs from `squeue` / `qstat`.
So `r.success` on a scheduler job means "no longer queued," not "exited 0."
Confirm real success by checking the job log and `fort.*` output, or by relying
on a solver sentinel file. See the
[completion-detection note](running-on-hpc.md#waittrue-vs-waitfalse-and-monitoring).

---

## `ValueError: All parameter lists must have the same length`

`zip_sweep` requires every parameter list to be the same length. Either pad the
lists so they match, or switch to `product_sweep` if you actually want every
combination. See [Parameter sweeps](parameter-sweeps.md).

---

## `ValueError: Job … has no prefix set`

Every job needs a unique `self.prefix` before it reaches the controller — it
becomes the output directory name. Set it in your `Job.__init__`, or let a
[sweep helper](parameter-sweeps.md) set it via `namer`.
