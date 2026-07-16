# Post-processing & analysis

There are two distinct kinds of post-processing, and `batch` handles them in
different places:

- **Per-job** work (plot this run, convert its output) → the `post_run` hook,
  which fires as each job completes.
- **Cross-run** work (compare all members, compute ensemble statistics) → the
  `results` list returned by `ctrl.run()`, after the batch is done.

← Back to the [documentation index](index.md).

---

## Per-job post-processing: the `post_run` hook

Override `Job.post_run(result)` to run plotting, data conversion, or any other
work immediately after a job completes **successfully**. The default is a no-op.

```python
from pathlib import Path
from batch import Job, plot_job


class MyJob(Job):
    def post_run(self, result) -> None:
        plot_job(result, setplot=Path(__file__).parent / "setplot.py")
```

`post_run` receives the job's `JobResult`, giving access to `result.paths` and
`result.returncode`. Its timing depends on the executor:

- **`ParallelExecutor`** — fires as each job is harvested, so post-processing for
  a finished job runs *concurrently* with jobs still in flight.
- **`SerialExecutor`** — fires right after each job returns.
- **`SLURMExecutor` / `PBSExecutor`** — fires as each job leaves the queue during
  `wait_all` (i.e. only when you call `run(wait=True)`).

> **It only runs on success.** `post_run` is called only when `result.success`
> is `True`. Exceptions raised inside it are logged and swallowed — a failing
> post-processing step never aborts the rest of the batch.

---

## `plot_job`: run plotclaw after a job

`plot_job` is the ready-made tool for the most common `post_run` task — running
plotclaw. It runs plotclaw as a **subprocess** so all output, including C-level
output from matplotlib, is captured to the job's log file rather than your
terminal; a `--- plotclaw ---` separator is written to the log between solver
and plotting output.

```python
plot_job(result, setplot="setplot.py", format="ascii", verbose=False) -> bool
```

- `setplot` — a file path (str or `Path`) or a callable. A relative string is
  resolved against the job directory if that file exists; a `Path` is resolved to
  an absolute path. A **callable** can't cross the subprocess boundary, so it
  triggers an in-process fallback with a logged warning (output not captured to
  the log).
- `format` — Clawpack output format passed to plotclaw (default `"ascii"`).
- Returns `True` on success, `False` on failure — including when
  `clawpack.visclaw` is not importable, in which case it logs a warning and
  returns `False` rather than raising. This means a missing VisClaw never crashes
  your batch.

On a cluster you can instead plot **on the compute node** by setting
`PBSResources.plot=True` — see
[the HPC guide](running-on-hpc.md#self-plotting-on-the-compute-node). That avoids
a long-lived login-node plotting process for large ensembles.

---

## Cross-run analysis: the `results` list

Anything that needs data from more than one job at once belongs *after* the
batch, not in `post_run`. Use the `results` list from `ctrl.run(wait=True)`:
filter to the successful jobs and iterate to load output files, compute metrics,
or produce comparison plots spanning the whole ensemble.

For a quick pass/fail tally first, `batch.report_results(results)` prints a
`Completed: X/Y successful, Z failed` summary (with the log path of each failure)
and returns a `ResultSummary` you can branch on — see
[Driving a batch from the command line](running-on-hpc.md#driving-a-batch-from-the-command-line).

```python
import matplotlib.pyplot as plt
import numpy as np

results = ctrl.run(wait=True)
successful = [r for r in results if r.success]

fig, ax = plt.subplots()
for r in successful:
    gauge_file = r.paths.job / "fort.gauge"
    if not gauge_file.exists():
        continue
    data = np.loadtxt(gauge_file)
    # fort.gauge columns: gauge_num, level, time, q[0], q[1], q[2], eta
    ax.plot(data[:, 2], data[:, 6], label=r.job.prefix)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Surface elevation (m)")
ax.legend()
fig.savefig(successful[0].paths.job.parent / "ensemble_comparison.png")
```

The `plot_ensemble` function in `examples/local_ensemble/run_batch.py` is a
complete version of this pattern.

> **Note.** The `batch.analysis` module is a placeholder and currently raises
> `NotImplementedError` — there is no built-in cross-run analysis API yet.
> Cross-run analysis is done manually via the `results` list as shown above.

---

## Next steps

- Choose an executor and understand when `post_run` fires: [Running on HPC](running-on-hpc.md).
- Compile per job or add a custom backend: [Extending batch](extending.md).
- Plotting produced nothing? [Troubleshooting](troubleshooting.md).
