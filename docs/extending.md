# Extending batch

Two extension points cover most needs beyond a plain data-file-driven run:
compiling a fresh binary per job, and teaching `batch` about a new scheduler.

← Back to the [documentation index](index.md).

---

## Per-job compilation

By default all jobs share one pre-built binary and `Job.build()` is a no-op.
Override it when a parameter is *compiled into* the Fortran source rather than
read from a data file, so each job needs its own executable.

The controller calls `job.build(paths)` after writing data files and **before**
submitting the job. Place the compiled binary at `paths.job / self.executable`,
or set `self.executable` to an absolute path, so the executor can find it.

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

> **Where `build()` runs matters.** For the scheduler backends, `build()` runs
> in your submitting process (typically the login node), not on the compute
> node — which is the correct place to compile before `sbatch` / `qsub`. Keep
> compilation cheap, or gate it so a resumed batch doesn't rebuild everything.

If you only need to write extra auxiliary files (not compile), override
`write_data_objects(path)` instead and call `super().write_data_objects(path)`
first — it writes the standard `.data` files via `rundata.write(out_dir=path)`.

---

## Custom executors

Executors are the pluggable backend. Any object satisfying the `Executor`
protocol works — it's a `typing.Protocol`, so **no inheritance is required**;
you just implement two methods:

```python
class Executor(Protocol):
    def submit(self, job: Job, paths: JobPaths) -> JobResult: ...
    def wait_all(self, results: list[JobResult]) -> list[JobResult]: ...
```

- `submit(job, paths)` — start or queue one job and return a `JobResult`. For a
  blocking backend, run it to completion and set `returncode`. For a queuing
  backend, return immediately with `returncode=None` and a `job_id`.
- `wait_all(results)` — block until everything finishes. Blocking backends make
  this a no-op; queuing backends poll here and set `returncode` as jobs finish.

A minimal blocking executor:

```python
import subprocess
from batch import JobResult
from batch.executors.local import _build_run_args   # builds the runclaw argv


class MyExecutor:
    def submit(self, job, paths) -> JobResult:
        with open(paths.log, "a") as log:
            proc = subprocess.run(_build_run_args(job, paths), stdout=log, stderr=log)
        result = JobResult(job=job, paths=paths, returncode=proc.returncode)
        if result.success:
            job.post_run(result)          # honor the post_run contract
        return result

    def wait_all(self, results):
        return results                     # nothing to wait for
```

Then pass an instance to the controller like any built-in backend:

```python
ctrl = BatchController(jobs=jobs, executor=MyExecutor(), experiment="custom")
```

### What a well-behaved executor should do

Study `batch/executors/local.py`, `slurm.py`, and `pbs.py` as references. To fit
in cleanly, an executor should:

- write solver/stdout to `paths.log`;
- call `job.post_run(result)` for successful jobs, wrapping it in `try/except`
  so a failing hook never aborts the batch (see [post-processing](postprocessing.md));
- honor `job.restart` (via `_build_run_args`, which already encodes it);
- read a per-job resource/config attribute with
  `getattr(job, "<name>", self.default)` if it needs one — that's how the SLURM
  and PBS backends implement [per-job overrides](running-on-hpc.md#per-job-resource-overrides-no-subclassing)
  without subclassing.

For a new scheduler specifically, mirror the SLURM/PBS structure: a resource
dataclass, a **pure** `render_*_script()` function (easy to unit-test without a
cluster), and an executor that submits and polls.

---

## Next steps

- See the hooks fire in context: [Post-processing & analysis](postprocessing.md).
- Compare against the shipped backends in `batch/executors/`.
