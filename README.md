# batch

Utilities for running [Clawpack](https://www.clawpack.org) / GeoClaw batch jobs.

`batch` manages the directory layout, data file generation, and job submission
for parameter sweeps and ensemble simulations. Execution backends are
pluggable: the same job definition runs locally (serial or parallel) or on a
SLURM / PBS cluster without changing any application code — you swap the
*executor*, not the job.

> **v2 breaking change.** The v1 API (`batch.Job`, `batch.BatchController` with
> scheduler-specific subclasses) is preserved on the `v1.0.0` tag. See the
> [CHANGELOG](CHANGELOG.md) for the full migration guide.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Clawpack must be importable at runtime but is not listed
as a hard dependency (it is assumed to be present in the environment).

---

## Quick start

Subclass `Job`, hand a list of jobs to a `BatchController` with an executor, and
call `run()`:

```python
from batch import Job, BatchController, ParallelExecutor


class MyGeoClawJob(Job):
    def __init__(self, manning: float) -> None:
        super().__init__()
        self.prefix = f"n{manning:.3f}"     # names the output directory
        self.executable = "xgeoclaw"
        self.rundata = setrun()             # your Clawpack ClawRunData
        self.rundata.geo_data.manning_coefficient = manning


jobs = [MyGeoClawJob(n) for n in (0.020, 0.025, 0.030)]

ctrl = BatchController(
    jobs=jobs,
    executor=ParallelExecutor(max_workers=3),
    experiment="manning_sensitivity",
)
results = ctrl.run()

for r in results:
    print(f"  {r.job.prefix}  {'ok' if r.success else 'FAILED'}  ->  {r.paths.job}")
```

Output lands under `OUTPUT_PATH/experiment/prefix/`. To run the same jobs on a
cluster, swap `ParallelExecutor` for a `SchedulerExecutor` with a
`SlurmScheduler` or `PBSScheduler` backend and a per-machine `env_file` — the job
definition is unchanged.

---

## Documentation

Task-oriented guides live in [`docs/`](docs/index.md):

- [**Documentation index**](docs/index.md) — core concepts and choosing an executor.
- [Getting started](docs/getting-started.md) — define a job and run it locally.
- [Running on HPC](docs/running-on-hpc.md) — submit to SLURM or PBS / Derecho.
- [Parameter sweeps](docs/parameter-sweeps.md) — build job grids and resume killed batches.
- [Post-processing & analysis](docs/postprocessing.md) — plotting and ensemble analysis.
- [Extending batch](docs/extending.md) — per-job compilation and custom executors.
- [Troubleshooting](docs/troubleshooting.md) — common failures and oversubscription tuning.

---

## Examples

- [`examples/local_ensemble/`](examples/local_ensemble/) — Manning's n
  sensitivity sweep run locally with `ParallelExecutor`.
- [`examples/storm_surge/`](examples/storm_surge/) — 100-member storm ensemble
  submitted to SLURM.
- [`examples/derecho_ensemble/`](examples/derecho_ensemble/) — PBS / Derecho
  submission with the pbs `SchedulerExecutor` backend.

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite has no dependency on an installed Clawpack or a running
scheduler. All executor and scheduler behavior is tested via mocks. Integration
tests that exercise the actual solver are marked `@pytest.mark.integration` and
are skipped by default.

---

## License

MIT — see [LICENSE](LICENSE).
