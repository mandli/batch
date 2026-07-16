# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `batch.analysis`: `parse_timing(job_dir)` reads a GeoClaw run's `timing.txt`
  into a structured dict (per-level, per-component, and total wall/cpu, plus the
  OpenMP thread count); `plot_performance(job_dirs, labels, out_path)` builds a
  three-panel wall-time / CPU-efficiency comparison across runs.  `parse_timing`
  is standard-library only; `plot_performance` needs the optional `[analysis]`
  extra (matplotlib + numpy) and degrades gracefully when it is absent.
  Replaces the `NotImplementedError` stub.
- `batch.cli`: shared command-line surface for driver scripts, so projects stop
  copy-pasting the scheduler/resource argparse block and the arg→executor glue.
  `add_execution_args(parser)` contributes the generic flag group;
  `executor_from_args(args)` / `clobber_from_args(args)` map args to objects;
  `report_results(results)` prints and returns a `ResultSummary`; and
  `execute(args, jobs, ...)` is a thin dispatch over local / scheduler / packed
  submission.  Projects keep their own `main()` and domain flags.
- `batch.packed`: fan a parameter sweep across several exclusive nodes, packing
  many runs onto each via the local pool with CPU pinning.  `submit_packed()`
  renders one per-node wrapper (PBS or SLURM) and submits it; the pure
  `render_packed_pbs_wrapper()` / `render_packed_slurm_wrapper()` functions are
  unit-testable without a cluster.  `PackedResources` is the scheduler-agnostic
  per-node resource request.
- `batch.sweep.shard_jobs(jobs, i, n)` and `parse_shard_spec("i/n")`: split a job
  list into disjoint round-robin shards, one per packed node.

## [2.0.0] — breaking API change

### Added
- SLURM compute-node self-plotting parity: `SLURMResources.plot` / `setplot`
  now append a `plotclaw` call to the generated script, matching
  `PBSResources.plot`.
- `PBSExecutor`, `PBSResources`, and `render_pbs_script()`: PBS Pro backend
  (NCAR Derecho `qsub`/`qstat`), mirroring the SLURM backend.  Submits via
  `qsub`, captures the job ID, polls `qstat` in `wait_all`.  Per-job override
  by attaching `job.pbs_resources`.  `PBSResources.plot=True` appends a
  `plotclaw` call so jobs self-plot on the compute node.  `render_pbs_script`
  is a pure function, unit-tested without a cluster.
- `Job.post_run(result)`: hook called after successful job completion.
  No-op default; override for per-job plotting or data conversion.
  Fires concurrently with remaining jobs in ParallelExecutor.
  Exceptions in post_run are logged and do not abort the batch.
- `batch.plot.plot_job`: runs plotclaw as a subprocess, capturing all
  output (including C-level I/O) to the job log file. Callable setplot
  falls back to in-process with a logged warning.
- `JobPaths` dataclass: typed, named filesystem layout replacing the `dict`
  returned by the old `run()`.
- `JobResult` dataclass: carries `job`, `paths`, `returncode`, and `job_id`.
- `ClobberPolicy` enum: `OVERWRITE` (default), `ERROR`, `SKIP`.
  `SKIP` gives free batch resumability — re-run the same script and only
  unfinished jobs are submitted.
- `BatchController.experiment` attribute: replaces the `job.type` / `job.name`
  two-level grouping with a single experiment subdirectory set once on the
  controller.
- `BatchController.setup()`: writes `.data` files without running the solver.
  Replaces the `run(only_write_data=True)` flag.
- `Job.build(paths)`: hook for per-job compilation before submission.
  No-op default; override for jobs that compile Fortran source.
- `Executor` protocol: defines `submit()` and `wait_all()`.  New schedulers
  are added by implementing this protocol rather than subclassing
  `BatchController`.
- `SerialExecutor`: sequential local runner.
- `ParallelExecutor`: concurrent local runner replacing the hand-rolled process
  queue.  Fixes the modify-list-while-iterating bug and propagates
  `returncode`.
- `SLURMExecutor`: submits via `sbatch --parsable`, captures job ID,
  polls `squeue` in `wait_all`.  Replaces `StampedeBatchController`.
- `SLURMResources` dataclass: typed SLURM resource request.  Per-job override
  by attaching `job.slurm_resources`.
- `render_slurm_script()`: pure function for SLURM script generation —
  independently testable without a cluster.
- `batch.sweep.product_sweep()`, `zip_sweep()`: build job lists from parameter
  grids.
- `pyproject.toml` (PEP 517/518, hatchling backend).
- pytest test suite covering all public components without requiring a Clawpack
  installation or a running scheduler.

### Changed
- `Job.write_data_objects()` now accepts an explicit `path: Path` argument and
  calls `rundata.write(out_dir=path)`.  The `os.chdir` pattern is eliminated.
- `Job.restart` is now a first-class attribute on `Job`, not accessed through
  `job.rundata.clawdata.restart` in the controller.
- `BatchController.run()` now defaults to `wait=True` (blocking).  The old
  default of `wait=False` silently killed background subprocesses when the
  calling script exited.
- `max_processes` no longer defaults from `$OMP_NUM_THREADS`.  Use
  `$BATCH_MAX_JOBS` or pass `max_workers` explicitly to `ParallelExecutor`.
- Flattened directory layout: data files, solver output, and log all share one
  directory (`OUTPUT_PATH/experiment/prefix/`).  Only plots get a subdirectory.
- `OUTPUT_PATH` is the only environment-variable default for output location.
  `DATA_PATH` is no longer used.
- All `subprocess` calls use explicit argument lists (`shell=False`).

### Removed
- `Job.type`, `Job.name` (replaced by `BatchController.experiment`).
- `Job.output_path`, `Job.data_path`, `Job.log_path` (dead attributes).
- `BatchController.parallel`, `BatchController.terminal_output`,
  `BatchController.runclaw_cmd`, `BatchController.plotclaw_cmd`,
  `BatchController.max_processes`, `BatchController.poll_interval` — all
  moved into the executor or removed.
- `StampedeBatchController` and `StampedeJob` — superseded by `SLURMExecutor`
  and `SLURMResources`.
- `from __future__ import` statements.
- Python 2 `super(ClassName, self)` style.

### Fixed
- Modify-list-while-iterating in the parallel process drain loop caused every
  other completed process to be silently skipped.
- Log file handle was never closed in parallel mode.
- `OMP_NUM_THREADS` was incorrectly used as the number of parallel jobs.
- `#SBATCH -t` in `StampedeBatchController` was hardcoded to `9:00:00`,
  ignoring `job.time`.
- Missing `\n` in the Stampede MIC environment export line.

Tagged as v2.  The v1 API (original `batch.py`, `stampede.py`) is preserved
on the `v1.0.0` tag.  Claude-Code assisted with this rewrite.

## [1.0.0]

Original implementation.  Tagged for historical reference.
See `batch.py` and `stampede.py` on the `v1.0.0` tag.
