"""Shared pytest fixtures for the batch test suite.

All fixtures avoid any dependency on an installed Clawpack or a running
scheduler.  A ``MockJob`` with a mock ``rundata`` is the primary test double.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from batch.job import Job, JobPaths


class MockJob(Job):
    """Minimal concrete Job for testing.

    ``write_data_objects`` writes a dummy ``.data`` file so directory-setup
    logic in the controller has real filesystem state to work with.
    ``rundata`` is a MagicMock so attribute access never raises.
    """

    def __init__(self, prefix: str = "job_001") -> None:
        super().__init__()
        self.prefix = prefix
        self.rundata = MagicMock()
        # Track calls for assertion in tests
        self._write_calls: list[Path] = []

    def write_data_objects(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "claw.data").write_text("mock claw data\n")
        self._write_calls.append(path)


@pytest.fixture
def mock_job() -> MockJob:
    return MockJob(prefix="job_001")


@pytest.fixture
def three_jobs() -> list[MockJob]:
    return [MockJob(prefix=f"job_{i:03d}") for i in range(3)]


@pytest.fixture
def job_paths(tmp_path: Path) -> JobPaths:
    job_dir = tmp_path / "job_001"
    job_dir.mkdir()
    return JobPaths(
        job=job_dir,
        plots=job_dir / "plots",
        log=job_dir / "job_001_log.txt",
    )
