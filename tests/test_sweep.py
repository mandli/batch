"""Tests for batch.sweep — product_sweep and zip_sweep."""

from __future__ import annotations

import pytest

from batch.job import Job
from batch.sweep import parse_shard_spec, product_sweep, shard_jobs, zip_sweep
from tests.conftest import MockJob


def simple_factory(**params) -> MockJob:
    """Factory that records the params it received."""
    job = MockJob(prefix="to_be_set")
    job._params = params
    return job


def simple_namer(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


# ---------------------------------------------------------------------------
# product_sweep
# ---------------------------------------------------------------------------


class TestProductSweep:
    def test_cartesian_product_count(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=simple_namer,
            manning=[0.020, 0.025, 0.030],
            level=[4, 5],
        )
        assert len(jobs) == 6  # 3 × 2

    def test_single_parameter_list(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=simple_namer,
            manning=[0.020, 0.025],
        )
        assert len(jobs) == 2

    def test_prefix_set_by_namer(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=lambda p: f"n{p['manning']:.3f}",
            manning=[0.020, 0.025],
        )
        assert jobs[0].prefix == "n0.020"
        assert jobs[1].prefix == "n0.025"

    def test_all_combinations_present(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=simple_namer,
            a=[1, 2],
            b=["x", "y"],
        )
        prefixes = {j.prefix for j in jobs}
        assert prefixes == {"a1_bx", "a1_by", "a2_bx", "a2_by"}

    def test_all_returned_objects_are_jobs(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=simple_namer,
            x=[1, 2, 3],
        )
        assert all(isinstance(j, Job) for j in jobs)

    def test_empty_param_grid_returns_one_job(self):
        # product of zero iterators is one empty combination
        jobs = product_sweep(
            factory=simple_factory,
            namer=lambda p: "only",
        )
        assert len(jobs) == 1
        assert jobs[0].prefix == "only"

    def test_factory_receives_correct_params(self):
        jobs = product_sweep(
            factory=simple_factory,
            namer=simple_namer,
            manning=[0.020],
            level=[4],
        )
        assert jobs[0]._params == {"manning": 0.020, "level": 4}


# ---------------------------------------------------------------------------
# zip_sweep
# ---------------------------------------------------------------------------


class TestZipSweep:
    def test_paired_count(self):
        jobs = zip_sweep(
            factory=simple_factory,
            namer=simple_namer,
            storm_id=[1, 2, 3],
            intensity=["low", "mid", "high"],
        )
        assert len(jobs) == 3

    def test_prefix_set_by_namer(self):
        jobs = zip_sweep(
            factory=simple_factory,
            namer=lambda p: f"{p['storm_id']}_{p['intensity']}",
            storm_id=[1, 2],
            intensity=["low", "high"],
        )
        assert jobs[0].prefix == "1_low"
        assert jobs[1].prefix == "2_high"

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            zip_sweep(
                factory=simple_factory,
                namer=simple_namer,
                a=[1, 2, 3],
                b=["x", "y"],  # length mismatch
            )

    def test_single_parameter_list(self):
        jobs = zip_sweep(
            factory=simple_factory,
            namer=lambda p: str(p["x"]),
            x=[10, 20, 30],
        )
        assert len(jobs) == 3
        assert [j.prefix for j in jobs] == ["10", "20", "30"]

    def test_factory_receives_paired_params(self):
        jobs = zip_sweep(
            factory=simple_factory,
            namer=simple_namer,
            a=[1, 2],
            b=["x", "y"],
        )
        assert jobs[0]._params == {"a": 1, "b": "x"}
        assert jobs[1]._params == {"a": 2, "b": "y"}


# ---------------------------------------------------------------------------
# parse_shard_spec
# ---------------------------------------------------------------------------


class TestParseShardSpec:
    def test_empty_is_no_sharding(self):
        assert parse_shard_spec("") == (1, 1)

    def test_valid_spec(self):
        assert parse_shard_spec("3/16") == (3, 16)

    @pytest.mark.parametrize("spec", ["1", "1/2/3", "a/2", "1-2"])
    def test_malformed_raises(self, spec):
        with pytest.raises(ValueError):
            parse_shard_spec(spec)

    @pytest.mark.parametrize("spec", ["0/4", "5/4", "1/0", "-1/4"])
    def test_out_of_range_raises(self, spec):
        with pytest.raises(ValueError):
            parse_shard_spec(spec)


# ---------------------------------------------------------------------------
# shard_jobs
# ---------------------------------------------------------------------------


class TestShardJobs:
    def test_single_shard_returns_all(self):
        jobs = [MockJob(prefix=str(k)) for k in range(10)]
        assert shard_jobs(jobs, 1, 1) == jobs

    def test_shards_are_disjoint_and_cover_all(self):
        jobs = [MockJob(prefix=str(k)) for k in range(10)]
        n = 3
        collected = []
        for i in range(1, n + 1):
            collected.extend(shard_jobs(jobs, i, n))
        # Every job appears exactly once across the shards.
        assert sorted(j.prefix for j in collected) == sorted(j.prefix for j in jobs)
        assert len(collected) == len(jobs)

    def test_round_robin_order(self):
        jobs = [MockJob(prefix=str(k)) for k in range(6)]
        assert [j.prefix for j in shard_jobs(jobs, 1, 3)] == ["0", "3"]
        assert [j.prefix for j in shard_jobs(jobs, 2, 3)] == ["1", "4"]
        assert [j.prefix for j in shard_jobs(jobs, 3, 3)] == ["2", "5"]

    def test_more_shards_than_jobs_gives_empty_tail(self):
        jobs = [MockJob(prefix=str(k)) for k in range(2)]
        assert [j.prefix for j in shard_jobs(jobs, 1, 4)] == ["0"]
        assert [j.prefix for j in shard_jobs(jobs, 2, 4)] == ["1"]
        assert shard_jobs(jobs, 3, 4) == []

    @pytest.mark.parametrize("i,n", [(0, 3), (4, 3), (1, 0)])
    def test_out_of_range_raises(self, i, n):
        with pytest.raises(ValueError):
            shard_jobs([], i, n)
