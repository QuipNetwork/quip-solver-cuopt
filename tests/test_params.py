"""anneal_time_us resolution: per-job > SetTarget > default; 0 is unset."""

from __future__ import annotations

from quip_solver_core import miner_pb2

from quip_miner_cuopt import DEFAULT_TIME_LIMIT_S
from quip_miner_cuopt.client import resolve_time_limit_s
from quip_miner_cuopt.job import sampling_params


def _ising(*, num_reads: int = 0, anneal_time_us: int = 0) -> miner_pb2.IsingProblem:
    return miner_pb2.IsingProblem(
        num_reads=num_reads,
        anneal_time_us=anneal_time_us,
    )


def _target(*, num_reads: int = 0, anneal_time_us: int = 0) -> miner_pb2.SetTarget:
    return miner_pb2.SetTarget(
        num_reads=num_reads,
        anneal_time_us=anneal_time_us,
    )


def test_per_job_anneal_wins():
    ising = _ising(anneal_time_us=2_500_000)
    target = _target(anneal_time_us=1_000_000)
    num_reads, anneal_us, limit_s = sampling_params(
        ising, target, default_time_limit_s=30.0
    )
    assert anneal_us == 2_500_000
    assert abs(limit_s - 2.5) < 1e-12
    assert num_reads == 1


def test_set_target_anneal_when_job_unset():
    ising = _ising(anneal_time_us=0)
    target = _target(anneal_time_us=4_000_000)
    _nr, anneal_us, limit_s = sampling_params(ising, target, default_time_limit_s=30.0)
    assert anneal_us == 4_000_000
    assert abs(limit_s - 4.0) < 1e-12


def test_default_when_both_unset():
    ising = _ising(anneal_time_us=0)
    target = _target(anneal_time_us=0)
    _nr, anneal_us, limit_s = sampling_params(ising, target, default_time_limit_s=12.0)
    assert anneal_us == 0
    assert abs(limit_s - 12.0) < 1e-12


def test_zero_is_unset_at_every_level():
    # Job 0, target 0, cli default 0 → documented DEFAULT_TIME_LIMIT_S
    ising = _ising(anneal_time_us=0)
    _nr, anneal_us, limit_s = sampling_params(ising, None, default_time_limit_s=0.0)
    assert anneal_us == 0
    assert abs(limit_s - DEFAULT_TIME_LIMIT_S) < 1e-12


def test_num_reads_precedence():
    ising = _ising(num_reads=0)
    target = _target(num_reads=7)
    num_reads, _a, _t = sampling_params(ising, target)
    assert num_reads == 7

    ising2 = _ising(num_reads=3)
    target2 = _target(num_reads=7)
    num_reads2, _a2, _t2 = sampling_params(ising2, target2)
    assert num_reads2 == 3

    ising3 = _ising(num_reads=0)
    num_reads3, _a3, _t3 = sampling_params(ising3, None)
    assert num_reads3 == 1


def test_resolve_time_limit_never_zero():
    assert abs(resolve_time_limit_s(0, 0.0) - DEFAULT_TIME_LIMIT_S) < 1e-12
    assert abs(resolve_time_limit_s(0, 5.0) - 5.0) < 1e-12
    assert abs(resolve_time_limit_s(2_500_000, 5.0) - 2.5) < 1e-12
