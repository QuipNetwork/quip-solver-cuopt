"""Session-loop log lines: the shared mixed-fleet miner format.

An operator watching a mixed fleet reads one line format from every backend.
These tests assert the emitted records, not a return value, because the log
line is the product.
"""

from __future__ import annotations

import logging

from quip_solver_cuopt.session_loop import log_attempt, log_progress

# First 8 bytes of a longer job id, matching the observed CPU miner line.
_JOB_ID = bytes.fromhex("1858b8caf35ae0df") + b"\x00\x01"


def test_attempt_line_matches_the_shared_format(caplog):
    # A finished attempt is info. Energy is milli internally and whole units
    # on the line. Wall and device use the shared duration buckets.
    with caplog.at_level(logging.INFO):
        log_attempt(
            _JOB_ID,
            energy_milli=-14_245_000,
            valid=0,
            total=106,
            wall_ms=21_000,
            device_ms=21_000,
        )
    assert (
        "[quip-solver-cuopt] attempt 1858b8caf35ae0df..: "
        "energy -14245, valid 0/106 | 21.0s wall, 21.0s device" in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO


def test_rejected_attempt_line_matches_the_shared_format(caplog):
    with caplog.at_level(logging.WARNING):
        log_attempt(_JOB_ID, rejected="MALFORMED", wall_ms=21_000)
    assert (
        "[quip-solver-cuopt] attempt 1858b8caf35ae0df..: "
        "rejected MALFORMED | 21.0s wall" in caplog.text
    )
    assert caplog.records[0].levelno == logging.WARNING


def test_cancelled_attempt_line_is_debug(caplog):
    with caplog.at_level(logging.DEBUG):
        log_attempt(_JOB_ID, cancelled=True, wall_ms=21_000)
    assert (
        "[quip-solver-cuopt] attempt 1858b8caf35ae0df..: cancelled after 21.0s"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.DEBUG


def test_progress_line_matches_the_shared_format(caplog):
    # 10 jobs in 20 seconds is 0.5 jobs/s. Best and the requirement use
    # whole energy units, not milli.
    with caplog.at_level(logging.INFO):
        log_progress(
            jobs_done=10,
            elapsed_s=20.0,
            reads=1,
            sweeps=0,
            best_energy_milli=-14_245_000,
            max_energy_milli=-14_000_000,
            min_solutions=1,
        )
    assert (
        "[quip-solver-cuopt] progress: 10 jobs | 0.5 jobs/s | reads=1 sweeps=0 | "
        "best=-14245 | requires energy<=-14000, solutions>=1" in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO


def test_progress_line_without_a_target_says_so(caplog):
    # Before the coordinator sends SetTarget there is no requirement clause
    # to print. The line must still carry the prefix, reads, and sweeps.
    with caplog.at_level(logging.INFO):
        log_progress(
            jobs_done=10,
            elapsed_s=20.0,
            reads=1,
            sweeps=0,
            best_energy_milli=None,
            max_energy_milli=None,
            min_solutions=0,
        )
    assert (
        "[quip-solver-cuopt] progress: 10 jobs | 0.5 jobs/s | reads=1 sweeps=0 | "
        "best=n/a | no target set" in caplog.text
    )
