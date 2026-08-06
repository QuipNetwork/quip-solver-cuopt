"""Job validation, MILP solve, and Result/Reject construction."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

from quip_proto import miner_pb2, wire

from quip_miner_cuopt import DEFAULT_TIME_LIMIT_S
from quip_miner_cuopt.client import (
    CuoptClient,
    CuoptError,
    log_dual_bound,
    resolve_time_limit_s,
)
from quip_miner_cuopt.linearize import ising_energy_milli, linearize, x_to_spins

logger = logging.getLogger(__name__)


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def decode_milli_i32(raw: bytes) -> List[int]:
    """Decode little-endian i32 milli array. Raises ValueError if bad."""
    return list(wire.decode_i32_le(raw))


def decode_milli_f64(raw: bytes) -> List[float]:
    return [v / 1000.0 for v in decode_milli_i32(raw)]


def edges_of(ising: miner_pb2.IsingProblem) -> List[Tuple[int, int]]:
    which = ising.WhichOneof("graph")
    if which == "edges":
        e = ising.edges
        return list(zip(e.u, e.v))
    return []


def resolve_graph(
    ising: miner_pb2.IsingProblem,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    n_h: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (nodes, edges) for a job.

    Inline EdgeList uses dense 0..n-1 node ids from ``n_h``; topology-hash
    jobs use the session-cached Topology.
    """
    which = ising.WhichOneof("graph")
    if which == "edges":
        return list(range(n_h)), edges_of(ising)
    return list(session_nodes), list(session_edges)


def spins_to_bytes(spins: Sequence[int]) -> bytes:
    return wire.encode_spins(list(spins))


class _Rejected(Exception):
    """Raised by a validation step to abort the job with a reject reason."""

    def __init__(self, reason: int):
        super().__init__(reason)
        self.reason = reason


def _reject(job_id: bytes, reason: int) -> List[miner_pb2.MinerMsg]:
    return [miner_pb2.MinerMsg(reject=miner_pb2.Reject(job_id=job_id, reason=reason))]


def _validate_job(
    job: miner_pb2.Job,
    session_hash: Optional[bytes],
) -> Tuple[miner_pb2.IsingProblem, List[int], List[int]]:
    """Check job-level invariants and decode the milli arrays as integers."""
    if job.kind not in (
        miner_pb2.ISING_SAMPLE,
        miner_pb2.JOB_KIND_UNSPECIFIED,
    ):
        raise _Rejected(miner_pb2.UNSUPPORTED_KIND)

    ising = job.ising if job.HasField("ising") else miner_pb2.IsingProblem()

    try:
        h_milli = decode_milli_i32(ising.h_milli_le32)
    except ValueError:
        raise _Rejected(miner_pb2.MALFORMED) from None
    try:
        j_milli = decode_milli_i32(ising.j_milli_le32) if ising.j_milli_le32 else []
    except ValueError:
        raise _Rejected(miner_pb2.MALFORMED) from None

    if job.deadline_ms and job.deadline_ms < now_unix_ms():
        raise _Rejected(miner_pb2.EXPIRED)

    if ising.WhichOneof("graph") == "topology_hash":
        if session_hash is None:
            raise _Rejected(miner_pb2.TOPOLOGY_MISSING)
        if bytes(ising.topology_hash) != bytes(session_hash):
            raise _Rejected(miner_pb2.TOPOLOGY_MISMATCH)

    return ising, h_milli, j_milli


def _resolve_problem(
    ising: miner_pb2.IsingProblem,
    h_milli: Sequence[int],
    j_milli: Sequence[int],
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    *,
    job_id: bytes,
) -> Tuple[List[int], List[Tuple[int, int]], List[int], List[int]]:
    """Resolve graph; return (nodes, edges, h_milli, j_milli) aligned to nodes."""
    nodes, edges = resolve_graph(ising, session_nodes, session_edges, len(h_milli))
    if not nodes and h_milli:
        nodes = list(range(len(h_milli)))

    if len(h_milli) != len(nodes):
        logger.warning(
            "job %s: %d biases for a %d-node graph; rejecting MALFORMED",
            job_id.hex(),
            len(h_milli),
            len(nodes),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    if len(j_milli) != len(edges):
        logger.warning(
            "job %s: %d couplings for a %d-edge graph; rejecting MALFORMED",
            job_id.hex(),
            len(j_milli),
            len(edges),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    # Dense positional edges for MILP / scoring (0..n-1).
    if edges and nodes and (min(nodes) != 0 or set(nodes) != set(range(len(nodes)))):
        # Map sparse qubit labels to dense positions.
        index = {int(n): i for i, n in enumerate(nodes)}
        dense_edges: List[Tuple[int, int]] = []
        for u, v in edges:
            if int(u) not in index or int(v) not in index:
                raise _Rejected(miner_pb2.MALFORMED)
            dense_edges.append((index[int(u)], index[int(v)]))
        edges = dense_edges
    else:
        edges = [(int(u), int(v)) for u, v in edges]

    return nodes, edges, list(h_milli), list(j_milli)


def sampling_params(
    ising: miner_pb2.IsingProblem,
    session_target: Optional["miner_pb2.SetTarget"],
    *,
    default_time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Tuple[int, int, float]:
    """Resolve ``(num_reads, anneal_time_us, time_limit_s)`` for one job.

    Both ``num_reads`` and ``anneal_time_us`` follow the same precedence:
    per-job override, then the session's ``SetTarget``, then the default.
    ``0`` means UNSET at every level (never a zero time limit).

    ``anneal_time_us`` becomes the cuOpt MILP ``time_limit`` in seconds.
    """
    num_reads = int(ising.num_reads)
    if num_reads == 0 and session_target is not None and session_target.num_reads:
        num_reads = int(session_target.num_reads)
    if num_reads == 0:
        num_reads = 1

    anneal_time_us = int(ising.anneal_time_us)
    if (
        anneal_time_us == 0
        and session_target is not None
        and session_target.anneal_time_us
    ):
        anneal_time_us = int(session_target.anneal_time_us)

    time_limit_s = resolve_time_limit_s(anneal_time_us, default_time_limit_s)
    return num_reads, anneal_time_us, time_limit_s


def handle_job(
    job: miner_pb2.Job,
    client: CuoptClient,
    *,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    session_hash: Optional[bytes] = None,
    session_target: Optional["miner_pb2.SetTarget"] = None,
    default_time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> List[miner_pb2.MinerMsg]:
    """Validate and solve one job; return Result+JobRequest or Reject messages."""
    job_id = job.job_id
    try:
        ising, h_milli, j_milli = _validate_job(job, session_hash)
        nodes, edges, h_milli, j_milli = _resolve_problem(
            ising,
            h_milli,
            j_milli,
            session_nodes,
            session_edges,
            job_id=job_id,
        )
    except _Rejected as exc:
        return _reject(job_id, exc.reason)

    num_reads, _anneal_us, time_limit_s = sampling_params(
        ising, session_target, default_time_limit_s=default_time_limit_s
    )
    # num_reads is accepted for protocol parity. cuOpt has no MILP batch mode;
    # we return the single best incumbent from one solve.
    _ = num_reads

    if not h_milli:
        # Empty problem: energy 0, no variables.
        sol = miner_pb2.Solution(spins_bytes=b"", energy_milli=0)
        meta = miner_pb2.SamplerMeta(reads=1, sweeps=0, device_access_time_us=0)
        return [
            miner_pb2.MinerMsg(
                result=miner_pb2.Result(job_id=job_id, solutions=[sol], meta=meta)
            ),
            miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
        ]

    h_f = [v / 1000.0 for v in h_milli]
    j_f = [v / 1000.0 for v in j_milli]
    model = linearize(h_f, j_f, edges)

    t0 = time.monotonic()
    try:
        milp_sol = client.solve_milp(model, time_limit_s)
    except CuoptError as exc:
        logger.error("cuOpt solve failed: %s", exc.message)
        return _reject(job_id, miner_pb2.OVERLOADED)
    device_us = int((time.monotonic() - t0) * 1_000_000)

    log_dual_bound(milp_sol)

    if len(milp_sol.primal) < model.n:
        logger.error(
            "cuOpt returned %d primal values, need at least %d",
            len(milp_sol.primal),
            model.n,
        )
        return _reject(job_id, miner_pb2.MALFORMED)

    x = milp_sol.primal[: model.n]
    spins = x_to_spins(x)
    # Report energy from decoded spins in integer milli arithmetic — never
    # from the MILP objective.
    energy = ising_energy_milli(spins, h_milli, j_milli, edges)

    solution = miner_pb2.Solution(
        spins_bytes=spins_to_bytes(spins),
        energy_milli=energy,
    )
    meta = miner_pb2.SamplerMeta(
        reads=1,
        sweeps=0,
        device_access_time_us=device_us,
        qpu_access_us=device_us,
    )
    return [
        miner_pb2.MinerMsg(
            result=miner_pb2.Result(
                job_id=job_id,
                solutions=[solution],
                meta=meta,
            )
        ),
        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
    ]
