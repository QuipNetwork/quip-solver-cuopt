"""gRPC session loop: Hello → Welcome → Configure → credit/job cycle.

Mirrors ``quip_miner_dwave.session_loop`` using the ``quip_proto`` Python SDK.
Uses the synchronous gRPC client with a request queue (reliable over UDS).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Tuple

import grpc
from quip_proto import miner_pb2, miner_pb2_grpc
from quip_proto import session as session_sdk

from quip_miner_cuopt import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_INTERNAL_FATAL,
    EXIT_TOKEN_REJECTED,
)
from quip_miner_cuopt.client import CuoptClient
from quip_miner_cuopt.config import warn_unknown_backend_keys
from quip_miner_cuopt.job import handle_job

logger = logging.getLogger(__name__)

_STOP = object()

# Operator log token. Distinct from BACKEND ("cuopt-milp"), which is the
# protocol advertisement. Mixed-fleet lines use [quip-miner-cuopt].
_LOG_BACKEND = "cuopt"


def _short_job_id(job_id: bytes) -> str:
    """Render the leading bytes of a job id, matching quip-miner-core."""
    head = bytes(job_id[:8]).hex()
    if len(job_id) > 8:
        return head + ".."
    return head


def _energy_units(milli: int) -> int:
    """Floor-divide a milli-energy to whole units for a log line."""
    return milli // 1000


def _format_duration_ms(ms: int) -> str:
    """Render a millisecond duration with the shared miner buckets."""
    if ms < 1_000:
        return f"{ms}ms"
    if ms < 60_000:
        tenths = (ms + 50) // 100
        if tenths >= 600:
            return "1m 0s"
        secs = tenths // 10
        frac = tenths % 10
        return f"{secs}.{frac}s"
    if ms < 3_600_000:
        mins = ms // 60_000
        rem = ms % 60_000
        secs = (rem + 500) // 1_000
        if secs == 60:
            mins += 1
            secs = 0
        if mins >= 60:
            return f"{mins // 60}h {mins % 60}m"
        return f"{mins}m {secs}s"
    hours = ms // 3_600_000
    rem = ms % 3_600_000
    mins = (rem + 30_000) // 60_000
    if mins == 60:
        hours += 1
        mins = 0
    return f"{hours}h {mins}m"


def _reject_reason_name(reason: int) -> str:
    try:
        return miner_pb2.RejectReason.Name(reason)
    except ValueError:
        return str(reason)


def log_attempt(
    job_id: bytes,
    *,
    energy_milli: Optional[int] = None,
    valid: int = 0,
    total: int = 0,
    wall_ms: int = 0,
    device_ms: int = 0,
    rejected: Optional[str] = None,
    cancelled: bool = False,
) -> None:
    """Emit one per-attempt line in the shared miner format."""
    job = _short_job_id(job_id)
    wall = _format_duration_ms(wall_ms)
    if cancelled:
        logger.debug(
            "[quip-miner-%s] attempt %s: cancelled after %s",
            _LOG_BACKEND,
            job,
            wall,
        )
        return
    if rejected is not None:
        logger.warning(
            "[quip-miner-%s] attempt %s: rejected %s | %s wall",
            _LOG_BACKEND,
            job,
            rejected,
            wall,
        )
        return
    energy = "n/a" if energy_milli is None else str(_energy_units(energy_milli))
    device = _format_duration_ms(device_ms)
    logger.info(
        "[quip-miner-%s] attempt %s: energy %s, valid %d/%d | %s wall, %s device",
        _LOG_BACKEND,
        job,
        energy,
        valid,
        total,
        wall,
        device,
    )


def log_progress(
    *,
    jobs_done: int,
    elapsed_s: float,
    reads: int,
    sweeps: int,
    best_energy_milli: Optional[int],
    max_energy_milli: Optional[int],
    min_solutions: int,
) -> None:
    """Emit the shared progress line every N completed jobs."""
    rate = jobs_done / elapsed_s if elapsed_s > 0 else 0.0
    best = "n/a" if best_energy_milli is None else str(_energy_units(best_energy_milli))
    if max_energy_milli is None:
        requirement = "no target set"
    else:
        requirement = "requires energy<={}, solutions>={}".format(
            _energy_units(max_energy_milli),
            min_solutions,
        )
    logger.info(
        "[quip-miner-%s] progress: %d jobs | %.1f jobs/s | reads=%d sweeps=%d"
        " | best=%s | %s",
        _LOG_BACKEND,
        jobs_done,
        rate,
        reads,
        sweeps,
        best,
        requirement,
    )


def _status(
    miner_id: str, jobs_done: int = 0, abandoned: int = 0
) -> miner_pb2.MinerMsg:
    return miner_pb2.MinerMsg(
        status=miner_pb2.Status(
            miner_id=miner_id,
            utilization=0.0,
            jobs_done=jobs_done,
            abandoned_generation=abandoned,
        )
    )


def _is_abandoned(generation: int, watermark: int) -> bool:
    """True if a job's ``generation`` was abandoned by a Cancel(``watermark``)."""
    return generation != 0 and generation <= watermark


def _unix_target(uri: str) -> str:
    """Normalize ``unix:///path`` or bare path to a grpc UDS target."""
    if uri.startswith("unix://"):
        path = uri[len("unix://") :]
    elif uri.startswith("unix:"):
        path = uri[len("unix:") :]
    else:
        path = uri
    if not path.startswith("/"):
        path = "/" + path
    return f"unix://{path}"


def run_session(
    coordinator_uri: str,
    miner_id: str,
    client: CuoptClient,
    *,
    default_time_limit_s: float,
) -> int:
    """Run one miner session; return a process exit code."""
    target = _unix_target(coordinator_uri)
    try:
        hello = session_sdk.build_hello(
            miner_id,
            BACKEND,
            ALGORITHM,
            [miner_pb2.ISING_SAMPLE],
        )
    except session_sdk.MissingToken:
        logger.error("QUIP_SESSION_TOKEN unset")
        return EXIT_TOKEN_REJECTED

    out_q: queue.Queue = queue.Queue()
    out_q.put(miner_pb2.MinerMsg(hello=hello))

    def request_iter() -> Iterator[miner_pb2.MinerMsg]:
        while True:
            item = out_q.get()
            if item is _STOP:
                return
            yield item

    jobs_done = 0
    cancel_watermark = 0
    grace_ms = 5000
    config: Optional[session_sdk.SessionConfig] = None
    session_nodes: list[int] = []
    session_edges: list[Tuple[int, int]] = []
    session_hash: Optional[bytes] = None
    session_target: Optional[miner_pb2.SetTarget] = None
    state_lock = threading.Lock()
    job_pool: Optional[ThreadPoolExecutor] = None
    session_start = time.monotonic()
    best_energy_milli: Optional[int] = None
    progress_log_interval = 10

    def process_job(job, s_nodes, s_edges, s_hash, s_target):
        nonlocal jobs_done, best_energy_milli
        started = time.monotonic()
        replies = handle_job(
            job,
            client,
            session_nodes=s_nodes,
            session_edges=s_edges,
            session_hash=s_hash,
            session_target=s_target,
            default_time_limit_s=default_time_limit_s,
        )
        wall_ms = int((time.monotonic() - started) * 1000)
        for reply in replies:
            kind = reply.WhichOneof("msg")
            with state_lock:
                if kind == "result":
                    jobs_done += 1
                    meta = reply.result.meta
                    job_best = min(
                        (s.energy_milli for s in reply.result.solutions),
                        default=None,
                    )
                    if job_best is not None and (
                        best_energy_milli is None or job_best < best_energy_milli
                    ):
                        best_energy_milli = job_best
                    if s_target is not None:
                        valid = sum(
                            1
                            for s in reply.result.solutions
                            if s.energy_milli <= s_target.max_energy_milli
                        )
                    else:
                        valid = len(reply.result.solutions)
                    log_attempt(
                        job.job_id,
                        energy_milli=job_best,
                        valid=valid,
                        total=len(reply.result.solutions),
                        wall_ms=wall_ms,
                        device_ms=meta.device_access_time_us // 1000,
                    )
                    if jobs_done % progress_log_interval == 0:
                        log_progress(
                            jobs_done=jobs_done,
                            elapsed_s=time.monotonic() - session_start,
                            reads=meta.reads,
                            sweeps=meta.sweeps,
                            best_energy_milli=best_energy_milli,
                            max_energy_milli=(
                                s_target.max_energy_milli
                                if s_target is not None
                                else None
                            ),
                            min_solutions=(
                                int(s_target.min_solutions)
                                if s_target is not None
                                else 0
                            ),
                        )
                elif kind == "reject":
                    log_attempt(
                        job.job_id,
                        rejected=_reject_reason_name(reply.reject.reason),
                        wall_ms=wall_ms,
                    )
            out_q.put(reply)

    exit_code = EXIT_CLEAN

    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.default_authority", "localhost"),
            ("grpc.enable_http_proxy", 0),
        ],
    )
    try:
        deadline = time.time() + 10.0
        while True:
            try:
                grpc.channel_ready_future(channel).result(
                    timeout=max(0.05, deadline - time.time())
                )
                break
            except grpc.FutureTimeoutError:
                if time.time() >= deadline:
                    logger.error("timed out waiting for coordinator at %s", target)
                    return EXIT_INTERNAL_FATAL

        stub = miner_pb2_grpc.MinerServiceStub(channel)
        responses = stub.Session(request_iter())

        last_activity = time.monotonic()
        for cm in responses:
            last_activity = time.monotonic()
            which = cm.WhichOneof("msg")
            if which == "welcome":
                ver = cm.welcome.protocol_version
                if ver not in (0, 1):
                    logger.error("bad Welcome protocol_version=%s", ver)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            fatal=miner_pb2.Fatal(
                                exit_code=EXIT_INTERNAL_FATAL,
                                reason="bad welcome protocol_version",
                                restart_required=False,
                            )
                        )
                    )
                    exit_code = EXIT_INTERNAL_FATAL
                    break
            elif which == "configure":
                config = session_sdk.session_config_from_configure(
                    miner_id, cm.configure
                )
                warn_unknown_backend_keys(cm.configure.backend_toml)
                out_q.put(miner_pb2.MinerMsg(ready=miner_pb2.Ready()))
                depth = config.queue_depth if config else 3
                # Stream width 1 keeps cuOpt load predictable; still advertise
                # credits so the coordinator can pipeline.
                if job_pool is None:
                    job_pool = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="cuopt-job"
                    )
                out_q.put(
                    miner_pb2.MinerMsg(
                        job_request=miner_pb2.JobRequest(credits=max(1, depth))
                    )
                )
            elif which == "topology":
                topo = cm.topology
                session_nodes = list(topo.nodes)
                session_hash = bytes(topo.hash)
                if topo.HasField("edges"):
                    session_edges = list(zip(topo.edges.u, topo.edges.v))
                else:
                    session_edges = []
            elif which == "set_target":
                session_target = cm.set_target
            elif which == "job":
                with state_lock:
                    cancelled = _is_abandoned(cm.job.generation, cancel_watermark)
                if cancelled:
                    log_attempt(cm.job.job_id, cancelled=True, wall_ms=0)
                    out_q.put(
                        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1))
                    )
                    continue
                args = (
                    cm.job,
                    list(session_nodes),
                    list(session_edges),
                    session_hash,
                    session_target,
                )
                if job_pool is not None:
                    job_pool.submit(process_job, *args)
                else:
                    process_job(*args)
            elif which == "cancel":
                with state_lock:
                    cancel_watermark = max(cancel_watermark, cm.cancel.max_generation)
                    done = jobs_done
                out_q.put(_status(miner_id, done, abandoned=1))
            elif which == "ping":
                with state_lock:
                    done = jobs_done
                out_q.put(_status(miner_id, done))
            elif which == "shutdown":
                grace_ms = cm.shutdown.grace_ms or 5000
                break

            idle_s = config.idle_timeout_s if config else 300
            if time.monotonic() - last_activity > idle_s:
                logger.info("idle timeout (%ss) — clean exit", idle_s)
                break

        if job_pool is not None:
            job_pool.shutdown(wait=True)
        out_q.put(_STOP)
        time.sleep(min(0.05, grace_ms / 1000.0))
        return exit_code
    except grpc.RpcError as exc:
        code = exc.code() if hasattr(exc, "code") else None
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return EXIT_TOKEN_REJECTED
        logger.exception("rpc error: %s", exc)
        return EXIT_INTERNAL_FATAL
    except Exception:
        logger.exception("session failed")
        return EXIT_INTERNAL_FATAL
    finally:
        if job_pool is not None:
            job_pool.shutdown(wait=False, cancel_futures=True)
        try:
            out_q.put(_STOP)
        except Exception:  # noqa: BLE001
            pass
        channel.close()


def run_session_sync(
    coordinator_uri: str,
    miner_id: str,
    client: CuoptClient,
    *,
    default_time_limit_s: float,
) -> int:
    """Sync entry (session is already synchronous)."""
    return run_session(
        coordinator_uri,
        miner_id,
        client,
        default_time_limit_s=default_time_limit_s,
    )
