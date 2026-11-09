"""Session loop against an in-process fake coordinator on a Unix socket."""

from __future__ import annotations

import queue
import shutil
import tempfile
from concurrent import futures
from typing import Callable, Iterator, List

import grpc
import pytest
from quip_solver_core import miner_pb2, miner_pb2_grpc

from quip_solver_cuopt import EXIT_CLEAN, EXIT_CONFIG_INVALID
from quip_solver_cuopt.client import CuoptClient
from quip_solver_cuopt.session_loop import capabilities_message, run_session

# A script reads the solver's messages from `inbox` and yields coordinator
# messages. Everything the solver sends lands in `received`.
Script = Callable[["queue.Queue[miner_pb2.MinerMsg]"], Iterator[miner_pb2.CoordMsg]]


class _Coordinator(miner_pb2_grpc.MinerServiceServicer):
    def __init__(self, script: Script) -> None:
        self.script = script
        self.received: List[miner_pb2.MinerMsg] = []

    def Session(self, request_iterator, context):  # noqa: N802
        inbox: "queue.Queue[miner_pb2.MinerMsg]" = queue.Queue()

        def pump() -> None:
            for msg in request_iterator:
                self.received.append(msg)
                inbox.put(msg)

        pool = futures.ThreadPoolExecutor(max_workers=1)
        pool.submit(pump)
        yield from self.script(inbox)
        pool.shutdown(wait=False)


def _next_of(inbox, kind: str) -> miner_pb2.MinerMsg:
    while True:
        msg = inbox.get(timeout=5)
        if msg.WhichOneof("msg") == kind:
            return msg


@pytest.fixture
def coordinator(monkeypatch):
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "tok")
    # macOS caps a socket path near 104 bytes, which pytest's tmp_path exceeds.
    sock_dir = tempfile.mkdtemp(prefix="cuopt-")
    servers = []

    def start(script: Script) -> tuple[_Coordinator, str]:
        coord = _Coordinator(script)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        miner_pb2_grpc.add_MinerServiceServicer_to_server(coord, server)
        uri = f"unix://{sock_dir}/coord.sock"
        server.add_insecure_port(uri)
        server.start()
        servers.append(server)
        return coord, uri

    yield start
    for server in servers:
        server.stop(grace=None)
    shutil.rmtree(sock_dir, ignore_errors=True)


def _run(uri: str) -> int:
    client = CuoptClient("http://127.0.0.1:1")
    return run_session(uri, "miner-test", client, default_time_limit_s=1.0)


def test_hello_carries_the_capabilities(coordinator):
    def script(inbox):
        _next_of(inbox, "hello")
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=1))

    coord, uri = coordinator(script)
    assert _run(uri) == EXIT_CLEAN
    hello = next(m.hello for m in coord.received if m.WhichOneof("msg") == "hello")
    assert hello.capabilities == capabilities_message()


def test_bad_welcome_exits_config_invalid(coordinator):
    def script(inbox):
        _next_of(inbox, "hello")
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        _next_of(inbox, "fatal")

    coord, uri = coordinator(script)
    assert _run(uri) == EXIT_CONFIG_INVALID
    fatal = next(m.fatal for m in coord.received if m.WhichOneof("msg") == "fatal")
    assert fatal.exit_code == EXIT_CONFIG_INVALID


def test_get_capabilities_gets_a_reply(coordinator):
    replies: List[miner_pb2.Capabilities] = []

    def script(inbox):
        _next_of(inbox, "hello")
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=2))
        yield miner_pb2.CoordMsg(configure=miner_pb2.Configure(queue_depth=1))
        _next_of(inbox, "ready")
        yield miner_pb2.CoordMsg(get_capabilities=miner_pb2.GetCapabilities())
        replies.append(_next_of(inbox, "capabilities").capabilities)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=1))

    _coord, uri = coordinator(script)
    assert _run(uri) == EXIT_CLEAN
    assert replies == [capabilities_message()]
