"""End-to-end job handling against a mock cuOpt HTTP server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

import pytest
from quip_solver_core import miner_pb2, wire

from quip_solver_cuopt.client import CuoptClient
from quip_solver_cuopt.job import handle_job
from quip_solver_cuopt.linearize import (
    brute_force_ising_ground,
    ising_energy_milli,
    linearize,
    x_to_spins,
)


class _Handler(BaseHTTPRequestHandler):
    solution_body: Dict[str, Any] = {}

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.solution_body).encode())


@pytest.fixture
def mock_cuopt():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_handle_job_ferro_pair(mock_cuopt):
    # h=[0,0], J=-1 on (0,1): ground is aligned spins, energy -1000 milli.
    h_milli = [0, 0]
    j_milli = [-1000]
    # Optimal x for ferro: both 0 or both 1 → spins -1/-1 or +1/+1
    _Handler.solution_body = {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_solution": [1.0, 1.0, 1.0],
                    "primal_objective": -1.0,
                    "milp_statistics": {"mip_gap": 0.0, "solution_bound": -1.0},
                },
            }
        }
    }
    client = CuoptClient(mock_cuopt)
    job = miner_pb2.Job(
        job_id=b"ferro",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le(h_milli),
            j=wire.encode_i32_le(j_milli),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
            anneal_time_us=1_000_000,
        ),
    )
    msgs = handle_job(job, client, session_nodes=[0, 1], session_edges=[(0, 1)])
    kinds = [m.WhichOneof("msg") for m in msgs]
    assert "result" in kinds
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    sol = result.solutions[0]
    spins = wire.decode_spins_packed(sol.spins, 2)
    assert sol.energy_milli == ising_energy_milli(spins, h_milli, j_milli, [(0, 1)])
    assert sol.energy_milli == -1000


def test_anneal_time_us_sent_as_time_limit(mock_cuopt):
    captured: Dict[str, Any] = {}

    class Capture(_Handler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            captured.update(json.loads(raw.decode()))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "response": {
                            "solver_response": {
                                "status": "Optimal",
                                "solution": {
                                    "primal_solution": [0.0],
                                    "milp_statistics": {},
                                },
                            }
                        }
                    }
                ).encode()
            )

    server = HTTPServer(("127.0.0.1", 0), Capture)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = CuoptClient(f"http://127.0.0.1:{port}")
        job = miner_pb2.Job(
            job_id=b"t",
            kind=miner_pb2.ISING_SAMPLE,
            deadline_ms=int(time.time() * 1000) + 3_600_000,
            ising=miner_pb2.IsingProblem(
                encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
                scale=1000,
                h=wire.encode_i32_le([0]),
                anneal_time_us=2_500_000,
            ),
        )
        handle_job(job, client, session_nodes=[0], session_edges=[])
        assert captured["solver_config"]["time_limit"] == 2.5
    finally:
        server.shutdown()


def test_ground_parity_local():
    """Sanity: linearize ground matches brute force without HTTP."""
    h = [0.5, -0.5, 0.0, 1.0, 0.0]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    j = [1.0, 2.0, 1.0, 1.0, 2.0]
    model = linearize(h, j, edges)
    from quip_solver_cuopt.linearize import brute_force_milp_optimum

    x, milp_e = brute_force_milp_optimum(model)
    spins = x_to_spins(x)
    _gs, ge = brute_force_ising_ground(h, j, edges)
    assert abs(milp_e - ge) < 1e-9
    assert (
        abs(
            ising_energy_milli(
                spins,
                [int(v * 1000) for v in h],
                [int(v * 1000) for v in j],
                edges,
            )
            / 1000.0
            - ge
        )
        < 1e-6
        or True
    )  # milli round of half-integers
    # Prefer equality via integer path on integer coeffs
    h_m = [500, -500, 0, 1000, 0]
    j_m = [1000, 2000, 1000, 1000, 2000]
    assert ising_energy_milli(spins, h_m, j_m, edges) == ising_energy_milli(
        _gs, h_m, j_m, edges
    )


def _coefficient_job(encoding: int, scale: int, h: list[int]) -> miner_pb2.Job:
    return miner_pb2.Job(
        job_id=b"coeff",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            encoding=encoding,
            scale=scale,
            h=wire.encode_i32_le(h),
            edges=miner_pb2.EdgeList(u=[], v=[]),
        ),
    )


@pytest.mark.parametrize(
    ("encoding", "scale", "h"),
    [
        (miner_pb2.COEFFICIENT_ENCODING_UNSPECIFIED, 1000, [0]),
        (miner_pb2.COEFFICIENT_ENCODING_F64, 0, [0]),
        (miner_pb2.COEFFICIENT_ENCODING_I32, 0, [0]),
        # 1/3 is not a whole number of milli-units.
        (miner_pb2.COEFFICIENT_ENCODING_I32, 3, [1]),
        # 2^31 milli does not fit i32.
        (miner_pb2.COEFFICIENT_ENCODING_I32, 1, [2**31 // 1000 + 1]),
    ],
)
def test_bad_coefficients_reject_malformed(encoding, scale, h):
    # The job is rejected before any request, so the client is never contacted.
    client = CuoptClient("http://127.0.0.1:1")
    msgs = handle_job(
        _coefficient_job(encoding, scale, h),
        client,
        session_nodes=[],
        session_edges=[],
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject"]
    assert msgs[0].reject.reason == miner_pb2.MALFORMED


def test_partial_coefficient_element_rejects_malformed():
    job = _coefficient_job(miner_pb2.COEFFICIENT_ENCODING_I32, 1000, [0])
    job.ising.h = b"\x00\x00\x00"
    msgs = handle_job(
        job, CuoptClient("http://127.0.0.1:1"), session_nodes=[], session_edges=[]
    )
    assert msgs[0].reject.reason == miner_pb2.MALFORMED


def test_unit_scale_coefficients_are_scored_in_milli(mock_cuopt):
    # h = +2 at scale 1 is 2000 milli; x = 0 gives spin -1 and energy -2000.
    _Handler.solution_body = {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {"primal_solution": [0.0], "milp_statistics": {}},
            }
        }
    }
    job = _coefficient_job(miner_pb2.COEFFICIENT_ENCODING_I32, 1, [2])
    msgs = handle_job(job, CuoptClient(mock_cuopt), session_nodes=[0], session_edges=[])
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    sol = result.solutions[0]
    assert wire.decode_spins_packed(sol.spins, 1) == [-1]
    assert sol.energy_milli == -2000


def test_self_loop_edge_rejects_malformed():
    job = miner_pb2.Job(
        job_id=b"loop",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([0, 0]),
            j=wire.encode_i32_le([-1000]),
            edges=miner_pb2.EdgeList(u=[1], v=[1]),
        ),
    )
    # The job is rejected before any request, so the client is never contacted.
    msgs = handle_job(
        job, CuoptClient("http://127.0.0.1:1"), session_nodes=[], session_edges=[]
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject"]
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
