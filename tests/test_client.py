"""Mock HTTP server for cuOpt client (no GPU / no real cuOpt required)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import pytest

from quip_miner_cuopt.client import (
    CuoptClient,
    CuoptError,
    encode_milp_request,
    parse_solution,
    resolve_time_limit_s,
)
from quip_miner_cuopt.linearize import linearize


class _Handler(BaseHTTPRequestHandler):
    """Minimal cuOpt-shaped mock: health + request → immediate solution."""

    solution_body: Dict[str, Any] = {}
    last_request_body: Optional[Dict[str, Any]] = None
    fail_health: bool = False

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/cuopt/health") or self.path.endswith("/health"):
            if self.fail_health:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"status":"unhealthy"}')
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        if "/cuopt/solution/" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.solution_body).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            _Handler.last_request_body = json.loads(raw.decode())
        except json.JSONDecodeError:
            _Handler.last_request_body = None
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.solution_body).encode())


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    # Default optimal solution for a 2-var problem.
    _Handler.solution_body = {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_solution": [1.0, 0.0, 0.0],
                    "primal_objective": -1.0,
                    "milp_statistics": {
                        "mip_gap": 0.0,
                        "solution_bound": -1.0,
                    },
                },
            }
        },
        "reqId": "test-req",
    }
    _Handler.fail_health = False
    _Handler.last_request_body = None
    yield url
    server.shutdown()


def test_health_ok(mock_server):
    client = CuoptClient(mock_server)
    client.health_check()  # does not raise


def test_health_names_url_on_failure():
    client = CuoptClient("http://127.0.0.1:1")
    with pytest.raises(CuoptError) as ei:
        client.health_check()
    assert "http://127.0.0.1:1/cuopt/health" in ei.value.message


def test_solve_milp_roundtrip(mock_server):
    model = linearize([0.0, 0.0], [1.0], [(0, 1)])
    client = CuoptClient(mock_server)
    sol = client.solve_milp(model, 12.5)
    assert sol.primal[:2] == [1.0, 0.0]
    assert sol.mip_gap == 0.0
    assert sol.solution_bound == -1.0
    body = _Handler.last_request_body
    assert body is not None
    assert body["maximize"] is False
    assert body["solver_config"]["time_limit"] == 12.5
    assert body["variable_types"] == ["I", "I", "C"]
    assert body["objective_data"]["offset"] == model.constant


def test_encode_zero_time_limit_falls_back():
    model = linearize([0.0, 0.0], [1.0], [(0, 1)])
    body = encode_milp_request(model, 0.0)
    assert body["solver_config"]["time_limit"] == 30.0


def test_parse_milp_statistics_and_mip_statistics():
    milp = {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_solution": [1.0, 0.0],
                    "milp_statistics": {
                        "mip_gap": 0.0,
                        "solution_bound": -1.5,
                    },
                },
            }
        }
    }
    s = parse_solution(milp)
    assert s.primal == [1.0, 0.0]
    assert s.mip_gap == 0.0
    assert s.solution_bound == -1.5

    mip = {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_solution": [0.0, 1.0],
                    "mip_statistics": {
                        "mip_gap": 0.01,
                        "solution_bound": 2.0,
                    },
                },
            }
        }
    }
    s2 = parse_solution(mip)
    assert s2.mip_gap == 0.01
    assert s2.solution_bound == 2.0


def test_resolve_time_limit_never_zero():
    assert resolve_time_limit_s(0, 0.0) == 30.0
    assert resolve_time_limit_s(0, 5.0) == 5.0
    assert abs(resolve_time_limit_s(2_500_000, 5.0) - 2.5) < 1e-12
