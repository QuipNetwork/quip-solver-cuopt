"""CLI --capabilities and --check."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from quip_miner_cuopt import EXIT_CLEAN, EXIT_ENV_INCOMPATIBLE
from quip_miner_cuopt.cli import main


def test_capabilities():
    # Capture via return code; capabilities prints to stdout.
    code = main(["--capabilities"])
    assert code == EXIT_CLEAN


def test_capabilities_json_shape(capsys):
    assert main(["--capabilities"]) == EXIT_CLEAN
    data = json.loads(capsys.readouterr().out)
    assert data["backend"] == "cuopt"
    assert data["algorithm"] == "milp"
    assert "ISING_SAMPLE" in data["supported_kinds"]
    assert "anneal-time-us" in data["features"]


def test_version(capsys):
    assert main(["--version"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "quip-cuopt-milp" in out
    assert "0.3.0rc1" in out


def test_check_fail_names_url(capsys):
    code = main(["--check", "--cuopt-url", "http://127.0.0.1:1"])
    assert code == EXIT_ENV_INCOMPATIBLE
    err = capsys.readouterr().err
    assert "http://127.0.0.1:1/cuopt/health" in err


def test_check_ok():
    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    server = HTTPServer(("127.0.0.1", 0), H)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        code = main(["--check", "--cuopt-url", f"http://127.0.0.1:{port}"])
        assert code == EXIT_CLEAN
    finally:
        server.shutdown()


def test_session_requires_coordinator():
    from quip_miner_cuopt import EXIT_CONFIG_INVALID

    code = main([])
    assert code == EXIT_CONFIG_INVALID
