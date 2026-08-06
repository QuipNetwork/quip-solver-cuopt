"""REST client for a self-hosted NVIDIA cuOpt server (LP/MILP).

Schema verified against NVIDIA cuOpt user-guide **26.06** OpenAPI
(``/cuopt/request``, ``/cuopt/solution/{id}``, ``/cuopt/health``) and the MILP
Python examples. No CUDA/cuOpt package required — plain HTTP only.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from quip_miner_cuopt import DEFAULT_CUOPT_URL, DEFAULT_TIME_LIMIT_S
from quip_miner_cuopt.linearize import MilpModel

logger = logging.getLogger(__name__)

# cuOpt thin-client / docs recommend identifying non-SDK clients.
_CLIENT_VERSION = "custom"


class CuoptError(Exception):
    """Transport or decode failure talking to cuOpt."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class MilpSolution:
    """Successful MILP solve payload we care about."""

    primal: List[float]
    mip_gap: Optional[float] = None
    solution_bound: Optional[float] = None
    status: Optional[str] = None
    primal_objective: Optional[float] = None


def trim_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def resolve_time_limit_s(anneal_time_us: int, cli_default_s: float) -> float:
    """Resolve wall time limit in seconds.

    ``0`` means unset on the wire. Never returns a literal zero time limit.
    Precedence for the caller: per-job ``anneal_time_us`` already selected
    before this helper; here we only convert microseconds and apply the CLI
    / documented default fallback.
    """
    if anneal_time_us > 0:
        s = float(anneal_time_us) / 1_000_000.0
        if s > 0.0:
            return s
    if cli_default_s > 0.0:
        return float(cli_default_s)
    return DEFAULT_TIME_LIMIT_S


def encode_milp_request(model: MilpModel, time_limit_s: float) -> Dict[str, Any]:
    """Encode a :class:`MilpModel` as LPData JSON for ``POST /cuopt/request``."""
    time_limit = time_limit_s if time_limit_s > 0.0 else DEFAULT_TIME_LIMIT_S
    var_names = [f"x{i}" for i in range(model.n)] + [
        f"y{k}" for k in range(len(model.y_edges))
    ]
    return {
        "csr_constraint_matrix": {
            "offsets": list(model.csr_offsets),
            "indices": list(model.csr_indices),
            "values": list(model.csr_values),
        },
        "constraint_bounds": {
            "lower_bounds": [b.to_json() for b in model.lower_bounds],
            "upper_bounds": [b.to_json() for b in model.upper_bounds],
        },
        "objective_data": {
            "coefficients": list(model.obj),
            "scalability_factor": 1.0,
            "offset": model.constant,
        },
        "variable_bounds": {
            "upper_bounds": model.var_upper_bounds(),
            "lower_bounds": model.var_lower_bounds(),
        },
        "maximize": False,
        "variable_names": var_names,
        "variable_types": model.variable_types(),
        "solver_config": {
            "time_limit": time_limit,
        },
    }


def _solution_ready(value: Dict[str, Any]) -> bool:
    try:
        sol = value["response"]["solver_response"]["solution"]
    except (KeyError, TypeError):
        return False
    if not isinstance(sol, dict):
        return False
    primal = sol.get("primal_solution")
    if isinstance(primal, list) and len(primal) > 0:
        return True
    vars_ = sol.get("vars")
    return isinstance(vars_, dict) and len(vars_) > 0


def _extract_stats(sol: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Decode ``milp_statistics`` (docs) or ``mip_statistics`` (OpenAPI example)."""
    stats = sol.get("milp_statistics")
    if not isinstance(stats, dict) or not stats:
        stats = sol.get("mip_statistics")
    if not isinstance(stats, dict):
        return None, None
    mip_gap = stats.get("mip_gap")
    solution_bound = stats.get("solution_bound")
    gap_f = float(mip_gap) if isinstance(mip_gap, (int, float)) else None
    bound_f = (
        float(solution_bound) if isinstance(solution_bound, (int, float)) else None
    )
    return gap_f, bound_f


def parse_solution(value: Dict[str, Any]) -> MilpSolution:
    """Parse a cuOpt solution response into :class:`MilpSolution`."""
    try:
        solver = value["response"]["solver_response"]
        sol = solver["solution"]
    except (KeyError, TypeError) as exc:
        raise CuoptError(f"missing solver_response.solution in {value!r}") from exc

    primal: List[float]
    if isinstance(sol.get("primal_solution"), list) and sol["primal_solution"]:
        try:
            primal = [float(x) for x in sol["primal_solution"]]
        except (TypeError, ValueError) as exc:
            raise CuoptError(
                f"non-numeric primal_solution: {sol['primal_solution']}"
            ) from exc
    elif isinstance(sol.get("vars"), dict) and sol["vars"]:
        pairs: List[tuple[int, float]] = []
        for k, val in sol["vars"].items():
            if not isinstance(k, str) or not k.startswith("x"):
                continue
            rest = k[1:]
            if rest.isdigit():
                pairs.append((int(rest), float(val)))
        pairs.sort(key=lambda p: p[0])
        primal = [v for _, v in pairs]
    else:
        raise CuoptError(f"solution has neither primal_solution nor vars: {sol!r}")

    mip_gap, solution_bound = _extract_stats(sol)
    status = solver.get("status")
    if status is not None:
        status = str(status)
    po = sol.get("primal_objective")
    primal_objective = float(po) if isinstance(po, (int, float)) else None

    return MilpSolution(
        primal=primal,
        mip_gap=mip_gap,
        solution_bound=solution_bound,
        status=status,
        primal_objective=primal_objective,
    )


class CuoptClient:
    """HTTP client for one cuOpt base URL."""

    def __init__(self, base_url: str = DEFAULT_CUOPT_URL, timeout_s: float = 600.0):
        self.base_url = trim_trailing_slash(base_url)
        self.timeout_s = timeout_s

    def _headers(self, content_type: bool = False) -> Dict[str, str]:
        h = {
            "Accept": "application/json",
            "client-version": _CLIENT_VERSION,
        }
        if content_type:
            h["Content-Type"] = "application/json"
        return h

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> tuple[int, str]:
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(content_type=body is not None),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            return exc.code, text
        except urllib.error.URLError as exc:
            raise CuoptError(f"{method} {url} failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise CuoptError(f"{method} {url} timed out") from exc

    def health_check(self) -> None:
        """Probe ``GET {base}/cuopt/health``. Raises with the URL named."""
        url = f"{self.base_url}/cuopt/health"
        try:
            status, text = self._request("GET", url, timeout=5.0)
        except CuoptError as exc:
            raise CuoptError(
                f"cuOpt health check failed for {url}: {exc.message} "
                f"(is the self-hosted cuOpt server running?)"
            ) from exc
        if not (200 <= status < 300):
            raise CuoptError(
                f"cuOpt health check failed for {url}: HTTP {status}"
                + (f" body={text}" if text else "")
            )

    def solve_milp(self, model: MilpModel, time_limit_s: float) -> MilpSolution:
        """POST a MILP and poll until a solution is ready."""
        body_obj = encode_milp_request(model, time_limit_s)
        body = json.dumps(body_obj).encode("utf-8")
        url = f"{self.base_url}/cuopt/request"
        status, text = self._request("POST", url, body=body)
        if not (200 <= status < 300):
            raise CuoptError(f"POST {url} returned HTTP {status}: {text}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CuoptError(f"decode POST {url}: {exc}; body={text}") from exc

        if _solution_ready(value):
            return parse_solution(value)

        req_id = value.get("reqId")
        if not isinstance(req_id, str) or not req_id:
            raise CuoptError(
                f"POST {url}: response has neither solution nor reqId: {text}"
            )

        poll_url = f"{self.base_url}/cuopt/solution/{req_id}"
        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() > deadline:
                raise CuoptError(
                    f"timed out polling {poll_url} after {self.timeout_s}s"
                )
            time.sleep(0.2)
            pstatus, ptext = self._request("GET", poll_url)
            if not (200 <= pstatus < 300):
                raise CuoptError(f"GET {poll_url} returned HTTP {pstatus}: {ptext}")
            try:
                value = json.loads(ptext)
            except json.JSONDecodeError as exc:
                raise CuoptError(f"decode GET {poll_url}: {exc}; body={ptext}") from exc
            if _solution_ready(value):
                return parse_solution(value)


def log_dual_bound(sol: MilpSolution) -> None:
    """Log ``mip_gap`` and ``solution_bound`` to stderr (protocol has no field)."""
    import sys

    gap = "n/a" if sol.mip_gap is None else str(sol.mip_gap)
    bound = "n/a" if sol.solution_bound is None else str(sol.solution_bound)
    msg = f"cuopt milp: mip_gap={gap} solution_bound={bound}"
    logger.info(msg)
    print(msg, file=sys.stderr, flush=True)
