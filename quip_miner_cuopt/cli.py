"""CLI for ``quip-cuopt-milp`` / ``python -m quip_miner_cuopt``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from quip_miner_cuopt import (
    ALGORITHM,
    BACKEND,
    DEFAULT_CUOPT_URL,
    DEFAULT_TIME_LIMIT_S,
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
    EXIT_INTERNAL_FATAL,
    __version__,
)
from quip_miner_cuopt.client import CuoptClient, CuoptError
from quip_miner_cuopt.session_loop import run_session_sync


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quip-cuopt-milp",
        description="QuIP NVIDIA cuOpt MILP Ising miner (v0.3 protocol)",
    )
    p.add_argument("--quip-coordinator", default=None, help="unix:// path or host")
    p.add_argument("--miner-id", default="cuopt-0")
    p.add_argument(
        "--cuopt-url",
        default=None,
        help=f"cuOpt REST base URL (default: {DEFAULT_CUOPT_URL}; env CUOPT_URL)",
    )
    p.add_argument(
        "--time-limit-s",
        type=float,
        default=None,
        help=(
            f"default MILP time_limit in seconds when anneal_time_us is unset "
            f"(default: {DEFAULT_TIME_LIMIT_S}; env CUOPT_TIME_LIMIT_S)"
        ),
    )
    p.add_argument(
        "--capabilities",
        action="store_true",
        help="print capability JSON and exit",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="probe cuOpt server health, then exit",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    p.add_argument(
        "--log-level",
        default="info",
        help="logging level (default: info)",
    )
    return p


def resolve_cuopt_url(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    env = os.environ.get("CUOPT_URL")
    if env:
        return env
    return DEFAULT_CUOPT_URL


def resolve_time_limit_s(cli_value: float | None) -> float:
    if cli_value is not None and cli_value > 0.0:
        return float(cli_value)
    env = os.environ.get("CUOPT_TIME_LIMIT_S")
    if env:
        try:
            v = float(env)
            if v > 0.0:
                return v
        except ValueError:
            pass
    return DEFAULT_TIME_LIMIT_S


def print_capabilities() -> None:
    print(
        json.dumps(
            {
                "backend": BACKEND,
                "algorithm": ALGORITHM,
                "supported_kinds": ["ISING_SAMPLE"],
                "max_nodes": 20000,
                "max_edges": 200000,
                "features": ["milp", "anneal-time-us"],
            }
        )
    )


def run_check(cuopt_url: str) -> int:
    """``--check``: probe cuOpt health; fail with the URL named."""
    client = CuoptClient(cuopt_url)
    try:
        client.health_check()
    except CuoptError as exc:
        print(f"FAIL: {exc.message}", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    print(f"OK: cuOpt reachable at {client.base_url}")
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cuopt_url = resolve_cuopt_url(args.cuopt_url)
    time_limit_s = resolve_time_limit_s(args.time_limit_s)

    if args.version:
        print(f"quip-cuopt-milp {__version__} protocol 1")
        return EXIT_CLEAN
    if args.capabilities:
        print_capabilities()
        return EXIT_CLEAN
    if args.check:
        return run_check(cuopt_url)

    if not args.quip_coordinator:
        print("--quip-coordinator required for session mode", file=sys.stderr)
        return EXIT_CONFIG_INVALID

    client = CuoptClient(cuopt_url)
    try:
        client.health_check()
    except CuoptError as exc:
        logging.getLogger(__name__).error("%s", exc.message)
        return EXIT_ENV_INCOMPATIBLE

    try:
        return run_session_sync(
            args.quip_coordinator,
            args.miner_id,
            client,
            default_time_limit_s=time_limit_s,
        )
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else EXIT_INTERNAL_FATAL
        return code
    except Exception:
        logging.getLogger(__name__).exception("fatal")
        return EXIT_INTERNAL_FATAL


if __name__ == "__main__":
    sys.exit(main())
