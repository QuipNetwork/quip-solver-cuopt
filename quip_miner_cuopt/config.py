"""Uniform config-override discipline for the cuOpt miner.

Python mirror of the shared miner config helpers: config (from the
coordinator's ``Configure``) overrides CLI/env with a warning, and
unrecognized keys warn.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Keys consumed by the shared session layer, not any single backend; never
# warned about as unknown.
SESSION_KEYS = frozenset({"num_sweeps"})

# Backend TOML keys this miner understands.
KNOWN_BACKEND_KEYS = frozenset({"cuopt_url", "time_limit_s"})

T = TypeVar("T")


def config_override(name: str, cli: T, from_config: Optional[T]) -> T:
    """Resolve one setting with ``config > CLI`` precedence.

    Warns and returns the config value only on an effective change; otherwise
    returns ``cli`` silently.
    """
    if from_config is not None and from_config != cli:
        logger.warning(
            "config overrides %s: %s -> %s (from coordinator)", name, cli, from_config
        )
        return from_config
    return cli


def warn_unknown_fields(
    backend: str, present: Iterable[str], known: Iterable[str]
) -> None:
    """Warn once per unrecognized key. Session keys are never flagged."""
    known_all = set(known) | SESSION_KEYS
    for key in present:
        if key not in known_all:
            logger.warning("config: unknown field '%s' for %s (ignored)", key, backend)


def warn_unknown_backend_keys(backend_toml: str) -> None:
    """Parse a TOML fragment lightly and warn on unknown keys.

    Accepts either real TOML (if ``tomllib`` can parse it) or a best-effort
    key extract so a missing optional dependency never blocks mining.
    """
    if not backend_toml or not backend_toml.strip():
        return
    keys: list[str] = []
    try:
        import tomllib

        data = tomllib.loads(backend_toml)
        if isinstance(data, dict):
            keys = [str(k) for k in data]
    except Exception:  # noqa: BLE001 - fall back to line scan
        for line in backend_toml.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                keys.append(line.split("=", 1)[0].strip().strip("\"'"))
    warn_unknown_fields("cuopt", keys, KNOWN_BACKEND_KEYS)
