"""NVIDIA cuOpt MILP Ising miner for the quip.network v0.3 protocol.

Binary entry point: ``quip-cuopt-milp`` (``python -m quip_miner_cuopt``).
"""

__version__ = "0.3.0rc1"

# sysexits-style exit codes (mirrored by quip_protocol::session::ExitCode)
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77

BACKEND = "cuopt"
ALGORITHM = "milp"

DEFAULT_CUOPT_URL = "http://localhost:5000"
# Documented CLI/env default when anneal_time_us is unset at every level.
DEFAULT_TIME_LIMIT_S = 30.0
