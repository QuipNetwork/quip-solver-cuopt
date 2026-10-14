"""NVIDIA cuOpt MILP Ising miner for quip.network protocol version 2.

Binary entry point: ``quip-cuopt-milp`` (``python -m quip_solver_cuopt``).
"""

__version__ = "0.3.0rc1"

# sysexits-style exit codes (mirrored by quip_protocol::session::ExitCode)
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77

# Size caps advertised in Hello and --capabilities.
MAX_NODES = 20000
MAX_EDGES = 200000

DEFAULT_CUOPT_URL = "http://localhost:5000"
# Documented CLI/env default when anneal_time_us is unset at every level.
DEFAULT_TIME_LIMIT_S = 30.0
