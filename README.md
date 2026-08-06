# ising-solver-nvidia

NVIDIA cuOpt MILP Ising miner for the [quip.network](https://gitlab.com/quip.network)
v0.3 mining protocol. Package name: `quip_miner_cuopt`. Binary: `quip-cuopt-milp`.

Each Ising job is cast as a mixed-integer linear program (binary spins as `x`,
continuous products as `y`), solved over REST against a self-hosted cuOpt
server, and scored from the decoded spins in integer milli arithmetic. The
package does not require CUDA or the cuOpt Python package at install time; any
machine that can install `quip_proto` can install and test this miner.

## Binary

| binary | backend | algorithm |
|--------|---------|-----------|
| `quip-cuopt-milp` | `cuopt` | `milp` |

isingmark resolves the binary name `quip-cuopt-milp` from `$PATH`.

## Install

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# needs Rust + protoc for quip_proto (maturin)
```

## Running

Start a self-hosted cuOpt server (default `http://localhost:5000`), then:

```sh
export QUIP_SESSION_TOKEN=...
quip-cuopt-milp \
  --quip-coordinator unix:///run/quip/coord.sock \
  --cuopt-url http://localhost:5000
```

### Flags

| flag | env | default | meaning |
|------|-----|---------|---------|
| `--cuopt-url` | `CUOPT_URL` | `http://localhost:5000` | cuOpt REST base URL |
| `--time-limit-s` | `CUOPT_TIME_LIMIT_S` | `30` | MILP `time_limit` (seconds) when `anneal_time_us` is unset |
| `--capabilities` | | | print capabilities JSON and exit 0 |
| `--check` | | | probe cuOpt health; exit 0 or 69 |
| `--quip-coordinator` | | | session endpoint (required for mining) |

### Parameter mapping

Wire field `anneal_time_us` is the per-job MILP time budget (microseconds).
Resolution matches the D-Wave miner:

1. per-job `IsingProblem.anneal_time_us` if nonzero
2. else session `SetTarget.anneal_time_us` if nonzero
3. else `--time-limit-s` / `CUOPT_TIME_LIMIT_S` / built-in 30 s

A value of `0` means unset at every level and never becomes a zero time limit.

`num_reads` is accepted for protocol parity. cuOpt has no MILP batch mode; the
miner returns the single best incumbent from one solve.

## Dual bounds (stderr only)

Per job the miner logs `mip_gap` and `solution_bound` from `milp_statistics`
(also accepts the OpenAPI example name `mip_statistics`) to **stderr**. The
v0.3 `Result` message has no field for a dual bound, so these values are not
returned on the protocol today.

## Formulation (summary)

Ising energy `E(s) = Σ h_i s_i + Σ J_ij s_i s_j` with `s ∈ {-1,+1}` becomes,
via `s = 2x − 1` and continuous `y_ij ≈ x_i x_j`:

- objective coefficients `2 h_i − 2 Σ_{e∋i} J_e` on binary `x_i`
- coefficient `4 J_ij` on continuous `y_ij ∈ [0,1]`
- constant `Σ (−h_i) + Σ J_ij` (tracked in `objective_data.offset`)

Sign-aware Fortet cuts only:

- `J > 0`: `y ≥ x_i + x_j − 1`
- `J < 0`: `y ≤ x_i`, `y ≤ x_j`
- `J = 0`: no `y`

Only `x` is integer (`variable_types` `"I"`); every `y` is continuous (`"C"`).

Reported energy is always computed from decoded spins in integer milli
arithmetic, never from the MILP objective alone.

## Tests and gates

```sh
python -m pytest
ruff check .
ruff format --check .
```

The suite includes brute-force parity on small random Ising instances (both
J-sign branches), an objective cross-check, `anneal_time_us` precedence tests,
a mock HTTP server (no cuOpt, no GPU), and `--capabilities` / `--check`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
