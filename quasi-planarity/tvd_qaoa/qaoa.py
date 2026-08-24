"""pytket-based QAOA for the cubic PUBO cost function (spec §6).

The phase-separator is built directly from the PUBO monomial dict: no
ancilla qubits, exactly one qubit per candidate vertex. Degree-1 PUBO terms
become single-qubit Rz phases; degree-3 (triangle-penalty) terms become a
CNOT-ladder + Rz + CNOT-ladder multi-qubit ZZZ-phase gate. This is the
standard PUBO-QAOA / "polynomial unconstrained binary optimization" QAOA
technique -- the qubit count is exactly `n_qubits_used`, never inflated by
quadratization ancillas (those only exist in `formulation.build_qubo`, for
the classical QUBO exact solver).

Simulation backend is pluggable (`BackendConfig`): exact statevector
(`AerStateBackend`, the default -- fine up to ~20-24 qubits), or shot-sampled
`AerBackend` with a chosen Aer `simulation_method` ("statevector",
"matrix_product_state", ...) and `device` ("CPU"/"GPU"), for instances too
large for exact statevector simulation. See `get_backend` for the caveat on
matrix_product_state and this problem's connectivity.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from scipy.optimize import minimize

from .formulation import evaluate_pubo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PUBO (x in {0,1}) -> Ising (s in {+/-1}) conversion, s_v = 1 - 2 x_v
# ---------------------------------------------------------------------------


def pubo_to_ising(pubo: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    """Substitute x_v = (1 - s_v) / 2 into every monomial and expand, so the
    cost becomes a sum of Pauli-Z-string coefficients (plus a constant)."""
    ising: dict[tuple[int, ...], float] = {}

    def add(key: tuple[int, ...], coeff: float) -> None:
        ising[key] = ising.get(key, 0.0) + coeff

    for monomial, coeff in pubo.items():
        # Expand product over v in monomial of (1 - s_v)/2 = sum over subsets
        # T of monomial of  (-1)^|T| / 2^|monomial| * prod_{v in T} s_v
        k = len(monomial)
        for mask in range(1 << k):
            subset = tuple(sorted(monomial[i] for i in range(k) if mask & (1 << i)))
            sign = (-1) ** bin(mask).count("1")
            add(subset, coeff * sign / (2**k))
    return {k: c for k, c in ising.items() if abs(c) > 1e-14}


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------


def _add_zzz_phase(circ: Circuit, qubits: tuple[int, ...], angle_halfturns: float) -> None:
    """exp(-i * pi * angle_halfturns/2 * Z_q1 Z_q2 ... ) via CNOT ladder."""
    if len(qubits) == 1:
        circ.Rz(angle_halfturns, qubits[0])
        return
    ladder = list(zip(qubits[:-1], qubits[1:]))
    for a, b in ladder:
        circ.CX(a, b)
    circ.Rz(angle_halfturns, qubits[-1])
    for a, b in reversed(ladder):
        circ.CX(a, b)


def build_qaoa_circuit(
    pubo: dict[tuple[int, ...], float],
    n_qubits: int,
    p: int,
    betas: list[float],
    gammas: list[float],
) -> Circuit:
    """Build a p-layer QAOA circuit for the given cubic PUBO cost.

    Phase separator per layer: for every Ising term h_S * prod_{v in S} Z_v
    (S nonempty), apply exp(-i * gamma * h_S * Z_S) via a single Rz (degree 1)
    or a CNOT-ladder + Rz + CNOT-ladder (degree >= 2). Mixer: Rx(2*beta) on
    every qubit. pytket's Rz/Rx angle parameter is in half-turns, i.e.
    Rz(t) = exp(-i*pi*t*Z/2), so exp(-i*theta*Z) needs t = 2*theta/pi.
    """
    assert len(betas) == p and len(gammas) == p
    ising = pubo_to_ising(pubo)
    circ = Circuit(n_qubits)
    for q in range(n_qubits):
        circ.H(q)

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        for monomial, coeff in ising.items():
            if len(monomial) == 0:
                continue  # global phase, irrelevant to measurement statistics
            angle_halfturns = 2 * gamma * coeff / math.pi
            _add_zzz_phase(circ, monomial, angle_halfturns)
        for q in range(n_qubits):
            circ.Rx(2 * beta / math.pi, q)

    return circ


# ---------------------------------------------------------------------------
# Simulation backend selection
# ---------------------------------------------------------------------------

# Aer's `matrix_product_state` method truncates bond dimension between
# *adjacent* qubits in a 1D chain. This problem's Z_uZvZw phase terms are
# wired by triangle membership, i.e. arbitrary long-range 3-qubit gates, not
# nearest-neighbor -- so MPS is not guaranteed to be efficient here the way
# it would be for a geometrically local circuit. Benchmark accuracy/runtime
# vs statevector on a small instance before trusting MPS results at scale.
_SIMULATION_METHODS = {
    "aer_statevector": "statevector",   # exact, pytket AerStateBackend path
    "aer": "automatic",                 # shot-sampled qiskit-aer, CPU or GPU
    "aer_mps": "matrix_product_state",  # shot-sampled tensor-network method
}


@dataclass
class BackendConfig:
    """Which simulator to run QAOA circuits on.

    name: one of "aer_statevector" (exact, default, small n only), "aer"
        (shot-sampled qiskit-aer statevector/automatic method), "aer_mps"
        (shot-sampled matrix-product-state / tensor-network method).
    device: "CPU" or "GPU" (only meaningful for name in {"aer", "aer_mps"};
        requires a GPU-enabled qiskit-aer build + CUDA on the host).
    mps_max_bond_dimension / mps_truncation_threshold: passed straight
        through to Aer's `matrix_product_state_max_bond_dimension` /
        `matrix_product_state_truncation_threshold` options when name ==
        "aer_mps"; leave None to use Aer's defaults.
    """

    name: str = "aer_statevector"
    device: str = "CPU"
    mps_max_bond_dimension: int | None = None
    mps_truncation_threshold: float | None = None

    @property
    def exact(self) -> bool:
        return self.name == "aer_statevector"


def get_backend(config: BackendConfig, n_qubits: int):
    """Construct the pytket backend for `config`. `n_qubits` must be <= 40
    (AerBackend/AerStateBackend's own MaxNQubitsPredicate default)."""
    if config.name == "aer_statevector":
        return AerStateBackend()
    if config.name not in _SIMULATION_METHODS:
        raise ValueError(f"unknown QAOA backend {config.name!r}")

    method = _SIMULATION_METHODS[config.name]
    backend = AerBackend(simulation_method=method, n_qubits=max(n_qubits, 1))
    opts: dict = {"device": config.device}
    if config.name == "aer_mps":
        if config.mps_max_bond_dimension is not None:
            opts["matrix_product_state_max_bond_dimension"] = config.mps_max_bond_dimension
        if config.mps_truncation_threshold is not None:
            opts["matrix_product_state_truncation_threshold"] = config.mps_truncation_threshold
    # pytket's AerBackend constructor doesn't expose `device`/MPS options
    # itself; reach into the underlying qiskit-aer simulator, which does.
    backend._qiskit_backend.set_options(**opts)
    return backend


# ---------------------------------------------------------------------------
# Circuit evaluation: exact statevector, or shot-sampled (any AerBackend)
# ---------------------------------------------------------------------------


def _bitstring_probabilities(circ: Circuit, backend: AerStateBackend) -> dict[tuple[int, ...], float]:
    """Full computational-basis probability distribution via exact statevector sim."""
    compiled = circ.copy()
    backend.rebase_pass().apply(compiled)
    compiled = backend.get_compiled_circuit(compiled)
    handle = backend.process_circuit(compiled)
    state = backend.get_result(handle).get_state()
    n = circ.n_qubits
    probs = np.abs(state) ** 2
    # pytket statevector ordering: qubit 0 is the most significant bit (pytket
    # default ILO convention already matches Circuit qubit order via get_state).
    out: dict[tuple[int, ...], float] = {}
    for idx, prob in enumerate(probs):
        if prob < 1e-12:
            continue
        bits = tuple((idx >> (n - 1 - q)) & 1 for q in range(n))
        out[bits] = out.get(bits, 0.0) + float(prob)
    return out


def expected_pubo_cost(pubo: dict[tuple[int, ...], float], probs: dict[tuple[int, ...], float]) -> float:
    total = 0.0
    for bits, prob in probs.items():
        assignment = {v: bits[v] for v in range(len(bits))}
        total += prob * evaluate_pubo(pubo, assignment)
    return total


def best_sampled_cost(
    pubo: dict[tuple[int, ...], float],
    probs: dict[tuple[int, ...], float],
    n_shots: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[int, ...]]:
    """Sample n_shots bitstrings from an exact probability distribution and
    return (minimum PUBO cost observed, the bitstring that achieved it) --
    the "best-sampled" objective and its solution."""
    outcomes = list(probs.keys())
    weights = np.array([probs[o] for o in outcomes])
    weights = weights / weights.sum()
    draws = rng.choice(len(outcomes), size=n_shots, p=weights)
    best = math.inf
    best_bits: tuple[int, ...] = ()
    for idx in np.unique(draws):
        bits = outcomes[idx]
        assignment = {v: bits[v] for v in range(len(bits))}
        cost = evaluate_pubo(pubo, assignment)
        if cost < best:
            best, best_bits = cost, bits
    return best, best_bits


def _evaluate_circuit(
    circ: Circuit,
    pubo: dict[tuple[int, ...], float],
    backend,
    backend_config: BackendConfig,
    n_shots: int,
    rng: np.random.Generator,
) -> tuple[float, float, tuple[int, ...]]:
    """Returns (expected_cost, best_sampled_cost, best_sampled_bits) for one
    circuit under the given backend. Exact backend: computed from the full
    probability distribution. Shot backends (aer/aer_mps, CPU or GPU, incl.
    tensor network): all quantities are empirical, estimated from `n_shots`
    measurement outcomes actually drawn on that backend."""
    if backend_config.exact:
        probs = _bitstring_probabilities(circ, backend)
        expected = expected_pubo_cost(pubo, probs)
        best, best_bits = best_sampled_cost(pubo, probs, n_shots, rng)
        return expected, best, best_bits

    measured = circ.copy()
    measured.measure_all()
    compiled = backend.get_compiled_circuit(measured)
    handle = backend.process_circuit(compiled, n_shots=n_shots, seed=int(rng.integers(0, 2**31 - 1)))
    counts = backend.get_result(handle).get_counts()
    total_shots = sum(counts.values())
    expected = 0.0
    best = math.inf
    best_bits: tuple[int, ...] = ()
    for bits, count in counts.items():
        assignment = {v: bits[v] for v in range(len(bits))}
        cost = evaluate_pubo(pubo, assignment)
        expected += (count / total_shots) * cost
        if cost < best:
            best, best_bits = cost, bits
    return expected, best, best_bits


# ---------------------------------------------------------------------------
# Angle optimization for a fixed p
# ---------------------------------------------------------------------------


@dataclass
class QAOALayerResult:
    p: int
    betas: list[float]
    gammas: list[float]
    expected_cost: float
    best_sampled_objective: float
    best_sampled_vertices: list[int]
    approx_ratio: float
    wall_clock_seconds: float = 0.0


def optimize_qaoa_angles(
    pubo: dict[tuple[int, ...], float],
    n_qubits: int,
    p: int,
    classical_optimal: float,
    n_restarts: int,
    n_shots: int,
    rng: np.random.Generator,
    backend_config: BackendConfig,
    optimizer_method: str = "COBYLA",
    n_shots_optimization: int | None = None,
    instance_id: str = "",
) -> QAOALayerResult:
    """Optimize (beta, gamma) at fixed layer count p with `n_restarts` random
    restarts (seeded from `rng`), minimizing the expected PUBO cost on
    `backend_config`. Returns the restart with the lowest expected cost,
    evaluated for its best-sampled objective and approximation ratio.

    `n_shots_optimization` (default: `n_shots`) lets the inner optimization
    loop use fewer shots than the final reported evaluation, which matters
    for shot-sampled backends (aer/aer_mps) where every COBYLA iteration
    re-runs the circuit -- irrelevant for the exact statevector backend.
    """
    layer_start = time.perf_counter()
    tag = f"[{instance_id}] " if instance_id else ""
    logger.info(
        "%sQAOA p=%d: starting %d restart(s) on backend=%s (n_qubits=%d)",
        tag, p, n_restarts, backend_config.name, n_qubits,
    )
    backend = get_backend(backend_config, n_qubits)
    opt_shots = n_shots_optimization or n_shots

    def objective(x: np.ndarray) -> float:
        betas, gammas = list(x[:p]), list(x[p:])
        circ = build_qaoa_circuit(pubo, n_qubits, p, betas, gammas)
        expected, _, _ = _evaluate_circuit(circ, pubo, backend, backend_config, opt_shots, rng)
        return expected

    best_result = None
    for restart in range(1, n_restarts + 1):
        restart_start = time.perf_counter()
        x0 = rng.uniform(0, 2 * math.pi, size=2 * p)
        res = minimize(objective, x0, method=optimizer_method, options={"maxiter": 200})
        restart_elapsed = time.perf_counter() - restart_start
        logger.info(
            "%sQAOA p=%d restart %d/%d: expected_cost=%.4f (%.2fs)",
            tag, p, restart, n_restarts, res.fun, restart_elapsed,
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    betas, gammas = list(best_result.x[:p]), list(best_result.x[p:])
    circ = build_qaoa_circuit(pubo, n_qubits, p, betas, gammas)
    _, best_obj, best_bits = _evaluate_circuit(circ, pubo, backend, backend_config, n_shots, rng)
    best_vertices = [v for v, bit in enumerate(best_bits) if bit == 1]
    ratio = 1.0 if classical_optimal == 0 else classical_optimal / best_obj if best_obj > 0 else 1.0
    wall_clock = time.perf_counter() - layer_start
    logger.info(
        "%sQAOA p=%d done: best_sampled_objective=%.4f approx_ratio=%.4f (%.2fs)",
        tag, p, best_obj, ratio, wall_clock,
    )

    return QAOALayerResult(
        p=p,
        betas=betas,
        gammas=gammas,
        expected_cost=float(best_result.fun),
        best_sampled_objective=best_obj,
        best_sampled_vertices=best_vertices,
        approx_ratio=ratio,
        wall_clock_seconds=wall_clock,
    )


# ---------------------------------------------------------------------------
# Layer-count (p) selection sweep (spec §6)
# ---------------------------------------------------------------------------


@dataclass
class QAOASweepResult:
    optimal_p: int
    optimal_betas: list[float]
    optimal_gammas: list[float]
    approx_ratio_at_optimal_p: float
    best_sampled_objective_at_optimal_p: float
    best_sampled_vertices_at_optimal_p: list[int]
    threshold_met: bool
    sweep: list[QAOALayerResult] = field(default_factory=list)


def qaoa_p_sweep(
    pubo: dict[tuple[int, ...], float],
    n_qubits: int,
    p_max: int,
    classical_optimal: float,
    approx_ratio_threshold: float = 0.99,
    n_restarts: int = 3,
    n_shots: int = 1024,
    seed: int = 0,
    backend_config: BackendConfig | None = None,
    n_shots_optimization: int | None = None,
    instance_id: str = "",
) -> QAOASweepResult:
    """Sweep p = 1..p_max. Select the smallest p whose approximation ratio
    (classical-optimal / best-QAOA-sampled-objective, both minimizations of a
    non-negative cost so ratio <= 1) meets `approx_ratio_threshold`. If no p
    reaches it, select the p with the best ratio achieved and set
    threshold_met=False.

    `backend_config` selects the simulator (default: exact statevector).
    """
    backend_config = backend_config or BackendConfig()
    rng = np.random.default_rng(seed)
    tag = f"[{instance_id}] " if instance_id else ""
    sweep_start = time.perf_counter()
    logger.info(
        "%sQAOA p-sweep: p=1..%d, threshold=%.3f, backend=%s",
        tag, p_max, approx_ratio_threshold, backend_config.name,
    )
    sweep: list[QAOALayerResult] = []
    for p in range(1, p_max + 1):
        layer_result = optimize_qaoa_angles(
            pubo,
            n_qubits,
            p,
            classical_optimal,
            n_restarts,
            n_shots,
            rng,
            backend_config,
            n_shots_optimization=n_shots_optimization,
            instance_id=instance_id,
        )
        sweep.append(layer_result)
        if layer_result.approx_ratio >= approx_ratio_threshold:
            logger.info(
                "%sQAOA p-sweep done: threshold met at p=%d (%.2fs total)",
                tag, p, time.perf_counter() - sweep_start,
            )
            return QAOASweepResult(
                optimal_p=p,
                optimal_betas=layer_result.betas,
                optimal_gammas=layer_result.gammas,
                approx_ratio_at_optimal_p=layer_result.approx_ratio,
                best_sampled_objective_at_optimal_p=layer_result.best_sampled_objective,
                best_sampled_vertices_at_optimal_p=layer_result.best_sampled_vertices,
                threshold_met=True,
                sweep=sweep,
            )

    best = max(sweep, key=lambda r: r.approx_ratio)
    logger.info(
        "%sQAOA p-sweep done: threshold NOT met by p=%d, best ratio=%.4f at p=%d (%.2fs total)",
        tag, p_max, best.approx_ratio, best.p, time.perf_counter() - sweep_start,
    )
    return QAOASweepResult(
        optimal_p=best.p,
        optimal_betas=best.betas,
        optimal_gammas=best.gammas,
        approx_ratio_at_optimal_p=best.approx_ratio,
        best_sampled_objective_at_optimal_p=best.best_sampled_objective,
        best_sampled_vertices_at_optimal_p=best.best_sampled_vertices,
        threshold_met=False,
        sweep=sweep,
    )
