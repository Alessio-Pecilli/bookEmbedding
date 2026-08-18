from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import config


def get_num_qubits(num_edges: int) -> int:
    return int(num_edges) * int(config.NUM_PAGES)


@dataclass(frozen=True)
class CostModel:
    n_qubits: int
    constant: float
    z_terms: Dict[int, float]
    zz_terms: Dict[Tuple[int, int], float]
    num_pages: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    weighted_crossings: Tuple[Tuple[int, int, float], ...] = ()


def _safe_penalty(weighted_crossings: Sequence[Tuple[int, int, float]]) -> float:
    """Choose an instance-safe one-hot penalty.

    If W is the sum of all non-negative weighted crossing terms, every valid
    assignment costs at most W while every one-hot violation costs at least
    alpha.  Thus alpha > W is a sufficient (conservative) guarantee that an
    invalid state cannot be a ground state.  ``config.ALPHA`` remains the
    minimum configured value.
    """
    beta = float(config.BETA)
    if beta < 0:
        raise ValueError("BETA must be non-negative for this minimization model")
    total = beta * sum(float(w) for _, _, w in weighted_crossings)
    margin = max(1e-9, abs(total) * 1e-9)
    return max(float(config.ALPHA), total + margin)


def build_cost_model(
    edges: List[Tuple[int, int]],
    weighted_crossings: Iterable[Tuple[int, int, float]],
) -> CostModel:
    """Build the diagonal Z/ZZ Hamiltonian for the fixed-order embedding."""
    weighted_crossings = tuple(
        (int(e), int(f), float(w)) for e, f, w in weighted_crossings
    )
    num_edges = len(edges)
    k = int(config.NUM_PAGES)
    if k < 1:
        raise ValueError("NUM_PAGES must be positive")
    for e_idx, f_idx, weight in weighted_crossings:
        if not (0 <= e_idx < num_edges and 0 <= f_idx < num_edges) or e_idx == f_idx:
            raise ValueError(f"Invalid crossing edge indices: {(e_idx, f_idx)}")
        if weight < 0:
            raise ValueError("Crossing weights must be non-negative")
    n_qubits = get_num_qubits(num_edges)
    alpha = _safe_penalty(weighted_crossings)

    constant = 0.0
    z_terms: Dict[int, float] = {}
    zz_terms: Dict[Tuple[int, int], float] = {}

    # alpha * (sum_p x[e,p] - 1)^2, x=(1-Z)/2.
    for e_idx in range(num_edges):
        qubits_e = [e_idx * k + p for p in range(k)]
        constant += alpha * (1 - 0.75 * k + 0.25 * (k**2))

        c_lin = alpha * (1 - k / 2.0)
        for q in qubits_e:
            z_terms[q] = z_terms.get(q, 0.0) + c_lin

        c_quad = 0.5 * alpha
        for i, q1 in enumerate(qubits_e):
            for q2 in qubits_e[i + 1 :]:
                zz_terms[(q1, q2)] = zz_terms.get((q1, q2), 0.0) + c_quad

    # beta*w*x[e,p]*x[f,p].
    for e_idx, f_idx, weight in weighted_crossings:
        w_eff = float(config.BETA) * weight
        for p in range(k):
            q_ep = e_idx * k + p
            q_fp = f_idx * k + p
            constant += w_eff * 0.25
            z_terms[q_ep] = z_terms.get(q_ep, 0.0) - w_eff * 0.25
            z_terms[q_fp] = z_terms.get(q_fp, 0.0) - w_eff * 0.25
            pair = (q_ep, q_fp) if q_ep < q_fp else (q_fp, q_ep)
            zz_terms[pair] = zz_terms.get(pair, 0.0) + w_eff * 0.25

    z_terms = {i: c for i, c in z_terms.items() if abs(c) > 1e-12}
    zz_terms = {ij: c for ij, c in zz_terms.items() if abs(c) > 1e-12}
    return CostModel(
        n_qubits=n_qubits,
        constant=float(constant),
        z_terms=z_terms,
        zz_terms=zz_terms,
        num_pages=k,
        alpha=float(alpha),
        beta=float(config.BETA),
        weighted_crossings=weighted_crossings,
    )


def _validate_logical_bitstring(bitstring: str, n_qubits: int) -> None:
    if len(bitstring) != n_qubits or any(ch not in "01" for ch in bitstring):
        raise ValueError(f"Expected a {n_qubits}-bit logical bitstring, got {bitstring!r}")


def energy_of_bitstring(model: CostModel, logical_bitstring: str) -> float:
    """Evaluate H for a canonical bitstring in q0, q1, ..., q(n-1) order."""
    _validate_logical_bitstring(logical_bitstring, model.n_qubits)
    z = [1.0 if bit == "0" else -1.0 for bit in logical_bitstring]
    return float(
        model.constant
        + sum(float(c) * z[int(i)] for i, c in model.z_terms.items())
        + sum(float(c) * z[int(i)] * z[int(j)] for (i, j), c in model.zz_terms.items())
    )


def direct_binary_objective(model: CostModel, logical_bitstring: str) -> float:
    """Evaluate the original binary objective independently of Z expansion."""
    _validate_logical_bitstring(logical_bitstring, model.n_qubits)
    k = model.num_pages
    x = [int(bit) for bit in logical_bitstring]
    one_hot = sum(
        (sum(x[e * k : (e + 1) * k]) - 1) ** 2
        for e in range(model.n_qubits // k)
    )
    crossing = sum(
        weight * sum(x[e * k + p] * x[f * k + p] for p in range(k))
        for e, f, weight in model.weighted_crossings
    )
    return float(model.alpha * one_hot + model.beta * crossing)


def decode_logical_bitstring(
    logical_bitstring: str, num_edges: int, num_pages: int
) -> Dict[int, int]:
    """Decode canonical q0,q1,... bits into an edge-to-page assignment."""
    _validate_logical_bitstring(logical_bitstring, num_edges * num_pages)
    assignment: Dict[int, int] = {}
    for edge in range(num_edges):
        block = logical_bitstring[edge * num_pages : (edge + 1) * num_pages]
        active = [page for page, bit in enumerate(block) if bit == "1"]
        assignment[edge] = active[0] if len(active) == 1 else -1
    return assignment


def backend_key_to_logical_bitstring(key: object, n_qubits: int) -> str:
    """Convert one pytket/Qiskit-Aer result key to canonical logical order.

    Experimentally, the supported ``pytket.extensions.qiskit.AerBackend``
    returns ``Result.get_counts()`` keys as ``(c[0], ..., c[n-1])``.  A
    textual Qiskit/Aer key is displayed most-significant bit first, so that
    fallback is reversed exactly once here.  No caller may reverse outcomes.
    """
    if isinstance(key, str):
        text = key.replace(" ", "")
        if len(text) != n_qubits or any(ch not in "01" for ch in text):
            raise ValueError(f"Unexpected backend bitstring {key!r}")
        return text[::-1]

    if isinstance(key, (tuple, list, np.ndarray)):
        bits = "".join(str(int(bit)) for bit in key)
        if len(bits) != n_qubits or any(ch not in "01" for ch in bits):
            raise ValueError(f"Unexpected backend bit tuple {key!r}")
        return bits

    raise TypeError(f"Unsupported backend count key type: {type(key).__name__}")


def backend_counts_to_logical_counts(
    counts: Dict[object, int], n_qubits: int
) -> Dict[str, int]:
    """Canonicalize backend keys and combine outcomes with the same value."""
    logical: Dict[str, int] = {}
    for key, count in counts.items():
        bitstring = backend_key_to_logical_bitstring(key, n_qubits)
        logical[bitstring] = logical.get(bitstring, 0) + int(count)
    return logical


def build_qaoa_circuit(
    model: CostModel,
    gammas: np.ndarray,
    betas: np.ndarray,
    measure: bool = True,
    prepare_uniform: bool = True,
) -> "Circuit":
    """Build a numeric QAOA circuit using pytket.

    ``gammas`` and ``betas`` are mathematical angles in radians.  pytket
    rotation parameters are expressed in half-turns, so conversion to pytket's
    convention happens only at the gate boundary below.
    """
    from pytket.circuit import Circuit

    n = model.n_qubits
    p = int(len(gammas))
    if len(betas) != p:
        raise ValueError("gammas and betas must have same length (layers).")

    # Circuit(n) has no classical register in current pytket; measurements
    # require the explicit one-to-one n-qubit/n-bit circuit.
    circ = Circuit(n, n)
    if prepare_uniform:
        for q in range(n):
            circ.H(q)

    for layer in range(p):
        g = float(gammas[layer])
        b = float(betas[layer])
        for i, coeff in model.z_terms.items():
            # Rz(t) = exp(-i*pi*t*Z/2), with t in half-turns.
            circ.Rz(2.0 * g * float(coeff) / np.pi, int(i))
        for (i, j), coeff in model.zz_terms.items():
            circ.CX(int(i), int(j))
            # The CX-Rz-CX gadget therefore implements exp(-i*g*c*ZiZj).
            circ.Rz(2.0 * g * float(coeff) / np.pi, int(j))
            circ.CX(int(i), int(j))
        for q in range(n):
            # Rx(t) = exp(-i*pi*t*X/2), again using half-turns.
            circ.Rx(2.0 * b / np.pi, q)

    if measure:
        for q in range(n):
            circ.Measure(q, q)
    return circ


def estimate_energy_from_counts(model: CostModel, counts: Dict[str, int]) -> float:
    """Estimate ``<H>`` from counts already in canonical logical order."""
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("No shots in counts.")
    return float(
        sum(energy_of_bitstring(model, bitstring) * (count / shots)
            for bitstring, count in counts.items())
    )


def sample_qaoa_counts(
    model: CostModel,
    gammas: np.ndarray,
    betas: np.ndarray,
    shots: int,
    seed: int | None = None,
    backend: object | None = None,
) -> Dict[str, int]:
    """Sample Aer and return counts in canonical logical order."""
    from pytket.extensions.qiskit import AerBackend

    if shots < 1:
        raise ValueError("shots must be positive")
    if backend is None:
        backend = AerBackend()
    circuit = build_qaoa_circuit(model, gammas, betas)
    kwargs = {"n_shots": int(shots)}
    if seed is not None:
        kwargs["seed"] = int(seed)
    # AerBackend accepts this gate set directly.  Avoiding a fresh pytket
    # compilation for every COBYLA objective evaluation is a major speedup for
    # the small simulators used here, while preserving the same backend and
    # count semantics tested above.
    handle = backend.process_circuit(circuit, **kwargs)
    raw = backend.get_result(handle).get_counts()
    return backend_counts_to_logical_counts(raw, model.n_qubits) if raw else {}


def most_probable_bitstring(counts: Dict[str, int]) -> str:
    if not counts:
        raise ValueError("Empty counts.")
    return max(counts.items(), key=lambda kv: kv[1])[0]
