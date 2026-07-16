from __future__ import annotations

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
    z_terms: Dict[int, float]  # i -> coeff * Z_i
    zz_terms: Dict[Tuple[int, int], float]  # (i,j) -> coeff * Z_i Z_j, with i<j


def build_cost_model(
    edges: List[Tuple[int, int]],
    weighted_crossings: Iterable[Tuple[int, int, float]],
) -> CostModel:
    """
    Build a diagonal cost Hamiltonian in the computational basis:
      H = constant*I + Σ_i a_i Z_i + Σ_{i<j} b_{ij} Z_i Z_j

    Encoding:
      qubit q = edge_idx * NUM_PAGES + page
      x = (1 - Z)/2

    H_PAGE:
      ALPHA * (Σ_p x_{e,p} - 1)^2  for each edge e

    H_CROSS (weighted):
      Σ_{(e,f,w)} Σ_p  (BETA*w) * x_{e,p} x_{f,p}
      where w = w_e * w_f
    """
    num_edges = len(edges)
    k = int(config.NUM_PAGES)
    n_qubits = get_num_qubits(num_edges)

    constant = 0.0
    z_terms: Dict[int, float] = {}
    zz_terms: Dict[Tuple[int, int], float] = {}

    # --- H_PAGE ---
    for e_idx in range(num_edges):
        qubits_e = [e_idx * k + p for p in range(k)]

        constant += float(config.ALPHA) * (1 - 0.75 * k + 0.25 * (k**2))

        c_lin = float(config.ALPHA) * (1 - k / 2.0)
        for q in qubits_e:
            z_terms[q] = z_terms.get(q, 0.0) + c_lin

        c_quad = 0.5 * float(config.ALPHA)
        for i, q1 in enumerate(qubits_e):
            for q2 in qubits_e[i + 1 :]:
                a, b = (q1, q2) if q1 < q2 else (q2, q1)
                zz_terms[(a, b)] = zz_terms.get((a, b), 0.0) + c_quad

    # --- H_CROSS (weighted) ---
    for (e_idx, f_idx, w) in weighted_crossings:
        w_eff = float(config.BETA) * float(w)
        for p in range(k):
            q_ep = int(e_idx) * k + p
            q_fp = int(f_idx) * k + p

            constant += w_eff * 0.25
            z_terms[q_ep] = z_terms.get(q_ep, 0.0) + (w_eff * -0.25)
            z_terms[q_fp] = z_terms.get(q_fp, 0.0) + (w_eff * -0.25)

            a, b = (q_ep, q_fp) if q_ep < q_fp else (q_fp, q_ep)
            zz_terms[(a, b)] = zz_terms.get((a, b), 0.0) + (w_eff * 0.25)

    # Remove near-zero coeffs from accumulation noise.
    z_terms = {i: c for i, c in z_terms.items() if abs(c) > 1e-12}
    zz_terms = {ij: c for ij, c in zz_terms.items() if abs(c) > 1e-12}

    return CostModel(
        n_qubits=n_qubits,
        constant=float(constant),
        z_terms=z_terms,
        zz_terms=zz_terms,
    )


def build_qaoa_circuit(
    model: CostModel,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> "Circuit":
    """
    Build a numeric (no-symbolic) QAOA circuit using pytket.
    Cost evolution uses Z and ZZ gadgets; mixer uses RX.
    """
    from pytket.circuit import Circuit

    n = model.n_qubits
    p = int(len(gammas))
    if len(betas) != p:
        raise ValueError("gammas and betas must have same length (layers).")

    circ = Circuit(n)

    for q in range(n):
        circ.H(q)

    for layer in range(p):
        g = float(gammas[layer])
        b = float(betas[layer])

        # exp(-i g * Σ a_i Z_i)
        for i, coeff in model.z_terms.items():
            angle = 2.0 * g * float(coeff)
            circ.Rz(angle, int(i))

        # exp(-i g * Σ b_ij Z_i Z_j)
        for (i, j), coeff in model.zz_terms.items():
            angle = 2.0 * g * float(coeff)
            ii, jj = int(i), int(j)
            circ.CX(ii, jj)
            circ.Rz(angle, jj)
            circ.CX(ii, jj)

        # exp(-i b * Σ X_i)
        for q in range(n):
            circ.Rx(2.0 * b, q)

    # Measurement for sampling-based estimation
    for q in range(n):
        circ.Measure(q, q)

    return circ


def estimate_energy_from_counts(model: CostModel, counts: Dict[str, int]) -> float:
    """
    Estimate E = <H> from sampled bitstrings.

    counts keys are bitstrings of length n_qubits.
    Convention: bitstring[0] is the most significant bit (common in simulators).
    We only use parity per-qubit, so we reverse if needed in main once verified.
    """
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("No shots in counts.")

    total = 0.0
    n = model.n_qubits

    for bitstring, c in counts.items():
        if len(bitstring) != n:
            raise ValueError(f"Unexpected bitstring length {len(bitstring)} != {n}")

        # Map bits to Z eigenvalues: 0 -> +1, 1 -> -1
        z = [1.0 if ch == "0" else -1.0 for ch in bitstring]

        e = model.constant
        for i, coeff in model.z_terms.items():
            e += float(coeff) * z[int(i)]
        for (i, j), coeff in model.zz_terms.items():
            e += float(coeff) * z[int(i)] * z[int(j)]

        total += e * (c / shots)

    return float(total)


def sample_qaoa_counts(
    model: CostModel,
    gammas: np.ndarray,
    betas: np.ndarray,
    shots: int,
    seed: int | None = None,
) -> Dict[str, int]:
    """
    Run the QAOA circuit and return shot counts.
    Requires pytket-qiskit (AerBackend).
    """
    from pytket.extensions.qiskit import AerBackend

    circ = build_qaoa_circuit(model, gammas, betas)
    backend = AerBackend()
    compiled = backend.get_compiled_circuit(circ)

    kwargs = {"n_shots": int(shots)}
    if seed is not None:
        kwargs["seed"] = int(seed)

    handle = backend.process_circuit(compiled, **kwargs)
    result = backend.get_result(handle)

    # result.get_counts() returns Dict[Tuple[int,...], int] or Dict[str,int] depending on backend.
    raw = result.get_counts()

    if not raw:
        return {}

    # Normalize to Dict[str,int] bitstring form.
    first_key = next(iter(raw.keys()))
    if isinstance(first_key, str):
        return {k: int(v) for k, v in raw.items()}

    # Assume tuple of bits (LSB/MSB ordering depends on backend); stringify in given order.
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(k, tuple):
            out["".join(str(int(b)) for b in k)] = int(v)
        else:
            out[str(k)] = int(v)
    return out


def most_probable_bitstring(counts: Dict[str, int]) -> str:
    if not counts:
        raise ValueError("Empty counts.")
    return max(counts.items(), key=lambda kv: kv[1])[0]

