import itertools
import numpy as np


def evaluate(bitstring, triangles, lam=5.0):
    x = np.array(bitstring, dtype=int)
    val = -np.sum(x)
    for (i,j,k) in triangles:
        val += lam * x[i]*x[j]*x[k]
    return val


def classical_ground_state_from_pubo(pubo_terms):
    """
    Compute exact ground state for a PUBO Hamiltonian by evaluating
    energies directly (diagonal) instead of building the matrix.

    Parameters:
        pubo_terms : dict
            PUBO terms, keys are tuples of variable indices, values are coefficients
            Example: {(0,): -1.0, (1,): -1.0, (0,1,2): 3.0}

    Returns:
        ground_bitstring : np.array of 0/1
        ground_energy : float
    """
    n = max([max(k) for k in pubo_terms.keys()]) + 1 if pubo_terms else 0
    print(f"Computing exact ground state for n={n} variables (diagonal evaluation)...")

    best_energy = float("inf")
    best_bitstring = None

    # Enumerate all 2^n bitstrings
    for bits in itertools.product([0,1], repeat=n):
        x = np.array(bits, dtype=int)
        energy = 0.0
        for vars_, coeff in pubo_terms.items():
            prod = np.prod([x[i] for i in vars_])
            energy += coeff * prod
        if energy < best_energy:
            best_energy = energy
            best_bitstring = x

    print("Classical minimum energy:", best_energy)
    print("Classical optimal bitstring:", best_bitstring)
    return best_bitstring, best_energy
