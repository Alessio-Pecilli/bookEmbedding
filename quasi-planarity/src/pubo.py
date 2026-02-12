import itertools
from collections import defaultdict
import pennylane as qml
import numpy as np


def build_pubo(C_sub, triangles_sub, lam=3.0):
    pubo = {}
    for i in C_sub.nodes():
        pubo[(i,)] = -1.0
    for t in triangles_sub:
        pubo[tuple(t)] = lam
    return pubo

def pubo_to_ising(pubo_terms):
    ising = defaultdict(float)
    for vars_, coeff in pubo_terms.items():
        k = len(vars_)
        for r in range(k+1):
            for subset in itertools.combinations(vars_, r):
                ising[tuple(sorted(subset))] += coeff*((-1)**r)/(2**k)
    coeffs = []
    ops = []
    for vars_, c in ising.items():
        if abs(c) < 1e-9 or len(vars_)==0:
            continue
        op = qml.PauliZ(vars_[0])
        for v in vars_[1:]:
            op = op @ qml.PauliZ(v)
        coeffs.append(c)
        ops.append(op)
    return qml.Hamiltonian(coeffs, ops)

# -------------------------------
# 4️⃣ Evaluate energy function
# -------------------------------
def evaluate(bitstring, triangles, lam=5.0):
    x = np.array(bitstring, dtype=int)
    val = -np.sum(x)
    for (i,j,k) in triangles:
        val += lam * x[i]*x[j]*x[k]
    return val