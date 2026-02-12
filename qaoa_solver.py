"""
=============================================================================
  qaoa_solver.py  —  Costruzione dell'Hamiltoniana e Circuito QAOA
                     (con lightning.qubit + adjoint differentiation)
=============================================================================
"""

import pennylane as qml
from pennylane import numpy as np
import config

def get_num_qubits(num_edges):
    return num_edges * config.NUM_PAGES

def build_hamiltonian(edges, crossing_pairs):
    num_edges = len(edges)
    k = config.NUM_PAGES
    n_qubits = get_num_qubits(num_edges)
    
    coeffs = []
    obs = []
    
    # --- H_PAGE (Vincolo: Un arco su una sola pagina) ---
    for e_idx in range(num_edges):
        qubits_e = [e_idx * k + p for p in range(k)]
        
        # 1. Costante
        coeffs.append(config.ALPHA * (1 - 0.75*k + 0.25*k**2))
        obs.append(qml.Identity(0))
        
        # 2. Lineari (Z_p)
        c_lin = config.ALPHA * (1 - k/2)
        for q in qubits_e:
            coeffs.append(c_lin)
            obs.append(qml.PauliZ(q))
            
        # 3. Quadratici (Z_p * Z_q)
        c_quad = 0.5 * config.ALPHA
        for i, q1 in enumerate(qubits_e):
            for q2 in qubits_e[i+1:]:
                coeffs.append(c_quad)
                obs.append(qml.prod(qml.PauliZ(q1), qml.PauliZ(q2)))

    # --- H_CROSS (Costo: Minimizzare incroci) ---
    for (e_idx, f_idx) in crossing_pairs:
        for p in range(k):
            q_ep = e_idx * k + p
            q_fp = f_idx * k + p
            
            # Espansione di beta * x_ep * x_fp
            coeffs.extend([config.BETA * 0.25, config.BETA * -0.25, 
                           config.BETA * -0.25, config.BETA * 0.25])
            obs.extend([qml.Identity(0), qml.PauliZ(q_ep), 
                        qml.PauliZ(q_fp), qml.prod(qml.PauliZ(q_ep), qml.PauliZ(q_fp))])

    return qml.simplify(qml.Hamiltonian(coeffs, obs)), n_qubits

def create_circuit(hamiltonian, n_qubits, layers):
    
    
    try:
        dev = qml.device("qulacs.simulator", wires=n_qubits)
        backend = "qulacs.simulator"  
    except Exception:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        backend = "lightning.qubit"
    print(dev)
    
    print(f"[CIRCUITO] Backend={backend}, Qubit={n_qubits}, Layers={layers}")
    
    mixer_h = qml.Hamiltonian(
        [1.0]*n_qubits, 
        [qml.PauliX(i) for i in range(n_qubits)]
    )
    
    # ── cost_function: adjoint diff per gradienti veloci ──
    @qml.qnode(dev, diff_method="best")
    def cost_function(params):
        gammas = params[0]
        betas = params[1]
        
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
            
        for l in range(layers):
            qml.qaoa.cost_layer(gammas[l], hamiltonian)
            qml.qaoa.mixer_layer(betas[l], mixer_h)
            
        return qml.expval(hamiltonian)

    # ── prob_function: non serve gradiente, best_method di default ──
    @qml.qnode(dev)
    def prob_function(params):
        gammas = params[0]
        betas = params[1]
        
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
            
        for l in range(layers):
            qml.qaoa.cost_layer(gammas[l], hamiltonian)
            qml.qaoa.mixer_layer(betas[l], mixer_h)
            
        return qml.probs(wires=range(n_qubits))
    
    return cost_function, prob_function