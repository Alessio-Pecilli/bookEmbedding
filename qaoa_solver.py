"""
=============================================================================
  qaoa_solver.py  —  Costruzione dell'Hamiltoniana e Circuito QAOA
=============================================================================
"""

import pennylane as qml
from pennylane import numpy as np
import config

def get_num_qubits(num_edges):
    return num_edges * config.NUM_PAGES

def build_hamiltonian(edges, crossing_pairs):
    num_edges = len(edges)
    n_qubits = get_num_qubits(num_edges)
    
    print("\n" + "="*60)
    print(f"[HAMILTONIANA] Costruzione H su {n_qubits} qubit")
    print("="*60)
    
    coeffs = []
    obs = []
    
    # --- H_PAGE ---
    print("  ... Aggiungendo H_page (Vincolo)")
    for e_idx in range(num_edges):
        qubits_e = [e_idx * config.NUM_PAGES + p for p in range(config.NUM_PAGES)]
        
        if config.NUM_PAGES == 2:
            # Caso ottimizzato k=2: H = A * (I + Z0 Z1) / 2
            q0, q1 = qubits_e
            # Termine I
            coeffs.append(config.ALPHA * 0.5)
            obs.append(qml.Identity(0))
            # Termine Z0 Z1
            coeffs.append(config.ALPHA * 0.5)
            obs.append(qml.prod(qml.PauliZ(q0), qml.PauliZ(q1)))
        else:
            # Caso generico (omesso per brevità, usa logica precedente se serve k>2)
            pass

    # --- H_CROSS ---
    print("  ... Aggiungendo H_cross (Costo)")
    for (e_idx, f_idx) in crossing_pairs:
        for p in range(config.NUM_PAGES):
            q_ep = e_idx * config.NUM_PAGES + p
            q_fp = f_idx * config.NUM_PAGES + p
            
            # x_ep * x_fp = (I - Z_ep - Z_fp + Z_ep Z_fp) / 4
            # Termini I, Z_ep, Z_fp, Z_ep Z_fp
            coeffs.append(config.BETA * 0.25)
            obs.append(qml.Identity(0))
            
            coeffs.append(config.BETA * -0.25)
            obs.append(qml.PauliZ(q_ep))
            
            coeffs.append(config.BETA * -0.25)
            obs.append(qml.PauliZ(q_fp))
            
            coeffs.append(config.BETA * 0.25)
            obs.append(qml.prod(qml.PauliZ(q_ep), qml.PauliZ(q_fp)))

    hamiltonian = qml.Hamiltonian(coeffs, obs)
    hamiltonian = qml.simplify(hamiltonian)
    return hamiltonian, n_qubits

def create_circuit(hamiltonian, n_qubits, layers):
    
    print(f"[CIRCUITO] Creazione QNode (Qubit={n_qubits}, Layers={layers})")
    
    mixer_h = qml.Hamiltonian(
        [1.0]*n_qubits, 
        [qml.PauliX(i) for i in range(n_qubits)]
    )
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # === CORREZIONE QUI: params unico argomento ===
    @qml.qnode(dev)
    def cost_function(params):
        gammas = params[0]
        betas = params[1]
        
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
            
        for l in range(layers):
            qml.qaoa.cost_layer(gammas[l], hamiltonian)
            qml.qaoa.mixer_layer(betas[l], mixer_h)
            
        return qml.expval(hamiltonian)

    # === CORREZIONE QUI: params unico argomento ===
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