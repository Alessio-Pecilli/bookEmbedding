"""
=============================================================================
  main.py  —  Orchestratore (FIXED)
=============================================================================
"""

import time
import pennylane as qml
from pennylane import numpy as np
import config
from graph_manager import get_graph, precompute_crossings, draw_book_embedding
from qaoa_solver import build_hamiltonian, create_circuit

def decode_solution(bitstring, edges):
    assignment = {}
    for e_idx in range(len(edges)):
        start = e_idx * config.NUM_PAGES
        end = start + config.NUM_PAGES
        bits = bitstring[start:end]
        active_pages = [p for p in range(config.NUM_PAGES) if bits[p] == 1]
        if len(active_pages) == 1:
            assignment[e_idx] = active_pages[0]
        else:
            assignment[e_idx] = -1
    return assignment

def count_crossings_in_solution(assignment, crossing_pairs):
    count = 0
    for (e_idx, f_idx) in crossing_pairs:
        p_e = assignment.get(e_idx, -1)
        p_f = assignment.get(f_idx, -1)
        if p_e >= 0 and p_e == p_f:
            count += 1
    return count

def main():
    print("\n" + "="*60)
    print("   QAOA SOLVER — Fixed Order Book Embedding (ADAM + FIXED)")
    print("="*60)
    
    nodes, edges, node_order = get_graph()
    crossing_pairs = precompute_crossings(edges, node_order)
    H, n_qubits = build_hamiltonian(edges, crossing_pairs)
    cost_fn, prob_fn = create_circuit(H, n_qubits, config.LAYERS)
    
    print("\n[FASE 5] INIZIO OTTIMIZZAZIONE")
    
    np.random.seed(config.SEED)
    # Params shape: (2, LAYERS) -> params[0]=gammas, params[1]=betas
    params = np.random.uniform(0, 2*np.pi, (2, config.LAYERS), requires_grad=True)
    
    optimizer = qml.AdamOptimizer(stepsize=config.LEARNING_RATE)
    
    best_energy = float('inf')
    best_params = params.copy()
    
    start_time = time.time()
    
    for step in range(config.STEPS):
        # L'ottimizzatore ora passa 'params' intero a cost_fn, che ora lo accetta!
        params, energy = optimizer.step_and_cost(cost_fn, params)
        e_val = float(energy)
        
        if e_val < best_energy:
            best_energy = e_val
            best_params = params.copy()
            marker = "*"
        else:
            marker = ""
            
        if step % 10 == 0 or step == config.STEPS - 1:
            print(f"  Step {step:3d} | E: {e_val:8.4f} | Best: {best_energy:8.4f} {marker}")

    print(f"\n[INFO] Best Energy: {best_energy:.4f}")

    # --- CORREZIONE QUI ---
    print("\n[FASE 6] DECODIFICA (Best Params)")
    
    # Ora passiamo best_params intero, perché prob_fn è stata aggiornata
    probs = prob_fn(best_params) 
    
    best_idx = np.argmax(probs)
    best_bs = [int(b) for b in format(best_idx, f'0{n_qubits}b')]
    
    print(f"  Stato vincente: |{''.join(map(str, best_bs))}>")
    
    assignment = decode_solution(best_bs, edges)
    crossings = count_crossings_in_solution(assignment, crossing_pairs)
    violated = sum(1 for v in assignment.values() if v == -1)
    
    print("-" * 40)
    for e, p in assignment.items():
        stat = "✓" if p >= 0 else "✗"
        pg = f"Pagina {p}" if p >= 0 else "ERRORE"
        print(f"  {stat} Arco {edges[e]} -> {pg}")
    print("-" * 40)
    
    if crossings == 0 and violated == 0:
        print("\n🎉🎉🎉  SUCCESSO! 0 INCROCI  🎉🎉🎉")
    else:
        print(f"\n⚠️  Fallimento. Incroci: {crossings}, Violazioni: {violated}")
    if crossings == 0:
        draw_book_embedding(nodes, edges, node_order, assignment)

if __name__ == "__main__":
    main()
    