"""
=============================================================================
  main.py  —  Orchestratore con Gradient Clipping, LR Scheduler,
              Plateau Detection & Crash Recovery
=============================================================================
"""

import os
import copy
import signal
import time
import numpy as standard_np                     # numpy puro (per deep copy sicure)
import pennylane as qml
from pennylane import numpy as np               # autograd numpy (per ottimizzazione)
import config
from graph_manager import get_graph, precompute_crossings
from qaoa_solver import build_hamiltonian, create_circuit
from book_viz import draw_book_embedding


# ─────────────────────────────────────────────────────────────────
#  Utility: deep copy parametri staccando da autograd
# ─────────────────────────────────────────────────────────────────
def _detach_params(params):
    """
    Converte params PennyLane in un array numpy puro (float64),
    spezzando QUALSIASI legame col grafo autograd.
    """
    return standard_np.array(params, dtype=standard_np.float64, copy=True)


def _to_grad_params(raw):
    """Riconverte un array numpy puro in un tensore PennyLane grad-enabled."""
    return np.array(raw, requires_grad=True)


# ─────────────────────────────────────────────────────────────────
#  Crash Recovery globale
# ─────────────────────────────────────────────────────────────────
_recovery_state = {
    "best_params": None,      # numpy puro, mai autograd
    "best_energy": float('inf'),
    "edges": None,
    "crossing_pairs": None,
    "nodes": None,
    "node_order": None,
    "n_qubits": None,
    "prob_fn": None,
    "interrupted": False,
}


def save_checkpoint():
    """Salva i migliori parametri trovati su disco."""
    bp = _recovery_state["best_params"]
    if bp is not None:
        path = os.path.join(os.path.dirname(__file__) or ".", config.CHECKPOINT_FILE)
        standard_np.savez(path,
                          best_params=bp,
                          best_energy=standard_np.array(_recovery_state["best_energy"]))
        print(f"\n  💾 Checkpoint salvato in '{path}'")


def _signal_handler(sig, frame):
    """Gestisce Ctrl+C: salva checkpoint e attiva il flag di interruzione."""
    print("\n\n⚡ INTERRUZIONE (Ctrl+C) — salvataggio in corso...")
    save_checkpoint()
    _recovery_state["interrupted"] = True


# ─────────────────────────────────────────────────────────────────
#  Utilità decodifica
# ─────────────────────────────────────────────────────────────────
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


def finalize_results():
    """
    Decodifica e mostra i risultati usando i MIGLIORI parametri trovati.
    Chiamata sia in caso di completamento normale, sia dopo crash/interrupt.
    """
    bp = _recovery_state["best_params"]
    be = _recovery_state["best_energy"]
    edges = _recovery_state["edges"]
    crossing_pairs = _recovery_state["crossing_pairs"]
    nodes = _recovery_state["nodes"]
    node_order = _recovery_state["node_order"]
    n_qubits = _recovery_state["n_qubits"]
    prob_fn = _recovery_state["prob_fn"]

    if bp is None or prob_fn is None:
        print("\n❌ Nessun parametro valido trovato. Impossibile decodificare.")
        return

    print(f"\n{'='*60}")
    print(f"  DECODIFICA FINALE (Best Energy: {be:.4f})")
    print(f"{'='*60}")

    try:
        bp_grad = _to_grad_params(bp)
        probs = prob_fn(bp_grad)
        best_idx = standard_np.argmax(probs)
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
            print(f"\n⚠️  Risultato parziale. Incroci: {crossings}, Violazioni: {violated}")

        # Mostra SEMPRE il grafico finale (anche con errori)
        print("[INFO] Visualizzo il risultato finale...")
        #draw_book_embedding(nodes, edges, node_order, assignment)
    except Exception as ex:
        print(f"\n❌ Errore nella decodifica finale: {ex}")


# ─────────────────────────────────────────────────────────────────
#  LR Scheduler
# ─────────────────────────────────────────────────────────────────
def get_scheduled_lr(initial_lr, step):
    """
    Exponential decay: lr = initial_lr * (decay_rate ^ (step // decay_every))
    Con floor a LR_MIN.
    """
    exponent = step // config.LR_DECAY_EVERY
    lr = initial_lr * (config.LR_DECAY_RATE ** exponent)
    return max(lr, config.LR_MIN)


# ─────────────────────────────────────────────────────────────────
#  Plateau Detection & Recovery
# ─────────────────────────────────────────────────────────────────
def detect_plateau(energy_window):
    if len(energy_window) < config.PLATEAU_WINDOW:
        return False
    arr = standard_np.array(energy_window[-config.PLATEAU_WINDOW:])
    span = float(arr.max() - arr.min())
    mean_abs = float(abs(arr.mean()))
    if mean_abs < 1e-12:
        return True
    return (span / mean_abs) < config.PLATEAU_THRESHOLD


def apply_recovery(best_params_raw, plateau_count, current_lr):
    """
    Applica la strategia di recovery.
    best_params_raw: array numpy PURO (non autograd).
    Restituisce (new_params_grad, new_optimizer, strategy_name).
    """
    strategies = [
        "LR Decay (÷2)",
        "Parameter Perturbation (σ=0.1)",
        "Partial Random Restart (50%)",
    ]
    idx = min(plateau_count - 1, len(strategies) - 1)
    strategy = strategies[idx]

    if idx == 0:
        new_lr = current_lr * 0.5
        new_optimizer = qml.AdamOptimizer(stepsize=new_lr)
        new_params = _to_grad_params(best_params_raw)
        print(f"        → LR: {current_lr:.5f} → {new_lr:.5f}")
        return new_params, new_optimizer, strategy

    elif idx == 1:
        noise = standard_np.random.normal(0, 0.1, size=best_params_raw.shape)
        new_raw = best_params_raw + noise
        new_optimizer = qml.AdamOptimizer(stepsize=current_lr)
        new_params = _to_grad_params(new_raw)
        return new_params, new_optimizer, strategy

    else:
        mask = standard_np.random.random(best_params_raw.shape) > 0.5
        random_vals = standard_np.random.uniform(0, 2 * standard_np.pi, size=best_params_raw.shape)
        mixed = standard_np.where(mask, random_vals, best_params_raw)
        new_optimizer = qml.AdamOptimizer(stepsize=config.LEARNING_RATE * 0.5)
        new_params = _to_grad_params(mixed)
        return new_params, new_optimizer, strategy


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   QAOA SOLVER — Fixed Order Book Embedding")
    print("   (Gradient Clipping + LR Scheduler + Crash Recovery)")
    print("="*60)

    signal.signal(signal.SIGINT, _signal_handler)

    nodes, edges, node_order = get_graph()
    crossing_pairs = precompute_crossings(edges, node_order)

    # ─── VISUALIZZAZIONE INIZIALE ───
    print("\n[INFO] Visualizzo il grafo iniziale...")
    print("       (Tutti gli archi su Pagina 0 per evidenziare gli incroci)")
    print("       >>> CHIUDI LA FINESTRA DEL GRAFICO PER CONTINUARE <<<")

    initial_assignment = {i: 0 for i in range(len(edges))}
    #_book_embedding(nodes, edges, node_order, initial_assignment)

    # ─── QAOA SETUP ───
    H, n_qubits = build_hamiltonian(edges, crossing_pairs)
    cost_fn, prob_fn = create_circuit(H, n_qubits, config.LAYERS)
    grad_fn = qml.grad(cost_fn)  # funzione gradiente pre-compilata

    # Popola recovery state
    _recovery_state["edges"] = edges
    _recovery_state["crossing_pairs"] = crossing_pairs
    _recovery_state["nodes"] = nodes
    _recovery_state["node_order"] = node_order
    _recovery_state["n_qubits"] = n_qubits
    _recovery_state["prob_fn"] = prob_fn

    print(f"\n[FASE 5] INIZIO OTTIMIZZAZIONE")
    print(f"  Qubit={n_qubits}, Layers={config.LAYERS}, LR={config.LEARNING_RATE}")
    print(f"  Grad Clip={config.GRAD_CLIP}, LR Decay={config.LR_DECAY_RATE} ogni {config.LR_DECAY_EVERY} step")

    np.random.seed(config.SEED)
    # Inizializzazione quasi-adiabatica: angoli piccoli vicini a 0
    # Simula evoluzione lenta che evita minimi locali di violazione totale
    params = np.array(
        np.random.uniform(-config.INIT_SCALE, config.INIT_SCALE, (2, config.LAYERS)),
        requires_grad=True
    )

    current_lr = config.LEARNING_RATE
    optimizer = qml.AdamOptimizer(stepsize=current_lr)

    best_energy = float('inf')
    best_params_raw = _detach_params(params)  # numpy puro
    energy_window = []
    plateau_count = 0
    divergence_cooldown = 0

    start_time = time.time()

    # ─── OPTIMIZATION LOOP ───
    try:
        for step in range(config.STEPS):
            if _recovery_state["interrupted"]:
                print("\n  🛑 Interruzione rilevata — esco dal loop.")
                break

            # ── 1. Calcola energia e gradiente separatamente ──
            energy = float(cost_fn(params))
            grad = grad_fn(params)

            # ── 2. Gradient Clipping (per-component) ──
            grad_clipped = np.clip(grad, -config.GRAD_CLIP, config.GRAD_CLIP)

            # ── 3. LR Scheduler: aggiorna LR con exponential decay ──
            scheduled_lr = get_scheduled_lr(config.LEARNING_RATE, step)
            if abs(optimizer.stepsize - scheduled_lr) > 1e-8:
                optimizer = qml.AdamOptimizer(stepsize=scheduled_lr)
                current_lr = scheduled_lr

            # ── 4. Applica gradiente clippato via Adam ──
            params, = optimizer.apply_grad([grad_clipped], [params])

            e_val = energy
            energy_window.append(e_val)

            # ── 5. Aggiorna best (numpy puro, NO autograd) ──
            if e_val < best_energy:
                best_energy = e_val
                best_params_raw = _detach_params(params)
                _recovery_state["best_params"] = best_params_raw.copy()
                _recovery_state["best_energy"] = best_energy
                marker = " *"
            else:
                marker = ""

            # ── 6. Divergence guard (con cooldown) ──
            if divergence_cooldown > 0:
                divergence_cooldown -= 1
            elif e_val > best_energy + config.DIVERGENCE_DELTA:
                print(f"\n  🔥 DIVERGENZA al passo {step} "
                      f"(E={e_val:.4f} >> Best={best_energy:.4f})")
                new_lr = current_lr * 0.5
                print(f"     → Ripristino best_params, LR: {current_lr:.5f} → {new_lr:.5f}")
                params = _to_grad_params(best_params_raw)
                optimizer = qml.AdamOptimizer(stepsize=new_lr)
                current_lr = new_lr
                energy_window.clear()
                divergence_cooldown = config.DIVERGENCE_COOLDOWN
                continue

            # ── 7. Plateau Detection ──
            if detect_plateau(energy_window):
                plateau_count += 1
                if plateau_count > config.MAX_PLATEAU_HITS:
                    print(f"\n  ⏹️  MAX PLATEAU HITS ({config.MAX_PLATEAU_HITS}). Early stop.")
                    break

                print(f"\n  ⚠️  PLATEAU al passo {step} "
                      f"(hit #{plateau_count}/{config.MAX_PLATEAU_HITS})")
                params, optimizer, strat = apply_recovery(
                    best_params_raw, plateau_count, current_lr)
                current_lr = optimizer.stepsize
                divergence_cooldown = config.DIVERGENCE_COOLDOWN
                print(f"     → Strategia: {strat}")
                energy_window.clear()

            # ── Log ──
            if step % 10 == 0 or step == config.STEPS - 1:
                elapsed = time.time() - start_time
                print(f"  Step {step:3d} | E: {e_val:8.4f} | Best: {best_energy:8.4f} "
                      f"| LR: {current_lr:.5f}{marker}  [{elapsed:.1f}s]")

    except Exception as ex:
        print(f"\n\n💥 CRASH: {ex}")
        print("   Salvataggio e decodifica con i migliori parametri trovati...")
        save_checkpoint()

    # ─── RISULTATI FINALI (sempre eseguiti) ───
    elapsed = time.time() - start_time
    print(f"\n[INFO] Terminato in {elapsed:.1f}s")
    print(f"[INFO] Best Energy: {best_energy:.4f}")
    print(f"[INFO] Plateau recovery: {plateau_count}")

    save_checkpoint()
    finalize_results()


if __name__ == "__main__":
    main()