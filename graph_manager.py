"""
=============================================================================
  graph_manager.py  —  Gestione del Grafo e Calcolo Classico degli Incroci
=============================================================================

Questo modulo contiene la logica CLASSICA del problema:
  1) Costruire (o generare) il grafo G con un ordine fisso dei nodi.
  2) Precomputare l'insieme C di tutte le coppie di archi che SI INCROCIANO
     se posti sulla stessa pagina, secondo la regola:

       Due archi e=(u,v) e f=(x,y)  (con u<v e x<y) si incrociano
       nella stessa pagina ⟺  u < x < v < y   (interlacciamento)

     Questa è la condizione geometrica fondamentale nel Book Embedding
     a ordine fisso.  Solo queste coppie "pericolose" generano penalità
     nell'Hamiltoniana quantistica.

Nessuna dipendenza da PennyLane: tutto è NumPy / Python puro.
=============================================================================
"""

import random
import itertools

# Importiamo la configurazione globale
import config

def assign_edge_weights(edges, low=None, high=None, seed=None):
    """
    Assegna un peso a ciascun arco (indice arco -> peso).
    Distribuzione: Uniform(low, high).
    """
    if low is None:
        low = config.WEIGHT_LOW
    if high is None:
        high = config.WEIGHT_HIGH

    rng = random.Random(seed if seed is not None else config.SEED)
    return {e_idx: rng.uniform(low, high) for e_idx in range(len(edges))}


# ─────────────────────────────────────────────────────────────────────────────
# get_graph()
# ─────────────────────────────────────────────────────────────────────────────
def get_graph():
    """
    Restituisce (nodes, edges, node_order) in base alla modalità scelta
    in config.py.

    Ritorni
    -------
    nodes : list[int]
        Lista dei nodi del grafo.
    edges : list[tuple[int,int]]
        Lista degli archi, ciascuno (u, v) con u < v.
    node_order : list[int]
        L'ordine fisso dei nodi sulla spina del libro.
        Nel Book Embedding, la spina ordina i vertici e gli archi
        vengono disegnati come semicerchi sopra/sotto (= le "pagine").
    """

    if config.USE_PLANAR_DEMO:
        # ─── CASO DEMO PLANARE ─────────────────────────────────────────
        # Grafo con 4 nodi e 2 archi specifici:
        #
        #   Nodi: 0 — 1 — 2 — 3   (spina del libro, ordine fisso)
        #   Archi: (0,2) e (1,3)
        #
        # Verifica incrocio:
        #   e=(0,2), f=(1,3) →  0 < 1 < 2 < 3  →  u<x<v<y  ✓
        #   Questi due archi SI INCROCIANO se messi sulla stessa pagina.
        #   La soluzione ottimale: metterli su pagine diverse → 0 incroci.
        # ────────────────────────────────────────────────────────────────
        # Nota: scegliamo un esempio con incrocio garantito:
        # e0=(0,2), e1=(1,3) con ordine [0,1,2,3] → 0<1<2<3 ⇒ incrocio se stessa pagina.
        nodes = [0, 1, 2, 3]
        edges = [(0, 2), (1, 3)]
        node_order = [0, 1, 2, 3]

        print("=" * 60)
        print("[GRAPH] Modalità: DEMO PLANARE")
        print(f"[GRAPH] Nodi: {nodes}")
        print(f"[GRAPH] Archi: {edges}")
        print(f"[GRAPH] Ordine sulla spina: {node_order}")
        print("[GRAPH] Questi archi si incrociano sulla stessa pagina.")
        print("[GRAPH] Soluzione attesa: archi su pagine DIVERSE → costo 0")
        print("=" * 60)

    else:
        # ─── CASO RANDOM ───────────────────────────────────────────────
        # Genera un grafo casuale con parametri da config.py.
        # L'ordine dei nodi è semplicemente 0, 1, ..., N-1.
        # ────────────────────────────────────────────────────────────────
        random.seed(config.SEED)
        nodes = list(range(config.NUM_NODES))
        node_order = list(nodes)   # ordine naturale sulla spina

        # Genera tutti i possibili archi e ne estrae NUM_EDGES a caso
        all_possible = [(i, j) for i in nodes for j in nodes if i < j]
        num_to_pick = min(config.NUM_EDGES, len(all_possible))
        edges = sorted(random.sample(all_possible, num_to_pick))

        print("=" * 60)
        print("[GRAPH] Modalità: RANDOM")
        print(f"[GRAPH] Nodi: {nodes}")
        print(f"[GRAPH] Archi ({len(edges)}): {edges}")
        print(f"[GRAPH] Ordine sulla spina: {node_order}")
        print(f"[GRAPH] Seed: {config.SEED}")
        print("=" * 60)

    return nodes, edges, node_order


# ─────────────────────────────────────────────────────────────────────────────
# precompute_crossings(edges, node_order)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_crossings(edges, node_order, edge_weights=None):
    """
    Precomputa l'insieme C di coppie di archi che si incrociano secondo
    la condizione geometrica del Book Embedding a ordine fisso:

        Due archi e=(u,v) e f=(x,y) si incrociano ⟺  u < x < v < y
        (dove u<v e x<y nell'ordine della spina)

    Questa è la parte CLASSICA del pre-processing.  Le coppie restituite
    saranno poi usate per costruire H_cross nell'Hamiltoniana quantistica.

    Parametri
    ---------
    edges : list[tuple[int,int]]
        Lista degli archi del grafo.
    node_order : list[int]
        Ordine fisso dei nodi sulla spina del libro.

    Ritorna
    -------
    crossing_pairs : list[tuple[int,int,float]]
        Lista di triple (i, j, w_ij) dove i,j sono INDICI degli archi in `edges`
        che si incrociano. w_ij è il peso dell'incrocio (w_i * w_j).
    """

    print("\n" + "=" * 60)
    print("[CROSSINGS] Inizio precomputo degli incroci...")
    print("=" * 60)

    # Mappa nodo → posizione sulla spina del libro.
    # Questo ci permette di confrontare la posizione relativa dei vertici.
    pos = {node: idx for idx, node in enumerate(node_order)}

    if edge_weights is None:
        edge_weights = assign_edge_weights(edges)

    crossing_pairs = []

    # Esaminiamo tutte le coppie distinte di archi
    for i, j in itertools.combinations(range(len(edges)), 2):
        e = edges[i]
        f = edges[j]

        # Normalizziamo: assicuriamoci che (u < v) e (x < y)
        # nell'ordine della spina (non nell'ordine numerico del nodo,
        # ma nella posizione sulla spina).
        u, v = sorted([pos[e[0]], pos[e[1]]])
        x, y = sorted([pos[f[0]], pos[f[1]]])

        # ── Condizione di incrocio: u < x < v < y ──
        # Se un arco "abbraccia" un vertice dell'altro e viceversa,
        # allora i due archi si incrociano geometricamente sulla pagina.
        #
        # Notiamo che la condizione è simmetrica nelle due coppie
        # (basta scambiarle), per cui controlliamo entrambi i casi.
        crosses = (u < x < v < y) or (x < u < y < v)

        # ── Log verboso ──
        edge_e_str = f"e{i}={edges[i]}"
        edge_f_str = f"e{j}={edges[j]}"
        pos_e_str = f"pos=({u},{v})"
        pos_f_str = f"pos=({x},{y})"

        if crosses:
            print(f"  [✓ INCROCIO]  {edge_e_str} {pos_e_str}  ×  "
                  f"{edge_f_str} {pos_f_str}")
            w = float(edge_weights[i]) * float(edge_weights[j])
            crossing_pairs.append((i, j, w))
        else:
            print(f"  [  nessuno ]  {edge_e_str} {pos_e_str}  ∥  "
                  f"{edge_f_str} {pos_f_str}")

    print(f"\n[CROSSINGS] Totale coppie pericolose |C| = {len(crossing_pairs)}")
    print("=" * 60)

    return crossing_pairs

import matplotlib.pyplot as plt
import numpy as np

def draw_book_embedding(nodes, edges, node_order, assignment):
    """
    Disegna il grafo in stile 'Book Embedding'.
    - Nodi sulla linea orizzontale (Spina).
    - Pagina 0: Archi sopra (Blu).
    - Pagina 1: Archi sotto (Rosso).
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Mappiamo i nodi alle coordinate X in base all'ordine
    # node_order = [0, 1, 2, 3] -> x=0, x=1, x=2, x=3
    pos = {node: i for i, node in enumerate(node_order)}
    
    # 2. Disegniamo la Spina (Linea nera)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.3)
    
    # 3. Disegniamo i Nodi
    x_vals = [pos[n] for n in nodes]
    y_vals = [0] * len(nodes)
    plt.scatter(x_vals, y_vals, s=200, color='black', zorder=5)
    
    # Etichette dei nodi
    for n in nodes:
        plt.text(pos[n], -0.1, str(n), ha='center', va='top', fontsize=12, fontweight='bold')

    # 4. Disegniamo gli Archi come Semicerchi
    for e_idx, page in assignment.items():
        if page == -1: continue # Salta archi invalidi
        
        u, v = edges[e_idx]
        x_u, x_v = pos[u], pos[v]
        
        # Calcoli per il semicerchio
        center = (x_u + x_v) / 2
        radius = abs(x_v - x_u) / 2
        
        # Generiamo i punti dell'arco
        theta = np.linspace(0, np.pi, 100)
        x_arc = center + radius * np.cos(theta)
        y_arc = radius * np.sin(theta)
        
        # Se è Pagina 1, ribaltiamo l'arco sotto (y negativi)
        color = 'blue'
        label = "Pagina 0"
        if page == 1:
            y_arc = -y_arc
            color = 'red'
            label = "Pagina 1"
        
        # Disegno arco
        plt.plot(x_arc, y_arc, color=color, linewidth=2, alpha=0.7)
        
        # Aggiungiamo etichetta a metà arco per capire che arco è
        mid_idx = 50
        plt.text(x_arc[mid_idx], y_arc[mid_idx], f"e{e_idx}", 
                 color=color, fontsize=9, ha='center', va='center', 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # Decorazioni
    plt.title("Visualizzazione Book Embedding (QAOA Result)", fontsize=14)
    plt.yticks([]) # Nascondi asse Y
    plt.xlabel("Ordine dei Nodi sulla Spina")
    
    # Legend trick
    plt.plot([], [], color='blue', label='Pagina 0 (Sopra)')
    plt.plot([], [], color='red', label='Pagina 1 (Sotto)')
    plt.legend()
    
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.show()
