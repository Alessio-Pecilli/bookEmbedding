"""
=============================================================================
  book_viz.py  —  Modulo Grafico (Rinomina questo file!)
=============================================================================
"""
import matplotlib.pyplot as plt
import numpy as np

def draw_book_embedding(nodes, edges, node_order, assignment):
    """
    Disegna il grafo:
    - Nodi sulla linea centrale (Spina).
    - Pagina 0: Archi blu (Sopra).
    - Pagina 1: Archi rossi (Sotto).
    """
    plt.figure(figsize=(10, 6))
    
    # Posizione dei nodi sulla spina
    pos = {node: i for i, node in enumerate(node_order)}
    
    # Disegna la spina
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.3)
    
    # Disegna i nodi
    x_vals = [pos[n] for n in nodes]
    y_vals = [0] * len(nodes)
    plt.scatter(x_vals, y_vals, s=300, color='black', zorder=5)
    
    for n in nodes:
        plt.text(pos[n], -0.1, str(n), ha='center', va='top', fontsize=12, color='black', fontweight='bold')

    # Disegna gli archi
    for e_idx, page in assignment.items():
        if page == -1: continue 
        
        u, v = edges[e_idx]
        x_u, x_v = pos[u], pos[v]
        
        # Geometria dell'arco
        center = (x_u + x_v) / 2
        radius = abs(x_v - x_u) / 2
        theta = np.linspace(0, np.pi, 100)
        
        x_arc = center + radius * np.cos(theta)
        y_arc = radius * np.sin(theta)
        
        color = 'blue'
        if page == 1:
            y_arc = -y_arc  # Ribalta sotto
            color = 'red'
        
        plt.plot(x_arc, y_arc, color=color, linewidth=2, alpha=0.8)

    plt.title("Book Embedding Visualization", fontsize=14)
    plt.yticks([])
    plt.box(False)
    plt.tight_layout()
    plt.show()