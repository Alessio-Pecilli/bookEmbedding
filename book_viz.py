"""
=============================================================================
  book_viz.py  —  Visualizzazione Book Embedding (N pagine)
=============================================================================
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import config


def draw_book_embedding(nodes, edges, node_order, assignment):
    """
    Disegna il book embedding per un numero arbitrario di pagine.

    Layout:
      - Pagine pari (0, 2, 4, ...): archi SOPRA la spina.
      - Pagine dispari (1, 3, 5, ...): archi SOTTO la spina.
      - Ogni livello successivo sullo stesso lato ha un fattore di
        scala crescente per evitare sovrapposizioni.

    Colori generati automaticamente da una colormap.
    """
    k = config.NUM_PAGES
    fig, ax = plt.subplots(figsize=(12, 7))

    # ── Colori distinti per ogni pagina ──
    if k <= 10:
        cmap = cm.get_cmap("tab10", k)
    else:
        cmap = cm.get_cmap("hsv", k)
    page_colors = [cmap(i) for i in range(k)]

    # ── Posizione dei nodi sulla spina ──
    pos = {node: i for i, node in enumerate(node_order)}

    # Spina
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.3)

    # Nodi
    x_vals = [pos[n] for n in nodes]
    y_vals = [0] * len(nodes)
    ax.scatter(x_vals, y_vals, s=300, color='black', zorder=5)

    for n in nodes:
        ax.text(pos[n], -0.15, str(n), ha='center', va='top',
                fontsize=12, color='black', fontweight='bold')

    # ── Disegna gli archi ──
    for e_idx, page in assignment.items():
        if page == -1:
            continue

        u, v = edges[e_idx]
        x_u, x_v = pos[u], pos[v]

        # Semicerchio
        center = (x_u + x_v) / 2
        radius = abs(x_v - x_u) / 2
        theta = np.linspace(0, np.pi, 100)

        x_arc = center + radius * np.cos(theta)
        y_arc = radius * np.sin(theta)

        # Direzione: pari → sopra, dispari → sotto
        # Livello: quante pagine dello stesso lato ci sono prima di questa
        level = page // 2          # 0→0, 1→0, 2→1, 3→1, 4→2, ...
        scale = 1.0 + level * 0.15  # offset crescente per evitare sovrapposizioni

        if page % 2 == 1:
            y_arc = -y_arc  # pagine dispari sotto

        y_arc = y_arc * scale

        color = page_colors[page]
        ax.plot(x_arc, y_arc, color=color, linewidth=2, alpha=0.8)

    # ── Legenda pagine ──
    for p in range(k):
        ax.plot([], [], color=page_colors[p], linewidth=3,
                label=f"Pagina {p}")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_title("Book Embedding Visualization", fontsize=14)
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    plt.show()