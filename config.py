"""
=============================================================================
  config.py  —  Configurazione
=============================================================================
"""

import time

# ── Problema: fixed-order book embedding ──
USE_PLANAR_DEMO = True
NUM_PAGES = 2

# Qubit = NUM_EDGES * NUM_PAGES
MAX_QUBITS = 40

# ── Pesi archi (Uniform distribution) ──
# Ogni arco e ha un peso w_e ~ Uniform(WEIGHT_LOW, WEIGHT_HIGH)
WEIGHT_LOW = 1.0
WEIGHT_HIGH = 10.0

# ── Hamiltoniana ──
ALPHA = 35.0  # Penalty per vincolo one-hot
# Il costo incroci ora è pesato (w_e * w_f). BETA è solo uno scaling opzionale.
BETA = 1.0

# ── QAOA / ottimizzazione ──
LAYERS = 5
STEPS = 200
LEARNING_RATE = 0.001
LAYER_SWEEP = False

# ── Grafo ──
NUM_NODES = 6
NUM_EDGES = 7
SEED = int(time.time())

# ── Gradient clipping ──
GRAD_CLIP = 0.3

# ── LR scheduler (Exponential Decay) ──
LR_DECAY_RATE = 0.95
LR_DECAY_EVERY = 20
LR_MIN = 0.0005

# ── Plateau detection & recovery ──
PLATEAU_WINDOW = 20
PLATEAU_THRESHOLD = 1e-4
MAX_PLATEAU_HITS = 3

# ── Divergence detection ──
DIVERGENCE_DELTA = 5.0
DIVERGENCE_COOLDOWN = 3

# ── Crash recovery ──
CHECKPOINT_FILE = "qaoa_checkpoint.npz"

# ── Inizializzazione parametri QAOA ──
INIT_SCALE = 0.01
