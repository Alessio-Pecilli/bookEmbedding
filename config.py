"""
=============================================================================
  config.py  —  Configurazione "Hardened"
=============================================================================
"""
import time
USE_PLANAR_DEMO = True
NUM_PAGES = 2

# --- Pesi Hamiltoniana (bilanciati per convergenza) ---
ALPHA = 35.0   # Alto per forzare vincolo one-hot (evita violazioni)
BETA = 5.0     # Costo incroci più significativo

# --- Ottimizzazione ---
LAYERS = 5             # 5 layers = sweet spot per 20 qubit
STEPS = 200            
LEARNING_RATE = 0.001  # Leggermente più veloce, sicuro con clipping

# Parametri grafo
NUM_NODES = 6
NUM_EDGES = 7
SEED = int(time.time())

# ── Gradient Clipping ──
GRAD_CLIP = 0.3  # Stretto per gestire ALPHA alto

# ── LR Scheduler (Exponential Decay) ──
LR_DECAY_RATE  = 0.95
LR_DECAY_EVERY = 20
LR_MIN         = 0.0005

# ── Plateau Detection & Recovery ──
PLATEAU_WINDOW    = 20
PLATEAU_THRESHOLD = 1e-4
MAX_PLATEAU_HITS  = 3

# ── Divergence Detection ──
DIVERGENCE_DELTA    = 5.0  # Più tollerante con ALPHA alto
DIVERGENCE_COOLDOWN = 3

# ── Crash Recovery ──
CHECKPOINT_FILE = "qaoa_checkpoint.npz"

# ── Inizializzazione Intelligente ──
INIT_SCALE = 0.01  # Angoli piccoli (quasi-adiabatico) invece di 0-2π