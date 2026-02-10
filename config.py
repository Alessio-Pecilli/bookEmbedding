"""
=============================================================================
  config.py
=============================================================================
"""
USE_PLANAR_DEMO = True
NUM_PAGES = 2

# Alpha 15 è sufficiente con Adam. Non serve 40 (che crea muri troppo alti).
ALPHA = 15.0  
BETA = 2.0    # Aumentiamo un po' il costo degli incroci

# Adam funziona bene con LR più alti di GD, ma teniamolo stabile.
LAYERS = 3
STEPS = 200
LEARNING_RATE = 0.05 

# Parametri Random
NUM_NODES = 5
NUM_EDGES = 5
SEED = 42