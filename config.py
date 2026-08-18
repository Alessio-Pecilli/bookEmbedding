"""Configuration for the weighted fixed-order book-embedding experiment.

Values in this module are deliberately plain constants: changing a seed or a
shot count is visible in the source and is also copied to the result JSON.
"""

# Problem
NUM_PAGES = 2
MAX_QUBITS = 40

# Graph generation
USE_PLANAR_DEMO = True
NUM_NODES = 6
NUM_EDGES = 7
SEED = 42

# Edge weights
WEIGHT_LOW = 1.0
WEIGHT_HIGH = 10.0

# Hamiltonian.  ALPHA is the configured minimum.  build_cost_model raises it
# when the instance needs a larger penalty to protect the valid ground state.
ALPHA = 35.0
BETA = 1.0

# QAOA optimization
LAYERS = 5
STEPS = 200
LAYER_SWEEP = False
INIT_SCALE = 0.01

# Sampling and classical baseline
QAOA_OPTIMIZATION_SHOTS = 5_000
QAOA_FINAL_SHOTS = 20_000
CLASSICAL_TIME_LIMIT_S = 10.0
CLASSICAL_NUM_WORKERS = 1
CLASSICAL_OBJECTIVE_SCALE = 1_000_000_000

# Output / presentation
# Plotting is opt-in so batch runs and CI remain headless by default.
SHOW_PLOTS = False
