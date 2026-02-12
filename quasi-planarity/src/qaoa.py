import pennylane as qml
import numpy as np
from src.pubo import build_pubo, pubo_to_ising
from .classical_solver import evaluate, classical_ground_state_from_pubo

from .utils import done, stage

def run_qaoa_component(C_sub, triangles_sub, original_nodes,
                       edge_labels, p=1, lam=3.0):

    n = C_sub.number_of_nodes()
    stage_time = stage(f"Running QAOA on component of size {n}")

    pubo = build_pubo(C_sub, triangles_sub, lam=lam)
    cost_h = pubo_to_ising(pubo)
    dev = qml.device("default.qubit", wires=n)


    def apply_cost(coeff, wires, gamma):
        angle = 2*gamma*coeff
        if len(wires)==1:
            qml.RZ(angle, wires=wires[0])
        elif len(wires)==2:
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.RZ(angle, wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
        elif len(wires)==3:
            qml.CNOT(wires=[wires[0],wires[2]])
            qml.CNOT(wires=[wires[1],wires[2]])
            qml.RZ(angle, wires=wires[2])
            qml.CNOT(wires=[wires[1],wires[2]])
            qml.CNOT(wires=[wires[0],wires[2]])

    def cost_layer(gamma):
        for c,op in zip(cost_h.coeffs, cost_h.ops):
            apply_cost(c, op.wires.tolist(), gamma)

    def mixer_layer(beta):
        for i in range(n):
            qml.RX(2*beta, wires=i)

    @qml.qnode(dev)
    def qaoa(params):
        p_len = len(params)//2
        gammas = params[:p_len]
        betas = params[p_len:]
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p_len):
            cost_layer(gammas[l])
            mixer_layer(betas[l])
        return qml.expval(cost_h)

    # Initialize parameters
    params = qml.numpy.array(qml.numpy.random.uniform(0,1,2*p), requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    for step in range(100):
        params = opt.step(qaoa, params)
    done(stage_time)

    # Sampling
    stage_time2 = stage("Sampling QAOA bitstrings")
    @qml.qnode(dev)
    def sample_qaoa(params):
        p_len = len(params)//2
        gammas = params[:p_len]
        betas = params[p_len:]
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p_len):
            cost_layer(gammas[l])
            mixer_layer(betas[l])
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]

    sample_qaoa = qml.set_shots(sample_qaoa, shots=2000)
    samples = sample_qaoa(params)
    z_samples = np.array(samples).T
    bitstrings = (z_samples==-1).astype(int)
    best_qaoa = np.array(bitstrings[0], dtype=int)
    energy_qaoa = evaluate(best_qaoa, triangles_sub)

    # Classical exact diagonalization
    pubo = build_pubo(C_sub, triangles_sub, lam=lam)

    best_classical, energy_classical = classical_ground_state_from_pubo(
        pubo
    )

    print("\n--- Comparison ---")
    print("QAOA energy:", energy_qaoa)
    print("QAOA bitstring:", best_qaoa)
    print("Exact ground energy:", energy_classical)
    print("Exact ground bitstring:", best_classical)

    if np.isclose(energy_qaoa, energy_classical):
        print("✔ QAOA found ground state!")
    else:
        print("✘ QAOA did NOT reach ground state.")



    return best_qaoa, energy_qaoa
