import numpy as np

from pytket.circuit import Circuit

from qaoa_solver import (
    build_cost_model,
    build_qaoa_circuit,
    energy_of_bitstring,
)


def assert_unitaries_equal_up_to_global_phase(actual, expected, atol=1e-10):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    overlap = np.vdot(expected.ravel(), actual.ravel())
    assert abs(overlap) > 0.0
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase * expected, atol=atol, rtol=atol)


def test_rz_implements_single_qubit_cost_evolution():
    gamma = 0.3719
    coefficient = 1.2847
    circuit = Circuit(1)
    circuit.Rz(2.0 * gamma * coefficient / np.pi, 0)

    expected = np.diag([
        np.exp(-1j * gamma * coefficient),
        np.exp(+1j * gamma * coefficient),
    ])
    assert_unitaries_equal_up_to_global_phase(circuit.get_unitary(), expected)


def test_rx_implements_single_qubit_mixer_evolution():
    beta = 0.4173
    circuit = Circuit(1)
    circuit.Rx(2.0 * beta / np.pi, 0)

    expected = np.cos(beta) * np.eye(2) - 1j * np.sin(beta) * np.array([[0, 1], [1, 0]])
    assert_unitaries_equal_up_to_global_phase(circuit.get_unitary(), expected)


def test_cx_rz_cx_implements_zz_cost_evolution():
    gamma = 0.2931
    coefficient = 1.713
    circuit = Circuit(2)
    circuit.CX(0, 1)
    circuit.Rz(2.0 * gamma * coefficient / np.pi, 1)
    circuit.CX(0, 1)

    # pytket's two-qubit matrix indexing uses q0 as the least-significant
    # basis bit, hence ZZ eigenvalues are (+1, -1, -1, +1).
    expected = np.diag([
        np.exp(-1j * gamma * coefficient),
        np.exp(+1j * gamma * coefficient),
        np.exp(+1j * gamma * coefficient),
        np.exp(-1j * gamma * coefficient),
    ])
    assert_unitaries_equal_up_to_global_phase(circuit.get_unitary(), expected)


def test_project_cost_layer_matches_exp_minus_i_gamma_h_cost():
    gamma = 0.227
    model = build_cost_model(
        [(0, 2), (1, 3)],
        [(0, 1, 1.7)],
    )
    circuit = build_qaoa_circuit(
        model,
        np.array([gamma]),
        np.array([0.0]),
        measure=False,
        prepare_uniform=False,
    )

    energies = []
    for basis_state in range(2 ** model.n_qubits):
        logical_bits = "".join(
            str((basis_state >> qubit) & 1) for qubit in range(model.n_qubits)
        )
        energies.append(energy_of_bitstring(model, logical_bits))
    expected = np.diag(np.exp(-1j * gamma * np.asarray(energies)))
    assert_unitaries_equal_up_to_global_phase(circuit.get_unitary(), expected)


def test_project_mixer_layer_matches_exp_minus_i_beta_sum_x():
    beta = 0.319
    model = build_cost_model([(0, 1), (1, 2)], [])
    circuit = build_qaoa_circuit(
        model,
        np.array([0.0]),
        np.array([beta]),
        measure=False,
        prepare_uniform=False,
    )

    n = model.n_qubits
    expected = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for column in range(2 ** n):
        for mask in range(2 ** n):
            row = column ^ mask
            flips = mask.bit_count()
            expected[row, column] = (
                np.cos(beta) ** (n - flips)
                * (-1j * np.sin(beta)) ** flips
            )
    assert_unitaries_equal_up_to_global_phase(circuit.get_unitary(), expected)
