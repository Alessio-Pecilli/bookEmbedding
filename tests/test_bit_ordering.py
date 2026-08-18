from qaoa_solver import (
    backend_counts_to_logical_counts,
    backend_key_to_logical_bitstring,
)


def test_textual_backend_fallback_is_reversed_once():
    # Qiskit textual counts are printed MSB -> LSB; canonical order is q0 -> qn.
    assert backend_key_to_logical_bitstring("100", 3) == "001"
    assert backend_counts_to_logical_counts({"100": 4}, 3) == {"001": 4}


def test_pytket_aer_known_computational_state_has_canonical_order():
    """Empirical contract: AerBackend returns tuple keys as (c[0], c[1], ...)."""
    from pytket.circuit import Circuit
    from pytket.extensions.qiskit import AerBackend

    circuit = Circuit(3, 3)
    circuit.X(0)
    circuit.X(2)
    for q in range(3):
        circuit.Measure(q, q)

    backend = AerBackend()
    handle = backend.process_circuit(
        backend.get_compiled_circuit(circuit), n_shots=64, seed=123
    )
    raw_counts = backend.get_result(handle).get_counts()

    assert raw_counts == {(1, 0, 1): 64}
    assert backend_counts_to_logical_counts(raw_counts, 3) == {"101": 64}
