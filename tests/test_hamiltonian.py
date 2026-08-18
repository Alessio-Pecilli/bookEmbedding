import itertools

import pytest

import config
from qaoa_solver import (
    build_cost_model,
    direct_binary_objective,
    energy_of_bitstring,
    estimate_energy_from_counts,
)


@pytest.fixture(autouse=True)
def two_pages(monkeypatch):
    monkeypatch.setattr(config, "NUM_PAGES", 2)
    monkeypatch.setattr(config, "ALPHA", 35.0)
    monkeypatch.setattr(config, "BETA", 1.0)


def model(weight=6.0):
    return build_cost_model([(0, 2), (1, 3)], [(0, 1, weight)])


def test_crossing_on_different_pages_is_zero():
    m = model()
    assert energy_of_bitstring(m, "1001") == pytest.approx(0.0)
    assert direct_binary_objective(m, "1001") == pytest.approx(0.0)


def test_crossing_on_same_page_has_weighted_cost():
    m = model()
    assert energy_of_bitstring(m, "1010") == pytest.approx(6.0)
    assert direct_binary_objective(m, "1010") == pytest.approx(6.0)


def test_zero_pages_has_one_hot_penalty():
    m = model()
    # e0 has no active page and e1 is on page 1; there is no crossing term.
    assert energy_of_bitstring(m, "0001") == pytest.approx(m.alpha)


def test_two_pages_for_one_edge_has_one_hot_penalty():
    m = model()
    # e0 is assigned to both pages; e1 is on page 1, so its crossing term is 6.
    assert energy_of_bitstring(m, "1101") == pytest.approx(m.alpha + 6.0)


def test_z_hamiltonian_matches_direct_binary_objective_for_every_state():
    m = model(4.5)
    for bits in itertools.product("01", repeat=m.n_qubits):
        bitstring = "".join(bits)
        assert energy_of_bitstring(m, bitstring) == pytest.approx(
            direct_binary_objective(m, bitstring), abs=1e-10
        )


def test_multiple_crossings_match_direct_objective_for_every_state():
    m = build_cost_model(
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1, 2.0), (1, 2, 3.5), (0, 2, 4.25)],
    )
    for bits in itertools.product("01", repeat=m.n_qubits):
        bitstring = "".join(bits)
        assert energy_of_bitstring(m, bitstring) == pytest.approx(
            direct_binary_objective(m, bitstring), abs=1e-10
        )


def test_energy_estimator_uses_the_same_logical_order():
    m = model()
    assert estimate_energy_from_counts(m, {"1001": 1}) == pytest.approx(
        energy_of_bitstring(m, "1001")
    )


def test_effective_penalty_protects_ground_state_from_invalid_assignments():
    m = model(100.0)
    assert m.alpha > 100.0
    states = ["".join(bits) for bits in itertools.product("01", repeat=m.n_qubits)]
    minimum = min(energy_of_bitstring(m, state) for state in states)
    ground_states = [state for state in states if energy_of_bitstring(m, state) == pytest.approx(minimum)]
    assert ground_states
    assert all(
        all(block.count("1") == 1 for block in (state[0:2], state[2:4]))
        for state in ground_states
    )
