import random

import pytest

from classical_solver import (
    is_valid_assignment,
    solve_book_embedding_bruteforce,
    solve_book_embedding_cpsat,
    weighted_crossing_cost,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_bruteforce_and_cpsat_agree(seed):
    rng = random.Random(seed)
    crossings = [(0, 1, rng.uniform(1.0, 10.0)), (1, 2, rng.uniform(1.0, 10.0)), (0, 2, rng.uniform(1.0, 10.0))]
    brute = solve_book_embedding_bruteforce(3, 2, crossings)
    cpsat = solve_book_embedding_cpsat(3, 2, crossings, time_limit_s=10.0, num_workers=1)

    assert brute.weighted_cost == pytest.approx(cpsat.weighted_cost, abs=1e-9)
    assert is_valid_assignment(cpsat.assignment, 3, 2)
    assert cpsat.weighted_cost == pytest.approx(
        weighted_crossing_cost(cpsat.assignment, crossings), abs=1e-12
    )


def test_bruteforce_is_independent_ground_truth_for_a_larger_tiny_instance():
    crossings = [(0, 2, 2.0), (1, 3, 3.0), (0, 3, 5.0)]
    brute = solve_book_embedding_bruteforce(4, 2, crossings)
    cpsat = solve_book_embedding_cpsat(4, 2, crossings, time_limit_s=10.0, num_workers=1)
    assert cpsat.weighted_cost == pytest.approx(brute.weighted_cost)


def test_cpsat_preserves_sub_milliscale_weight_ordering():
    crossings = [(0, 1, 1.0005), (0, 2, 1.0006), (1, 2, 1.0004)]
    brute = solve_book_embedding_bruteforce(3, 2, crossings)
    cpsat = solve_book_embedding_cpsat(3, 2, crossings, num_workers=1)
    assert cpsat.weighted_cost == pytest.approx(brute.weighted_cost, abs=1e-12)
