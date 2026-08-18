import pytest

import config
from main import evaluate_sampled_solutions


def test_metrics_use_all_counts_and_classical_ground_truth(monkeypatch):
    monkeypatch.setattr(config, "NUM_PAGES", 2)
    counts = {"1001": 3, "1010": 1, "0000": 2}
    metrics = evaluate_sampled_solutions(
        counts, 2, iter([(0, 1, 6.0)]), optimal_cost=0.0
    )

    assert metrics["probability_valid_solution"] == pytest.approx(4 / 6)
    assert metrics["probability_optimal_solution"] == pytest.approx(3 / 6)
    assert metrics["best_valid_weighted_cost_sampled"] == pytest.approx(0.0)
    assert metrics["best_valid_bitstring_sampled"] == "1001"
    assert metrics["expected_weighted_cost_valid"] == pytest.approx(1.5)
    assert metrics["most_probable_bitstring"] == "1001"
    assert metrics["most_probable_weighted_cost"] == pytest.approx(0.0)
    assert metrics["approximation_gap_vs_classical"] == pytest.approx(0.0)
    assert metrics["approximation_ratio"] is None
