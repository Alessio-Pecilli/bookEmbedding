import csv

from results_io import append_summary_csv


def test_summary_csv_remains_rectangular_when_schema_grows(tmp_path):
    first = append_summary_csv({"seed": 42, "cost": 0.0}, base_dir=str(tmp_path))
    second = append_summary_csv(
        {"seed": 43, "cost": 1.0, "probability_valid": 0.5},
        base_dir=str(tmp_path),
    )
    assert first == second
    with open(second, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"seed": "42", "cost": "0.0", "probability_valid": ""},
        {"seed": "43", "cost": "1.0", "probability_valid": "0.5"},
    ]
