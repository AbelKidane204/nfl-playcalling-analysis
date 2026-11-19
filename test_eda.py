"""Unit tests for eda.py."""

import pandas as pd
from eda import compute_down_efficiency, compute_fourth_down_trends


def test_compute_down_efficiency() -> None:
    """Verify down-level aggregation outputs expected columns and values."""
    sample = pd.DataFrame({
        "down": [1.0, 1.0, 3.0, 4.0],
        "yards_gained": [8.0, 2.0, 12.0, 4.0],
        "first_down": [1.0, 0.0, 1.0, 1.0],
    })
    result = compute_down_efficiency(sample)

    assert list(result["down"]) == [1, 3, 4]
    first_down_row = result[result["down"] == 1].iloc[0]
    assert first_down_row["play_count"] == 2
    assert first_down_row["mean_yards"] == 5.0
    assert first_down_row["first_down_rate"] == 0.5


def test_compute_fourth_down_trends() -> None:
    """Verify 4th-down seasonal summary filters and aggregates accurately."""
    sample = pd.DataFrame({
        "season": [2020, 2020, 2021, 2021],
        "down": [4, 3, 4, 4],
        "yards_gained": [5.0, 15.0, 0.0, 6.0],
        "first_down": [1.0, 1.0, 0.0, 1.0],
    })
    result = compute_fourth_down_trends(sample)

    assert list(result["season"]) == [2020, 2021]
    row_2020 = result[result["season"] == 2020].iloc[0]
    assert row_2020["fourth_down_plays"] == 1
    assert row_2020["first_down_rate"] == 1.0

    row_2021 = result[result["season"] == 2021].iloc[0]
    assert row_2021["fourth_down_plays"] == 2
    assert row_2021["first_down_rate"] == 0.5


def main() -> None:
    """Run all unit tests."""
    test_compute_down_efficiency()
    test_compute_fourth_down_trends()
    print("All EDA tests passed!")


if __name__ == "__main__":
    main()
