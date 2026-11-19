"""Exploratory Data Analysis for NFL passing play-by-play data."""

from pathlib import Path
import pandas as pd
import plotly.express as px


DATA_PATH = "nfl_plays.parquet"
OUTPUT_DIR = Path("outputs")


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load passing play data and filter to valid down and yardage rows."""
    df = pd.read_parquet(file_path)
    valid_rows = (
        df["play_type"].notna()
        & df["down"].isin([1.0, 2.0, 3.0, 4.0])
        & df["ydstogo"].notna()
        & df["yards_gained"].notna()
        & df["first_down"].notna()
    )
    cleaned = df[valid_rows].copy()
    cleaned["down"] = cleaned["down"].astype(int)
    cleaned["first_down"] = cleaned["first_down"].astype(float)
    return cleaned


def compute_down_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute passing counts, yardage stats, and first-down rate by down."""
    grouped = (
        df.groupby("down", as_index=False)
        .agg(
            play_count=("yards_gained", "count"),
            mean_yards=("yards_gained", "mean"),
            median_yards=("yards_gained", "median"),
            first_down_rate=("first_down", "mean"),
        )
        .sort_values("down")
        .reset_index(drop=True)
    )
    return grouped


def compute_fourth_down_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute seasonal 4th-down pass volume, yards, and conversion rate."""
    fourth_down = df[df["down"] == 4]
    trends = (
        fourth_down.groupby("season", as_index=False)
        .agg(
            fourth_down_plays=("yards_gained", "count"),
            mean_yards=("yards_gained", "mean"),
            first_down_rate=("first_down", "mean"),
        )
        .sort_values("season")
        .reset_index(drop=True)
    )
    return trends


def create_charts(
    down_df: pd.DataFrame, fourth_df: pd.DataFrame, output_dir: Path
) -> None:
    """Save Plotly charts for down efficiency and 4th-down season trends."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_down = px.bar(
        down_df,
        x="down",
        y="mean_yards",
        title="Mean Yards Gained on Passing Plays by Down",
        labels={"down": "Down", "mean_yards": "Mean Yards Gained"},
    )
    fig_down.write_html(str(output_dir / "passing_efficiency_by_down.html"))

    fig_fourth = px.line(
        fourth_df,
        x="season",
        y="first_down_rate",
        markers=True,
        title="4th-Down Pass First-Down Conversion Rate by Season",
        labels={"season": "Season", "first_down_rate": "First-Down Rate"},
    )
    fig_fourth.write_html(
        str(output_dir / "fourth_down_conversion_by_season.html")
    )


def main() -> None:
    """Run data cleaning, summary calculations, and chart generation."""
    df = load_and_clean_data(DATA_PATH)
    down_summary = compute_down_efficiency(df)
    fourth_summary = compute_fourth_down_trends(df)

    print("Cleaned dataset shape:", df.shape)
    print("\nPassing Efficiency by Down:")
    print(down_summary.to_string(index=False))

    print("\n4th-Down Passing Trends by Season:")
    print(fourth_summary.to_string(index=False))

    create_charts(down_summary, fourth_summary, OUTPUT_DIR)


if __name__ == "__main__":
    main()
