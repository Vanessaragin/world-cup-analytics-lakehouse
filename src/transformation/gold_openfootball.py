"""
Gold transformation for World Cup analytics.

Input:
- silver_teams.csv
- silver_world_cups.csv

Output:
- gold_team_ranking.csv
- gold_world_cup_summary.csv
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "openfootball"

GOLD_DIR.mkdir(parents=True, exist_ok=True)


def create_gold_team_ranking(teams: pd.DataFrame) -> pd.DataFrame:
    df = teams.copy()

    df["ranking_score"] = (
        df["wins"] * 3
        + df["draws"]
        + df["goal_difference"] * 0.1
    )

    df = df.sort_values(
        by="ranking_score",
        ascending=False
    ).reset_index(drop=True)

    df["ranking_position"] = df.index + 1

    columns = [
        "ranking_position",
        "team",
        "games",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "win_rate",
        "goals_per_game",
        "goals_against_per_game",
        "ranking_score"
    ]

    return df[columns]


def create_gold_world_cup_summary(world_cups: pd.DataFrame) -> pd.DataFrame:
    df = world_cups.copy()

    df["offensive_rank"] = (
        df["avg_goals_per_game"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    df = df.sort_values(
        by="avg_goals_per_game",
        ascending=False
    )

    return df


def main() -> None:
    teams = pd.read_csv(SILVER_DIR / "silver_teams.csv")
    world_cups = pd.read_csv(SILVER_DIR / "silver_world_cups.csv")

    gold_team_ranking = create_gold_team_ranking(teams)
    gold_world_cup_summary = create_gold_world_cup_summary(world_cups)

    gold_team_ranking.to_csv(
        GOLD_DIR / "gold_team_ranking.csv",
        index=False
    )

    gold_world_cup_summary.to_csv(
        GOLD_DIR / "gold_world_cup_summary.csv",
        index=False
    )

    print("Gold created successfully.")
    print(f"Team ranking: {gold_team_ranking.shape}")
    print(f"World Cup summary: {gold_world_cup_summary.shape}")


if __name__ == "__main__":
    main()