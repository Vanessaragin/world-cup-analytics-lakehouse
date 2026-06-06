"""
Simulate World Cup 2026 group stage using predicted match results.

Input:
- data/modeling/world_cup_2026_predictions.csv

Output:
- data/modeling/world_cup_2026_group_standings.csv
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_DIR = PROJECT_ROOT / "data" / "modeling"


def add_team_result(standings, group, team, points, win, draw, loss):
    """
    Adds one match result to a team's group standing.
    """
    key = (group, team)

    if key not in standings:
        standings[key] = {
            "group": group,
            "team": team,
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "games": 0,
        }

    standings[key]["points"] += points
    standings[key]["wins"] += win
    standings[key]["draws"] += draw
    standings[key]["losses"] += loss
    standings[key]["games"] += 1


def simulate_group_stage(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Converts predicted match results into group standings.
    """
    group_matches = predictions[
        predictions["round"].str.contains("Matchday", na=False)
    ].copy()

    standings = {}

    for _, match in group_matches.iterrows():
        group = match["group"]
        team1 = match["team1"]
        team2 = match["team2"]
        predicted_result = match["predicted_result"]

        if predicted_result == "team1_win":
            add_team_result(standings, group, team1, 3, 1, 0, 0)
            add_team_result(standings, group, team2, 0, 0, 0, 1)

        elif predicted_result == "team2_win":
            add_team_result(standings, group, team1, 0, 0, 0, 1)
            add_team_result(standings, group, team2, 3, 1, 0, 0)

        else:
            add_team_result(standings, group, team1, 1, 0, 1, 0)
            add_team_result(standings, group, team2, 1, 0, 1, 0)

    standings_df = pd.DataFrame(standings.values())

    standings_df = standings_df.sort_values(
        by=["group", "points", "wins"],
        ascending=[True, False, False],
    )

    standings_df["group_position"] = (
        standings_df
        .groupby("group")
        .cumcount()
        + 1
    )

    return standings_df


def main() -> None:
    predictions = pd.read_csv(
        MODELING_DIR / "world_cup_2026_predictions.csv"
    )

    standings = simulate_group_stage(predictions)

    standings.to_csv(
        MODELING_DIR / "world_cup_2026_group_standings.csv",
        index=False
    )

    print("World Cup 2026 group stage simulated successfully.")
    print()
    print(standings.head(30))


if __name__ == "__main__":
    main()