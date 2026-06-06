"""
Build machine learning features for World Cup match prediction.

Input:
- data/silver/openfootball/silver_matches.csv
- data/silver/openfootball/silver_teams.csv

Output:
- data/modeling/training_dataset.csv

Goal:
Create one row per historical match with features from both teams.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"
MODELING_DIR = PROJECT_ROOT / "data" / "modeling"

MODELING_DIR.mkdir(parents=True, exist_ok=True)


def create_target(row):
    """
    Creates the target variable.

    target = what happened in the match from team1 perspective.
    """
    if row["team1_score_final"] > row["team2_score_final"]:
        return "team1_win"

    if row["team1_score_final"] < row["team2_score_final"]:
        return "team2_win"

    return "draw"


def build_training_dataset(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Combines match data with team historical metrics.

    We merge the teams table twice:
    - once for team1
    - once for team2
    """
    played_matches = matches[
        matches["match_played"] == True
    ].copy()

    played_matches = played_matches[
        played_matches["team1_score_final"].notna()
        & played_matches["team2_score_final"].notna()
    ].copy()

    played_matches["target"] = played_matches.apply(
        create_target,
        axis=1
    )

    team_features = teams[
        [
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
        ]
    ].copy()

    team1_features = team_features.add_prefix("team1_")
    team2_features = team_features.add_prefix("team2_")

    dataset = played_matches.merge(
        team1_features,
        left_on="team1_normalized",
        right_on="team1_team",
        how="left"
    )

    dataset = dataset.merge(
        team2_features,
        left_on="team2_normalized",
        right_on="team2_team",
        how="left"
    )

    dataset["win_rate_diff"] = (
        dataset["team1_win_rate"] - dataset["team2_win_rate"]
    )

    dataset["goals_per_game_diff"] = (
        dataset["team1_goals_per_game"] - dataset["team2_goals_per_game"]
    )

    dataset["goals_against_per_game_diff"] = (
        dataset["team1_goals_against_per_game"]
        - dataset["team2_goals_against_per_game"]
    )

    dataset["goal_difference_diff"] = (
        dataset["team1_goal_difference"]
        - dataset["team2_goal_difference"]
    )

    dataset["experience_diff"] = (
        dataset["team1_games"] - dataset["team2_games"]
    )

    selected_columns = [
        "match_id",
        "year",
        "round",
        "team1_normalized",
        "team2_normalized",
        "team1_score_final",
        "team2_score_final",
        "target",

        "team1_games",
        "team1_wins",
        "team1_goals_for",
        "team1_goals_against",
        "team1_goal_difference",
        "team1_win_rate",
        "team1_goals_per_game",
        "team1_goals_against_per_game",

        "team2_games",
        "team2_wins",
        "team2_goals_for",
        "team2_goals_against",
        "team2_goal_difference",
        "team2_win_rate",
        "team2_goals_per_game",
        "team2_goals_against_per_game",

        "win_rate_diff",
        "goals_per_game_diff",
        "goals_against_per_game_diff",
        "goal_difference_diff",
        "experience_diff",
    ]

    return dataset[selected_columns]


def main() -> None:
    matches = pd.read_csv(SILVER_DIR / "silver_matches.csv")
    teams = pd.read_csv(SILVER_DIR / "silver_teams.csv")

    training_dataset = build_training_dataset(matches, teams)

    training_dataset.to_csv(
        MODELING_DIR / "training_dataset.csv",
        index=False
    )

    print("Training dataset created successfully.")
    print(f"Rows: {training_dataset.shape[0]}")
    print(f"Columns: {training_dataset.shape[1]}")
    print(training_dataset["target"].value_counts())


if __name__ == "__main__":
    main()