"""
Predict World Cup 2026 matches.

Input:
- models/match_result_model.pkl
- data/silver/openfootball/silver_matches.csv
- data/silver/openfootball/silver_teams.csv

Output:
- data/modeling/world_cup_2026_predictions.csv

Important:
Some 2026 teams may not exist in the historical dataset.
For these teams, we use the average historical metrics.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"
MODELING_DIR = PROJECT_ROOT / "data" / "modeling"
MODELS_DIR = PROJECT_ROOT / "models"

MODELING_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
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


def is_placeholder(team: str) -> bool:
    """
    Checks if a team name is a placeholder instead of a real team.

    Examples:
    - W101
    - L102
    - 2A
    - 3A/B/C/D/F
    """
    if pd.isna(team):
        return True

    team = str(team)

    if "/" in team:
        return True

    if team.startswith("W") and team[1:].isdigit():
        return True

    if team.startswith("L") and team[1:].isdigit():
        return True

    if len(team) == 2 and team[0].isdigit() and team[1].isalpha():
        return True

    return False


def get_team_data(team: str, teams: pd.DataFrame) -> pd.Series:
    """
    Returns historical metrics for a team.

    If the team does not exist in the historical table,
    returns the average metrics of all teams.

    This allows predictions for new or low-history teams.
    """
    team_row = teams[teams["team"] == team]

    if not team_row.empty:
        return team_row.iloc[0]

    average_data = teams.mean(numeric_only=True)
    average_data["team"] = team

    return average_data


def build_match_features(
    team1: str,
    team2: str,
    teams: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates model features for one match.
    """
    team1_data = get_team_data(team1, teams)
    team2_data = get_team_data(team2, teams)

    row = {
        "team1_games": team1_data["games"],
        "team1_wins": team1_data["wins"],
        "team1_goals_for": team1_data["goals_for"],
        "team1_goals_against": team1_data["goals_against"],
        "team1_goal_difference": team1_data["goal_difference"],
        "team1_win_rate": team1_data["win_rate"],
        "team1_goals_per_game": team1_data["goals_per_game"],
        "team1_goals_against_per_game": team1_data["goals_against_per_game"],

        "team2_games": team2_data["games"],
        "team2_wins": team2_data["wins"],
        "team2_goals_for": team2_data["goals_for"],
        "team2_goals_against": team2_data["goals_against"],
        "team2_goal_difference": team2_data["goal_difference"],
        "team2_win_rate": team2_data["win_rate"],
        "team2_goals_per_game": team2_data["goals_per_game"],
        "team2_goals_against_per_game": team2_data["goals_against_per_game"],
    }

    row["win_rate_diff"] = (
        row["team1_win_rate"] - row["team2_win_rate"]
    )

    row["goals_per_game_diff"] = (
        row["team1_goals_per_game"]
        - row["team2_goals_per_game"]
    )

    row["goals_against_per_game_diff"] = (
        row["team1_goals_against_per_game"]
        - row["team2_goals_against_per_game"]
    )

    row["goal_difference_diff"] = (
        row["team1_goal_difference"]
        - row["team2_goal_difference"]
    )

    row["experience_diff"] = (
        row["team1_games"] - row["team2_games"]
    )

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_2026_matches() -> pd.DataFrame:
    """
    Predicts 2026 matches where both sides are real teams.

    Placeholder knockout matches are skipped.
    """
    model = joblib.load(MODELS_DIR / "match_result_model.pkl")

    matches = pd.read_csv(SILVER_DIR / "silver_matches.csv")
    teams = pd.read_csv(SILVER_DIR / "silver_teams.csv")

    matches_2026 = matches[
        matches["year"] == 2026
    ].copy()

    predictions = []

    for _, match in matches_2026.iterrows():
        team1 = match["team1_normalized"]
        team2 = match["team2_normalized"]

        if is_placeholder(team1) or is_placeholder(team2):
            continue

        features = build_match_features(
            team1=team1,
            team2=team2,
            teams=teams
        )

        probabilities = model.predict_proba(features)[0]
        classes = model.classes_

        probability_dict = {
            class_name: probability
            for class_name, probability in zip(classes, probabilities)
        }

        predicted_result = max(
            probability_dict,
            key=probability_dict.get
        )

        predictions.append(
            {
                "match_id": match["match_id"],
                "year": match["year"],
                "round": match["round"],
                "group": match["group"],
                "date": match["date"],
                "time": match["time"],
                "team1": team1,
                "team2": team2,
                "prob_draw": probability_dict.get("draw", 0),
                "prob_team1_win": probability_dict.get("team1_win", 0),
                "prob_team2_win": probability_dict.get("team2_win", 0),
                "predicted_result": predicted_result,
            }
        )

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(
        MODELING_DIR / "world_cup_2026_predictions.csv",
        index=False
    )

    return predictions_df


def main() -> None:
    predictions = predict_2026_matches()

    print("World Cup 2026 predictions created successfully.")
    print(f"Predicted matches: {len(predictions)}")
    print()

    print(
        predictions[
            [
                "group",
                "team1",
                "team2",
                "prob_team1_win",
                "prob_draw",
                "prob_team2_win",
                "predicted_result",
            ]
        ].head(30)
    )


if __name__ == "__main__":
    main()