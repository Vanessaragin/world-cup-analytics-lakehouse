from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"
MODELS_DIR = PROJECT_ROOT / "models"


FEATURE_COLUMNS = [
    "team1_games", "team1_wins", "team1_goals_for", "team1_goals_against",
    "team1_goal_difference", "team1_win_rate", "team1_goals_per_game",
    "team1_goals_against_per_game",
    "team2_games", "team2_wins", "team2_goals_for", "team2_goals_against",
    "team2_goal_difference", "team2_win_rate", "team2_goals_per_game",
    "team2_goals_against_per_game",
    "win_rate_diff", "goals_per_game_diff",
    "goals_against_per_game_diff", "goal_difference_diff", "experience_diff",
]


def build_match_features(team1: str, team2: str, teams: pd.DataFrame) -> pd.DataFrame:
    team1_data = teams[teams["team"] == team1].iloc[0]
    team2_data = teams[teams["team"] == team2].iloc[0]

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

    row["win_rate_diff"] = row["team1_win_rate"] - row["team2_win_rate"]
    row["goals_per_game_diff"] = row["team1_goals_per_game"] - row["team2_goals_per_game"]
    row["goals_against_per_game_diff"] = (
        row["team1_goals_against_per_game"] - row["team2_goals_against_per_game"]
    )
    row["goal_difference_diff"] = row["team1_goal_difference"] - row["team2_goal_difference"]
    row["experience_diff"] = row["team1_games"] - row["team2_games"]

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_match(team1: str, team2: str) -> None:
    teams = pd.read_csv(SILVER_DIR / "silver_teams.csv")
    model = joblib.load(MODELS_DIR / "match_result_model.pkl")

    features = build_match_features(team1, team2, teams)

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    print(f"\nPrediction: {team1} vs {team2}\n")

    for class_name, probability in zip(classes, probabilities):
        print(f"{class_name}: {probability:.2%}")


if __name__ == "__main__":
    predict_match("Brazil", "Germany")