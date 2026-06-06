"""
Train a machine learning model to predict World Cup match results.

Input:
- data/modeling/training_dataset.csv

Output:
- models/match_result_model.pkl

Target:
- team1_win
- draw
- team2_win
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_DIR = PROJECT_ROOT / "data" / "modeling"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


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


def main() -> None:

    dataset = pd.read_csv(
        MODELING_DIR / "training_dataset.csv"
    )

    X = dataset[FEATURE_COLUMNS]

    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 50)
    print("MODEL TRAINED SUCCESSFULLY")
    print("=" * 50)
    print()

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    model_path = (
        MODELS_DIR
        / "match_result_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(
        f"Model saved: {model_path}"
    )
    print()

    print(
        "Target distribution:"
    )

    print(
        dataset["target"]
        .value_counts()
    )


if __name__ == "__main__":
    main()