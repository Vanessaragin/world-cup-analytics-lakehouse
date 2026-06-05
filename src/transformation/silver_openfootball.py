from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openfootball"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"


def create_silver_matches() -> pd.DataFrame:
    matches_df = pd.read_csv(BRONZE_DIR / "bronze_matches.csv")

    matches_df["date"] = pd.to_datetime(matches_df["date"])

    matches_df["total_goals"] = (
        matches_df["team1_score"] + matches_df["team2_score"]
    )

    matches_df["goal_difference"] = (
        matches_df["team1_score"] - matches_df["team2_score"]
    ).abs()

    matches_df["is_draw"] = (
        matches_df["team1_score"] == matches_df["team2_score"]
    )

    matches_df["winner"] = matches_df.apply(
        lambda row: row["team1"]
        if row["team1_score"] > row["team2_score"]
        else row["team2"]
        if row["team2_score"] > row["team1_score"]
        else "Draw",
        axis=1
    )

    return matches_df

def create_silver_goals() -> pd.DataFrame:
    goals_df = pd.read_csv(BRONZE_DIR / "bronze_goals.csv")

    goals_df["player_name"] = goals_df["player_name"].str.strip()
    goals_df["team"] = goals_df["team"].str.strip()
    goals_df["opponent"] = goals_df["opponent"].str.strip()

    goals_df["minute"] = pd.to_numeric(
        goals_df["minute"],
        errors="coerce"
    )

    goals_df = goals_df.dropna(subset=["player_name", "minute"])

    return goals_df

def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    silver_matches_df = create_silver_matches()

    silver_matches_df.to_csv(
        SILVER_DIR / "silver_matches.csv",
        index=False
    )

    silver_goals_df = create_silver_goals()

    silver_goals_df.to_csv(
        SILVER_DIR / "silver_goals.csv",
        index=False
    )

    print("Silver matches criada com sucesso.")
    print(silver_matches_df.head())


if __name__ == "__main__":
    main()