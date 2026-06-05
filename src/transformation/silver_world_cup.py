"""
Silver transformation for World Cup analytics.

Input:
- data/bronze/openfootball/bronze_matches.csv
- data/bronze/openfootball/bronze_team_match_stats.csv
- data/bronze/openfootball/bronze_goals.csv

Output:
- data/silver/openfootball/silver_matches.csv
- data/silver/openfootball/silver_teams.csv
- data/silver/openfootball/silver_world_cups.csv
- data/silver/openfootball/silver_goals.csv
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openfootball"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"

SILVER_DIR.mkdir(parents=True, exist_ok=True)


COUNTRY_MAPPING = {
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Czechoslovakia": "Czech Republic",
}


def normalize_country(series: pd.Series) -> pd.Series:
    """
    Standardizes historical country names.
    """
    return series.replace(COUNTRY_MAPPING)


def create_silver_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an analytical match table.

    One row = one match.
    """
    df = matches.copy()

    df["team1_normalized"] = normalize_country(df["team1"])
    df["team2_normalized"] = normalize_country(df["team2"])

    df["match_played"] = df["has_score"] == True

    df["is_draw"] = (
        df["team1_score_final"] == df["team2_score_final"]
    )

    df["winner"] = None
    df["loser"] = None

    team1_won = df["team1_score_final"] > df["team2_score_final"]
    team2_won = df["team2_score_final"] > df["team1_score_final"]

    df.loc[team1_won, "winner"] = df.loc[team1_won, "team1_normalized"]
    df.loc[team1_won, "loser"] = df.loc[team1_won, "team2_normalized"]

    df.loc[team2_won, "winner"] = df.loc[team2_won, "team2_normalized"]
    df.loc[team2_won, "loser"] = df.loc[team2_won, "team1_normalized"]

    df["goal_difference"] = (
        df["team1_score_final"] - df["team2_score_final"]
    ).abs()

    selected_columns = [
        "match_id",
        "match_num",
        "year",
        "cup_name",
        "round",
        "date",
        "time",
        "team1",
        "team2",
        "team1_normalized",
        "team2_normalized",
        "team1_score_final",
        "team2_score_final",
        "total_goals_final",
        "winner",
        "loser",
        "is_draw",
        "goal_difference",
        "group",
        "ground",
        "status",
        "match_played",
        "has_goals_events",
    ]

    return df[selected_columns]


def create_silver_teams(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an analytical team table.

    One row = one national team.
    """
    df = team_stats.copy()

    df["team_normalized"] = normalize_country(df["team"])
    df["opponent_normalized"] = normalize_country(df["opponent"])

    silver_teams = (
        df.groupby("team_normalized")
        .agg(
            games=("match_id", "count"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_difference=("goal_difference", "sum"),
            wins=("is_winner", "sum"),
            draws=("is_draw", "sum"),
            losses=("is_loser", "sum"),
        )
        .reset_index()
        .rename(columns={"team_normalized": "team"})
    )

    silver_teams["win_rate"] = (
        silver_teams["wins"] / silver_teams["games"]
    )

    silver_teams["goals_per_game"] = (
        silver_teams["goals_for"] / silver_teams["games"]
    )

    silver_teams["goals_against_per_game"] = (
        silver_teams["goals_against"] / silver_teams["games"]
    )

    silver_teams = silver_teams.sort_values(
        by=["goals_for", "wins"],
        ascending=False,
    )

    return silver_teams


def create_silver_world_cups(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an analytical World Cup table.

    One row = one World Cup edition.
    """
    played_matches = matches[matches["has_score"] == True].copy()

    silver_world_cups = (
        played_matches.groupby(["year", "cup_name"])
        .agg(
            games=("match_id", "count"),
            goals=("total_goals_final", "sum"),
            avg_goals_per_game=("total_goals_final", "mean"),
            max_goals_in_match=("total_goals_final", "max"),
        )
        .reset_index()
    )

    return silver_world_cups


def create_silver_goals(goals: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a clean goals table.

    One row = one goal.
    """
    df = goals.copy()

    df["team_normalized"] = normalize_country(df["team"])
    df["opponent_normalized"] = normalize_country(df["opponent"])

    df["player_name"] = df["player_name"].fillna("Unknown")

    df["minute"] = pd.to_numeric(
        df["minute"],
        errors="coerce",
    )

    df["offset"] = pd.to_numeric(
        df["offset"],
        errors="coerce",
    )

    return df


def main() -> None:
    """
    Runs the Silver transformation.
    """
    matches = pd.read_csv(BRONZE_DIR / "bronze_matches.csv")
    team_stats = pd.read_csv(BRONZE_DIR / "bronze_team_match_stats.csv")
    goals = pd.read_csv(BRONZE_DIR / "bronze_goals.csv")

    silver_matches = create_silver_matches(matches)
    silver_teams = create_silver_teams(team_stats)
    silver_world_cups = create_silver_world_cups(matches)
    silver_goals = create_silver_goals(goals)

    silver_matches.to_csv(
        SILVER_DIR / "silver_matches.csv",
        index=False,
    )

    silver_teams.to_csv(
        SILVER_DIR / "silver_teams.csv",
        index=False,
    )

    silver_world_cups.to_csv(
        SILVER_DIR / "silver_world_cups.csv",
        index=False,
    )

    silver_goals.to_csv(
        SILVER_DIR / "silver_goals.csv",
        index=False,
    )

    print("Silver created successfully.")
    print(f"Matches: {silver_matches.shape}")
    print(f"Teams: {silver_teams.shape}")
    print(f"World Cups: {silver_world_cups.shape}")
    print(f"Goals: {silver_goals.shape}")


if __name__ == "__main__":
    main()