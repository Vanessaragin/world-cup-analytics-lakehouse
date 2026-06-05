"""
Bronze transformation for OpenFootball World Cup data.

Creates:
1. bronze_matches.csv
2. bronze_goals.csv
3. bronze_team_match_stats.csv

Important rule:
- FT = score at 90 minutes
- ET = score after extra time
- P  = penalty shootout

For total match goals, we use:
- ET when it exists
- otherwise FT

Penalty shootout goals are NOT counted as match goals.
"""

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "openfootball"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openfootball"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def get_score_value(score: dict, key: str, index: int):
    """
    Safely gets one score value.

    Example:
        score = {"ft": [2, 1]}
        key = "ft"
        index = 0
        result = 2
    """
    values = score.get(key)

    if values is None:
        return None

    return values[index]


def sum_score(score: dict, key: str):
    """
    Sums both teams' score for a score type.

    Examples:
        ft = full time
        ht = half time
        et = extra time
        p  = penalties
    """
    values = score.get(key)

    if values is None:
        return None

    return values[0] + values[1]


def get_final_score(score: dict):
    """
    Gets the final match score excluding penalty shootout.

    Priority:
    1. Extra time score, if available
    2. Full time score, otherwise

    Penalties are not counted as match goals.
    """
    if score.get("et") is not None:
        return score.get("et")

    return score.get("ft")


def extract_year_from_file(file_path: Path) -> int:
    """
    Extracts the year from file name.

    Example:
        worldcup_1930_raw.json -> 1930
    """
    return int(file_path.stem.split("_")[1])


def normalize_goal(match_id, year, team, opponent, goal, goal_side):
    """
    Converts a real goal event into a standard row.
    """
    return {
        "match_id": match_id,
        "year": year,
        "team": team,
        "opponent": opponent,
        "player_name": goal.get("name"),
        "minute": goal.get("minute"),
        "offset": goal.get("offset"),
        "penalty": goal.get("penalty", False),
        "owngoal": goal.get("owngoal", False),
        "goal_side": goal_side,
        "goal_source": "event"
    }


def create_unknown_goal(match_id, year, team, opponent, goal_side):
    """
    Creates one inferred goal when the match has a score,
    but does not have goal event details.

    player_name = Unknown because the scorer is not available.
    """
    return {
        "match_id": match_id,
        "year": year,
        "team": team,
        "opponent": opponent,
        "player_name": "Unknown",
        "minute": None,
        "offset": None,
        "penalty": False,
        "owngoal": False,
        "goal_side": goal_side,
        "goal_source": "inferred_from_score"
    }


def process_goals(
    match,
    match_id,
    year,
    team1_score_final,
    team2_score_final,
    has_score
):
    """
    Creates goal rows for one match.

    Rule:
    1. If goals1/goals2 exist, use real events.
    2. If goals1/goals2 do not exist but score exists,
       infer goals from final score.
    3. If score does not exist, create no goals.
    """
    goals_rows = []

    team1 = match.get("team1")
    team2 = match.get("team2")

    goals1 = match.get("goals1")
    goals2 = match.get("goals2")

    has_goal_events = goals1 is not None or goals2 is not None

    if has_goal_events:
        goals1 = goals1 or []
        goals2 = goals2 or []

        for goal in goals1:
            goals_rows.append(
                normalize_goal(
                    match_id=match_id,
                    year=year,
                    team=team1,
                    opponent=team2,
                    goal=goal,
                    goal_side="team1"
                )
            )

        for goal in goals2:
            goals_rows.append(
                normalize_goal(
                    match_id=match_id,
                    year=year,
                    team=team2,
                    opponent=team1,
                    goal=goal,
                    goal_side="team2"
                )
            )

    elif (
        has_score
        and team1_score_final is not None
        and team2_score_final is not None
    ):
        for _ in range(int(team1_score_final)):
            goals_rows.append(
                create_unknown_goal(
                    match_id=match_id,
                    year=year,
                    team=team1,
                    opponent=team2,
                    goal_side="team1"
                )
            )

        for _ in range(int(team2_score_final)):
            goals_rows.append(
                create_unknown_goal(
                    match_id=match_id,
                    year=year,
                    team=team2,
                    opponent=team1,
                    goal_side="team2"
                )
            )

    return goals_rows


def create_team_stats_rows(
    match,
    match_id,
    year,
    cup_name,
    team1_score_final,
    team2_score_final,
    has_score
):
    """
    Creates one analytical row per team per match.

    Example:
        Brazil 2 x 1 France

    Output:
        Brazil | goals_for=2 | goals_against=1 | goal_difference=1
        France | goals_for=1 | goals_against=2 | goal_difference=-1
    """
    if (
        not has_score
        or team1_score_final is None
        or team2_score_final is None
    ):
        return []

    team1 = match.get("team1")
    team2 = match.get("team2")

    common_fields = {
        "match_id": match_id,
        "year": year,
        "cup_name": cup_name,
        "round": match.get("round"),
        "date": match.get("date")
    }

    return [
        {
            **common_fields,
            "team": team1,
            "opponent": team2,
            "goals_for": team1_score_final,
            "goals_against": team2_score_final,
            "goal_difference": team1_score_final - team2_score_final,
            "is_winner": team1_score_final > team2_score_final,
            "is_draw": team1_score_final == team2_score_final,
            "is_loser": team1_score_final < team2_score_final,
            "match_played": True
        },
        {
            **common_fields,
            "team": team2,
            "opponent": team1,
            "goals_for": team2_score_final,
            "goals_against": team1_score_final,
            "goal_difference": team2_score_final - team1_score_final,
            "is_winner": team2_score_final > team1_score_final,
            "is_draw": team2_score_final == team1_score_final,
            "is_loser": team2_score_final < team1_score_final,
            "match_played": True
        }
    ]


def process_worldcup_file(file_path: Path):
    """
    Processes one World Cup JSON file.
    """
    year = extract_year_from_file(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    cup_name = data.get("name")
    matches = data.get("matches", [])

    matches_rows = []
    goals_rows = []
    team_stats_rows = []

    for index, match in enumerate(matches, start=1):
        match_id = f"{year}_{index}"

        score = match.get("score", {})
        has_score = "score" in match

        final_score = get_final_score(score)

        team1_score_final = final_score[0] if final_score is not None else None
        team2_score_final = final_score[1] if final_score is not None else None

        team1_score_ft = get_score_value(score, "ft", 0)
        team2_score_ft = get_score_value(score, "ft", 1)

        status = match.get(
            "status",
            "scheduled" if not has_score else "played"
        )

        match_row = {
            "match_id": match_id,
            "match_num": match.get("num"),
            "year": year,
            "cup_name": cup_name,

            "round": match.get("round"),
            "date": match.get("date"),
            "time": match.get("time"),

            "team1": match.get("team1"),
            "team2": match.get("team2"),

            "team1_score_ft": team1_score_ft,
            "team2_score_ft": team2_score_ft,
            "total_goals_ft": sum_score(score, "ft"),

            "team1_score_ht": get_score_value(score, "ht", 0),
            "team2_score_ht": get_score_value(score, "ht", 1),
            "total_goals_ht": sum_score(score, "ht"),

            "team1_score_et": get_score_value(score, "et", 0),
            "team2_score_et": get_score_value(score, "et", 1),
            "total_goals_et": sum_score(score, "et"),

            "team1_score_final": team1_score_final,
            "team2_score_final": team2_score_final,
            "total_goals_final": (
                team1_score_final + team2_score_final
                if team1_score_final is not None
                and team2_score_final is not None
                else None
            ),

            "team1_score_p": get_score_value(score, "p", 0),
            "team2_score_p": get_score_value(score, "p", 1),
            "total_goals_p": sum_score(score, "p"),

            "group": match.get("group"),
            "ground": match.get("ground"),
            "status": status,

            "has_score": has_score,
            "has_goals_events": "goals1" in match or "goals2" in match
        }

        matches_rows.append(match_row)

        goals_rows.extend(
            process_goals(
                match=match,
                match_id=match_id,
                year=year,
                team1_score_final=team1_score_final,
                team2_score_final=team2_score_final,
                has_score=has_score
            )
        )

        team_stats_rows.extend(
            create_team_stats_rows(
                match=match,
                match_id=match_id,
                year=year,
                cup_name=cup_name,
                team1_score_final=team1_score_final,
                team2_score_final=team2_score_final,
                has_score=has_score
            )
        )

    return matches_rows, goals_rows, team_stats_rows


def main():
    """
    Runs the Bronze transformation.
    """
    all_matches = []
    all_goals = []
    all_team_stats = []

    json_files = sorted(RAW_DIR.glob("worldcup_*_raw.json"))

    for file_path in json_files:
        matches_rows, goals_rows, team_stats_rows = process_worldcup_file(
            file_path
        )

        all_matches.extend(matches_rows)
        all_goals.extend(goals_rows)
        all_team_stats.extend(team_stats_rows)

    matches_df = pd.DataFrame(all_matches)
    goals_df = pd.DataFrame(all_goals)
    team_stats_df = pd.DataFrame(all_team_stats)

    matches_df.to_csv(
        BRONZE_DIR / "bronze_matches.csv",
        index=False
    )

    goals_df.to_csv(
        BRONZE_DIR / "bronze_goals.csv",
        index=False
    )

    team_stats_df.to_csv(
        BRONZE_DIR / "bronze_team_match_stats.csv",
        index=False
    )

    print("Bronze created successfully.")
    print(f"JSON files processed: {len(json_files)}")
    print(f"Matches: {len(matches_df)}")
    print(f"Goals: {len(goals_df)}")
    print(f"Team match stats rows: {len(team_stats_df)}")

    if not matches_df.empty:
        total_goals_from_matches = matches_df["total_goals_final"].sum()
        total_goals_from_goals = len(goals_df)

        print("Data quality check:")
        print(f"Total goals from matches: {total_goals_from_matches}")
        print(f"Total goals from goals table: {total_goals_from_goals}")
        print(f"Difference: {total_goals_from_matches - total_goals_from_goals}")


if __name__ == "__main__":
    main()