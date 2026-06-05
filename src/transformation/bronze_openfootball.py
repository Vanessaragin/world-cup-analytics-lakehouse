from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "openfootball"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openfootball"


def extract_year_from_filename(file_path: Path) -> int:
    return int(file_path.stem.split("_")[1])


def load_json(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def process_file(file_path: Path) -> tuple[list, list]:
    data = load_json(file_path)

    year = extract_year_from_filename(file_path)
    cup_name = data.get("name")
    matches = data.get("matches", [])

    matches_rows = []
    goals_rows = []

    for match_index, match in enumerate(matches, start=1):
        match_id = f"{year}_{match_index}"

        score = match.get("score", {})
        full_time = score.get("ft", [None, None])
        half_time = score.get("ht", [None, None])

        team1 = match.get("team1")
        team2 = match.get("team2")

        matches_rows.append({
            "match_id": match_id,
            "year": year,
            "cup_name": cup_name,
            "round": match.get("round"),
            "date": match.get("date"),
            "team1": team1,
            "team2": team2,
            "team1_score": full_time[0],
            "team2_score": full_time[1],
            "team1_ht_score": half_time[0],
            "team2_ht_score": half_time[1],
            "group": match.get("group"),
            "ground": match.get("ground")
        })

        for goal in match.get("goals1", []):
            goals_rows.append({
                "match_id": match_id,
                "year": year,
                "team": team1,
                "opponent": team2,
                "player_name": goal.get("name"),
                "minute": goal.get("minute"),
                "goal_side": "team1"
            })

        for goal in match.get("goals2", []):
            goals_rows.append({
                "match_id": match_id,
                "year": year,
                "team": team2,
                "opponent": team1,
                "player_name": goal.get("name"),
                "minute": goal.get("minute"),
                "goal_side": "team2"
            })

    return matches_rows, goals_rows


def main() -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    all_matches = []
    all_goals = []

    json_files = sorted(RAW_DIR.glob("worldcup_*_raw.json"))

    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo JSON encontrado em {RAW_DIR}")

    for file_path in json_files:
        matches_rows, goals_rows = process_file(file_path)

        all_matches.extend(matches_rows)
        all_goals.extend(goals_rows)

        print(f"Processado: {file_path.name}")

    matches_df = pd.DataFrame(all_matches)
    goals_df = pd.DataFrame(all_goals)

    matches_df.to_csv(
        BRONZE_DIR / "bronze_matches.csv",
        index=False
    )

    goals_df.to_csv(
        BRONZE_DIR / "bronze_goals.csv",
        index=False
    )

    print("Bronze criada com sucesso.")
    print(f"Total de jogos: {len(matches_df)}")
    print(f"Total de gols: {len(goals_df)}")


if __name__ == "__main__":
    main()