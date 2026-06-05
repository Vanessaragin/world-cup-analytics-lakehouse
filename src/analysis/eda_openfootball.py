from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "openfootball"


matches_df = pd.read_csv(
    SILVER_DIR / "silver_matches.csv"
)

goals_df = pd.read_csv(
    SILVER_DIR / "silver_goals.csv"
)


print("\n=== MATCHES ===")
print(matches_df.shape)

print("\n=== GOALS ===")
print(goals_df.shape)

print("\n=== COPAS ===")
print(
    matches_df.groupby("year")
    .size()
)
